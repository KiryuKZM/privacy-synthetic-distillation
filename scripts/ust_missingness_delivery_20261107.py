"""CPU-only closure audit and local delivery for the frozen background study.

Never trains, selects recipes, calls services, modifies frozen inputs, or reads
original protected evaluation cohorts. Repeated views share a scene cluster.
"""
from pathlib import Path
from decimal import Decimal
import argparse, ast, collections, hashlib, json, os, re, statistics, subprocess, time
R = Path(__file__).resolve().parents[1]
S = R / 'runs/2026-11-07-ust-missingness-repair-v1'
P = R / 'runs/2026-11-07-ust-joint-generator-repair-v1/student-utility-v1'
REPORT = R / 'reports/ust-missingness-repair-20261107.md'
FIELDS = ('coverage', 'demand', 'distance', 'waiting_time', 'charging_time')
PARSER = R / 'runs/2026-11-06-ust-retention-information-v1/independent-audit/real-own-runtime-and-closed-audit-v2.py'

def load(p):
    return json.loads(Path(p).read_text())

def pin(p):
    p = Path(p)
    return dict(path=str(p.relative_to(R)), bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

def put_report(text):
    tmp = REPORT.with_suffix('.tmp')
    tmp.write_text(text); tmp.replace(REPORT)

def sections(question):
    m=re.search(r'(?ms)^## Data\s*\n(.*?)(?=^## |\Z)',question);assert m
    data=m.group(1)
    blocks = re.split(r'(?m)^location ([1-4]):[ \t]*$', data)
    assert len(blocks) == 9
    obs = {}
    for j in range(1, len(blocks), 2):
        values = {}
        for field, value in re.findall(r'(?m)^\s*- ([a-z_]+): ([^\n]+)$', blocks[j+1]):
            value = value.strip()
            if value in ('NaN', 'NaNh', 'NaN h'):
                assert field in ('waiting_time', 'charging_time'); values[field] = None
            else:
                unit = 'm' if field == 'distance' else 'h' if field.endswith('_time') else ''
                m = re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))' + unit, value)
                assert m, (field, value)
                values[field] = Decimal(m.group(1))
        assert set(values) == set(FIELDS)
        obs[int(blocks[j])] = values
    options = {a: tuple(int(n) for n in re.findall(r'\d+', s))
               for a, s in re.findall(r'(?m)^([A-D])\.\s*([^\n]+)', question)}
    assert len(options) == 4 and all(sorted(v) == [1, 2, 3, 4] for v in options.values())
    return obs, options

def effective_map(original, joint):
    a, ao = sections(original); b, bo = sections(joint)
    rows_a = {tuple(a[k][f] for f in FIELDS): k for k in a}
    assert len(rows_a) == 4
    b_to_a = {k: rows_a[tuple(b[k][f] for f in FIELDS)] for k in b}
    assert set(b_to_a.values()) == {1, 2, 3, 4}
    bo_original = {k: tuple(b_to_a[v] for v in order) for k, order in bo.items()}
    mapping = {k: next(c for c, v in bo_original.items() if v == order) for k, order in ao.items()}
    assert set(mapping.values()) == set('ABCD')
    return mapping

def subgroup_plan():
    p = load(P / 'target-subgroup-plan.json')
    return {r['source_id']: r for r in p['features']['freshINIT']}

def stats(rows):
    return dict(n=len(rows), scenes=len({r['source_id'] for r in rows}),
                numeric_groups=len({r['group'] for r in rows}),
                correct=sum(r['correct'] for r in rows),
                compatible=sum(r['compatible_correct'] for r in rows),
                valid=sum(r['valid'] for r in rows), capped=sum(r['capped'] for r in rows))

def mechanism(q, prediction):
    obs, options = sections(q['question']); metric = q['metric']
    if prediction not in options:
        return dict(valid=False, nan_first=False, finite_order=False)
    seq = [obs[i][metric] for i in options[prediction]]
    nan_first = all(x is None for x in seq[:sum(x is None for x in seq)])
    finite = [x for x in seq if x is not None]
    return dict(valid=True, nan_first=nan_first,
                finite_order=all(x >= y for x, y in zip(finite, finite[1:])))

def transition(a, b):
    aa = {r['id']: r for r in a}; bb = {r['id']: r for r in b}
    assert aa.keys() == bb.keys()
    return dict(n=len(aa), before_correct=sum(x['correct'] for x in aa.values()),
                after_correct=sum(x['correct'] for x in bb.values()),
                lost=sum(x['correct'] and not bb[k]['correct'] for k,x in aa.items()),
                gained=sum(not x['correct'] and bb[k]['correct'] for k,x in aa.items()))

def prepare():
    assert not list((S / 'diagnostic').glob('*/raw/[0-9][0-9][0-9][0-9].json'))
    maps = {}; disagrees = 0
    for cohort in ('Select', 'Confirm'):
        qs = load(S / 'data' / (cohort + '-queries.local.json'))
        answers = load(S / 'data' / (cohort + '-answers.local.json'))
        groups = collections.defaultdict(dict)
        for q in qs:
            groups[(q['case_id'],q['family'],q['spelling'])][q['view']] = q
        for pair in groups.values():
            a,b = pair['original'],pair['joint']; m = effective_map(a['question'],b['question'])
            assert m[answers[a['id']]['answer']] == answers[b['id']]['answer']
            maps[b['id']] = dict(original_id=a['id'], old_to_new_letter=m)
            disagrees += m != b['old_to_new_letter']
    # Input-defined current train coverage, with the same historical Fit thresholds.
    old = load(P / 'target-subgroup-plan.json'); thresholds = load(R / old['thresholds']['path'])['positive_field_q01_q99']
    coverage = {}
    for name in load(S / 'data-summary.json')['datasets']:
        src = load(S / 'data' / name / 'source68.local.json'); c = collections.Counter()
        for row in src:
            obs = row['obs']; j = FIELDS.index(row['metric'])
            def tail(k):
                lo,hi = thresholds[k]
                return any(v[k] is not None and v[k]>0 and (v[k]<lo or v[k]>hi) for v in obs)
            c['target_tail'] += tail(j); c['any_tail'] += any(tail(k) for k in range(5))
            c['target_NaN'] += any(v[j] is None for v in obs)
        coverage[name] = dict(c)
    assert coverage['real_native']['target_tail']==5 and coverage['real_native']['target_NaN']==1
    # Data-independent paired aggregation and mechanism fixtures.
    a = [dict(id=str(i),correct=i<2) for i in range(4)]
    b = [dict(id=str(i),correct=i%2==0) for i in range(4)]
    assert transition(a,b) == dict(n=4,before_correct=2,after_correct=2,lost=1,gained=1)
    save(S/'effective-joint-maps.local.json', dict(maps=maps, inherited_map_metadata_mismatches=disagrees,
         explanation='Synthetic query old_to_new_letter was an unused identity placeholder. Reconstructed here from complete numeric rows and option permutations. Prompts, references, generated answers, raw scoring and training unchanged; the original metadata is preserved.'))
    save(S/'delivery-plan.json', dict(source=pin(__file__), original_parser=pin(PARSER),
         study_plan=pin(S/'study-plan.json'), effective_maps=pin(S/'effective-joint-maps.local.json'),
         target_subgroup_plan=pin(P/'target-subgroup-plan.json'), training_coverage=coverage,
         paired_bootstrap=dict(seed=2026110735,replicates=5000,clusters='original Dev82 numeric groups shared by all three fixed orders',
              interpretation='Conditional input variability for the mean of these three fixed order differences; not population training-seed uncertainty or equivalence.'),
         no_model_outputs_read=True, no_recipe_or_data_changes=True, API=0, Final_access=False))
    print('DELIVERY_PREPARED',len(maps),'maps;',disagrees,'metadata placeholders clarified',flush=True)

def parse_function():
    nodes=[x for x in ast.parse(PARSER.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='independent_prediction']
    assert len(nodes)==1
    namespace=dict(re=re)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(PARSER),'exec'),namespace)
    return namespace['independent_prediction']

def closed_phase(phase):
    folder=S/phase; terminal=load(folder/'terminal.json');assert terminal['status']=='COMPLETE'
    assert terminal['results']==pin(folder/'results.json')
    result=load(folder/'results.json'); protocol=load(folder/'protocol.json')
    pins=load(folder/'all-raw-frozen-before-score.json')['raw']
    assert len(pins)==protocol['total_raw']
    for p in pins: assert pin(R/p['path'])==p
    answer=load(folder/'answers.local.json'); parser=parse_function(); descriptors=subgroup_plan()
    allrows={}; extra={}; samples={}; effective=load(S/'effective-joint-maps.local.json')['maps']
    for key in protocol['units']:
        qs=load(folder/key/'queries.local.json'); saved=load(folder/key/'graded.local.json');assert len(qs)==len(saved)
        rows=[]; defects=[]; mechanisms=[]
        for i,(q,g) in enumerate(zip(qs,saved)):
            raw=load(folder/key/'raw'/f'{i:04d}.json');v=parser(raw['raw_generated'],raw['finish_reason']);a=answer[q['id']]
            assert raw['id']==g['id']==q['id'] and v==g['prediction']
            assert (v in 'ABCD' if isinstance(v,str) and len(v)==1 else False)==g['valid']
            assert g['correct']==bool(g['valid'] and v==a['answer'])
            assert g['compatible_correct']==bool(g['valid'] and v in a['compatible'])
            rows.append(g)
            if q['family']=='missing':
                me=mechanism(q,v);mechanisms.append(dict(id=q['id'],source_id=q['source_id'],**me))
                if not g['correct']:
                    defects.append(dict(query=q,reference=a,raw=raw,grade=g,mechanism=me))
        allrows[key]=rows;samples[key]=defects
        dev={}
        for view in ('original','joint'):
            rr=[x for x in rows if x['cohort']=='Dev' and x['view']==view]
            dev[view]=dict(overall=stats(rr),by_metric={m:stats([x for x in rr if x['metric']==m]) for m in ('distance','charging_time','waiting_time')},
                subgroups={f:stats([x for x in rr if descriptors[x['source_id']][f]]) for f in ('target_tail','target_NaN','displayed_tie','ambiguous')},
                finite_untied=stats([x for x in rr if not descriptors[x['source_id']]['target_NaN'] and not descriptors[x['source_id']]['displayed_tie']]))
        byid={x['id']:x for x in rows}; missing=[x for x in rows if x['family']=='missing']; worlds=collections.defaultdict(list)
        for row in missing: worlds[row['source_id']].append(row)
        dg=[x for x in rows if x['cohort'] in ('Select','Confirm')]
        equiv={}
        for family in ('missing','finite_twin','ordinary'):
            jj=[x for x in dg if x['view']=='joint' and x['family']==family]
            equiv[family]=dict(n=len(jj),equivariant=sum(x['valid'] and byid[effective[x['id']]['original_id']]['valid'] and effective[x['id']]['old_to_new_letter'].get(byid[effective[x['id']]['original_id']]['prediction'])==x['prediction'] for x in jj))
        extra[key]=dict(Dev=dev, diagnostic=stats(dg), missing=stats(missing),
            missing_by_metric={m:stats([x for x in missing if x['metric']==m]) for m in ('waiting_time','charging_time')},
            missing_by_spelling={s:stats([x for x in missing if x['spelling']==s]) for s in ('bare','unit','spaced')},
            missing_world_correct_counts=sorted(sum(x['correct'] for x in v) for v in worlds.values()),
            missing_all12_views_correct_worlds=sum(all(x['correct'] for x in v) for v in worlds.values()),
            mechanism=dict(n=len(mechanisms),invalid=sum(not x['valid'] for x in mechanisms),nan_first=sum(x['valid'] and x['nan_first'] for x in mechanisms),finite_order=sum(x['valid'] and x['finite_order'] for x in mechanisms),both=sum(x['valid'] and x['nan_first'] and x['finite_order'] for x in mechanisms)),
            equivariance=equiv,own=stats([x for x in rows if x['cohort']=='Own']))
        claimed=result['summaries'][key]['diagnostic']['all']
        assert all(extra[key]['diagnostic'][f]==claimed[f] for f in ('n','numeric_groups','correct','compatible','valid','capped'))
        assert extra[key]['diagnostic']['scenes']==claimed['source_worlds']
    return dict(result=pin(folder/'results.json'),phase=phase,raw=len(pins),grade_disagreements=0,models=extra),allrows,samples

def comparison_statistics(rows):
    import numpy as np
    bp=load(S/'delivery-plan.json')['paired_bootstrap']; seeds=load(S/'study-plan.json')['comparison']['order_seeds'];desc=subgroup_plan();out={}
    for view in ('original','joint'):
        out[view]={}
        for group in ('all','target_tail','target_NaN','displayed_tie','ambiguous'):
            pairs=[];differences=[]
            for seed in seeds:
                aa=[x for x in rows[f'real_order{seed}'] if x['cohort']=='Dev' and x['view']==view and (group=='all' or desc[x['source_id']][group])]
                bb=[x for x in rows[f'synthetic_order{seed}'] if x['cohort']=='Dev' and x['view']==view and (group=='all' or desc[x['source_id']][group])]
                t=transition(aa,bb); pairs.append(dict(seed=seed,**t));differences.append(t['after_correct']-t['before_correct'])
            entry=dict(pairs=pairs,synthetic_minus_real_correct_counts=differences,mean_count_difference=statistics.mean(differences),minimum=min(differences),maximum=max(differences))
            if group=='all':
                groups=sorted({x['group'] for x in aa});num=[];den=[]
                for g in groups:
                    delta=0
                    for seed in seeds:
                        rr=[x for x in rows[f'real_order{seed}'] if x['cohort']=='Dev' and x['view']==view and x['group']==g]
                        ss=[x for x in rows[f'synthetic_order{seed}'] if x['cohort']=='Dev' and x['view']==view and x['group']==g]
                        delta+=sum(x['correct'] for x in ss)-sum(x['correct'] for x in rr)
                    num.append(delta/len(seeds));den.append(sum(x['group']==g for x in aa))
                num=np.asarray(num);den=np.asarray(den);idx=np.random.default_rng(bp['seed']).integers(0,len(groups),(bp['replicates'],len(groups)));boot=num[idx].sum(axis=1)/den[idx].sum(axis=1)
                entry['conditional_input_bootstrap']=dict(groups=len(groups),questions=len(aa),difference=float(num.sum()/den.sum()),percentile95=np.quantile(boot,[.025,.975]).tolist(),interpretation=bp['interpretation'])
            out[view][group]=entry
    return out

def resources():
    completedfits=[];updates=raws=startedraw=0;fitseconds=0
    for phase in ('diagnostic','fallback','comparison'):
        folder=S/phase
        for p in folder.glob('*/fit-terminal.json'):
            t=load(p);completedfits.append(str(p.parent.relative_to(S)));fitseconds+=t['seconds']
        updates+=len(list(folder.glob('*/updates/[0-9][0-9][0-9][0-9].json')))
        raws+=len(list(folder.glob('*/raw/[0-9][0-9][0-9][0-9].json')))
        startedraw+=len(list(folder.glob('*/raw/[0-9][0-9][0-9][0-9].started.json')))
    processes=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv=p.read_bytes().decode(errors='replace').split('\0'); command=' '.join(argv)
            if Path(argv[0]).name.startswith('python') and str(R/'scripts/ust_missingness_run_20261107.py') in argv:
                processes.append(dict(PID=int(p.parent.name),command=command))
        except (OSError,ValueError): pass
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
    return dict(epoch=time.time(),completed_fits=completedfits,completed_updates=updates,microexamples=updates*4,raw_completed=raws,raw_started=startedraw,fit_seconds=fitseconds,owned_GPU_pipeline_processes=processes,gpu_snapshot=gpu.stdout.strip(),new_API_calls=0,new_API_CNY=0,cumulative_API_CNY=0,API_cap_CNY=500,remaining_API_CNY=500,cloud_charges='user managed/excluded by explicit user instruction',Final_access=False,reporter_PID=os.getpid())

def headline_status():
    if (S/'terminal.json').exists():return load(S/'terminal.json')['status']
    launch=load(S/'background-launch.json'); p=Path('/proc')/str(launch['PID'])/'cmdline'
    try: command=p.read_bytes().decode(errors='replace')
    except OSError:command=''
    return 'RUNNING' if 'ust_missingness_run_20261107.py' in command else 'STOPPED_BEFORE_TERMINAL_CHECK_FAILURE_LOGS'

def render(status, audits, paired, resource):
    lines=['# UST 固定生成器：缺失值修复与三个训练顺序对照','',f'当前状态：`{status}`。统计时间 UTC：{time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime())}。','',
    '生成器参数与校准表保持不变。旧合成68题没有目标字段缺失题；真实68题只有1题。上轮三道真实缺失题的有限值替换改善明显，而仅改 NaNh 单位写法不足以消除错误。本轮用同场景的2×2对照检验缺失覆盖和显式规则，随后才检验真实／合成训练对比。','',
    '## 已冻结设置','',
    '四个诊断模型：目标缺失0或8场景 × 原生表示或明确NaN排序规则。0覆盖控制仅把被问字段NaN换成保持同一排序答案的有限大值，其余数值、选项、答案相同；这是数学控制，不是物理有效性证明。每组68题（距离42、充电14、等待12），四次选项轮换、两遍训练，136次Adam更新、544个微样本。所有模型使用相同原始初始化、学习率0.0001、固定8条真实示例和原有确定性计算设置。','',
    'Select与预留合成Confirm各256题：12个缺失场景产生144视图，另有48个有限值对应题、32个普通场景的64视图。缺失视图跨两个时间目标、位置/选项重排及NaN、NaNh、NaN h三种写法。144视图仅来自12场景，不能按144独立样本计算可信度。两集合及训练数值场景不相交。此处Confirm是新建合成诊断，不是原有封存的真实Confirm。','',
    '选择只用Select：缺失≥130/144，原生NaN≥44/48，有限值≥101/112，且相对同表示0覆盖控制有限题损失≤5。先选8场景原生，再选8场景规则；均未通过才运行预定16场景规则方案。仍未通过则停止对比并保留失败。原Dev和合成Confirm成绩不参与选择。','',
    '通过后，真实68与选定合成68分别运行三个配对训练顺序（2026110731、732、733），共六模型，初始化不变。原版Dev155题为主，同步重排155题为辅助；原题文字和发布答案不变。8道原题存在多个弱排序兼容选项，另报兼容答案分数。合成Confirm只在方案冻结后评分。Final3670及其他原有受保护评测材料继续封存。','',
    '这属于最终答案监督微调，不是IPD。比较是在共同8条真实示例条件下替换68条目标，不是零真实数据，也不是在真实68条上追加合成题。生成器拟合另用原有145个真实场景。合成组对缺失场景有意增加覆盖，真实组仍为原68题；不应声称它们的缺失比例相同。','',
    '## 执行与结果','']
    if not audits:lines += ['尚未完成第一阶段统一评分；目前不能宣布缺失值已修好或合成数据可替代真实数据。','']
    for phase,audit in audits.items():
        lines += [f'### {phase}','',f'闭合原始输出 {audit["raw"]} 条；独立重读解析与参考评分差异为0。','',
            '| 模型 | 缺失视图 | 12视图全对的缺失场景 | 有限值视图 | NaN位置正确 | 其余有限值顺序正确 | 已见题 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for key,x in audit['models'].items():
            n=x['missing']['n']; finite_n=x['diagnostic']['n']-n;finite_c=x['diagnostic']['correct']-x['missing']['correct']; me=x['mechanism'];own=x['own']
            lines.append(f'| {key} | {x["missing"]["correct"]}/{n} | {x["missing_all12_views_correct_worlds"]}/12 | {finite_c}/{finite_n} | {me["nan_first"]}/{n} | {me["finite_order"]}/{n} | {own["correct"]}/{own["n"]} |')
        lines += ['','| 模型 | 原Dev | 重排Dev | 目标数值尾部 原/重排 | 目标缺失 原/重排 | 原Dev兼容答案 |','|---|---:|---:|---:|---:|---:|']
        for key,x in audit['models'].items():
            if not x['Dev']['original']['overall']['n']:continue
            a,b=x['Dev']['original'],x['Dev']['joint'];t=a['subgroups']['target_tail'];u=b['subgroups']['target_tail'];na=a['subgroups']['target_NaN'];nb=b['subgroups']['target_NaN']
            lines.append(f'| {key} | {a["overall"]["correct"]}/155 | {b["overall"]["correct"]}/155 | {t["correct"]}/{t["n"]} · {u["correct"]}/{u["n"]} | {na["correct"]}/{na["n"]} · {nb["correct"]}/{nb["n"]} | {a["overall"]["compatible"]}/155 |')
        lines.append('')
    if (S/'recipe-decision.json').exists():
        d=load(S/'recipe-decision.json');lines += [f'冻结选择：`{d["selected"]}`；资格通过：{d["qualified"]}。详细每项门槛结果见 recipe-decision.json。','']
    if paired:
        for view,p in paired.items():
            a=p['all'];ci=a['conditional_input_bootstrap'];lines += [f'{view} 的三次“合成−真实”答对题数差为 {a["synthetic_minus_real_correct_counts"]}；均值 {a["mean_count_difference"]:.2f}/155。按共同82数值组重采样的条件输入区间为 [{100*ci["percentile95"][0]:.2f}, {100*ci["percentile95"][1]:.2f}] 个百分点。这个区间条件于三个固定顺序，不是训练种子总体不确定性，也不是等价检验。','']
    lines += ['## 结论边界与后续判断','',
    '原版Dev是已用于开发的数据，不是独立最终验收。三个顺序复用同155题，不能说成465道独立真实题。目标数值尾部仅14题；真实目标缺失仅3题。旧统计离群候选仅2题、1个数值组，且没有人工确认真实异常，无法可靠估计真实outlier保留。各题型、写法、缺失错误类型、逐题原始输出和场景分组统计保存在本轮分析目录。','',
    '即使Select通过，也应继续看未参与选方案的合成Confirm是否稳定，以及原Dev三个配对顺序是否一致；只在合成缺失视图改善而真实缺失未改善时，应报告分布迁移缺口。若资格未通过，优先按保留的NaN位置/有限值排序错误分解继续诊断，不把失败阶段当作修复完成。即使所有开发检查通过，仍缺独立真实验收和更充足、可信的尾部/outlier证据，不能宣称全UST替代。','',
    '## 资源与产物','',f'完成训练 {len(resource["completed_fits"])} 个，已保存Adam更新 {resource["completed_updates"]} 次，原始输出 {resource["raw_completed"]} 条。API本轮和累计均0元，500元额度剩余500元；云机计费由用户管理。当前本任务GPU流水线进程数 {len(resource["owned_GPU_pipeline_processes"])}。GPU快照：{resource["gpu_snapshot"]}。','',
    f'本轮目录：`{S}`。配置 study-plan.json；固定数据 data/；运行诊断 diagnostic/；必要时 fallback/；三顺序 comparison/；日志各阶段目录；模型各单元 checkpoints/step0136；原始输出 raw/；独立分析 analysis/；最终资源记录 delivery-resources.json。原数据、旧生成器及旧失败结果保留。','',
    '生成诊断题的old_to_new_letter元数据曾保留无效的恒等占位；独立检查从数值行和选项重建真实映射，保存在effective-joint-maps.local.json。原映射未用于训练、提示、参考答案或得分，因此不影响本轮模型与正确率；等变性分析使用重建映射。','']
    return '\n'.join(lines)

def finish():
    p=load(S/'delivery-plan.json');assert p['source']==pin(__file__) and p['original_parser']==pin(PARSER)
    assert p['effective_maps']==pin(S/'effective-joint-maps.local.json') and p['target_subgroup_plan']==pin(P/'target-subgroup-plan.json')
    status=headline_status();assert status!='RUNNING';audits={};allrows={};failures={}
    for phase in ('diagnostic','fallback','comparison'):
        if (S/phase/'terminal.json').exists():
            audits[phase],allrows[phase],failures[phase]=closed_phase(phase)
            save(S/'analysis'/(phase+'-independent-review.json'),audits[phase]);save(S/'analysis'/(phase+'-missing-failure-cases.local.json'),failures[phase])
    paired=comparison_statistics(allrows['comparison']) if 'comparison' in allrows else {}
    if paired:save(S/'analysis/three-paired-orders.json',paired)
    if 'diagnostic' in allrows:
        rr=allrows['diagnostic'];contrasts={}
        for a,b in [('coverage0_native','coverage8_native'),('coverage0_rule','coverage8_rule'),('coverage0_native','coverage0_rule'),('coverage8_native','coverage8_rule')]:
            contrasts[a+'__'+b]={f:transition([x for x in rr[a] if x['cohort']=='Select' and (x['family']=='missing')==(f=='missing')],[x for x in rr[b] if x['cohort']=='Select' and (x['family']=='missing')==(f=='missing')]) for f in ('missing','finite')}
        save(S/'analysis/factorial-paired-contrasts.json',contrasts)
    resource=resources();save(S/'delivery-resources.json',resource)
    put_report(render(status,audits,paired,resource));save(S/'delivery-terminal.json',dict(status='REPORT_AND_AVAILABLE_CLOSED_PHASES_COMPLETE',pipeline_status=status,report=pin(REPORT),resources=pin(S/'delivery-resources.json'),source=pin(__file__),epoch=time.time(),API=0,Final_access=False))
    with (R/'progress.md').open('a') as f:f.write(f'\n\n## {time.strftime("%Y-%m-%d %H:%M UTC",time.gmtime())} — Missingness background delivery\nPipeline {status}; completed fits {len(resource["completed_fits"])}, saved updates {resource["completed_updates"]}, raw {resource["raw_completed"]}. Independently reparsed each closed phase. Report reports/ust-missingness-repair-20261107.md; no Final or API access. No unknown process stopped. See delivery-resources.json for remaining process/GPU snapshot.\n')
    print('DELIVERY_COMPLETE',status,flush=True)

def watch():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    save(S/'delivery-started.json',dict(PID=os.getpid(),epoch=time.time(),independent_session=os.getsid(0)==os.getpid(),source=pin(__file__)))
    launch=load(S/'background-launch.json');limit=launch['epoch']+load(S/'study-plan.json')['runtime']['maximum_seconds']+300
    while headline_status()=='RUNNING' and time.time()<limit:time.sleep(15)
    if headline_status()=='RUNNING':
        save(S/'delivery-watch-timeout.json',dict(epoch=time.time(),reason='Training controller still active after own hard bound; reporter stops, does not stop unknown processes.'));return
    # Let the controller exit after writing terminal; collect an honest resource snapshot.
    for _ in range(20):
        if not resources()['owned_GPU_pipeline_processes']:break
        time.sleep(1)
    finish()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','watch','finish','status']);a=p.parse_args()
    if a.action=='status':print(json.dumps(resources(),ensure_ascii=False,indent=2))
    else:
        try:globals()[a.action]()
        except BaseException as e:
            save(S/('delivery-failure-'+str(time.time_ns())+'.json'),dict(type=type(e).__name__,error=str(e),epoch=time.time()));raise
