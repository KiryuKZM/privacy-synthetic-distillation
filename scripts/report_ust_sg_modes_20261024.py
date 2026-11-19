#!/usr/bin/env python3
"""Consolidate already-scored UST/SG results; never open gold or run models."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sqlite3

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/2026-10-24-ust100-sg60-modes'
REPORT=ROOT/'reports/ust100-sg60-modes-2026-10-24/results.md'
COUNTS={'USTBench':100,'SGQA':60}
MODES=('off','on')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def bind(p):return {'path':Path(p).relative_to(ROOT).as_posix(),'sha256':sha(p)}
def atomic(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(data,encoding='utf-8');os.replace(tmp,p)
def js(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
def localpath(p):
    path=Path(p)
    return path if path.is_absolute() else ROOT/path

def root_data_review():
    path=RUN/'root-data-review.json'
    if not path.exists():return None
    review=load(path)
    if review.get('status')!='FINAL':return None
    if set(review.get('tasks',{}))!=set(COUNTS):raise ValueError('Root data review must declare both tasks')
    for value in review['tasks'].values():
        if value.get('status') not in ('PASS','BLOCKED') or not isinstance(value.get('reason',''),str):
            raise ValueError('Unknown root data review status')
        if value['status']=='BLOCKED' and not value.get('reason','').strip():raise ValueError('Blocked task needs a recorded reason')
    return review

def approved_smaller_tasks(required=False):
    path=RUN/'conditional-08b/selection.json'
    if not path.exists():
        if required:raise ValueError('Small model outputs require an immutable approved selection')
        return None
    selection=load(path)
    tasks=selection.get('tasks',[])
    if (not tasks or len(tasks)!=len(set(tasks)) or tasks!=[t for t in COUNTS if t in tasks]
        or selection.get('model')!='Qwen/Qwen3.5-0.8B'
        or selection.get('revision')!='2fc06364715b967f1860aea9cf38778875588b17'
        or selection.get('root_preapproved_conditional_route') is not True
        or selection.get('both_modes') is not True):
        raise ValueError('Small model declared selection differs from approved route')
    required_paths={ROOT/'scripts/eval_ust_sg_qwen_modes_20261024.py',ROOT/'scripts/report_ust_sg_modes_20261024.py',
        ROOT/'scripts/finalize_ust_sg_conditional_08b_20261024.py',RUN/'local-qwen-manifest.json',RUN/'root-data-review.json'}
    bindings={localpath(name):expected for name,expected in selection.get('source_bindings',{}).items()}
    if not required_paths<=set(bindings):raise ValueError('Small model selection lacks required source bindings')
    for source,expected in bindings.items():
        if not source.resolve().is_relative_to(ROOT.resolve()) or sha(source)!=expected:
            raise ValueError('Small model selection source binding changed')
    window=load(RUN/'local-qwen-original2b/queue-start.json')
    if (selection.get('deadline_epoch')!=window['deadline_epoch'] or not window.get('only_local')
        or window.get('api_calls')!=0):raise ValueError('Small model selection changed the original local window')
    gatepath=RUN/'smaller-model-trigger.json'
    if sha(gatepath)!=selection.get('trigger_sha256'):raise ValueError('Small model trigger binding changed')
    packet=load(RUN/'execution-packet.json')
    ds=read_scores(ROOT/packet['run_dir'],'DS')
    original=read_scores(RUN/'local-qwen-original2b','Qwen2B')
    gate=trigger(ds,original)
    review=root_data_review()
    if review is None or selection.get('data_review_sha256')!=sha(RUN/'root-data-review.json'):
        raise ValueError('Small model selection needs the unchanged final root data review')
    eligible=[t for t in gate['qualifying_tasks'] if review['tasks'][t]['status']=='PASS']
    blocked={t:v.get('reason','') for t,v in review['tasks'].items() if v['status']=='BLOCKED'}
    if (gate['status']!='FROZEN_TRIGGER_EVALUATED' or gate!=load(gatepath) or tasks!=eligible
        or selection.get('numerical_qualifying_tasks')!=gate['qualifying_tasks']
        or selection.get('data_eligible_tasks')!=eligible or selection.get('blocked_tasks')!=blocked):
        raise ValueError('Small model selection does not match the numeric and data-review gates')
    return tasks

def ds_length_counts(frozen):
    """Derived reporting only; original immutable score files stay unchanged."""
    packet=load(RUN/'execution-packet.json')
    reusepath=localpath(packet['reuse']['path'])
    if sha(reusepath)!=packet['reuse']['sha256']:raise ValueError('DS reuse binding changed')
    records=[load(localpath(spec['path'])) for spec in frozen['predictions']]
    for line in reusepath.read_text(encoding='utf-8').splitlines():
        row=json.loads(line);spec=row['original_prediction'];path=localpath(spec['path'])
        if sha(path)!=spec['sha256']:raise ValueError('Original reused DS prediction changed')
        original=load(path)
        if original.get('body_sha256')!=row['body_sha256'] or original.get('identity')!=row['identity']:
            raise ValueError('Reused DS identity/body differs')
        records.append(dict(original,task=row['task'],arm=row['arm']))
    counts={f'{t}/{m}':{'total':0,'length_capped':0} for t in COUNTS for m in MODES}
    for record in records:
        arm=record['arm'];mode={'adapted-off':'off','adapted-on-high':'on'}.get(arm)
        key=f"{record['task']}/{mode}"
        if key not in counts:raise ValueError('Out-of-scope DS mode/task')
        counts[key]['total']+=1
        counts[key]['length_capped']+=record.get('finish_reason')=='length' or record.get('status')=='LENGTH_TRUNCATION'
    if any(value['total']!=COUNTS[key.split('/')[0]] for key,value in counts.items()):
        raise ValueError('DS derived truncation denominator differs')
    return counts

def read_scores(directory,model):
    scorepath=next((directory/name for name in ('scores.json','score.json') if (directory/name).exists()),None)
    progress={'saved_predictions':len([p for p in directory.glob('*/*/responses/*.json') if not p.name.endswith('.started.json')])}
    if model=='DS':progress['saved_new_predictions']=len(list(directory.glob('predictions/*.json')))
    for name in ('queue-status.json','global-stop.json'):
        if (directory/name).exists():progress[name]=load(directory/name)
    scope=list(COUNTS)
    if model=='Qwen0.8B':
        scope=approved_smaller_tasks(required=scorepath is not None or (directory/'preflight.json').exists())
    task_status={t:('NOT_TRIGGERED' if scope is not None and t not in scope else 'PENDING_UNSCORED') for t in COUNTS}
    if model=='Qwen0.8B':
        review=root_data_review()
        if review:
            for task,value in review['tasks'].items():
                if value['status']=='BLOCKED':task_status[task]='NOT_RUN_DATA_REVIEW'
    if scorepath is None:return {'status':'PENDING_UNSCORED','progress':progress,'scores':{},'scope_tasks':scope,'task_status':task_status}
    result=load(scorepath);freeze=directory/'predictions-frozen.json'
    if not freeze.exists():raise ValueError('Score exists without frozen predictions')
    frozen=load(freeze)
    expected_hash=result.get('freeze_sha256',result.get('prediction_freeze_sha256'))
    if expected_hash!=sha(freeze) or not frozen.get('complete'):raise ValueError('Score/freeze binding mismatch')
    if model=='DS':
        if result['packet_sha256']!=sha(RUN/'execution-packet.json') or len(frozen['predictions'])!=200:raise ValueError('DS exact package/freeze mismatch')
    elif model=='Qwen2B':
        if result['manifest_sha256']!=sha(RUN/'local-qwen-manifest.json') or frozen['expected']!=320 or len(frozen['predictions'])!=320:raise ValueError('Qwen2B full320 manifest/freeze mismatch')
    elif model=='Qwen0.8B':
        preflight=directory/'preflight.json';audit=load(preflight)
        expected=2*sum(COUNTS[t] for t in scope)
        if (audit.get('tasks')!=scope or audit.get('manifest_sha256')!=sha(RUN/'local-qwen-manifest.json')
            or result.get('manifest_sha256')!=audit['manifest_sha256']
            or frozen.get('manifest_sha256')!=audit['manifest_sha256']
            or frozen.get('preflight_sha256')!=sha(preflight)
            or frozen.get('expected')!=expected or len(frozen['predictions'])!=expected):
            raise ValueError('Small model full declared task scope/preflight/freeze differs')
    observed_small=set()
    for spec in frozen['predictions']:
        if sha(localpath(spec['path']))!=spec['sha256']:raise ValueError('Saved frozen prediction changed')
        if model=='Qwen0.8B':
            record=load(localpath(spec['path']))
            identity=(record.get('task'),record.get('thinking'),str(record.get('id')))
            if identity in observed_small or record.get('preflight_sha256')!=sha(preflight) or record.get('manifest_sha256')!=audit['manifest_sha256']:
                raise ValueError('Small model duplicate/unbound prediction')
            observed_small.add(identity)
    if model=='Qwen0.8B':
        expected_small={(t,m,str(row['id'])) for t in scope for m in MODES for row in audit['audit'][t][m]}
        if len(expected_small)!=expected or observed_small!=expected_small:
            raise ValueError('Small model prediction membership differs from the bound preflight')
    summaries={}
    for key,value in result['scores'].items():
        normalized=key.replace('/adapted-on-high','/on').replace('/adapted-off','/off')
        n=value['total'];c=value['correct'];v=value['valid']
        if not 0<=c<=v<=n:raise ValueError('Invalid score counts')
        if normalized.count('/')==1 and n!=COUNTS[normalized.split('/')[0]]:raise ValueError('Fixed denominator changed')
        if model=='Qwen0.8B' and normalized not in {f'{t}/{m}' for t in scope for m in MODES}:
            raise ValueError('Small model score outside approved task scope')
        summaries[normalized]=dict(value)
    if model=='DS':
        lengths=ds_length_counts(frozen)
        for key,value in lengths.items():
            if key in summaries:summaries[key]['length_capped']=value['length_capped']
    complete=all(f'{task}/{mode}' in summaries for task in scope for mode in MODES)
    for task in scope:
        task_status[task]='FROZEN_SCORED' if all(f'{task}/{m}' in summaries for m in MODES) else 'PARTIAL_SCORED'
    return {'status':'FROZEN_SCORED' if complete else 'PARTIAL_SCORED','scores':summaries,
        'scope_tasks':scope,'task_status':task_status,'score_binding':bind(scorepath),'freeze_binding':bind(freeze),'progress':progress}

def trigger(ds,qwen):
    if ds['status']!='FROZEN_SCORED' or qwen['status']!='FROZEN_SCORED':
        return {'status':'PENDING_FULL_FROZEN_SCORES','download_authorized':False,'qualifying_tasks':[]}
    comparisons={};qualifying=[]
    for task in COUNTS:
        passes=[]
        for mode in MODES:
            key=f'{task}/{mode}';d=ds['scores'][key];q=qwen['scores'][key]
            acc=Fraction(q['correct'],q['total']);valid=Fraction(q['valid'],q['total']);gap=Fraction(d['correct'],d['total'])-acc
            checks={'accuracy_ge_80pct':acc>=Fraction(4,5),'valid_ge_95pct':valid>=Fraction(19,20),'ds_minus_qwen_le_10pp':gap<=Fraction(1,10)}
            comparisons[key]={'qwen_correct':q['correct'],'qwen_valid':q['valid'],'total':q['total'],'ds_correct':d['correct'],'ds_total':d['total'],'ds_minus_qwen_gap_pp':float(100*gap),'checks':checks,'pass':all(checks.values())}
            passes.append(all(checks.values()))
        if any(passes):qualifying.append(task)
    return {'status':'FROZEN_TRIGGER_EVALUATED','download_authorized':False,'root_selection_required':True,
        'rule':'At least one matched mode per task: Qwen accuracy>=80%, valid>=95%, DS minus Qwen accuracy<=10pp; negative gap allowed. Qualifying tasks run smaller model in both modes.',
        'qualifying_tasks':qualifying,'comparisons':comparisons,'ds_scores':ds['score_binding'],'qwen2b_scores':qwen['score_binding'],
        'gate_manifest':bind(RUN/'local-qwen-manifest.json')}

def budget(packet):
    path=ROOT/packet['budget']['path']
    if os.name=='nt':return {'status':'UNAVAILABLE_REQUIRES_NATIVE_LINUX_READONLY_SQL','observed_epoch':None}
    try:
        with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=5) as db:
            db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
            rows=db.execute('SELECT COUNT(*),COALESCE(SUM(pico),0),COALESCE(SUM(1-usage_known),0),COALESCE(SUM(1-settled),0),COALESCE(SUM(input_tokens),0),COALESCE(SUM(output_tokens),0) FROM calls').fetchone()
            package=db.execute("SELECT COUNT(*),COALESCE(SUM(pico),0),COALESCE(SUM(1-usage_known),0),COALESCE(SUM(1-settled),0) FROM calls WHERE tag LIKE ?",('%/screen100-modes-20261024/%',)).fetchone()
        return {'status':'LIVE_READONLY_SQL','all_route_calls':rows[0],'all_route_conservative_usd':rows[1]/10**12,'unknown_usage':rows[2],'inflight':rows[3],'accounted_input_tokens':rows[4],'accounted_output_tokens':rows[5],
            'package_calls':package[0],'package_conservative_usd':package[1]/10**12,'package_unknown_usage':package[2],'package_inflight':package[3],
            'scope':'Canonical task-sweep route only; includes reservations and unknown full reserves; not provider invoice or all historical project cost.'}
    except sqlite3.Error as exc:return {'status':'READONLY_SQL_UNAVAILABLE','error':str(exc)}

def fraction(c,n):return f'{c}/{n} ({100*c/n:.2f}%)'
def render(status):
    lines=['# UST100 / SGQA60 模式对比', '',f"更新时间：{status['updated_utc']}。只汇总已冻结并评分的结果；未评分的响应不计算正确率。",'',
        'USTBench：根据完整公开城市时空数据、问题与选项，回答理解、规划、反思题；100题分别为34/33/33。保留旧60题及其顺序，新增40题来自既有Screen中未使用的40个独立组件；合计99组件。', '',
        'TSG-SGQA：根据有序场景图与问题作答；原60题保持不变，12个场景图分组。两任务均使用冻结的8个公开Fit示例及相同输入。', '',
        '这是已暴露Screen上的模式/模型比较；旧DS关闭思考响应复用，存在提供方时间混杂。关闭/开启思考分别使用256/8192输出上限；不是等计算量思考因果实验、独立Final评估或隐私证据。','',
        '| 模型 | 任务 | 模式 | 正确/总数 | 有效/总数 | 截断 | 输入/输出token | 生成秒 | 状态 |',
        '|---|---|---|---|---|---|---|---|---|']
    for model,record in status['models'].items():
        for task,n in COUNTS.items():
            for mode in MODES:
                score=record['scores'].get(f'{task}/{mode}')
                if score:
                    cols=[fraction(score['correct'],score['total']),fraction(score['valid'],score['total']),str(score.get('length_capped','未记录')),f"{score.get('input_tokens','—')}/{score.get('output_tokens','—')}",str(round(score['generation_seconds'],2)) if 'generation_seconds' in score else '—',record['status']]
                elif record.get('task_status',{}).get(task)=='NOT_RUN_DATA_REVIEW':cols=['数据审查阻止运行','数据审查阻止运行','—','—','—','NOT_RUN_DATA_REVIEW']
                elif record.get('task_status',{}).get(task)=='NOT_TRIGGERED':cols=['未触发','未触发','—','—','—','NOT_TRIGGERED']
                else:cols=['待评分','待评分','—','—','—',record['status']]
                lines.append('| '+' | '.join([model,task,mode]+cols)+' |')
    lines+=['','UST旧60 / 新40（仅使用DS冻结评分中的分组结果）：','', '| 模式 | 分组 | 正确/总数 | 有效/总数 |','|---|---|---|---|']
    for mode in MODES:
        for cohort in ('old60','new40'):
            score=status['models']['DS']['scores'].get(f'USTBench/{mode}/{cohort}')
            lines.append(f"| {mode} | {cohort} | "+(f"{fraction(score['correct'],score['total'])} | {fraction(score['valid'],score['total'])}" if score else '待评分 | 待评分')+' |')
    gate=status['smaller_model_trigger'];b=status['budget']
    lines+=['',f"0.8B条件：{gate['status']}；至少一个匹配模式满足数值门槛的任务：{', '.join(gate['qualifying_tasks']) or '尚无'}。每任务至少一个匹配模式达到门槛即触发该任务的小模型双模式评估。门槛为Qwen2B正确率≥80%、有效率≥95%、同模式DS减Qwen差距≤10个百分点（负差允许）。还须通过已冻结的主线程数据审查；NOT_RUN_DATA_REVIEW表示该任务因记录的数据问题未运行小模型，不改变原始数值结果。主线程冻结数值门槛与数据审查共同允许的任务后才能下载。",'',f"费用状态：{b['status']}。"]
    if b['status']=='LIVE_READONLY_SQL':lines.append(f"本包已记账{b['package_calls']}/200次，保守费用${b['package_conservative_usd']:.8f}；整个既有任务路线{b['all_route_calls']}次/${b['all_route_conservative_usd']:.8f}，未知用量{b['unknown_usage']}，进行中{b['inflight']}。费用包括未知请求的完整预留，并非账单或全项目历史费用。")
    lines+=['','进度（响应数量不表示正确率）：','', '```json',json.dumps({k:v['progress'] for k,v in status['models'].items()},ensure_ascii=False,indent=2),'```','']
    return '\n'.join(lines)

def self_test():
    def rec(c=80,v=95,ds=False):
        return {'status':'FROZEN_SCORED','score_binding':{'path':'fake','sha256':'fake'},'scores':{f'{t}/{m}':{'correct':(c*n)//100,'valid':(v*n)//100,'total':n} for t,n in COUNTS.items() for m in MODES}}
    pending={'status':'PENDING_UNSCORED'}
    assert trigger(pending,pending)['qualifying_tasks']==[]
    # Stub binding only: no manifest/gold/model/API access in fixtures.
    global bind
    original=bind;bind=lambda p:{'path':'fake','sha256':'fake'}
    try:
        assert trigger(rec(90,100),rec())['qualifying_tasks']==['USTBench','SGQA']
        assert trigger(rec(100,100),rec())['qualifying_tasks']==[]
        assert trigger(rec(70,100),rec())['qualifying_tasks']==['USTBench','SGQA']
        assert trigger(rec(90,100),rec(79,100))['qualifying_tasks']==[]
        assert trigger(rec(90,100),rec(80,94))['qualifying_tasks']==[]
        d=rec(90,100);q=rec();q['scores']['USTBench/on']['correct']=79
        assert trigger(d,q)['qualifying_tasks']==['USTBench','SGQA']
        q['scores']['USTBench/off']['correct']=79
        assert trigger(d,q)['qualifying_tasks']==['SGQA']
    finally:bind=original
    print('PASS8 trigger boundary fixtures; no gold/model/API/SQL')

def main():
    packet=load(RUN/'execution-packet.json')
    models={'DS':read_scores(ROOT/packet['run_dir'],'DS'),'Qwen2B':read_scores(RUN/'local-qwen-original2b','Qwen2B'),
        'Qwen0.8B':read_scores(RUN/'local-qwen-original08b','Qwen0.8B')}
    gate=trigger(models['DS'],models['Qwen2B'])
    review=root_data_review()
    if gate['status']=='FROZEN_TRIGGER_EVALUATED':
        for task in COUNTS:
            if task not in gate['qualifying_tasks'] and models['Qwen0.8B'].get('task_status',{}).get(task)!='NOT_RUN_DATA_REVIEW':
                models['Qwen0.8B'].setdefault('task_status',{})[task]='NOT_TRIGGERED'
    if gate['status']=='FROZEN_TRIGGER_EVALUATED':
        receipt=RUN/'smaller-model-trigger.json';payload=js(gate)
        if receipt.exists() and receipt.read_text(encoding='utf-8')!=payload:raise ValueError('Existing immutable trigger receipt differs')
        if not receipt.exists():
            with receipt.open('x',encoding='utf-8') as f:f.write(payload)
    status={'updated_utc':datetime.now(timezone.utc).isoformat(),'packet':bind(RUN/'execution-packet.json'),'models':models,'smaller_model_trigger':gate,'root_data_review':review,'budget':budget(packet),'gold_read':False,'api_calls':0}
    atomic(RUN/'consolidated-status.json',js(status));atomic(REPORT,render(status))
    print(json.dumps({'report':str(REPORT),'models':{k:v['status'] for k,v in models.items()},'trigger':gate['status'],'budget':status['budget']['status']}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test()
    else:main()

