#!/usr/bin/env python3
"""Deterministic CPU1 audit of exactly frozen UST100 query inputs and labels.

No model/API/tokenizer imports, predictions, SGQA gold, or Final reads.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from decimal import Decimal,InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/2026-10-24-ust100-sg60-modes'
OUT=RUN/'ust100-zero-api-audit'
SOURCE=ROOT/'runs/2026-10-23-llm-task-sweep/preparation/text-qa/sources'
ABILITIES=('st_understanding','planning','reflection')
EXPECTED_PACKET='8d3045db9ccc6480bcc8d6a03a1104a66bfb6b40a052e3571909305883c2cbfd'

def encode(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def bind(p):return {'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)}
def bound(spec):
    p=ROOT/spec['path'];assert sha(p)==spec['sha256'],spec['path'];return p

def once(p,x):
    data=encode(x);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert p.read_text(encoding='utf-8')==data,'Existing immutable output differs: '+str(p)
    else:
        with p.open('x',encoding='utf-8') as f:f.write(data)

def cpu_one():
    if hasattr(os,'sched_getaffinity'):
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    elif os.name=='nt':
        import ctypes
        kernel=ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype=ctypes.c_void_p
        kernel.SetProcessAffinityMask.argtypes=(ctypes.c_void_p,ctypes.c_size_t)
        assert kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(),1)

def selected_source_objects(text,wanted):
    """Lexically skip other objects; JSON-decode only selected array members."""
    depth=0;quoted=False;escaped=False;start=None;index=-1;found={}
    for pos,char in enumerate(text):
        if quoted:
            if escaped:escaped=False
            elif char=='\\':escaped=True
            elif char=='"':quoted=False
            continue
        if char=='"':quoted=True;continue
        if char in '[{':
            if depth==1 and char=='{':start=pos;index+=1
            depth+=1
        elif char in ']}':
            depth-=1
            if depth==1 and char=='}' and start is not None:
                if index in wanted:found[index]=json.loads(text[start:pos+1])
                start=None
    assert depth==0 and not quoted and set(found)==set(wanted)
    return found

def sections(question):
    result={}
    parts=re.split(r'^## ([^\r\n]+)\r?\n',question,flags=re.MULTILINE)
    for name,body in zip(parts[1::2],parts[2::2]):result[name.strip()]=body.strip()
    return result

def parse_data(data):
    result={}
    chunks=re.split(r'^location ([1-4]):\s*$',data,flags=re.MULTILINE)
    for name,body in zip(chunks[1::2],chunks[2::2]):
        fields={}
        for match in re.finditer(r'^- ([a-z_]+):\s*([^\r\n]+)',body,re.MULTILINE):
            raw=match.group(2).strip()
            number=re.fullmatch(r'(NaN|[-+]?(?:\d+(?:\.\d*)?|\.\d+))(m|h)?',raw,re.IGNORECASE)
            if number:fields[match.group(1)]={'display':raw,'number':number.group(1),'unit':number.group(2),'decimal_places':len(number.group(1).split('.')[1]) if '.' in number.group(1) else 0}
        assert int(name) not in result
        result[int(name)]=fields
    return result

def option_texts(question_section):
    return {m.group(1):m.group(2).strip() for m in re.finditer(r'^([A-D])\.\s*(.+?)\s*$',question_section,re.MULTILINE)}

def metric_from_request(question):
    first=question.split('\n',1)[0].lower()
    pairs=(('average charging time','charging_time'),('average waiting time','waiting_time'),('average distance','distance'))
    matches=[metric for phrase,metric in pairs if phrase in first]
    descending='from the longest to the shortest' in first or 'from the farthest to the closest' in first
    return matches[0] if len(matches)==1 and descending and 'rank the locations' in first else None

def rank_audit(sec,gold):
    flags=[];metric=metric_from_request(sec.get('Question',''));data=parse_data(sec.get('Data',''));options=option_texts(sec.get('Question',''))
    orders={k:tuple(int(n) for n in re.findall(r'\d+',v)) for k,v in options.items() if re.fullmatch(r'[1-4](?:\s*>\s*[1-4]){3}',v)}
    coverage={'metric_recognized':metric is not None,'four_data_locations':set(data)=={1,2,3,4},'four_option_letters':set(options)==set('ABCD'),'four_complete_rank_permutations':len(orders)==4 and all(set(v)=={1,2,3,4} and len(v)==4 for v in orders.values())}
    available=metric is not None and all(metric in data.get(i,{}) for i in range(1,5))
    coverage['requested_metric_present_all4']=available
    result={'requested_metric':metric,'coverage':coverage,'option_texts':options,'option_orders':{k:list(v) for k,v in orders.items()},'flags':flags,'semantic_correctness_status':'UNKNOWN_OUTSIDE_RULE_COVERAGE'}
    duplicates=defaultdict(list)
    for k,order in orders.items():duplicates[order].append(k)
    duplicate_groups=[letters for letters in duplicates.values() if len(letters)>1]
    result['duplicate_option_groups']=duplicate_groups
    if duplicate_groups:flags.append('duplicate_or_equivalent_rank_options')
    if any(gold in g for g in duplicate_groups):flags.append('duplicate_or_equivalent_options_include_gold')
    if not all(coverage.values()):
        flags.append('rank_rule_coverage_incomplete');return result
    vals={};unknown=[]
    for i in range(1,5):
        token=data[i][metric]['number']
        if token.lower()=='nan':
            if metric in ('waiting_time','charging_time') and 'NaN indicates no station nearby and the highest' in sec.get('Data Schema',''):vals[i]=Decimal('Infinity')
            else:unknown.append(i)
        else:vals[i]=Decimal(token)
    if unknown:
        result['unknown_numeric_semantics_locations']=unknown;flags.append('numeric_state_semantics_unknown');return result
    tied=defaultdict(list)
    for i,v in vals.items():tied[str(v)].append(i)
    ties=[g for g in tied.values() if len(g)>1]
    compatible=[k for k,order in orders.items() if all(vals[a]>=vals[b] for a,b in zip(order,order[1:]))]
    distinct={orders[k] for k in compatible}
    result.update(displayed_metric={str(i):data[i][metric] for i in range(1,5)},requested_metric_tied_groups=ties,compatible_options=compatible,compatible_distinct_orders=len(distinct),gold_compatible=gold in compatible,
        explicit_tie_break_rule_detected=bool(re.search(r'tie[- ]break|in case of (?:a )?tie|ties? (?:are|should)',sec.get('Question',''),re.I)),
        semantic_correctness_status='CHECKED_DISPLAYED_RANK_CONSISTENCY_ONLY')
    equivalent=defaultdict(list)
    for letter,order in orders.items():equivalent[tuple(str(vals[i]) for i in order)].append(letter)
    equivalent_groups=[letters for letters in equivalent.values() if len(letters)>1]
    result['requested_metric_equivalent_option_groups']=equivalent_groups
    if equivalent_groups:flags.append('requested_metric_equivalent_options')
    if any(gold in g for g in equivalent_groups):flags.append('requested_metric_equivalent_options_include_gold')
    if equivalent_groups and not any(gold in g for g in equivalent_groups):flags.append('equivalent_distractors_only_not_gold_ambiguity')
    if ties:flags.append('requested_metric_has_displayed_tie')
    if len(distinct)>1 and ties:flags.append('precision_tie_multiple_compatible_rank_orders')
    if len(compatible)>1:flags.append('multiple_compatible_answer_letters')
    if len(compatible)==0:flags.append('no_option_matches_displayed_weak_order')
    if gold not in compatible:flags.append('gold_violates_displayed_strict_comparison')
    result['violated_gold_comparisons']=[{'left':a,'right':b,'left_value':str(vals[a]),'right_value':str(vals[b])} for a,b in zip(orders.get(gold,()),orders.get(gold,())[1:]) if vals[a]<vals[b]]
    return result

def planning_audit(sec,record,gold):
    flags=['structural_counterfactual_environment_dependency']
    options=option_texts(sec.get('Question',''))
    option_location={k:int(m.group(1)) for k,text in options.items() if (m:=re.fullmatch(r'location ([1-4])',text,re.I))}
    feedback=record.get('feedbacks');required={'cov_gain','travel_gain','wait_gain','chg_gain'}
    schema=isinstance(feedback,list) and len(feedback)==4 and all(isinstance(x,dict) and required<=set(x) and all(isinstance(x[k],(int,float)) and Decimal(str(x[k])).is_finite() for k in required) for x in feedback)
    target=record.get('target');target_keys={'reducing overall waiting time':'wait_gain','reducing overall charging time':'chg_gain','reducing overall travel time':'travel_gain','increasing charging station coverage':'cov_gain','maximizing coverage':'cov_gain','enhancing overall station coverage':'cov_gain','minimizing overall travel time':'travel_gain'}
    key=target_keys.get(target)
    result={'flags':flags,'requested_target':target,'option_location':option_location,'feedback_schema_four_rows_four_finite_gains':schema,'feedback_keys':sorted({k for x in feedback for k in x}) if isinstance(feedback,list) and all(isinstance(x,dict) for x in feedback) else [],
        'feedbacks_in_model_query':bool(re.search(r'wait_gain|chg_gain|cov_gain|travel_gain',sec.get('Question','')+sec.get('Data',''))),
        'counterfactual_state_availability':'Four source feedback outcomes are metadata, not model-query evidence; complete rollout state/transition not reconstructed.',
        'label_semantic_status':'UNKNOWN_AUTHOR_H5_TRAJECTORY_CORRESPONDENCE_NOT_VERIFIED',
        'mapping_assumption':'Author reflection constructor read_utils.py explicitly maps A/B/C/D to feedback0/1/2/3; visible options map A/B/C/D to location1/2/3/4. Target gain argmax remains a conditional diagnostic, not reconstruction of author H5 label mechanism.'}
    if not schema:flags.append('planning_feedback_schema_incomplete_or_unknown')
    if key and schema and set(option_location)==set('ABCD') and set(option_location.values())=={1,2,3,4}:
        vals=[Decimal(str(x[key])) for x in feedback];best=max(vals);locations=[i+1 for i,v in enumerate(vals) if v==best];letters=[k for k,v in option_location.items() if v in locations]
        result.update(conditional_gain_key=key,conditional_gain_values=[str(x) for x in vals],conditional_best_locations=locations,conditional_best_letters=letters,conditional_gold_agrees=gold in letters)
        if gold not in letters:flags.append('conditional_feedback_argmax_gold_disagreement_not_proven_label_error')
    else:result['conditional_argmax_coverage']='UNKNOWN_TARGET_OR_SCHEMA_OR_OPTION_MAPPING'
    return result

def reflection_audit(sec,gold):
    changes=sec.get('Environment Changes','')
    parsed={m.group(1).lower():{'direction':m.group(2),'percent':str(Decimal(m.group(3)))} for m in re.finditer(r'(Station coverage|Charging time|Travel time|Waiting time) (increased|decreased) by ([0-9.]+)%',changes)}
    question=sec.get('Question','').lower()
    target='waiting time' if 'waiting time' in question else 'charging time' if 'charging time' in question else 'travel time' if 'travel time' in question else 'station coverage' if 'coverage' in question else None
    selected=sec.get('Decision','').strip()
    effect=parsed.get(target)
    direction=None
    if effect:
        if Decimal(effect['percent'])==0:direction='DISPLAYED_ZERO_CHANGE'
        elif target=='station coverage':direction='DISPLAYED_IMPROVEMENT' if effect['direction']=='increased' else 'DISPLAYED_DETERIORATION'
        else:direction='DISPLAYED_IMPROVEMENT' if effect['direction']=='decreased' else 'DISPLAYED_DETERIORATION'
    return {'flags':['structural_counterfactual_environment_dependency'],'decision_present':'Decision' in sec,'reason_present':'Reason' in sec,'environment_change_present':'Environment Changes' in sec,
        'requested_target_metric':target,'visible_selected_decision':selected,'gold_keeps_visible_decision':selected==gold,'environment_changes_parsed':parsed,'target_effect_direction':direction or 'UNKNOWN_PARSE_OR_TARGET',
        'selected_feedback_record_consistency':'UNKNOWN_NO_FEEDBACKS_FIELD_IN_RELEASED_REFLECTION_RECORD; no unselected planning source row opened',
        'label_semantic_status':'UNKNOWN_ALTERNATIVE_ACTION_ROLLOUT_OUTCOMES_NOT_VERIFIED',
        'counterfactual_state_availability':'Given selected-decision changes are visible; alternatives/transition model not established. Absolute deterioration does not prove gold is wrong.'}


def selftest():
    def sec(v,options):
        return {'Question':'Based on the data provided, rank the locations by average waiting time at near charging stations, from the longest to the shortest.\n\n'+options,'Data':'\n\n'.join(f'location {i}:\n- waiting_time: {val}h' for i,val in enumerate(v,1)),'Data Schema':'NaN indicates no station nearby and the highest waiting time.'}
    standard='A. 1 > 2 > 3 > 4\nB. 1 > 2 > 4 > 3\nC. 2 > 1 > 3 > 4\nD. 4 > 3 > 2 > 1'
    x=rank_audit(sec(['.2','.1','0','0'],standard),'A');assert 'precision_tie_multiple_compatible_rank_orders' in x['flags'] and x['compatible_options']==['A','B']
    x=rank_audit(sec(['.2','.1','.05','0'],standard),'A');assert x['compatible_options']==['A'] and not x['requested_metric_tied_groups']
    x=rank_audit(sec(['.2','.1','0','0'],standard.replace('1 > 2 > 4 > 3','4 > 2 > 1 > 3')),'A');assert 'precision_tie_multiple_compatible_rank_orders' not in x['flags']
    x=rank_audit(sec(['.2','.1','.05','0'],standard.replace('1 > 2 > 4 > 3','1 > 2 > 3 > 4')),'A');assert 'duplicate_or_equivalent_options_include_gold' in x['flags'] and 'precision_tie_multiple_compatible_rank_orders' not in x['flags']
    x=rank_audit(sec(['.2','.1','.05','0'],standard),'C');assert 'gold_violates_displayed_strict_comparison' in x['flags']
    x=rank_audit(sec(['NaN','.1','.05','0'],standard),'A');assert x['compatible_options']==['A']
    x=rank_audit({'Question':standard,'Data':''},'A');assert 'rank_rule_coverage_incomplete' in x['flags']
    text='[{"question":"brace } and \\\" [", "answer":"unread"},{"answer":"selected","nested":[{}]}]';assert selected_source_objects(text,{1})=={1:{'answer':'selected','nested':[{}]}}
    x=rank_audit(sec(['.2','.1','0','0'],standard),'A');assert x['requested_metric_equivalent_option_groups']==[['A','B']]
    r=reflection_audit({'Question':'reducing overall charging time','Decision':'B','Environment Changes':'Charging time increased by 4.76%'},'D');assert r['target_effect_direction']=='DISPLAYED_DETERIORATION' and not r['gold_keeps_visible_decision']
    print('PASS10 focused parser/tie/duplicate/strict-order/NaN/coverage/selected-only/reflection fixtures')

def main():
    cpu_one()
    packetpath=RUN/'execution-packet.json';assert sha(packetpath)==EXPECTED_PACKET;p=load(packetpath)
    spec=p['tasks']['USTBench'];requestpath=bound(spec['queries']);goldpath=bound(spec['gold'])
    canonical=[json.loads(x) for x in requestpath.read_text(encoding='utf-8').splitlines()]
    assert len(canonical)==100 and Counter(r['metadata']['ability'] for r in canonical)=={'st_understanding':34,'planning':33,'reflection':33}
    gold=load(goldpath);assert set(gold)=={r['id'] for r in canonical}
    members=load(RUN/'data-freeze/membership-freeze.json');old=set(members['old_members_ordered'])
    assert [r['id'] for r in canonical]==members['old_members_ordered']+[r['id'] for r in members['new_members_ordered']]
    source_records={};sources=[]
    for ability in ABILITIES:
        path=SOURCE/f'ust-data-poi_placement__{ability}_QA.json';assert path.stat().st_size<64*1024*1024
        wanted={int(r['id'].rsplit(':',1)[1]) for r in canonical if r['metadata']['ability']==ability}
        extracted=selected_source_objects(path.read_text(encoding='utf-8'),wanted);sources.append(dict(bind(path),selected_indices=sorted(wanted),byte_size=path.stat().st_size))
        for index,obj in extracted.items():source_records[f'poi_placement:{ability}:{index}']=obj
    query_only=[{'id':r['id'],'group':r['group'],'ability':r['metadata']['ability'],'question':r['adapted_messages'][-1]['content']} for r in canonical]
    once(OUT/'queries-only.local.json',query_only)
    findings=[];extracted=[]
    for r in canonical:
        rid=r['id'];ability=r['metadata']['ability'];question=r['adapted_messages'][-1]['content'];record=source_records[rid]
        assert question==record['question'] and str(gold[rid])==str(record['answer'])
        sec=sections(question)
        f={'id':rid,'group':r['group'],'ability':ability,'cohort':'old60' if rid in old else 'new40','gold_option':gold[rid],
            'query_sha256':hashlib.sha256(question.encode()).hexdigest(),'source_index_zero_based':int(rid.rsplit(':',1)[1]),
            'query_section_names':list(sec),'source_field_names':sorted(record),'query_only_rule_scope':True,
            'original_unrounded_precision_available_in_record':'NOT_ESTABLISHED; source question reproduces displayed precision; no raw four-location state identified.',
            'missing_state_classification':'UNKNOWN_NOT_AUTOMATIC_LABEL_ERROR'}
        if ability=='st_understanding':f.update(rank_audit(sec,gold[rid]))
        elif ability=='planning':f.update(planning_audit(sec,record,gold[rid]))
        else:f.update(reflection_audit(sec,gold[rid]))
        findings.append(f)
        extracted.append({'id':rid,'source_index_zero_based':f['source_index_zero_based'],'question_sha256':f['query_sha256'],'metadata':{k:v for k,v in record.items() if k!='question'}})
    once(OUT/'selected100-source-metadata.local.json',extracted)
    once(OUT/'findings-v3.local.json',findings)
    flagsets={flag:{f['id'] for f in findings if flag in f['flags']} for flag in sorted({x for f in findings for x in f['flags']})}
    flagcounts={k:len(v) for k,v in flagsets.items()}
    direct_flags={'precision_tie_multiple_compatible_rank_orders','duplicate_or_equivalent_options_include_gold','gold_violates_displayed_strict_comparison','no_option_matches_displayed_weak_order','requested_metric_equivalent_options_include_gold'}
    direct=set().union(*(flagsets.get(k,set()) for k in direct_flags));structural=flagsets.get('structural_counterfactual_environment_dependency',set())
    union=set().union(*flagsets.values()) if flagsets else set()
    overlaps={a+' & '+b:len(flagsets[a]&flagsets[b]) for i,a in enumerate(flagsets) for b in list(flagsets)[i+1:] if flagsets[a]&flagsets[b]}
    summary={'status':'FROZEN100_MACHINE_AUDIT_COMPLETE_NOT_SEMANTIC_CERTIFICATION','count':100,'ability_counts':dict(Counter(f['ability'] for f in findings)),
        'flag_counts':flagcounts,'flag_pair_nonzero_overlap_counts':overlaps,'any_flag_union':len(union),'direct_displayed_answer_issue_union':len(direct),'structural_counterfactual_dependency_union':len(structural),'direct_or_structural_union':len(direct|structural),'direct_and_structural_overlap':len(direct&structural),
        'rank_rule_full_coverage':sum(f['ability']=='st_understanding' and all(f.get('coverage',{}).values()) for f in findings),'not_flagged_is_not_pass':True,
        'planning_feedback_schema_valid':sum(f.get('feedback_schema_four_rows_four_finite_gains') is True for f in findings),
        'planning_conditional_argmax_covered':sum('conditional_gold_agrees' in f for f in findings),'planning_conditional_argmax_gold_agrees':sum(f.get('conditional_gold_agrees') is True for f in findings),
        'limitations':['Exact displayed decimals; latent rounding intervals unknown.','Only descending requested-metric rank questions are covered; no general semantic reasoning certification.','Conditional feedback argmax is not author H5-policy reconstruction or proof of label error.','No input/score/member/relabel changes; no model calls, prediction reads, SG gold or Final.'],
        'queries':bind(requestpath),'gold':bind(goldpath),'membership':bind(RUN/'data-freeze/membership-freeze.json'),'sources':sources,'all100_source_query_and_gold_exact_match':True,'revision':'v3 final: exact target mapping, descriptive reflection directions, and requested-metric-equivalent distractors; v1/v2 retained',
        'components':{'count':len({r['group'] for r in canonical}),'limitation':'Numeric-record fingerprint connected components do not prove independent physical places; source omits stable physical identity.'},
        'reflection_target_effect_counts':dict(Counter(f.get('target_effect_direction') for f in findings if f['ability']=='reflection')),
        'author_reflection_mapping_code':bind(OUT/'recovery-sources/read_utils.py'),
        'findings':bind(OUT/'findings-v3.local.json'),'query_only':bind(OUT/'queries-only.local.json'),'selected_metadata':bind(OUT/'selected100-source-metadata.local.json'),'script':bind(Path(__file__).resolve()),'cpu_limit':1}
    once(OUT/'summary-v3.json',summary)
    report=['# UST100输入/标签机器审计（事后，零API）','',f"冻结100题全部完成来源问题/标签一致性验证。理解34、规划33、反思33；排序规则完整覆盖{summary['rank_rule_full_coverage']}/34。未修改输入、标签、评分或成员，未读取任何Qwen预测。",'', '| 标记 | 数量/100 |','|---|---|']
    report += [f'| {k} | {v}/100 ({v:.2f}%) |' for k,v in flagcounts.items()]
    report += ['',f"直接涉及显示答案可判定性的四类标记并集：{len(direct)}/100；规划/反思结构性反事实依赖：{len(structural)}/100；两者并集{len(direct|structural)}/100、交集{len(direct&structural)}/100。所有机器标记并集{len(union)}/100。requested_metric_has_displayed_tie本身只是描述，不自动算答案问题。",'',f"规划四反馈schema有效{summary['planning_feedback_schema_valid']}/33；条件argmax可覆盖{summary['planning_conditional_argmax_covered']}/33，其中与gold一致{summary['planning_conditional_argmax_gold_agrees']}/33。此检查假设反馈索引对应location1..4、目标对应gain最大化，不能证明H5 rollout标签生成逻辑，也不能把不一致直接当标签错误。",'', '并列问题只在所请求指标使不同排序选项同时兼容时标记；不扫描示例中的数字或其他非目标指标。重复排序选项另计，保留交集。gold违反显示严格比较与“并列次序未唯一规定”分别记录。没有标记不等于PASS；原始未舍入数值、完整环境状态/反事实转移仍未建立。','', '反思可见环境方向仅是事实描述，不能用绝对变差推断该动作不是候选中最优。已发布反思记录没有feedbacks字段，未打开不在本100中的规划记录，因此原始动作反馈一致性仍为UNKNOWN。99个组件为数值记录指纹连通分组，不证明99个独立物理地点。','', '反思可见环境方向仅是事实描述，不能用绝对变差推断该动作不是候选中最优。已发布反思记录没有feedbacks字段，未打开不在本100中的规划记录，因此原始动作反馈一致性仍为UNKNOWN。99个组件为数值记录指纹连通分组，不证明99个独立物理地点。','', '非零标记交集：','', '```json',encode(overlaps).strip(),'```','', '逐题详情和完整源绑定仅保存于被忽略的 `runs/2026-10-24-ust100-sg60-modes/ust100-zero-api-audit/`。本报告为机器规则审计，独立人工/作者方法审查另行报告。','']
    reportpath=ROOT/'reports/ust100-sg60-modes-2026-10-24/ust100-input-audit-machine.md'
    data='\n'.join(report)
    reportpath.write_text(data,encoding='utf-8')
    print(encode({'summary':bind(OUT/'summary-v3.json'),'flags':flagcounts,'direct_issue_union':len(direct),'structural_union':len(structural),'rank_coverage':summary['rank_rule_full_coverage'],'planning_covered':summary['planning_conditional_argmax_covered'],'planning_agrees':summary['planning_conditional_argmax_gold_agrees']}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:selftest()
    else:main()
