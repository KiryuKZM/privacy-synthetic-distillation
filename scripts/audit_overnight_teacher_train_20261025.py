#!/usr/bin/env python3
"""Strict descriptive Train-only audit after complete200 teacher-answer freeze."""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import run_llm_task_sweep_20261023 as base
RUN=ROOT/'runs/2026-10-25-overnight-onpolicy'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def bind(p,root):return {'path':Path(p).relative_to(root).as_posix(),'sha256':sha(p)}
def resolve(b,root):
 p=(root/b['path']).resolve()
 if not p.is_relative_to(root.resolve()) or sha(p)!=b['sha256']:raise PermissionError('Bound file changed/outside root: '+b['path'])
 return p
def require(x,msg):
 if not x:raise PermissionError(msg)
def audit(task,root=ROOT,run=RUN,expected=200,gold_loader=load):
 root,run=Path(root),Path(run);out=run/'teacher'/task/'train';mp=out/'manifest.json';fp=out/'predictions-frozen.json';rp=out/'references-frozen.json'
 if not all(p.is_file() for p in (mp,fp,rp)):return {'status':'WAITING_FROZEN_TRAIN200','task':task,'gold_read':False}
 m,f,rf=load(mp),load(fp),load(rp)
 require(m.get('task')==task and m.get('count')==expected,'Teacher task/count mismatch')
 require(f.get('complete') is True and f.get('expected')==expected and len(f.get('predictions',[]))==expected,'Incomplete teacher freeze')
 require(rf.get('complete') is True and rf.get('count')==expected,'Incomplete reference freeze')
 require(resolve(rf['manifest'],root)==mp.resolve(),'Reference freeze manifest mismatch')
 for b in m['bindings']:resolve(b,root)
 packet_path=resolve(f['packet'],root);packet=load(packet_path)
 require(packet['task']==task and packet['stage']=='train' and packet['expected']==expected,'Packet task/stage/count mismatch')
 dataset_path=resolve(packet['dataset'],root);dataset=load(dataset_path);spec=dataset['phases']['train'];query_path=resolve(spec['queries'],root)
 require(spec['count']==expected and packet['queries']==spec['queries'],'Train query binding mismatch')
 queries=rows(query_path);requests=rows(resolve(packet['requests'],root));references=rows(resolve(m['records_file'],root))
 for name,rs in [('queries',queries),('requests',requests),('references',references)]:require(len(rs)==expected and len({str(x['id']) for x in rs})==expected,name+' member/denominator mismatch')
 byq={str(r['id']):r for r in queries};byr={str(r['id']):r for r in requests};byref={str(r['id']):r for r in references}
 require(set(byq)==set(byr)==set(byref),'Member sets differ')
 frozen_paths={resolve(b,root):b['sha256'] for b in f['predictions']};require(len(frozen_paths)==expected,'Duplicate frozen paths')
 verified=[]
 for rid,q in byq.items():
  req,ref=byr[rid],byref[rid];source=resolve(ref['source_result'],root)
  require(source in frozen_paths,'Reference outside frozen predictions')
  v=load(source)
  require(req['body_sha256']==base.digest(req['body'])==ref['body_sha256']==v['body_sha256'],'Body binding mismatch')
  require(v['packet_sha256']==sha(packet_path) and v['tag']==req['tag'] and str(v['id'])==rid and v['task']==task,'Prediction packet/tag/identity mismatch')
  require(ref.get('group')==q['group']==req['group'] and ref.get('metadata',{})==q.get('metadata',{})==req.get('metadata',{}),'Group/metadata mismatch')
  require(ref.get('text','')==(v.get('text') or '') and ref.get('valid')==bool(v.get('valid')) and ref.get('prediction')==v.get('prediction'),'Reference output changed')
  require(not v.get('stop'),'Technical-stop output not completed Train')
  verified.append((q,ref,v,ref['source_result']))
 # Gold bytes are neither opened nor hashed until all membership and output checks above pass.
 gp=resolve(spec['gold'],root);gold=gold_loader(gp);require(set(gold)==set(byq),'Train gold membership mismatch')
 total={'total':expected,'correct':0,'valid':0,'invalid':0,'wrong_valid':0,'truncated':0};families={};details=[]
 for q,ref,v,source in verified:
  correct,valid=base.score_prediction(v,{'kind':'qa','scorer':spec['scorer']},gold[str(q['id'])],q.get('metadata',{}));family=q.get('metadata',{}).get('ability',q.get('metadata',{}).get('question_type','UNKNOWN'))
  group=families.setdefault(family,{'total':0,'correct':0,'valid':0,'invalid':0,'wrong_valid':0,'truncated':0})
  for target in (total,group):
   if target is group:target['total']+=1
   target['correct']+=bool(correct);target['valid']+=bool(valid);target['invalid']+=not valid;target['wrong_valid']+=bool(valid and not correct);target['truncated']+=v.get('finish_reason')=='length'
  details.append({'id':q['id'],'group':q['group'],'family':family,'correct':bool(correct),'valid':bool(valid),'prediction':v.get('prediction'),'gold':gold[str(q['id'])],'finish_reason':v.get('finish_reason'),'source_result':source})
 for target in [total,*families.values()]:target['accuracy']=target['correct']/target['total'];target['valid_rate']=target['valid']/target['total']
 result={'status':'COMPLETE_DESCRIPTIVE_TRAIN_ONLY','task':task,'mode':m.get('mode'),'totals':total,'by_family':families,'items':details,'bindings':{'teacher_manifest':bind(mp,root),'predictions_frozen':bind(fp,root),'references_frozen':bind(rp,root),'original_teacher_references':m['records_file'],'query_members':spec['queries'],'train_gold':spec['gold'],'dataset':packet['dataset'],'packet':f['packet'],'scorer':bind(Path(base.__file__).resolve(),ROOT)},'all200_retained':True,'wrong_answers_not_filtered_or_reasked':True,'selection_gate':False,'gold_read_after_complete_freeze_and_all_hashes':True,'Eval_Final_read':False,'gpu_used':False,'api_calls':0}
 return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['USTBench','SGQA'],required=True);a=parser.parse_args();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 result=audit(a.task)
 if result['status'].startswith('WAITING'):print(json.dumps(result));return
 result['audit_source']=bind(Path(__file__).resolve(),ROOT);dest=RUN/'teacher'/a.task/'train'/'quality.local.json';payload=json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
 if dest.exists():require(dest.read_text()==payload,'Frozen diagnostic differs; no overwrite')
 else:
  with dest.open('x',encoding='utf-8') as f:f.write(payload)
 print(json.dumps({'status':result['status'],'task':a.task,'totals':result['totals'],'receipt':bind(dest,ROOT)},indent=2))
if __name__=='__main__':main()
