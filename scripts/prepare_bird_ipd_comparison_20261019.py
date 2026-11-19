"""Zero-API sample freeze: new dev300 and explicitly provisional train200."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import unicodedata

ROOT=Path('/home/kiryu/privacy-synthetic-distillation')
RUN=ROOT/'runs/2026-10-19-bird-ipd-comparison-design'
DATA=ROOT/'data/raw/candidate-qualification/bird'
SEED=20261019


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def fingerprint(row):
    def norm(x):return ' '.join(unicodedata.normalize('NFKC',str(x or '')).casefold().split())
    return digest([row['db_id'],norm(row['question']),norm(row.get('evidence'))])


def stratified(rows,n,key,seed):
    groups=defaultdict(list)
    for row in rows:groups[key(row)].append(row)
    total=len(rows)
    quotas={k:n*len(v)//total for k,v in groups.items()}
    remainders=sorted(groups,key=lambda k:(-(n*len(groups[k])%total),str(k)))
    for k in remainders[:n-sum(quotas.values())]:quotas[k]+=1
    chosen=[]
    for k in sorted(groups,key=str):
        values=sorted(groups[k],key=lambda x:x['source_index'])
        rng=random.Random(digest([seed,k]));rng.shuffle(values)
        chosen.extend(values[:quotas[k]])
    random.Random(seed).shuffle(chosen)
    assert len(chosen)==n
    return chosen,quotas


def main():
    if RUN.exists():raise RuntimeError('Do not overwrite a frozen selection')
    old_path=ROOT/'runs/2026-10-13-overnight-ds-datasets/frozen-items.json'
    dev=read(DATA/'dev.json');train=read(DATA/'train.json');old=read(old_path)['bird']
    by_id={r['question_id']:r for r in dev}
    assert len(dev)==1534 and len(train)==9428 and len(old)==200
    assert all(by_id[r['source_id']]['db_id']==r['db_id'] for r in old)
    old_ids={r['source_id'] for r in old};old_fp={fingerprint(by_id[i]) for i in old_ids}
    pool=[];seen=set()
    for i,row in enumerate(dev):
        fp=fingerprint(row)
        if row['question_id'] in old_ids or fp in old_fp or fp in seen:continue
        seen.add(fp)
        pool.append(dict(source_index=i,question_id=row['question_id'],db_id=row['db_id'],difficulty=row['difficulty'],question_evidence_sha256=fp))
    selected,quota=stratified(pool,300,lambda r:(r['db_id'],r['difficulty']),SEED)
    assert not old_ids & {r['question_id'] for r in selected}
    assert not old_fp & {r['question_evidence_sha256'] for r in selected}
    assert len({r['db_id'] for r in selected})==11
    recovery=ROOT/'runs/2026-10-19-official-teacher-support-ipd/migration'
    repaired=read(recovery/'recovery-train.json')+read(recovery/'recovery-validation.json')
    excluded={r['train_json_index'] for r in repaired}
    train_pool=[];seen=set()
    for i,row in enumerate(train):
        if row['db_id'] not in {'food_inspection','donor'} or i in excluded:continue
        fp=fingerprint(row)
        if fp in seen:continue
        seen.add(fp)
        train_pool.append(dict(source_index=i,db_id=row['db_id'],question_evidence_sha256=fp,synthetic_field_support='NOT_AUDITED'))
    candidates,train_quota=stratified(train_pool,200,lambda r:r['db_id'],SEED+1)
    assert not excluded & {r['source_index'] for r in candidates}
    assert not {r['db_id'] for r in candidates}&{r['db_id'] for r in selected}
    RUN.mkdir(parents=True)
    dump(RUN/'eval300-identities.json',selected)
    dump(RUN/'train200-candidate-identities.json',candidates)
    protocol=dict(status='DESIGN_AND_LOCAL_SELECTION_ONLY',seed=SEED,eval_denominator=300,
        dev_version='official_public_dev_20240627; no annotation migration',old200_exclusion='source questionID and normalized DB/question/evidence fingerprint',
        eval_selection='proportional DB × official difficulty, largest remainder; no model outputs/gold correctness used',
        train_status='CANDIDATES_ONLY_NOT_GENERATOR_QUALIFIED',train_selection='proportional to available food_inspection/donor question counts, excluding338 restoration source indices',
        train_coverage_reason='Both DBs had previous fitted generator pilots; acceptance NOT established',
        no_training_or_api=True,api_attempts=0,assets={str(p):sha(p) for p in [DATA/'dev.json',DATA/'train.json',old_path,recovery/'recovery-train.json',recovery/'recovery-validation.json']},
        code_sha256=sha(Path(__file__)),eval_sha256=sha(RUN/'eval300-identities.json'),train_sha256=sha(RUN/'train200-candidate-identities.json'))
    dump(RUN/'protocol.json',protocol)
    summary=dict(status='PASS_LOCAL_SELECTION',eval_total=300,old200_overlap=0,normalized_question_overlap=0,eval_by_db=dict(Counter(r['db_id'] for r in selected)),
        eval_by_difficulty=dict(Counter(r['difficulty'] for r in selected)),remaining_dev_pool=len(pool),train_candidate_total=200,
        train_by_db=dict(Counter(r['db_id'] for r in candidates)),train_candidate_pool=len(train_pool),restoration_train_overlap=0,
        train_not_certified_for_synthetic_arm=True,new_inference_or_api=0)
    dump(RUN/'selection-summary.json',summary)
    dump(ROOT/'reports/bird-ipd-comparison-selection-2026-10-19.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
