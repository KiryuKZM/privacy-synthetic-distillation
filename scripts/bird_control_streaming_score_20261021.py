"""Post-hoc, zero-API row-cap diagnostic; original fixed-cap scores stay intact.

Uses the original canonical row keys and EXACT set/multiset/ordered hashes.
Counts live on a bounded scratch SQLite database instead of a list in RAM.
This is supplemental scoring, never a filter or a student training decision.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from rescore_bird_student_inplace import load_bird
from ipd_native_migration_20261019 import read,dump,now
RUN=ROOT/'runs/2026-10-21-bird-tokenizer-generator-control'

def summarize_stream(rows,directory,deadline=None,max_rows=5000000,max_scratch_bytes=2*1024**3):
    bird=load_bird();path=Path(directory)/'counts.sqlite'
    db=sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=OFF');db.execute('PRAGMA synchronous=OFF')
    db.execute('PRAGMA cache_size=-8192');db.execute('PRAGMA temp_store=FILE')
    db.execute('CREATE TABLE counts(k TEXT PRIMARY KEY,n INTEGER) WITHOUT ROWID')
    ordered=hashlib.sha256();ordered.update(b'[');n=0;batch=[]
    def check():
        if deadline and time.monotonic()>deadline:raise TimeoutError('streaming time limit')
        if n>max_rows:raise ValueError('supplemental streaming row cap')
        if path.stat().st_size>max_scratch_bytes:raise ValueError('scratch disk cap')
    for row in rows:
        key=bird.row_key(row)
        if n:ordered.update(b', ')
        ordered.update(json.dumps(key,ensure_ascii=False).encode());n+=1
        batch.append((key,))
        if len(batch)==512:
            check();db.executemany('INSERT INTO counts VALUES(?,1) ON CONFLICT(k) DO UPDATE SET n=n+1',batch);batch=[]
    check()
    db.executemany('INSERT INTO counts VALUES(?,1) ON CONFLICT(k) DO UPDATE SET n=n+1',batch)
    db.commit();ordered.update(b']');unique=hashlib.sha256();multi=hashlib.sha256()
    unique.update(b'[');multi.update(b'[');distinct=0
    for key,count in db.execute('SELECT k,n FROM counts ORDER BY k COLLATE BINARY'):
        if distinct:unique.update(b', ');multi.update(b', ')
        unique.update(json.dumps(key,ensure_ascii=False).encode())
        multi.update(json.dumps((key,count),ensure_ascii=False).encode());distinct+=1
        if distinct%512==0:check()
    unique.update(b']');multi.update(b']');db.close()
    return dict(row_count=n,distinct_row_count=distinct,set_sha256=unique.hexdigest(),
        multiset_sha256=multi.hexdigest(),ordered_sha256=ordered.hexdigest())

def worker(pipe,path,sql,timeout):
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
        resource.setrlimit(resource.RLIMIT_CPU,(int(timeout)+1,int(timeout)+2))
        bird=load_bird()
        if not isinstance(sql,str) or not 0<len(sql)<=20000:raise ValueError('SQL length')
        lead=re.sub(r'\A(?:\s+|--[^\n]*(?:\n|$)|/\*.*?\*/)*','',sql,flags=re.S)
        if not re.match(r'(?i)(?:SELECT|WITH)\b',lead):raise ValueError('SELECT/WITH required')
        conn=bird.read_only_connection(path)
        allowed={sqlite3.SQLITE_SELECT,sqlite3.SQLITE_READ,sqlite3.SQLITE_FUNCTION}
        if hasattr(sqlite3,'SQLITE_RECURSIVE'):allowed.add(sqlite3.SQLITE_RECURSIVE)
        banned={'load_extension','readfile','writefile','fts3_tokenizer','randomblob','zeroblob'}
        conn.set_authorizer(lambda action,a,b,d,t:sqlite3.SQLITE_DENY if action not in allowed or (action==sqlite3.SQLITE_FUNCTION and str(b).lower() in banned) else sqlite3.SQLITE_OK)
        deadline=time.monotonic()+timeout
        conn.set_progress_handler(lambda:int(time.monotonic()>deadline),1000)
        cur=conn.execute(sql)
        if cur.description is None:raise ValueError('query has no result')
        scratch=RUN/'streaming-scratch';scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            result=summarize_stream(cur,directory,deadline)
        conn.close();pipe.send(dict(status='ok',summary=result))
    except BaseException as exc:pipe.send(dict(status='error',error_type=type(exc).__name__,error=str(exc)[:300]))
    finally:pipe.close()

def execute(path,sql,timeout=120):
    ctx=mp.get_context('spawn');recv,send=ctx.Pipe(duplex=False)
    process=ctx.Process(target=worker,args=(send,path,sql,timeout));process.start();send.close()
    result=None
    try:
        if recv.poll(timeout+3):
            try:result=recv.recv()
            except EOFError:result=dict(status='error',error_type='WorkerExit')
        else:result=dict(status='timeout')
    finally:
        if process.is_alive():process.terminate()
        process.join();recv.close()
    return result

def main():
    out=RUN/'supplemental-streaming';out.mkdir(exist_ok=True)
    cache=read(RUN/'teacher-gold-cache.local.json');gold=read(RUN/'teacher-gold.local.json')
    rows=read(RUN/'inputs/teacher-items.json');scores=read(RUN/'scores/teacher-per-item.json')
    results={x['id']:x for x in scores};queries={}
    for row in rows:
        g=gold[row['id']];key=hashlib.sha256((g['db_path']+'\n'+g['sql']).encode()).hexdigest()
        if cache[key].get('error')!='result row cap':continue
        path=out/('gold-'+key+'.json')
        if not path.exists():dump(path,execute(g['db_path'],g['sql']))
        expected=read(path)
        response=read(RUN/'responses/teacher'/(row['id']+'.json'))
        try:
            value=json.loads(response['content'])
            if not isinstance(value,dict) or set(value)!={'sql'} or not isinstance(value['sql'],str):raise ValueError('invalid SQL JSON')
            sql=value['sql']
        except (ValueError,TypeError):
            queries[row['id']]=dict(id=row['id'],status='invalid_json',correct=False);continue
        pred_key=hashlib.sha256((g['db_path']+'\n'+sql).encode()).hexdigest()
        pred_path=out/('prediction-'+pred_key+'.json')
        if not pred_path.exists():dump(pred_path,execute(g['db_path'],sql))
        prediction=read(pred_path)
        correct=expected['status']=='ok' and prediction['status']=='ok' and expected['summary']['set_sha256']==prediction['summary']['set_sha256']
        queries[row['id']]=dict(id=row['id'],gold=expected,prediction=prediction,correct=correct)
        print(json.dumps(dict(id=row['id'],gold_status=expected['status'],prediction_status=prediction['status'],correct=correct)),flush=True)
    by_source={}
    for source in ['G','R']:
        subset=[row for row in rows if row['source']==source]
        correct=sum(bool(queries.get(row['id'],results[row['id']]).get('correct',results[row['id']].get('official_ex'))) for row in subset)
        unresolved=sum(queries.get(row['id'],{}).get('gold',{}).get('status','ok')!='ok' for row in subset)
        by_source[source]=dict(denominator=200,correct=correct,unresolved_gold=unresolved)
    pairs=[]
    for task in sorted({row['task_id'] for row in rows}):
        pair={}
        for source in ['G','R']:
            ident=task+'-'+source;pair[source]=bool(queries[ident]['correct']) if ident in queries else bool(results[ident].get('official_ex'))
        pairs.append(dict(task_id=task,**pair))
    report=dict(at=now(),status='SUPPLEMENTAL_ONLY',method='Original canonical set hashes, external-memory streaming',
        original_scores_untouched=True,training_targets_unchanged=True,timeout=120,max_rows=5000000,
        repaired=queries,by_source=by_source,paired=dict(G_only=sum(x['G'] and not x['R'] for x in pairs),R_only=sum(x['R'] and not x['G'] for x in pairs)),
        no_extra_API_calls=True,no_GPU=True)
    dump(out/'teacher-summary.json',report)

if __name__=='__main__':main()
