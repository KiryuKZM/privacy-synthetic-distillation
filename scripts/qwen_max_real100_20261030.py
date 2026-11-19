"""Fixed public STAR graph-QA100; bounded official Max/off and shared USD ledger."""
from pathlib import Path
import argparse
from decimal import Decimal
import fcntl
import hashlib
import json
import re
import sqlite3
import ssl
import time
import urllib.error
import urllib.request

R=Path(__file__).resolve().parents[1]
D=R/'runs/2026-10-30-qwen-max-real100'
LEDGER=R/'runs/2026-10-26-generator-discovery/budget/new-api-ledger.sqlite'
URL='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
MODEL='qwen3.8-max-0902'
SCALE=10**12

def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,z):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(z,f,ensure_ascii=False,indent=2)
def atomic(p,z):
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(z,ensure_ascii=False,indent=2));t.replace(p)
def cost(inp,out):return int(((Decimal(inp)*Decimal('1.65')+Decimal(out)*Decimal('4.951'))/10**6*SCALE).to_integral_value(rounding='ROUND_CEILING'))


class MaxBudget:
    """Add a provider table while preserving legacy config/call rows and caps.

    A DB trigger protects the combined500 ceiling even if a legacy writer starts.
    Unreported usage keeps its entire peak reservation; unique tags prevent replay.
    """
    def __init__(self,path):
        self.path=path
        with sqlite3.connect(path,timeout=10) as db:
            db.execute('BEGIN IMMEDIATE')
            cfg=json.loads(db.execute('SELECT config FROM settings').fetchone()[0])
            assert cfg['global_pico']==500*SCALE
            db.execute('CREATE TABLE IF NOT EXISTS qwen_max_calls(id INTEGER PRIMARY KEY,tag TEXT UNIQUE NOT NULL,purpose TEXT NOT NULL,model TEXT NOT NULL,body_sha TEXT NOT NULL,inp INTEGER NOT NULL,out INTEGER NOT NULL,pico INTEGER NOT NULL,settled INTEGER NOT NULL DEFAULT0,usage_known INTEGER NOT NULL DEFAULT0)'.replace('DEFAULT0','DEFAULT 0'))
            db.execute('''CREATE TRIGGER IF NOT EXISTS sgqa_global500_qwen_insert BEFORE INSERT ON qwen_max_calls
                WHEN (SELECT COALESCE(SUM(pico),0) FROM calls)+(SELECT COALESCE(SUM(pico),0) FROM qwen_max_calls)+NEW.pico>500000000000000
                BEGIN SELECT RAISE(ABORT,'Cumulative500USDcap'); END''')
            db.execute('''CREATE TRIGGER IF NOT EXISTS sgqa_global500_legacy_insert BEFORE INSERT ON calls
                WHEN (SELECT COALESCE(SUM(pico),0) FROM calls)+(SELECT COALESCE(SUM(pico),0) FROM qwen_max_calls)+NEW.pico>500000000000000
                BEGIN SELECT RAISE(ABORT,'Cumulative500USDcap'); END''')
            # Previously authorized generic Flash probe is outside the legacy DB.
            # Keep a conservative $0.01 reserve, not a fabricated invoice amount.
            historical=R/'runs/2026-10-30-qwen-api-smoke/result.json'
            if historical.exists():
                db.execute("INSERT OR IGNORE INTO qwen_max_calls(tag,purpose,model,body_sha,inp,out,pico,settled,usage_known) VALUES('historical-flash-smoke','historical-smoke','qwen3.8-flash',?,33,1,10000000000,1,0)",(sha(historical),))

    def reserve(self,tag,purpose,model,body_sha,inp=32768,out=512):
        amount=cost(inp,out)
        with sqlite3.connect(self.path,timeout=10) as db:
            db.execute('BEGIN IMMEDIATE')
            n,spent=db.execute("SELECT COUNT(*),COALESCE(SUM(pico),0) FROM qwen_max_calls WHERE purpose!='historical-smoke'").fetchone()
            if n>=111 or spent+amount>10*SCALE:raise PermissionError('Max111HTTP/10USDcap')
            if purpose=='eval' and db.execute("SELECT COUNT(*) FROM qwen_max_calls WHERE purpose='eval'").fetchone()[0]>=110:raise PermissionError('EvalHTTPcap')
            return db.execute('INSERT INTO qwen_max_calls(tag,purpose,model,body_sha,inp,out,pico) VALUES(?,?,?,?,?,?,?)',
                (tag,purpose,model,body_sha,inp,out,amount)).lastrowid

    def settle(self,call,usage):
        with sqlite3.connect(self.path,timeout=10) as db:
            db.execute('BEGIN IMMEDIATE')
            inp,out,reserved,settled=db.execute('SELECT inp,out,pico,settled FROM qwen_max_calls WHERE id=?',(call,)).fetchone()
            if settled:raise PermissionError('Callalreadysettled')
            if usage is None:
                db.execute('UPDATE qwen_max_calls SET settled=1 WHERE id=?',(call,));return
            i,o=usage['prompt_tokens'],usage['completion_tokens'];charge=cost(i,o)
            if min(i,o)<0 or i>inp or o>out or charge>reserved:raise PermissionError('Usageexceededreservation')
            db.execute('UPDATE qwen_max_calls SET inp=?,out=?,pico=?,settled=1,usage_known=1 WHERE id=?',(i,o,charge,call))

    def summary(self):
        with sqlite3.connect(self.path) as db:
            old=db.execute('SELECT COUNT(*),COALESCE(SUM(pico),0),SUM(1-usage_known),SUM(1-settled) FROM calls').fetchone()
            new=db.execute('SELECT COUNT(*),COALESCE(SUM(pico),0),SUM(1-usage_known),SUM(1-settled) FROM qwen_max_calls').fetchone()
            return dict(legacy_calls=old[0],legacy_accounted_USD=old[1]/SCALE,legacy_unknown=old[2],legacy_inflight=old[3],
                qwen_attempt_rows=new[0],qwen_accounted_USD=new[1]/SCALE,qwen_unknown=new[2],qwen_inflight=new[3],
                cumulative_accounted_USD=(old[1]+new[1])/SCALE,cumulative_cap_USD=500,provider_invoice_reconciled=False)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


def verify():
    g=load(D/'root-grant.json');a=load(D/'user-authorization.json');m=load(D/'data/manifest.json')
    assert g['approved'] and g['source_sha256']==sha(Path(__file__)) and g['authorization_sha256']==sha(D/'user-authorization.json')
    assert g['requests_sha256']==sha(D/'data/requests.jsonl')==m['requests_sha256']
    assert g['manifest_sha256']==sha(D/'data/manifest.json') and m['n']==100 and m['unique_videos']==100
    assert m['protected_video_overlap']==m['protected_question_id_overlap']==0
    assert a['API_hardcap_USD']==10 and a['parent_total_USD']==500 and not (D/'STOP').exists()
    return g


def key():
    with (Path.home()/'.secrets/qwen.env').open(encoding='utf-8-sig') as f:
        for line in f:
            name,sep,value=line.strip().removeprefix('export ').partition('=')
            if sep and name.strip()=='DASHSCOPE_API_KEY':
                k=value.strip().strip('\"').strip("'")
                if k and not any(c.isspace() for c in k):return k
    raise PermissionError('Missinglocalcredential')


def body(messages,model=MODEL,smoke=False):
    return dict(model=model,max_tokens=32 if smoke else 512,enable_thinking=False,stream=False,temperature=0,messages=messages)


def request(tag,messages,purpose,model=MODEL):
    verify();payload=body(messages,model,purpose=='smoke');raw=json.dumps(payload,ensure_ascii=False,separators=(',',':')).encode()
    assert len(raw)<=32768
    output=D/'responses'/(tag+'.json');start=D/'responses'/(tag+'.started.json')
    if output.exists():return load(output)
    if start.exists():raise PermissionError('Unknownrequeststart_NOT_REPLAYED')
    token=key();budget=MaxBudget(LEDGER);digest=hashlib.sha256(raw).hexdigest()
    call=budget.reserve(tag,purpose,model,digest,1024 if purpose=='smoke' else 32768,32 if purpose=='smoke' else 512)
    put(start,dict(epoch=time.time(),model=model,body_sha256=digest,ledger_id=call,purpose=purpose,
        max_tokens=payload['max_tokens'],enable_thinking=False,request_bytes=len(raw),data='genericOK' if purpose=='smoke' else 'permittedpublicSTARgraphQA,noquerygold'))
    started=time.monotonic();usage=None;result=dict(model_requested=model,ledger_id=call,purpose=purpose,body_sha256=digest)
    try:
        req=urllib.request.Request(URL,data=raw,method='POST',headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
        opener=urllib.request.build_opener(NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        with opener.open(req,timeout=75) as reply:
            result['http_status']=reply.status;response=json.loads(reply.read(2**20))
        u=response.get('usage',{})
        if all(isinstance(u.get(k),int) for k in ('prompt_tokens','completion_tokens')):
            usage={k:u[k] for k in ('prompt_tokens','completion_tokens')}
        msg=response.get('choices',[{}])[0].get('message',{})
        returned=response.get('model','')
        if returned not in ('qwen3.8-max-0902','qwen3.8-max-2026-09-02','qwen3.8-max'):
            raise RuntimeError('Unexpectedmodelidentity')
        reasoning=msg.get('reasoning_content')
        result.update(status='RESPONDED',model_returned=returned,content=msg.get('content',''),
            thinking_content_present=bool(reasoning),finish_reason=response.get('choices',[{}])[0].get('finish_reason'),usage=usage)
    except urllib.error.HTTPError as exc:
        result.update(status='HTTP_ERROR',http_status=exc.code)
    except Exception as exc:
        result.update(status='MODEL_IDENTITY_CHANGED' if isinstance(exc,RuntimeError) else 'CALL_OUTCOME_UNKNOWN',error_type=type(exc).__name__)
    budget.settle(call,usage)
    result.update(seconds=time.monotonic()-started,epoch=time.time(),usage_known=usage is not None)
    put(output,result);atomic(D/'budget-live.json',budget.summary())
    return result


def smoke():
    z=request('max0902-generic-smoke',[dict(role='user',content='Reply with exactly OK and nothing else.')],'smoke')
    if z.get('http_status') in (400,404):
        z=request('max-alias-generic-smoke',[dict(role='user',content='Reply with exactly OK and nothing else.')],'smoke','qwen3.8-max')
    assert z.get('status')=='RESPONDED' and z.get('content','').strip()=='OK' and not z['thinking_content_present'], 'Flagship/offsmokenotpassed'
    path=D/'model-qualified.json'
    value=dict(status='FLAGSHIP_THINKING_OFF_GENERIC_PASS',model=z['model_requested'],returned=z['model_returned'],
        smoke_ledger_id=z['ledger_id'],reasoning_verified_empty=True,not_IPD_qualified=True,epoch=time.time())
    if not path.exists():put(path,value)
    return value


def run():
    verify();qualified=smoke();items=[json.loads(x) for x in (D/'data/requests.jsonl').read_text().splitlines() if x.strip()]
    assert len(items)==len({x['id'] for x in items})==len({x['video_id'] for x in items})==100
    with (D/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for index,row in enumerate(items):
            verify();tag=f'star-eval-{index:03d}'
            z=request(tag,row['messages'],'eval',qualified['model'])
            if z.get('status')=='MODEL_IDENTITY_CHANGED':raise RuntimeError('Modelidentitydrift')
            if z.get('thinking_content_present'):raise RuntimeError('Thinkingoffviolated')
            atomic(D/'progress.json',dict(committed=index+1,total=100,model=qualified['model'],epoch=time.time(),API_calls=MaxBudget(LEDGER).summary()['qwen_attempt_rows']))
            if z.get('http_status') in (401,403):raise PermissionError('Credentialoraccountblocked')
            time.sleep(.25)
        files=[D/'responses'/f'star-eval-{i:03d}.json' for i in range(100)]
        freeze=D/'ALL-outputs-frozen.json'
        if not freeze.exists():put(freeze,dict(count=100,files=[dict(path=str(p.relative_to(R)),sha256=sha(p)) for p in files],gold_read=False,epoch=time.time()))
        if (D/'scores.json').exists():return
        marker=D/'scoring.started.json';assert not marker.exists();put(marker,dict(all_outputs_sha256=sha(freeze),epoch=time.time()))
        gold=load(D/'data/localgold.json');correct=valid=0;families={}
        for i,row in enumerate(items):
            z=load(files[i]);answer=z.get('content','');match=re.fullmatch(r'\s*([A-D])\s*',answer) if isinstance(answer,str) else None
            expected=gold[row['id']]['correct_letter'];v=bool(match);c=v and match[1]==expected
            correct+=int(c);valid+=int(v)
            family=families.setdefault(row['family'],dict(correct=0,valid=0,total=0))
            family['correct']+=int(c);family['valid']+=int(v);family['total']+=1
        answers=[load(p) for p in files]
        result=dict(benchmark='STAR-public-realgraph-MCQA100',model=qualified['model'],thinking=False,correct=correct,total=100,valid=valid,
            percentage=correct,per_family=families,by_video_unique=100,API_budget=MaxBudget(LEDGER).summary(),
            HTTP200=sum(x.get('http_status')==200 for x in answers),technical_missing=sum(x.get('status')!='RESPONDED' for x in answers),
            request_seconds=sum(x['seconds'] for x in answers),input_tokens=sum((x.get('usage') or {}).get('prompt_tokens',0) for x in answers),
            output_tokens=sum((x.get('usage') or {}).get('completion_tokens',0) for x in answers),
            scoring='all100frozen_beforelocalgold_once',description='Graph-oracle multiple-choice descriptive reference; not TSG or video perception; no privacy/IPDproof')
        put(D/'scores.json',result);put(D/'terminal.json',dict(status='COMPLETE_100_FROZEN_AND_SCORED',epoch=time.time()))
        report=R/'reports/qwen-max-real100-results-2026-10-30.md'
        report.write_text(f'# Qwen Max：公开真实场景图100题，thinking off\n\n模型{qualified["model"]}，STAR真实Charades场景图四选一诊断，100个不同视频。正确{correct}/100（{correct:.2f}%），格式有效{valid}/100。\n\n四类结果：{json.dumps(families,ensure_ascii=False)}\n\n完整事实输入；未上传gold或保护TSG数据，100输出冻结后一次本地评分。它是独立STAR图事实基线，不与旧TSG/Common60或视频盲预测成绩混算，不代表IPD或隐私证明。\n\n费用账：{json.dumps(result["API_budget"],ensure_ascii=False)}。均为保守token价格核算，非已核提供方账单。原累计500USD非追加。\n',encoding='utf-8')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['smoke','run']);a=ap.parse_args()
    try:print(json.dumps(smoke() if a.action=='smoke' else run()))
    except Exception as exc:
        atomic(D/'failure.json',dict(status='AFFECTED_API_STAGE_DIAGNOSE_NO_UNKNOWN_REPLAY',error_type=type(exc).__name__,epoch=time.time()))
        raise SystemExit(type(exc).__name__)
