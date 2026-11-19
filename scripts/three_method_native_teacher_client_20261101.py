"""Local native Qwen logits pipe client. No API, token remapping or score imputation."""
from pathlib import Path
import hashlib,json,math,os,selectors,subprocess,time
SEMANTIC=248077;NATIVE=248320
class NativeTeacher:
    def __init__(self,binary,model,backend,stderr,ngl,threads=2,guard=lambda:None,timeout=300,qualification=False):
        self.guard=guard;self.timeout=timeout;self.proc=None;self.err=None;self.selector=None;self.calls=[]
        assert 0<=ngl<=64 and 1<=threads<=4 and 0<timeout<=900
        for path in (Path(binary),Path(model),Path(backend)):assert path.exists() and not path.is_symlink()
        # Caller must bind actual verified artifact/resource grant before constructing.
        # This class never declares a model qualified from a successful constructor.
        self.guard();self.err=Path(stderr).open('x');env=dict(os.environ)
        env['LD_LIBRARY_PATH']=str(backend)+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        try:
            self.proc=subprocess.Popen([str(binary),str(model),str(backend),str(ngl),str(threads),'0'],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,text=True,bufsize=1,env=env)
            self.selector=selectors.DefaultSelector();self.selector.register(self.proc.stdout,selectors.EVENT_READ)
            self.ready=self._receive();assert self.ready['status']=='READY' and self.ready['vocab_only'] is False
            assert self.ready['semantic_n']==SEMANTIC and self.ready['unused']==243
            self.template=bytes.fromhex(self.ready['template_hex']).decode('utf-8')
            self.template_SHA256=hashlib.sha256(self.template.encode()).hexdigest()
            self.qualification_mode=qualification
        except BaseException:self.close();raise
    def _receive(self):
        begin=time.monotonic()
        while True:
            self.guard()
            if self.proc.poll() is not None:raise RuntimeError('Native teacher exited; preserve stderr, no model restart here')
            if time.monotonic()-begin>self.timeout:raise TimeoutError('Bounded native teacher call; no fabricated feedback')
            if self.selector.select(.25):
                line=self.proc.stdout.readline(1048577)
                if not line:raise RuntimeError('Native teacher closed output')
                if len(line)>1048576 or not line.endswith('\n'):raise RuntimeError('Bounded complete native reply')
                return json.loads(line)
    def _call(self,line):
        self.guard();begin=time.monotonic();self.proc.stdin.write(line+'\n');self.proc.stdin.flush()
        value=self._receive();self.calls.append(dict(command=line.split(' ',1)[0],seconds=time.monotonic()-begin));return value
    @staticmethod
    def ids(ids,maximum):
        assert isinstance(ids,(list,tuple)) and 0<len(ids)<=maximum
        assert all(type(i) is int and 0<=i<SEMANTIC for i in ids)
        return ids
    @staticmethod
    def slot(slot):assert type(slot) is int and 0<=slot<2
    def score(self,prefix,slot=0):
        self.slot(slot);self.ids(prefix,16384)
        value=self._call('SCORE '+str(slot)+' '+str(len(prefix))+' '+' '.join(map(str,prefix)))
        assert value['status']=='RAW_NATIVE_LOGITS' and value['n_vocab']==NATIVE and value['semantic_n']==SEMANTIC
        assert len(value['top16_ids'])==len(set(value['top16_ids']))==len(value['top16_raw_logits'])==16
        self.ids(value['top16_ids'],16)
        assert all(math.isfinite(v) for v in value['top16_raw_logits'])
        assert math.isfinite(value['logZ_semantic']) and math.isfinite(value['logZ_native'])
        assert 0<=value['unused_mass_native']<=1 and value['logZ_semantic']<=value['logZ_native']+1e-7
        assert type(value['new_decoded_tokens']) is int and 0<=value['new_decoded_tokens']<=len(prefix)
        return value
    def gather(self,ids,slot=0):
        self.slot(slot);self.ids(ids,32);assert len(ids)==len(set(ids))
        value=self._call('GATHER '+str(slot)+' '+str(len(ids))+' '+' '.join(map(str,ids)))
        assert value['status']=='GATHER' and len(value['raw_logits'])==len(ids)
        assert math.isfinite(value['logZ_semantic']) and all(math.isfinite(v) for v in value['raw_logits'])
        return value
    def sample(self,uniform,slot=0):
        self.slot(slot);assert type(uniform) in (float,int) and math.isfinite(uniform) and 0<=uniform<1
        value=self._call('SAMPLE '+str(slot)+' 1 '+format(uniform,'.17g'))
        assert value['status']=='SAMPLE_FULL_REGISTERED' and value['native_semantic_domain']==SEMANTIC
        self.ids([value['token_id']],1)
        assert math.isfinite(value['log_probability_semantic']) and abs(value['uniform']-uniform)<=1e-16
        return value
    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait(timeout=5)
        if self.selector is not None:self.selector.close()
        if self.err is not None:self.err.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()