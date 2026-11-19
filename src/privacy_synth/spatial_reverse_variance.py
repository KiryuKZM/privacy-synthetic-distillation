"""Scheduler variance ablation without changing frozen denoising means or weights."""
import ast,copy,hashlib
from pathlib import Path

def make_schedule(mode='fixed_small',steps=64,device='cpu'):
    from privacy_synth import spatial_set_diffusion as core
    if mode=='fixed_small':
        schedule=core.Schedule(steps,device);schedule.adaptation_proof=dict(mode=mode,original=True);return schedule
    if mode!='fixed_large':raise ValueError('Explicit fixed_small/fixed_large mode required')
    path=Path(core.__file__);tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Schedule');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='posterior');original=copy.deepcopy(fn);previous=None;count=0
    for n in ast.walk(fn):
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='variance':
            previous=copy.deepcopy(n.value);n.value=ast.parse('self.beta[t]',mode='eval').body;count+=1
    if count!=1:raise PermissionError('Exactly one variance assignment required')
    inverse=copy.deepcopy(fn)
    for n in ast.walk(inverse):
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='variance':n.value=copy.deepcopy(previous)
    if ast.dump(inverse)!=ast.dump(original):raise PermissionError('Reverse mean/finalstep changed')
    namespace=dict(core.__dict__);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path),'exec'),namespace)
    derived=type('FixedLargeSchedule',(core.Schedule,),dict(posterior=namespace['posterior']));schedule=derived(steps,device)
    schedule.adaptation_proof=dict(mode=mode,original_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),only_variance_expression_changed=True,all_mean_math_unchanged=True,final_step_still_deterministic=True)
    return schedule
