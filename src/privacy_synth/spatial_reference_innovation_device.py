"""Reference-only schedule view; original full neural mean/schedule remain unchanged."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,json
import torch
from privacy_synth import spatial_reference_innovation as original
from privacy_synth.spatial_reference_innovation import reference_innovation

def reference_schedule_view(reference,schedule):
    if schedule.steps!=64 or reference.bar.shape!=schedule.bar.shape or reference.bar.dtype!=schedule.bar.dtype or reference.bar.device!=schedule.bar.device:
        raise ValueError('Matching original64-step schedule dtype/device required')
    error=(reference.bar-schedule.bar).abs();epsilon=torch.finfo(reference.bar.dtype).eps
    if not bool(torch.isfinite(error).all()) or bool((error>4*epsilon).any()) or bool((error/reference.bar>4*epsilon).any()):
        raise ValueError('Schedule discrepancy exceeds bounded floating roundoff')
    return SimpleNamespace(steps=schedule.steps,beta=schedule.beta,bar=reference.bar)

root=Path(__file__).resolve().parents[2];path=Path(original.__file__);protocol=json.loads((root/'runs/2026-10-31-spatial-reference-innovation-v1/protocol.json').read_text());binding=next(b for b in protocol['bindings'] if b['path']==path.relative_to(root).as_posix())
if hashlib.sha256(path.read_bytes()).hexdigest()!=binding['sha256']:raise PermissionError('Frozen v1 innovation math drift')
fn=copy.deepcopy(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='sample_reference_innovation'));baseline=copy.deepcopy(fn);changes=0
for n in ast.walk(fn):
    if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='reference_innovation':
        if not isinstance(n.args[4],ast.Name) or n.args[4].id!='schedule':raise PermissionError('Original innovation schedule seam required')
        n.args[4]=ast.Name(id='reference_schedule',ctx=ast.Load());changes+=1
loop_index=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.For));fn.body.insert(loop_index,ast.parse('reference_schedule=reference_schedule_view(model.reference,schedule)').body[0]);inverse=copy.deepcopy(fn)
inverse.body=[n for n in inverse.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='reference_schedule' for t in n.targets))]
for n in ast.walk(inverse):
    if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='reference_innovation':n.args[4]=ast.Name(id='schedule',ctx=ast.Load())
if changes!=1 or ast.dump(inverse)!=ast.dump(baseline):raise PermissionError('Only innovation schedule-view seam may differ; full mean/RNG/model queries unchanged')
namespace=dict(original.__dict__,reference_schedule_view=reference_schedule_view);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(__file__),'exec'),namespace);sample_reference_innovation=namespace['sample_reference_innovation']
ADAPTATION_PROOF=dict(innovation_schedule_argument_changes=1,reference_view_created_once_before_loop=True,full_mean_modelquery_categoryupdate_RNG_exact_inverse_AST=True,model_buffers_not_mutated=True,max_rounding_absolute_and_relative_eps=4)
