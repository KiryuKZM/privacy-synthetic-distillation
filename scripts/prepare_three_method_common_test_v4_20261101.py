"""Current actual v4/C2 dependency adaptation; original1569 QA construction unchanged."""
from pathlib import Path
import argparse,ast,sys
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
from three_method_native_ipd_v4_20261101 import e
import prepare_three_method_common_test_20261101 as b
ORIGINAL=R/'scripts/prepare_three_method_common_test_20261101.py';b.P=e.O/'common-test-v4';b.__file__=str(Path(__file__).resolve())
node=next(x for x in ast.parse(ORIGINAL.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='actual_endpoints_ready')
changed=0
for x in ast.walk(node):
 if isinstance(x,ast.Constant) and isinstance(x.value,str):
  replacement={'student-controller-v3/terminal.json':'student-controller-v5/terminal.json','latest-ipd-controller-v1/terminal.json':'latest-ipd-controller-v4/terminal.json'}
  if x.value in replacement:x.value=replacement[x.value];changed+=1
assert changed==2
ns=dict(b.__dict__);exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(ORIGINAL)+'::exact2-dependency-strings','exec'),ns);b.actual_endpoints_ready=ns['actual_endpoints_ready']

def prepare():
 b.prepare();old=e.O/'common-test-v1/old-endpoints8-frozen.json';new=b.P/'old-endpoints8-frozen.json'
 with new.open('xb') as f:f.write(old.read_bytes())
 assert e.sha(old)==e.sha(new)
 e.once(b.P/'current-dependency-root.json',dict(approved=True,actor='Root',source=e.bind(Path(__file__)),
  bindings=[e.bind(p) for p in [ORIGINAL,b.P/'proposal.json',old,new,e.O/'student-controller-v5/root-grant.json',e.P/'root-delegation.json']],
  exactly_two_actual_dependency_strings_changed=True,question_construction_and_seeds_and_old_endpoints_unchanged=True,actual_Test_feature_payload_decoded=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','freeze']);v=p.parse_args()
 if v.action=='prepare':prepare()
 else:
  g=e.load(b.P/'current-dependency-root.json');assert g['source']==e.bind(Path(__file__))
  for x in g['bindings']:assert e.bind(R/x['path'])==x,x['path']
  b.freeze()