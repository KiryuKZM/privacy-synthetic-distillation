"""Uncoupled exact native sampler; avoids legacy CTX/logging/seed-reset wrapper.
Uses the already seeded current W/actor CUDA RNG stream, no extra per-draw reseeding.
"""
from pathlib import Path
import ast,hashlib,sys
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'scripts'),str(R/'src')]
import three_method_native_student_v2_20261101 as m
SOURCE=R/'scripts/train_ust_sg_native_gad_20261024.py'
EXPECTED='fd04244b6d288dd3b6412a87eaca46f319d7dfe950ae50346959b419b8c8ce16'

def uncoupled_sampler(overrides=None):
 assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==EXPECTED
 node=next(x for x in ast.parse(SOURCE.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='rollouts')
 ns=dict(m.lib.setup.__globals__)
 if overrides:ns.update(overrides)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE)+'::unchanged-native-rollouts','exec'),ns)
 return ns['rollouts']