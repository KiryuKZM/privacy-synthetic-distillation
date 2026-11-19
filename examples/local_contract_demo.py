"""Invented numeric fixture; no original UST row and no API/model call."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from privacy_synth.ust_task_contract import ranking_answers, ContractError
values = {'p1': '2', 'p2': '7', 'p3': '4', 'p4': '1'}
options = {'A': ['p2', 'p3', 'p1', 'p4'], 'B': ['p4', 'p1', 'p3', 'p2']}
assert ranking_answers(values, options) == ['A']
print('Invented descending-ranking fixture: A')
try:
    ranking_answers(dict(values, p2='NaN'), options)
except ContractError:
    print('Missing value without an explicit ordering rule: correctly rejected')
else:
    raise AssertionError('Ambiguous missing-value rule was accepted')
