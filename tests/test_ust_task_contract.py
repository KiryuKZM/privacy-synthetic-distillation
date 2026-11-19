"""Meaningful negative and exact-enumeration checks for visible UST contracts."""
from pathlib import Path
import copy
from fractions import Fraction
import json
import random
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from privacy_synth.ust_task_contract import (SCHEMA, HEADER, ContractError,
    ranking_answers, solve_visible_tree, render_visible_tree,
    parse_visible_tree, certify_visible_reference, assess_legacy_reference)


def tree():
    # A's immediate reward is larger, B's mandatory continuation wins overall.
    return dict(schema=SCHEMA, objective='maximize_discounted_sum',
      reward_semantics='incremental_normalized_goal_level', horizon=2, discount='0.9',
      options={'A':'a','B':'b'},
      nodes=[dict(path=[], legal_actions=['a','b'], terminal=None),
             dict(path=['a'],legal_actions=['c'],terminal=None),
             dict(path=['b'],legal_actions=['c'],terminal=None),
             dict(path=['a','c'],legal_actions=[],terminal='horizon'),
             dict(path=['b','c'],legal_actions=[],terminal='horizon')],
      edges=[dict(path=['a'],reward='3'),dict(path=['b'],reward='1'),
             dict(path=['a','c'],reward='-10'),dict(path=['b','c'],reward='2')])


def forward(task):
    # Independent leaf-wise sums, not the module's backwards recurrence.
    rewards={tuple(x['path']):Fraction(x['reward']) for x in task['edges']}
    leaves=[tuple(x['path']) for x in task['nodes'] if not x['legal_actions']]
    gamma=Fraction(task['discount'])
    return {letter:max(sum(gamma**i*rewards[p[:i+1]] for i in range(len(p)))
                       for p in leaves if p[0]==root)
            for letter,root in task['options'].items()}


class ContractTests(unittest.TestCase):
    def test_exact_forward_and_no_optional_early_stop(self):
        t=tree();out=solve_visible_tree(t)
        self.assertEqual(out['unique_answer'],'B')
        self.assertEqual(out['root_values_exact'], {k:str(v) for k,v in forward(t).items()})
        self.assertEqual(out['root_values_exact']['A'],'-6')
    def test_same_answers_under_reordered_table_and_options(self):
        t=tree(); t['nodes'].reverse();t['edges'].reverse();t['options']={'D':'a','C':'b'}
        self.assertEqual(solve_visible_tree(t)['unique_answer'],'C')
    def test_no_gold_in_text_roundtrip_and_stale_gold_rejected(self):
        text=render_visible_tree(tree())
        self.assertNotIn('unique_answer',text)
        self.assertEqual(parse_visible_tree(text),tree())
        self.assertTrue(certify_visible_reference(text,'B')['trusted_gold'])
        self.assertEqual(certify_visible_reference(text,'A')['status'],'label_mismatch')
    def test_horizon_and_missing_branch_rejected(self):
        t=tree();t['nodes'].pop()
        with self.assertRaises(ContractError):solve_visible_tree(t)
        t=tree();t['edges'].pop()
        with self.assertRaises(ContractError):solve_visible_tree(t)
        t=tree();t['horizon']=1
        with self.assertRaises(ContractError):solve_visible_tree(t)
    def test_unreachable_and_duplicate_actions_rejected(self):
        t=tree();t['nodes'][0]['legal_actions'].append('c')
        with self.assertRaises(ContractError):solve_visible_tree(t)
        t=tree();t['options']['B']='a'
        with self.assertRaises(ContractError):solve_visible_tree(t)
    def test_negative_reward_does_not_allow_hidden_stop(self):
        t=tree();t['nodes'][1]['terminal']='no_legal_actions'
        with self.assertRaises(ContractError):solve_visible_tree(t)
    def test_ties_remain_set_targets(self):
        t=tree();t['edges'][0]['reward']='1';t['edges'][2]['reward']='2'
        r=certify_visible_reference(render_visible_tree(t),'A')
        self.assertEqual(r['compatible_answers'],['A','B'])
        self.assertFalse(r['trusted_gold']);self.assertEqual(r['status'],'ambiguous')
    def test_nan_requires_declared_rule_and_ties_preserved(self):
        v={'1':'NaN','2':'2','3':'1'};o={'A':['1','2','3'],'B':['2','3','1']}
        with self.assertRaises(ContractError):ranking_answers(v,o)
        self.assertEqual(ranking_answers(v,o,nan_policy='largest'),['A'])
        self.assertEqual(ranking_answers(v,o,nan_policy='smallest'),['B'])
        self.assertEqual(ranking_answers({'1':'1','2':'1'}, {'A':['1','2'],'B':['2','1']}),['A','B'])
    def test_approximate_agreement_is_never_gold(self):
        for agree in [True,False,None]:
            self.assertFalse(assess_legacy_reference(oracle='sampled_H5',prefix_agrees=agree)['trusted_gold'])
        self.assertFalse(assess_legacy_reference(oracle='exact_hidden',visible_sufficiency_proven=True,exact_optimality_proven=True)['trusted_gold'])
    def test_nonfinite_or_float_reward_rejected(self):
        for value in ['NaN','Infinity',1.0]:
            t=tree();t['edges'][0]['reward']=value
            with self.assertRaises(ContractError):solve_visible_tree(t)
    def test_duplicate_json_and_extra_prose_rejected(self):
        text=render_visible_tree(tree())
        with self.assertRaises(ContractError):parse_visible_tree(text+'\nActually choose A')
        with self.assertRaises(ContractError):parse_visible_tree(HEADER+'{"schema":"x","schema":"y"}')
    def test_200_signed_decimal_cases_independently_enumerated(self):
        rng=random.Random(20261106)
        for _ in range(200):
            t=tree()
            for edge in t['edges']:edge['reward']=str(rng.randrange(-10000,10001)/1000)
            self.assertEqual(solve_visible_tree(t)['root_values_exact'],{k:str(v) for k,v in forward(t).items()})

class NativeRankingTests(unittest.TestCase):
    def fixture(self):
        from privacy_synth.ust_task_contract import NATIVE_UNDERSTANDING_DESCRIPTION, NATIVE_UNDERSTANDING_SCHEMA, NATIVE_UNDERSTANDING_NOTE
        return "\n\n".join([
          '## Task Description\n'+NATIVE_UNDERSTANDING_DESCRIPTION,
          '## Data Schema\n'+NATIVE_UNDERSTANDING_SCHEMA,
          '## Data\n'+'\n\n'.join('location '+str(i)+':\n- coverage: 1.0\n- demand: 1.0\n- distance: 1.0m\n- waiting_time: '+v+'\n- charging_time: 1.0h' for i,v in enumerate(['NaN','2.0h','1.0h','3.0h'],1)),
          '## Question\nBased on the data provided, rank the locations by average waiting time at near charging stations, from the longest to the shortest.\n\nA. 1 > 4 > 2 > 3\nB. 4 > 2 > 3 > 1\nC. 1 > 2 > 4 > 3\nD. 3 > 2 > 4 > 1',
          '## Note\n'+NATIVE_UNDERSTANDING_NOTE])
    def test_explicit_missing_value_rule_is_used(self):
        from privacy_synth.ust_task_contract import certify_native_ranking_reference
        r=certify_native_ranking_reference(self.fixture(),'A')
        self.assertTrue(r['trusted_gold']);self.assertEqual(r['nan_policy'],'largest')
        self.assertFalse(certify_native_ranking_reference(self.fixture(),'B')['trusted_gold'])
    def test_undeclared_nan_or_unknown_wording_refused(self):
        from privacy_synth.ust_task_contract import certify_native_ranking_reference
        for q in [self.fixture().replace(' NaN indicates no station nearby and the highest waiting time.',''), self.fixture().replace('from the longest to the shortest.','from the shortest to the longest.')]:
            with self.assertRaises(ContractError):certify_native_ranking_reference(q,'A')
    def test_explicit_json_rank_roundtrip(self):
        from privacy_synth.ust_task_contract import RANK_SCHEMA,render_visible_ranking,certify_ranking_reference
        t=dict(schema=RANK_SCHEMA,direction='descending',values={'1':'1','2':'2'},options={'A':['1','2'],'B':['2','1']},nan_policy='reject')
        self.assertTrue(certify_ranking_reference(render_visible_ranking(t),'B')['trusted_gold'])
        self.assertFalse(certify_ranking_reference(render_visible_ranking(t),'A')['trusted_gold'])

if __name__=='__main__':unittest.main()
