"""Local, model-free UST supervision contracts.

This validates a *displayed* finite decision problem, not the unrecovered
published UST oracle. Source simulators must separately verify their legal-action
lists. Historical state-based/Monte-Carlo references remain uncertain unless
answerability and optimality are independently established.
"""
from __future__ import annotations
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import json
import re

SCHEMA = 'UST_VISIBLE_DECISION_TREE_V1'
HEADER = ('This is an auxiliary, fully observed finite planning task, not the '
          'original published UST QA. The JSON below defines all legal actions '
          'and immediate rewards. Maximize the discounted sum, starting at '
          'exponent zero, and do not stop while a legal continuation exists. '
          'Return an option in <Answer> tags.\n')

class ContractError(ValueError):
    """A reference cannot be certified from the supplied visible contract."""


def rational(token):
    """Use displayed decimal tokens exactly; reject NaN/Infinity/binary floats."""
    if not isinstance(token, str):
        raise ContractError('Numbers must be explicit decimal strings')
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', token):
        raise ContractError('Nonfinite or unsupported numeric token')
    try:
        value = Decimal(token)
    except InvalidOperation as ex:
        raise ContractError('Invalid decimal') from ex
    if not value.is_finite() or abs(value.adjusted()) > 1000 or len(token) > 200:
        raise ContractError('Nonfinite or unbounded decimal')
    return Fraction(value)


def ranking_answers(values, options, *, nan_policy='reject'):
    """Descending weak order; missing ordering must be explicitly declared.

The caller must bind nan_policy to a visible task rule, never infer it from an
answer. The default refuses missing values rather than inventing their rank.
"""
    if nan_policy not in ('reject', 'largest', 'smallest'):
        raise ContractError('Unknown missing-value convention')
    if not values or not options:
        raise ContractError('Empty ranking problem')
    def key(v):
        if isinstance(v, str) and v.lower() == 'nan':
            if nan_policy == 'reject':
                raise ContractError('Missing value has no visible ordering rule')
            return (1 if nan_policy == 'largest' else -1, Fraction(0))
        return (0, rational(v))
    parsed = {name: key(value) for name, value in values.items()}
    answers = []
    for letter, order in options.items():
        if len(order) != len(parsed) or set(order) != set(parsed):
            raise ContractError('Ranking option is not a permutation')
        if all(parsed[a] >= parsed[b] for a, b in zip(order, order[1:])):
            answers.append(letter)
    return sorted(answers)


def _parse_tree(task):
    fields = {'schema', 'objective', 'reward_semantics', 'horizon', 'discount',
              'options', 'nodes', 'edges'}
    if not isinstance(task, dict) or set(task) != fields:
        raise ContractError('Missing or unknown task fields')
    if task['schema'] != SCHEMA or task['objective'] != 'maximize_discounted_sum':
        raise ContractError('Unknown task or objective')
    if task['reward_semantics'] not in ('incremental_normalized_goal_level',
                                      'post_action_normalized_goal_level',
                                      'declared_immediate_reward'):
        raise ContractError('Reward semantics must be explicit')
    horizon = task['horizon']
    if type(horizon) is not int or not 1 <= horizon <= 16:
        raise ContractError('Invalid horizon')
    discount = rational(task['discount'])
    if not 0 <= discount <= 1:
        raise ContractError('Invalid discount')
    options = task['options']
    if not isinstance(options, dict) or not options or set(options) - set('ABCDE'):
        raise ContractError('Unsupported options')
    if len(set(options.values())) != len(options):
        raise ContractError('Duplicate option action')
    nodes, rewards = {}, {}
    if len(task['nodes']) > 100000 or len(task['edges']) > 100000:
        raise ContractError('Unbounded visible tree')
    def path(row):
        seq = row['path']
        if not isinstance(seq, list) or not all(isinstance(x, str) and re.fullmatch(r'[a-zA-Z0-9_-]{1,30}', x) for x in seq):
            raise ContractError('Invalid action path')
        if len(seq) > horizon:
            raise ContractError('Path exceeds explicit horizon')
        return tuple(seq)
    for row in task['nodes']:
        if set(row) != {'path', 'legal_actions', 'terminal'}:
            raise ContractError('Invalid node fields')
        p = path(row)
        legal = row['legal_actions']
        if not isinstance(legal, list) or not all(isinstance(x, str) and re.fullmatch(r'[a-zA-Z0-9_-]{1,30}', x) for x in legal):
            raise ContractError('Invalid legal action')
        if p in nodes or len(legal) != len(set(legal)):
            raise ContractError('Duplicate node or action')
        terminal = row['terminal']
        if len(p) == horizon:
            if legal or terminal != 'horizon':
                raise ContractError('Horizon node must be terminal')
        elif legal:
            if terminal is not None:
                raise ContractError('A legal continuation cannot be omitted')
        elif terminal != 'no_legal_actions':
            raise ContractError('Early terminal requires no-legal-actions rule')
        nodes[p] = tuple(legal)
    if () not in nodes or set(options.values()) != set(nodes[()]):
        raise ContractError('Root options must equal the displayed legal roots')
    for row in task['edges']:
        if set(row) != {'path', 'reward'}:
            raise ContractError('Invalid edge fields')
        p = path(row)
        if not p or p in rewards:
            raise ContractError('Duplicate or empty edge')
        rewards[p] = rational(row['reward'])
    if set(rewards) != set(nodes) - {()}:
        raise ContractError('Every non-root state needs exactly one reward edge')
    for p, legal in nodes.items():
        if p and (p[:-1] not in nodes or p[-1] not in nodes[p[:-1]]):
            raise ContractError('Orphan or undeclared edge')
        if any(p + (action,) not in nodes for action in legal):
            raise ContractError('Missing declared legal branch')
    return nodes, rewards, discount


def solve_visible_tree(task):
    """Exact backward solution, using only displayed decimal reward tokens."""
    nodes, rewards, discount = _parse_tree(task)
    continuation = {}
    for p in sorted(nodes, key=len, reverse=True):
        continuation[p] = max((rewards[p+(a,)] + discount*continuation[p+(a,)]
                               for a in nodes[p]), default=Fraction(0))
    values = {letter: rewards[(a,)] + discount*continuation[(a,)]
              for letter, a in task['options'].items()}
    best = max(values.values())
    answers = sorted(k for k, value in values.items() if value == best)
    return {'compatible_answers': answers,
            'root_values_exact': {k: str(v) for k, v in values.items()},
            'unique_answer': answers[0] if len(answers) == 1 else None,
            'nodes': len(nodes), 'edges': len(rewards)}


def render_visible_tree(task):
    solve_visible_tree(task)
    return HEADER + json.dumps(task, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def parse_visible_tree(question):
    if not isinstance(question, str) or not question.startswith(HEADER):
        raise ContractError('Question is not the declared visible task')
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ContractError('Duplicate JSON key')
            value[key] = item
        return value
    try:
        task = json.loads(question[len(HEADER):], object_pairs_hook=unique_object)
    except (ValueError, TypeError) as ex:
        raise ContractError('Invalid visible task JSON') from ex
    solve_visible_tree(task)
    return task


def certify_visible_reference(question, proposed_answer):
    """Certification requires the actual rendered input, never hidden metadata."""
    solved = solve_visible_tree(parse_visible_tree(question))
    trusted = solved['unique_answer'] is not None and proposed_answer == solved['unique_answer']
    return dict(solved, trusted_gold=trusted,
                status='trusted_unique' if trusted else 'ambiguous' if solved['unique_answer'] is None else 'label_mismatch',
                scope='displayed_auxiliary_task_only_not_original_UST_gold')


def assess_legacy_reference(*, oracle, visible_sufficiency_proven=False,
                            exact_optimality_proven=False, prefix_agrees=None):
    """Conservative metadata assessment, not a certification API.

Even both True cannot issue trusted_gold: use an executable visible-input
verifier. MC prefix agreement is reported but never elevated to correctness.
"""
    reasons = []
    if not visible_sufficiency_proven:
        reasons.append('visible_input_sufficiency_unproven')
    if not exact_optimality_proven:
        reasons.append('optimality_unproven')
    if oracle == 'sampled_H5' and prefix_agrees is False:
        reasons.append('sampled_label_changes_with_search_budget')
    return {'oracle': oracle, 'trusted_gold': False, 'status': 'uncertain_reference',
            'prefix_agrees': prefix_agrees,
            'reasons': reasons or ['requires_executable_visible_input_verification']}

RANK_SCHEMA = 'UST_VISIBLE_RANKING_V1'
RANK_HEADER = ('This is a fully observed descending-ranking task. The JSON below '
               'defines all displayed values, candidate rankings and missing-value '
               'semantics. Return an option in <Answer> tags.\n')


def _ranking_task_answers(task):
    if not isinstance(task, dict) or set(task) != {'schema','direction','values','options','nan_policy'}:
        raise ContractError('Missing or unknown ranking fields')
    if task['schema'] != RANK_SCHEMA or task['direction'] != 'descending':
        raise ContractError('Unsupported ranking semantics')
    return ranking_answers(task['values'], task['options'], nan_policy=task['nan_policy'])


def render_visible_ranking(task):
    _ranking_task_answers(task)
    return RANK_HEADER + json.dumps(task,sort_keys=True,separators=(',',':'),allow_nan=False)


def certify_ranking_reference(question, proposed_answer):
    if not isinstance(question, str) or not question.startswith(RANK_HEADER):
        raise ContractError('Question is not the declared visible ranking task')
    def unique_object(pairs):
        obj = {}
        for k,v in pairs:
            if k in obj:raise ContractError('Duplicate JSON key')
            obj[k] = v
        return obj
    try:
        task = json.loads(question[len(RANK_HEADER):],object_pairs_hook=unique_object)
    except (ValueError,TypeError) as ex:
        raise ContractError('Invalid ranking JSON') from ex
    answers = _ranking_task_answers(task)
    unique = answers[0] if len(answers) == 1 else None
    trusted = unique is not None and proposed_answer == unique
    return {'compatible_answers':answers,'unique_answer':unique,'trusted_gold':trusted,
            'status':'trusted_unique' if trusted else 'ambiguous' if unique is None else 'label_mismatch',
            'scope':'displayed_ranking_task_only'}


# Static text from the already-used public UST understanding template.
# These allowlisted strings contain no task values, answers, or protected rows.
NATIVE_UNDERSTANDING_DESCRIPTION = 'We are tasked with determining the optimal location for a new electric vehicle (EV) charging station to maximize benefits by effectively meeting demand. The candidate locations, along with their expected charging station coverage, demand (number of vehicles), average distance, travel time, and waiting time to existing charging stations, are provided below.'
NATIVE_UNDERSTANDING_SCHEMA = """- coverage: The number of locations that will be within the influence radius of the charging station after its construction.
- demand: The number of vehicles requiring charging near the location.
- distance: The average distance that vehicles near the location must travel to reach existing charging stations.
- waiting_time: The average waiting time at nearby existing charging stations. NaN indicates no station nearby and the highest waiting time.
- charging_time: The average charging time at nearby existing charging stations. NaN indicates no station nearby and the highest charging time."""
NATIVE_UNDERSTANDING_NOTE = 'Please answer the question using the information given. Let’s work this out in a step-by-step way to be sure that we have the right answer. Then provide your final answer within the XML tag <Answer>A/B/C/D</Answer>.'


# Separately allowlisted learned-generator template. The extra travel_time
# schema field has no displayed Data column; certification is target-specific
# ranking only and reports this historical defect explicitly.
LEARNED_UNDERSTANDING_SCHEMA = NATIVE_UNDERSTANDING_SCHEMA.replace(
    '- waiting_time:',
    '- travel_time: The average travel time for vehicles near the location to reach existing charging stations.\n- waiting_time:')
LEARNED_UNDERSTANDING_NOTE = 'Return only <Answer>A</Answer>, replacing A with your option.'


def _static_text(text):
    # Ignore only typography. Never delete prose, negation, punctuation or fields.
    return ' '.join(text.split())


def certify_native_ranking_reference(question, proposed_answer):
    """Verify supported native ranking text without using stored simulator state.

Exact recognized question wording and a finite schema keep this a deliberately
bounded verifier. Unknown forms raise rather than being guessed. NaN ordering
is accepted only when the corresponding visible schema explicitly states it.
"""
    if not isinstance(question, str):
        raise ContractError('Native question must be text')
    matches=list(re.finditer(r'^## ([^\n]+)\n',question,re.M))
    if not matches or question[:matches[0].start()].strip():
        raise ContractError('Unsupported native preamble')
    parts={}
    for i,m in enumerate(matches):
        name=m.group(1).strip()
        if name in parts:raise ContractError('Duplicate native section')
        parts[name]=question[m.end():matches[i+1].start() if i+1<len(matches) else len(question)].strip()
    families = (
      ('original_understanding', ['Task Description','Data Schema','Data','Question','Note'],
       NATIVE_UNDERSTANDING_SCHEMA, NATIVE_UNDERSTANDING_NOTE, None),
      ('learned_understanding', ['Task Description','Data Schema','Data','Analysis','Question','Note'],
       LEARNED_UNDERSTANDING_SCHEMA, LEARNED_UNDERSTANDING_NOTE, 'N/A'))
    matched=[]
    for family, names, schema, note, analysis in families:
        if (list(parts)==names and
            _static_text(parts['Task Description'])==_static_text(NATIVE_UNDERSTANDING_DESCRIPTION) and
            _static_text(parts['Data Schema'])==_static_text(schema) and
            _static_text(parts['Note'])==_static_text(note) and
            (analysis is None or parts['Analysis']==analysis)):
            matched.append(family)
    if len(matched)!=1:
        raise ContractError('Unrecognized or conflicting complete native template')
    template_family=matched[0]
    opts=list(re.finditer(r'^([A-D])\.\s*([1-4](?:\s*>\s*[1-4]){3})\s*$',parts['Question'],re.M))
    if len(opts)!=4 or {m.group(1) for m in opts}!=set('ABCD'):
        raise ContractError('Invalid native ranking options')
    if any(parts['Question'][a.end():b.start()].strip() for a,b in zip(opts,opts[1:])) or parts['Question'][opts[-1].end():].strip():
        raise ContractError('Unsupported trailing option text')
    prefix=parts['Question'][:opts[0].start()].strip()
    phrases={
      'distance':'average distance to nearby charging stations, from the farthest to the closest.',
      'waiting_time':'average waiting time at near charging stations, from the longest to the shortest.',
      'charging_time':'average charging time at near charging stations, from the longest to the shortest.'}
    metrics=[k for k,v in phrases.items() if prefix=='Based on the data provided, rank the locations by '+v]
    if len(metrics)!=1:raise ContractError('Unsupported or ambiguous native ranking objective')
    metric=metrics[0]
    blocks=list(re.finditer(r'^location ([1-4]):\s*$',parts['Data'],re.I|re.M))
    if len(blocks)!=4 or {m.group(1) for m in blocks}!=set('1234') or parts['Data'][:blocks[0].start()].strip():
        raise ContractError('Invalid native data locations')
    values={};missing=0;unit='m' if metric=='distance' else 'h'
    for i,m in enumerate(blocks):
        body=parts['Data'][m.end():blocks[i+1].start() if i+1<len(blocks) else len(parts['Data'])]
        # Consume the entire block, including every non-target field. Otherwise
        # a hidden correction after a numeric row could be silently ignored.
        lines=body.strip().splitlines()
        fields=('coverage','demand','distance','waiting_time','charging_time')
        if len(lines)!=len(fields):raise ContractError('Extra or missing native Data text')
        field_values={}
        for field,line in zip(fields,lines):
            full=re.fullmatch(r'- '+field+r': (NaN(?: ?h)?|[^\s]+)',line.strip())
            if not full:raise ContractError('Unrecognized, repeated or reordered native field')
            value=full.group(1)
            suffix='m' if field=='distance' else 'h' if field.endswith('_time') else ''
            if re.fullmatch(r'NaN(?: ?h)?',value):
                if field not in ('waiting_time','charging_time'):
                    raise ContractError('Undefined missing-value semantics')
            else:
                token=re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))'+suffix,value)
                if not token or rational(token.group(1))<0:
                    raise ContractError('Invalid native metric, unit or domain')
            field_values[field]=value
        raw=field_values[metric]
        if re.fullmatch(r'NaN(?: ?h)?',raw):values[m.group(1)]='NaN';missing+=1
        else:
            number=re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))'+unit,raw)
            if not number:raise ContractError('Unsupported native value or unit')
            values[m.group(1)]=number.group(1)
    nan_policy='reject'
    if missing:
        suffix={'waiting_time':'waiting time','charging_time':'charging time'}.get(metric)
        if suffix is None:raise ContractError('No explicit missing-distance order')
        lines=[x for x in parts['Data Schema'].splitlines() if x.startswith('- '+metric+':')]
        required='NaN indicates no station nearby and the highest '+suffix+'.'
        if len(lines)!=1 or not lines[0].endswith(required):
            raise ContractError('Missing native NaN ordering declaration')
        nan_policy='largest'
    options={m.group(1):re.findall('[1-4]',m.group(2)) for m in opts}
    answers=ranking_answers(values,options,nan_policy=nan_policy)
    unique=answers[0] if len(answers)==1 else None
    trusted=unique is not None and proposed_answer==unique
    return {'compatible_answers':answers,'unique_answer':unique,'trusted_gold':trusted,
      'status':'trusted_unique' if trusted else 'ambiguous' if unique is None else 'label_mismatch',
      'metric':metric,'missing_values':missing,'nan_policy':nan_policy,
      'template_family':template_family,
      'warnings':['schema_mentions_travel_time_not_present_in_Data'] if template_family=='learned_understanding' else [],
      'scope':'recognized_native_displayed_ranking_only_not_full_schema_fidelity'}
