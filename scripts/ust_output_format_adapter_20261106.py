"""Strict output-only UST interface experiment; never changes old raw/scoring.

Only a stopped sole XML A-D answer on planning/reflection can be converted to
JSON with an empty summary. No reference answer or prompt is an adapter input.
"""
from pathlib import Path
import argparse,collections,hashlib,json,re,sys,time,unittest
R=Path(__file__).resolve().parents[1]
B=R/'runs/2026-11-06-ust-autonomous-repair-v1'
O=B/'output-interface-repair-v1'
EOS=('<|im_end|>','<|endoftext|>')

def normalize_output(raw_generated, ability, finish_reason):
    if not isinstance(raw_generated,str):raise TypeError('raw_generated must be text')
    result=dict(text=raw_generated,converted=False,reason='not_eligible',extracted_answer=None,
                summary_synthesized=False,meaning='format-only interface; no reasoning or summary generation')
    if ability not in ('planning','reflection'):
        result['reason']='ability_unchanged';return result
    if finish_reason!='stop':
        result['reason']='nonstop_unchanged';return result
    body=raw_generated.strip();removed=None
    for token in EOS:
        if body.endswith(token):
            body=body[:-len(token)].rstrip();removed=token;break
    match=re.fullmatch(r'<Answer>([A-D])</Answer>',body)
    if not match:
        result['reason']='not_sole_exact_XML_unchanged';return result
    letter=match.group(1)
    result.update(text=json.dumps(dict(answer=letter,summary=''),separators=(',',':')),
                  converted=True,reason='sole_XML_to_JSON_empty_summary',extracted_answer=letter,
                  removed_terminal_EOS=removed)
    return result

class AdapterTests(unittest.TestCase):
    def test_positive_stopped_exact_singletons(self):
        for ability in ('planning','reflection'):
            for answer in 'ABCD':
                for eos in ('',*EOS):
                    raw='<Answer>'+answer+'</Answer>'+eos
                    z=normalize_output(raw,ability,'stop')
                    self.assertTrue(z['converted']);self.assertEqual(json.loads(z['text']),dict(answer=answer,summary=''))
                    self.assertFalse(z['summary_synthesized'])
                    again=normalize_output(z['text'],ability,'stop')
                    self.assertFalse(again['converted']);self.assertEqual(again['text'],z['text'])
    def test_allowed_outer_whitespace(self):
        z=normalize_output(' \n<Answer>C</Answer> \n<|im_end|> \n','reflection','stop')
        self.assertTrue(z['converted']);self.assertEqual(z['extracted_answer'],'C')
    def test_malformed_or_ambiguous_unchanged(self):
        fixtures=['<Answer>A</Answer><Answer>B</Answer>', '<Answer>A</Answer> because demand',
          'Answer: <Answer>A</Answer>', '<Answer>E</Answer>', '<Answer>a</Answer>', '<Answer> A </Answer>',
          '<answer>A</answer>', '<Answer>A/B</Answer>', '<Answer>A</Answer><|im_end|><|im_end|>',
          '<Answer>A</Answer><|im_end|><|endoftext|>', '<|im_end|><Answer>A</Answer>',
          '<Answer><|im_end|>A</Answer>', '<think>A</think><Answer>A</Answer>',
          '```xml\n<Answer>A</Answer>\n```', '<Answer>A</Answer>\n{"answer":"B","summary":""}',
          '<Answer>A</Answer><!-- note -->', '', '<Answer>Ａ</Answer>']
        for raw in fixtures:
            z=normalize_output(raw,'planning','stop');self.assertFalse(z['converted']);self.assertEqual(z['text'],raw)
    def test_valid_JSON_and_noneligible_unchanged(self):
        examples=['{"answer":"A","summary":""}', '{"summary":"A valid reason.","answer":"C"}<|im_end|>',
                  '{"answer":"A","answer":"B","summary":""}', '<Answer>B</Answer><|im_end|>']
        for raw in examples:
            if raw.startswith('{'):
                z=normalize_output(raw,'planning','stop');self.assertFalse(z['converted']);self.assertEqual(z['text'],raw)
            for ability,finish in [('st_understanding','stop'),('unknown','stop'),('planning','length'),('reflection','error')]:
                z=normalize_output(raw,ability,finish);self.assertFalse(z['converted']);self.assertEqual(z['text'],raw)

def load(p):return json.loads(Path(p).read_text())
def bind(p):
    p=Path(p);return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def checked(pin):
    p=R/pin['path'];assert bind(p)==pin;return p

def sources():
    pos=B/'position-training-v2';causal=B/'causal-training'
    return {**{k:(pos,pos/k) for k in ['fresh-init0','real-canonical200','real-balanced200','synthetic-balanced200']},
            **{k:(causal,causal/k) for k in ['A-original200-replay','B-understanding-real-template200','C-understanding-only68']}}

def selftest():
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AdapterTests))
    if not result.wasSuccessful():raise AssertionError('Adapter fixture failure')
    return dict(status='PASS',test_methods=result.testsRun,positive_exact_singletons=24,negative_ambiguous_fixtures=18,idempotence_checked=True)

def prepare():
    tests=selftest();O.mkdir(parents=True,exist_ok=False)
    units={}
    for name,(root,unit) in sources().items():
        term=load(unit/'evaluation-terminal.json');assert term['raw']==212
        units[name]=dict(root=str(root.relative_to(R)),unit=str(unit.relative_to(R)),
          queries=bind(root/'queries.local.json'),answers=bind(root/'answers.local.json'),
          old_grades=bind(unit/'graded.local.json'),closed_terminal=bind(unit/'evaluation-terminal.json'),
          raw=[bind(unit/'raw'/f'{i:04d}.json') for i in range(212)])
    scorer=[R/'scripts'/n for n in ['eval_ust_sg_qwen_modes_20261024.py','run_llm_task_sweep_20261023.py','build_llm_task_sweep_qa_20261023.py']]
    score_deps=scorer+[R/'src/privacy_synth/choice_answer.py']
    epoch=time.time()
    p=dict(status='RULE_AND_ALL1484_INPUTS_FROZEN_BEFORE_INTERFACE_SCORE',epoch=epoch,source=bind(Path(__file__)),
      rule='Only planning/reflection, finish_reason=stop, outer whitespace removed and at most one terminal nativeEOS removed, then fullmatch exact <Answer>[A-D]</Answer>. Converted JSON has answer and empty summary. All other raw text byte-identical.',
      adapter_inputs=['raw_generated','ability','finish_reason'],no_gold_or_question_in_adapter=True,
      interpretation='Posthoc interface repair on exposed Dev and Train resubstitution. Old strict scores remain primary historical artifacts. Empty summary does not establish reasoning or explanation ability.',
      units=units,unique_models=7,raw_count=1484,shared_reference_alias='A-original200-replay is synthetic-identity200-reused in position comparison; counted once',
      scorer_bindings=[bind(x) for x in score_deps],tests=tests,GPU=0,API=0,training=0)
    save(O/'protocol.json',p);save(O/'tests.json',tests)
    # Freeze use on the two previously planned orders without inspecting their
    # answer content. Existing raw counts document exact prospective limitations.
    rep=B/'position-order-replicates-v1';counts={}
    for seed in (2026110602,2026110603):
        for arm in ('real-canonical200','real-balanced200','synthetic-identity200','synthetic-balanced200'):
            unit=rep/str(seed)/arm
            counts[str(seed)+'/'+arm]=dict(raw_already_present=sum((unit/'raw'/f'{i:04d}.json').exists() for i in range(212)),
               evaluation_terminal_present=(unit/'evaluation-terminal.json').exists(),old_grades_present=(unit/'graded.local.json').exists())
    save(O/'replication-application-protocol.json',dict(status='UNCHANGED_RULE_FROZEN_BEFORE_REPLICATION_INTERFACE_SCORING',epoch=epoch,adapter=bind(Path(__file__)),
       original_interface_protocol=bind(O/'protocol.json'),replication_source=bind(rep/'protocol.json'),
       expected_models=8,expected_raw=1696,rule_identical=True,output_namespace='replications',state_at_rule_freeze=counts,
       timing_scope='Rule selected using first-order closed Dev outputs. Counts show any repeat raw already existed; no claim of being frozen before training or every decode. No repeat-answer contents inspected to select rule.',
       no_per_seed_rule_tuning=True,no_training_change=True))
    print(json.dumps(dict(status='PREPARED',models=7,raw=1484,repeat_state=counts)))

def aggregate(rows):
    return dict(total=len(rows),strict_correct=sum(x['strict']['correct'] for x in rows),strict_valid=sum(x['strict']['valid'] for x in rows),
      normalized_correct=sum(x['normalized']['correct'] for x in rows),normalized_valid=sum(x['normalized']['valid'] for x in rows),
      converted=sum(x['adapter']['converted'] for x in rows),capped=sum(x['strict']['capped'] for x in rows),
      gains=sum(not x['strict']['correct'] and x['normalized']['correct'] for x in rows),losses=sum(x['strict']['correct'] and not x['normalized']['correct'] for x in rows))
def summaries(rows):
    groups=collections.defaultdict(list)
    for x in rows:groups[(x['cohort'],x['variant'])].append(x)
    out={}
    for (c,v),z in groups.items():out.setdefault(c,{})[v]=dict(**aggregate(z),abilities={a:aggregate([x for x in z if x['ability']==a]) for a in sorted({x['ability'] for x in z})},
       planning_reflection=aggregate([x for x in z if x['ability'] in ('planning','reflection')]))
    return out

def run_unit(name,spec,output):
    import eval_ust_sg_qwen_modes_20261024 as parser
    queries=load(checked(spec['queries']));answers=load(checked(spec['answers']));old=load(checked(spec['old_grades']))
    assert len(queries)==len(old)==len(spec['raw'])==212
    rows=[]
    for q,g,pin in zip(queries,old,spec['raw']):
        raw=load(checked(pin));assert q['id']==g['id']==raw['id']
        answer=answers[q['id']]['answer'];assert answer==g['answer']
        adapted=normalize_output(raw['raw_generated'],q['metadata']['ability'],raw['finish_reason'])
        classified=parser.classify(adapted['text'],'off','ust',q['metadata'],raw['finish_reason'])
        normalized=dict(prediction=classified['prediction'],valid=bool(classified['valid']),correct=bool(classified['valid'] and classified['prediction']==answer),parser_reason=classified.get('reason'))
        if not adapted['converted']:
            assert normalized['prediction']==g['prediction'] and normalized['valid']==g['valid'] and normalized['correct']==g['correct']
            assert adapted['text']==raw['raw_generated']
        else:
            assert q['metadata']['ability'] in ('planning','reflection') and raw['finish_reason']=='stop'
            assert not g['valid'] and normalized['valid'] and normalized['prediction']==adapted['extracted_answer']
            assert json.loads(adapted['text'])['summary']==''
        rows.append(dict(id=q['id'],source_id=q['source_id'],cohort=q['cohort'],variant=q['variant'],ability=q['metadata']['ability'],answer=answer,
            raw_binding=pin,raw_generated=raw['raw_generated'],finish_reason=raw['finish_reason'],strict=g,adapter=adapted,normalized=normalized))
    save(output/(name+'.normalized.local.json'),rows)
    lookup={(x['source_id'],x['variant']):x for x in rows if x['cohort']=='RepairDev'}
    pairs=[]
    for q in queries:
        if q['cohort']=='RepairDev' and q['variant']=='both':
            x=lookup[(q['source_id'],'canonical')];y=lookup[(q['source_id'],'both')]
            valid=x['normalized']['valid'] and y['normalized']['valid']
            pairs.append(dict(source_id=q['source_id'],ability=y['ability'],both_valid=valid,
              equivariant=bool(valid and q['old_to_new_letter'][x['normalized']['prediction']]==y['normalized']['prediction'])))
    return dict(scores=summaries(rows),normalized_equivariance=dict(total=len(pairs),both_valid=sum(x['both_valid'] for x in pairs),equivariant=sum(x['equivariant'] for x in pairs)),conversions=aggregate(rows))

def score():
    p=load(O/'protocol.json');checked(p['source']);[checked(x) for x in p['scorer_bindings']]
    assert not (O/'score-started.json').exists(),'Once interface scoring; no automatic replay'
    # Recheck every complete raw pin before any new result is produced.
    for spec in p['units'].values():
        checked(spec['closed_terminal']);[checked(x) for x in spec['raw']];checked(spec['old_grades']);checked(spec['queries']);checked(spec['answers'])
    save(O/'score-started.json',dict(epoch=time.time(),protocol=bind(O/'protocol.json'),raw_verified=1484))
    sys.path[:0]=[str(R/'scripts'),str(R/'src')]
    results={name:run_unit(name,spec,O) for name,spec in p['units'].items()}
    # Original records remain byte-identical after all normalization/scoring.
    for spec in p['units'].values():
        checked(spec['old_grades']);[checked(x) for x in spec['raw']]
    save(O/'results.json',dict(status='COMPLETE_POSTHOC_INTERFACE_EXPERIMENT',models=results,raw=1484,old_raw_and_strict_grades_unchanged=True,
       scope=p['interpretation'],summary_ability_claim=False,GPU=0,API=0,new_training=0))
    save(O/'terminal.json',dict(status='COMPLETE',epoch=time.time(),results=bind(O/'results.json')))
    print(json.dumps({name:{v:(z['strict_correct'],z['normalized_correct'],z['total'],z['converted']) for v,z in x['scores'].get('RepairDev',{}).items()} for name,x in results.items()}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['selftest','prepare','score']);a=ap.parse_args();globals()[a.action]()
