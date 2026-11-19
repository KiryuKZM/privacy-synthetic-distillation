#!/usr/bin/env python3
"""One bounded, offline SGQA literal-support audit; never imports model code.
Freeze the Fit-video frequency definition before graph decoding. Results are
local/protected aggregates, not accuracy, semantic coverage, privacy or fit proof.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, os, statistics, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/2026-10-28-sgqa-2b-checkpoint-lr/analysis/longtail-20261029'
FORMAL = ROOT / 'runs/2026-10-27-generator-formal'
CACHE = FORMAL / 'preparation/realgraph-full-incidence-v1'
SOURCES = {
    'real_manifest': CACHE / 'manifest.json',
    'fit_metadata': FORMAL / 'data/GeneratorFitLocalRealTrain.metadata.json',
    'dev_metadata': FORMAL / 'data/LocalDev.metadata.json',
    'real_fit': CACHE / 'fit.graph-incidence.jsonl',
    'real_dev': CACHE / 'dev.graph-incidence.jsonl',
    'markov_train300': FORMAL / 'students-train300-v1/inputs/markov.json',
    'control_train300': FORMAL / 'students-train300-v1/inputs/control.json',
    'pool_manifest': FORMAL / 'teacher-bodies-cpu-v1/manifest.json',
    'markov_pool1000': FORMAL / 'teacher-bodies-cpu-v1/markov-trainpool.requests.jsonl',
    'control_pool1000': FORMAL / 'teacher-bodies-cpu-v1/control-trainpool.requests.jsonl',
    'markov_pool_members': FORMAL / 'teacher-bodies-cpu-v1/markov-trainpool.members.jsonl',
    'control_pool_members': FORMAL / 'teacher-bodies-cpu-v1/control-trainpool.members.jsonl',
    'markov_binding': ROOT / 'runs/2026-10-26-generator-discovery/branches/SGQA/statistical-markov-v2/model-binding.json',
    'markov_protocol': ROOT / 'runs/2026-10-26-generator-discovery/branches/SGQA/statistical-markov-v2/prepared/protocol.json',
    'markov_fit_summary': ROOT / 'runs/2026-10-26-generator-discovery/runtime/stages/sgqa-statistical-markov-countfit-v2/committed/fit-summary.json',
    'source': Path(__file__),
}
DIMS = ('action_literal', 'nonverb_relation_literal', 'entity_surface_literal')
def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def digest(x): return hashlib.sha256(json.dumps(x, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
def rows(p):
    with Path(p).open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(x, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
def bind(p): return {'path': str(Path(p).relative_to(ROOT)), 'sha256': sha(p), 'bytes': Path(p).stat().st_size}
def frozen_sources(): return {k: bind(p) for k, p in SOURCES.items()}
def freeze():
    if RUN.exists(): raise FileExistsError('Immutable namespace exists; use verify, never overwrite')
    p = dict(schema='SGQA-literal-longtail-local-v1', created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        status='FROZEN_BEFORE_ANALYSIS', purpose='Bounded literal support audit, not generator selection or causal utility test',
        frequency_source='Only frozen GeneratorFit/LocalRealTrain: 21 author graphs in 20 physical videos; union within each physical video',
        bins={'unseen': '0 Fit physical videos', 'rare': '1 or 2 Fit physical videos', 'head': '3 or more Fit physical videos'},
        definitions={'action_literal': 'Exact string object of relation == verb, without lemmatization or synonym normalization',
            'nonverb_relation_literal': 'Exact relation string != verb; all annotation variants retained',
            'entity_surface_literal': 'Exact subject/object strings minus all verb-object action strings within that whole graph; no inferred semantic category',
            'support': 'A literal type occurs in at least one graph in the frozen synthetic corpus; not relation-argument/motif/QA coverage',
            'video_feature_incidence': 'One (physical video, literal feature type) unit after union of its graphs/frames',
            'structural': 'Graph length/count ranges, per-frame nonverb non-action-subject and multi-dobj; reported separately from frequency rarity'},
        populations={'real_fit': {'graphs': 21, 'physical_videos': 20, 'attached_QA_not_independent_graphs': 105},
            'real_dev': {'graphs': 6, 'physical_videos': 6, 'attached_QA_not_independent_graphs': 30},
            'synthetic_train': 'Both actual frozen Train300 inputs',
            'synthetic_pool': 'Both eligible, deduplicated fixed Train1000 request pools, NOT the raw10000 draw pool'},
        limits={'Final_and_Common_body_or_accuracy_access': False, 'gold_or_answer_scoring': False,
            'API_calls': 0, 'GPU': False, 'model_imports': False, 'optimizer_updates': 0, 'new_samples': 0,
            'retained_all_graphs_no_accuracy_filter': True, 'threads': 1, 'maximum_input_file_bytes': 64*2**20,
            'entity_names_are_not_semantic_categories': True, 'no_test_based_method_selection': True,
            'literal_fit_coverage_utility_privacy_are_distinct': True},
        verification={'source_hashes': 'Local automatic hash checks; no human verification claimed', 'human_review': 'PENDING'},
        source_bindings=frozen_sources())
    for x in p['source_bindings'].values():
        if x['bytes'] > p['limits']['maximum_input_file_bytes']: raise ValueError('Input cap exceeded')
    write(RUN / 'protocol.json', p)
    with (RUN / 'source.py').open('xb') as f: f.write(Path(__file__).read_bytes())
    print(json.dumps({'status': p['status'], 'protocol_sha256': sha(RUN/'protocol.json'), 'graph_bodies_decoded': 0}))
def graph_features(gs):
    if not isinstance(gs, list): raise ValueError('Graph sequence required')
    actions = set(); surfaces = set(); rels = set(); count = collections.Counter()
    for frame in gs:
        vs = {o for s, p, o in frame if p == 'verb'}
        actions.update(vs)
        count['frames'] += 1; count['edges'] += len(frame)
        count['verb_first_frames'] += bool(frame and frame[0][:2] == ['person','verb'])
        dobj = collections.Counter()
        for edge in frame:
            if len(edge) != 3 or not all(isinstance(x,str) for x in edge): raise ValueError('Invalid edge')
            s, p, o = edge; surfaces.update((s,o)); count['verb_edges' if p == 'verb' else 'nonverb_edges'] += 1
            if p != 'verb':
                rels.add(p); count['non_action_subject_edges'] += s not in vs
            if p == 'dobj': dobj[s] += 1
        count['multi_dobj_frames'] += any(n > 1 for n in dobj.values())
    return {'action_literal': actions, 'nonverb_relation_literal': rels, 'entity_surface_literal': surfaces - actions}, count

def bin_of(n): return 'unseen' if n == 0 else 'rare' if n <= 2 else 'head'
def summary(stats):
    keys = ('frames','edges','verb_edges','nonverb_edges','non_action_subject_edges','multi_dobj_frames','verb_first_frames')
    return {'graphs': len(stats), 'totals': {k: sum(x[k] for x in stats) for k in keys},
        'per_graph_min_median_max': {k: [min(x[k] for x in stats), statistics.median(x[k] for x in stats), max(x[k] for x in stats)] for k in ('frames','edges')},
        'graphs_with_non_action_subject_edges': sum(x['non_action_subject_edges'] > 0 for x in stats),
        'graphs_with_multi_dobj_frames': sum(x['multi_dobj_frames'] > 0 for x in stats)}
def real(role):
    meta = load(SOURCES[role+'_metadata']); expected = {m['clip']: m['video_uid'] for m in meta['members']}
    seen = set(); byvideo = collections.defaultdict(lambda: {d: set() for d in DIMS}); stats = []
    for r in rows(SOURCES['real_'+role]):
        clip = r['clip']; video = r['video_uid']; gs = r['graphs']
        if clip in seen or expected.get(clip) != video or digest(gs) != r['graph_sha256']: raise ValueError('Real cache identity/hash mismatch')
        seen.add(clip); fs, st = graph_features(gs); stats.append(st)
        for d in DIMS: byvideo[video][d].update(fs[d])
    if seen != set(expected) or len(seen) != meta['clips'] or len(byvideo) != meta['videos']: raise ValueError('Real denominator mismatch')
    return byvideo, stats, meta

def from_query(text):
    if not text.startswith('Ordered scene graphs:\n') or '\nQuestion:\n' not in text: raise ValueError('Serialization mismatch')
    return json.loads(text.split('Ordered scene graphs:\n',1)[1].split('\nQuestion:\n',1)[0])
def synthetic(name):
    union = {d: set() for d in DIMS}; stats = []; seen = set(); hashes = set()
    ispool = name.endswith('pool1000')
    records = rows(SOURCES[name]) if ispool else iter(load(SOURCES[name])['selected'])
    for r in records:
        if r['id'] in seen: raise ValueError('Synthetic duplicate member ID')
        seen.add(r['id']); gs = from_query(r['body']['messages'][-1]['content'] if ispool else r['question'])
        hashes.add(digest(gs)); fs, st = graph_features(gs); stats.append(st)
        for d in DIMS: union[d].update(fs[d])
    if len(seen) != (1000 if ispool else 300): raise ValueError('Synthetic denominator mismatch')
    return union, stats, hashes

def frac(n,d): return {'covered': n, 'total': d, 'percentage': 100*n/d if d else None}
def check_bindings():
    p = load(RUN/'protocol.json')
    if p['source_bindings'] != frozen_sources() or sha(RUN/'source.py') != sha(__file__): raise ValueError('Frozen source drift')
    manifest = load(SOURCES['real_manifest'])
    for role in ('Fit','Dev'):
        if manifest['records'][role]['sha256'] != sha(SOURCES['real_'+role.lower()]): raise ValueError('Cache manifest drift')
        if manifest['role_bindings'][role]['sha256'] != sha(SOURCES[role.lower()+'_metadata']): raise ValueError('Role manifest drift')
    m = load(SOURCES['pool_manifest'])
    for route in ('markov','control'):
        packet = m['packages'][route+'-trainpool']
        if packet['request_sha256'] != sha(SOURCES[route+'_pool1000']) or packet['members_sha256'] != sha(SOURCES[route+'_pool_members']): raise ValueError('Pool binding drift')
    return p

def analyze():
    p = check_bindings(); fit, fs, fm = real('fit'); dev, ds, dm = real('dev')
    if len(fit) != 20 or len(dev) != 6 or set(fit) & set(dev): raise ValueError('Video disjointness/count changed')
    freq = {d: collections.Counter(x for v in fit.values() for x in v[d]) for d in DIMS}
    dfreq = {d: collections.Counter(x for v in dev.values() for x in v[d]) for d in DIMS}
    out = {'schema':'SGQA-literal-longtail-local-result-v1','protocol_sha256':sha(RUN/'protocol.json'),
        'status':'COMPLETE_LOCAL_COVERAGE_ONLY','accuracy_gold_Final_Common_access':False,
        'counts':{'Fit_graphs':len(fs),'Fit_physical_videos':len(fit),'Fit_attached_QA':fm['QA'],
            'Dev_graphs':len(ds),'Dev_physical_videos':len(dev),'Dev_attached_QA':dm['QA']},
        'real_frequency':{},'synthetic':{},'structure':{'real_fit':summary(fs),'real_dev':summary(ds)},
        'execution':{'API_calls':0,'GPU':False,'optimizer_updates':0,'new_samples':0,'large_weights_read':False,'correctness_outputs_read':False}}
    for d in DIMS:
        out['real_frequency'][d] = {
            'Fit_type_count':len(freq[d]),'Fit_video_frequency_histogram':dict(sorted(collections.Counter(freq[d].values()).items())),
            'Fit_types_by_bin':dict(collections.Counter(bin_of(n) for n in freq[d].values())),
            'Dev_type_count':len(dfreq[d]),'Dev_types_by_Fit_bin':dict(collections.Counter(bin_of(freq[d][x]) for x in dfreq[d])),
            'Dev_video_feature_incidences_by_Fit_bin':dict(collections.Counter({b:sum(n for x,n in dfreq[d].items() if bin_of(freq[d][x])==b) for b in ('unseen','rare','head')}))}
    fitmax = max(s['frames'] for s in fs)
    for name in ('markov_train300','control_train300','markov_pool1000','control_pool1000'):
        support, ss, hashes = synthetic(name); out['structure'][name]=summary(ss)
        entry={'graphs':len(ss),'exact_graph_count':len(hashes),'dimensions':{},'graphs_longer_than_Fit_max_frames':sum(x['frames']>fitmax for x in ss)}
        for d in DIMS:
            z={'synthetic_literal_types':len(support[d]), 'Fit':{},'Dev':{}}
            for b in ('rare','head'):
                types={x for x,n in freq[d].items() if bin_of(n)==b}
                z['Fit'][b]={'types':frac(len(types & support[d]),len(types)),
                    'Fit_video_feature_incidences':frac(sum(freq[d][x] for x in types & support[d]),sum(freq[d][x] for x in types))}
            for b in ('unseen','rare','head'):
                types={x for x in dfreq[d] if bin_of(freq[d][x])==b}
                z['Dev'][b]={'types':frac(len(types & support[d]),len(types)),
                    'Dev_video_feature_incidences':frac(sum(dfreq[d][x] for x in types & support[d]),sum(dfreq[d][x] for x in types))}
            entry['dimensions'][d]=z
        out['synthetic'][name]=entry
    out['structure']['length_reference']={'Fit_max_frames':fitmax,'Dev_graphs_longer_than_Fit_max_frames':sum(x['frames']>fitmax for x in ds),'not_frequency_longtail':True}
    mb=load(SOURCES['markov_binding']); mp=load(SOURCES['markov_protocol']); ms=load(SOURCES['markov_fit_summary'])
    if mb['protocol_sha256']!=sha(SOURCES['markov_protocol']) or mb['fit_summary_sha256']!=sha(SOURCES['markov_fit_summary']): raise ValueError('Markov source binding drift')
    out['literal_fit']={'Markov_training_source':mp['domain'],'Markov_countfit_worlds':ms['fit_worlds'],
        'Markov_model_sha256':mb['model_sha256'],'author_real_graph_fit_not_established_by_this_Markov':True}
    out['limits']=['Frequency is relative to only 20 frozen Fit videos, not the entire author/source population.',
        'Literal presence is weaker than matching frequency, relation argument composition, temporal motif or QA reasoning support.',
        'Entity surface names are not semantic categories; formal naming/suffix differences intentionally remain literal mismatches.',
        'The pool is the fixed eligible Train1000 candidate pool; raw10000 coverage is not claimed.',
        'No student tail utility or causal explanation of negative transfer was measured; all historical failures/results remain.',
        'No privacy, DP, non-copying, distribution-fit or author-scale reproduction claim follows from this audit.']
    write(RUN/'aggregate.json',out)
    write(RUN/'frequency.local.json',{d:{'Fit_video_frequency':dict(sorted(freq[d].items())), 'Dev_video_frequency':dict(sorted(dfreq[d].items()))} for d in DIMS})
    write(RUN/'completion.json',{'status':out['status'],'protocol':bind(RUN/'protocol.json'),'aggregate':bind(RUN/'aggregate.json'),
        'local_frequency':bind(RUN/'frequency.local.json'),'source':bind(RUN/'source.py'),'source_bindings_unchanged':True})
    print(json.dumps({'status':out['status'],'aggregate_sha256':sha(RUN/'aggregate.json'),'counts':out['counts'],'execution':out['execution']}))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','analyze','verify']);args=parser.parse_args()
    if args.action=='freeze': freeze()
    elif args.action=='analyze': analyze()
    else:
        check_bindings();c=load(RUN/'completion.json')
        for k in ('protocol','aggregate','local_frequency','source'):
            if sha(ROOT/c[k]['path'])!=c[k]['sha256']:raise ValueError('Result drift')
        print(json.dumps({'status':'FROZEN_INPUT_AND_OUTPUT_HASH_VERIFIED','execution':load(RUN/'aggregate.json')['execution']}))
if __name__=='__main__': main()
