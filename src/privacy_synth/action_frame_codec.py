"""Lossless person/action-frame factorization of full incidence payloads.

The one-person-verb-per-frame contract is an explicit domain assumption, not
learned quality. Non-verb assertions, multiplicities and directed roles remain.
"""
from __future__ import annotations

def factor(payload):
    shape=payload['shape'];names=payload['entity_names'];assertions=payload['assertions'];F=shape['frames']
    if len(set(names))!=len(names) or 'person' not in names:raise ValueError('Unique named person required')
    frames=[[] for _ in range(F)]
    for a in assertions:
        if not 0<=a['frame']<F:raise ValueError('Frame range')
        frames[a['frame']].append(a)
    actions=[];frame_action_names=[];verb_slots=[]
    for frame in frames:
        verbs=[(i,a) for i,a in enumerate(frame) if a['relation']=='verb']
        if len(verbs)!=1 or names[verbs[0][1]['subject']]!='person':raise ValueError('Exactly one person verb per frame required')
        slot,a=verbs[0];word=names[a['object']]
        if word=='person':raise ValueError('Person cannot be action label in this contract')
        if word not in actions:actions.append(word)
        frame_action_names.append(word);verb_slots.append(slot)
    objects=[n for n in names if n!='person' and n not in actions]
    canonical=['person',*actions,*objects];old_to_new={i:canonical.index(n) for i,n in enumerate(names)}
    nonverb=[]
    for f,frame in enumerate(frames):
        for a in frame:
            if a['relation']!='verb':nonverb.append(dict(frame=f,subject=old_to_new[a['subject']],relation=a['relation'],object=old_to_new[a['object']]))
    return dict(shape=dict(frames=F,action_types=len(actions),object_types=len(objects),
                           verb_slot_per_frame=verb_slots,nonverb_per_frame=[len(x)-1 for x in frames]),
                action_names=actions,object_names=objects,frame_actions=[actions.index(n) for n in frame_action_names],nonverb_assertions=nonverb)

def unfactor(x):
    s=x['shape'];F=s['frames'];actions=x['action_names'];objects=x['object_names'];names=['person',*actions,*objects]
    if len(actions)!=s['action_types'] or len(objects)!=s['object_types'] or not actions or len(set(names))!=len(names) or any(not isinstance(n,str) or not n for n in names):raise ValueError('Role roster mismatch/duplicate')
    if len(x['frame_actions'])!=F or len(s['verb_slot_per_frame'])!=F or len(s['nonverb_per_frame'])!=F:raise ValueError('Frame factor count')
    frames=[[] for _ in range(F)]
    for a in x['nonverb_assertions']:
        if not 0<=a['frame']<F or a['relation']=='verb' or not 0<=a['subject']<len(names) or not 0<=a['object']<len(names):raise ValueError('Full directed assertion contract')
        frames[a['frame']].append(dict(a))
    assertions=[]
    for f,frame in enumerate(frames):
        target=x['frame_actions'][f];slot=s['verb_slot_per_frame'][f]
        if not 0<=target<len(actions) or len(frame)!=s['nonverb_per_frame'][f] or not 0<=slot<=len(frame):raise ValueError('Frame/action shape mismatch')
        frame.insert(slot,dict(frame=f,subject=0,relation='verb',object=1+target));assertions.extend(frame)
    return dict(shape=dict(frames=F,entities=len(names),assertions=len(assertions)),entity_names=names,assertions=assertions)
