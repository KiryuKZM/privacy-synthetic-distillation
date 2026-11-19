"""Public-proxy factored Markov counts with generic last-nonnull registers.
Standard library only. No scripts/vendor, I/O, identifiers, QA, gradients or DP.
"""
from __future__ import annotations
import copy,json,math,random
from collections import Counter

KINDS=('object','tool','location','device')
ROLES=('object','tool','location')
START='<START>'
ALPHA_TOTAL=1
BINS=4


def key(context):return json.dumps(list(context),separators=(',',':'),ensure_ascii=False)

def require_family(family):
    if type(family) is not int or not 0<=family<10:raise ValueError('Family must be integer0..9')
    return family


def time_bin(index,length):
    if type(length) is not int or length<1 or not 0<=index<length:raise ValueError('Invalid relative time')
    return min(BINS-1,BINS*index//length)


class Counts:
    def __init__(self,levels):self.tables=[{} for _ in range(levels)]
    def add(self,contexts,value):
        assert len(contexts)==len(self.tables)
        for table,context in zip(self.tables,contexts):table.setdefault(key(context),Counter())[str(value)]+=1
    def probabilities(self,contexts,support):
        support=list(support)
        if not support or len(set(support))!=len(support):raise ValueError('Nonempty distinct legal support required')
        assert len(contexts)==len(self.tables)
        parent={x:1/len(support) for x in support}
        # At every level counts are restricted to the current legal support.
        for table,context in reversed(list(zip(self.tables,contexts))):
            counts=table.get(key(context),{})
            total=sum(counts.get(str(x),0) for x in support)
            parent={x:(counts.get(str(x),0)+parent[x])/(total+ALPHA_TOTAL) for x in support}
        if not all(math.isfinite(p) and p>=0 for p in parent.values()) or abs(sum(parent.values())-1)>1e-10:
            raise ArithmeticError('Non-normalized CPT; no resampling/repair')
        return parent
    def to_dict(self):return [{k:dict(sorted(v.items())) for k,v in sorted(t.items())} for t in self.tables]
    @classmethod
    def from_dict(cls,payload):
        obj=cls(len(payload));obj.tables=copy.deepcopy(payload)
        for table in obj.tables:
            for context,counts in table.items():
                assert isinstance(json.loads(context),list)
                if any(type(n) is not int or n<0 for n in counts.values()):raise ValueError('Invalid saved integer count')
        return obj


def draw(rng,probabilities):
    values=list(probabilities);return rng.choices(values,[probabilities[v] for v in values],k=1)[0]


def draw_nouns_without_replacement(rng,weights,count):
    """Restrict the SAME full-support unigram weights; do not re-smooth S."""
    if type(count) is not int or count<0 or count>len(weights):
        raise ValueError('Insufficient unique noun support; no duplication/refill')
    remaining=dict(weights);chosen=[]
    for _ in range(count):
        total=sum(remaining.values())
        probabilities={name:weight/total for name,weight in remaining.items()}
        noun=draw(rng,probabilities);chosen.append(noun);del remaining[noun]
    return chosen

def contexts(field,family,bin_index,previous_verb=START,previous_hand=-1,verb=None,available=False):
    if field=='verb':return [(family,bin_index,previous_verb),(family,bin_index),(family,),()]
    if field=='hand':return [(family,bin_index,verb,previous_hand),(family,verb),(verb,),()]
    if field in ROLES:return [(family,bin_index,verb,int(available)),(family,verb),(verb,),()]
    raise ValueError('Unknown factor')


def role_candidates(entities,role):
    if role=='object':return list(range(len(entities)))
    if role=='tool':return list(range(1,1+sum(e['kind']=='tool' for e in entities)))
    if role=='location':return list(range(1,1+2*sum(e['kind']=='location' for e in entities)))
    raise ValueError('Unknown role')


def role_class(entities,role,value):
    if role=='object':entity=entities[value];return json.dumps([entity['kind'],entity['noun']],separators=(',',':'))
    if role=='tool':entity=[e for e in entities if e['kind']=='tool'][value-1];return json.dumps([entity['kind'],entity['noun']],separators=(',',':'))
    locations=[e for e in entities if e['kind']=='location'];i=value-1
    entity=locations[i%len(locations)]
    return json.dumps(['into' if i>=len(locations) else 'on',entity['kind'],entity['noun']],separators=(',',':'))


def role_support(entities,role,register):
    candidates=role_candidates(entities,role);available=register is not None and register in candidates
    others=[v for v in candidates if not available or v!=register]
    support=(['NULL'] if role!='object' else [])+(['REUSE'] if available else [])+(['OTHER'] if others else [])
    if not support:raise ValueError('No legal role support')
    return support,others,available


def update_registers(registers,event):
    for role in ROLES:
        value=event[role]
        # object0 is an actual entity; only tool/location0 are null.
        if role=='object' or value!=0:registers[role]=value


class MarkovGenerator:
    def __init__(self,vocab,verbs,family_priors):
        self.vocab=copy.deepcopy(vocab);self.verbs=list(verbs);self.family_priors=copy.deepcopy(family_priors)
        if not self.verbs or any(not self.vocab[k] for k in KINDS):raise ValueError('Empty declared vocabulary')
        self.nouns=Counts(1);self.verb=Counts(4);self.hand=Counts(4)
        self.role_modes={r:Counts(4) for r in ROLES};self.role_classes={r:Counts(4) for r in ROLES}
        self.fit_worlds=0;self.fitted=False
    def fit(self,rows):
        if self.fitted or self.fit_worlds:raise ValueError('One closed-form fit per model; no continuation')
        for family,state in rows:
            family=require_family(family);entities=state['entities'];events=state['events']
            if set(state)!={'entities','events'} or not events:raise ValueError('State-only nonempty event sequence required')
            for entity in entities:
                if set(entity)!={'kind','noun'} or entity['noun'] not in self.vocab[entity['kind']]:raise ValueError('Typed noun outside frozen vocabulary')
                self.nouns.add([(entity['kind'],)],entity['noun'])
            pv=START;ph=-1;registers={r:None for r in ROLES}
            for i,event in enumerate(events):
                if set(event)!={'verb','hand','object','tool','location'}:raise ValueError('No full-tuple auxiliary fields')
                verb=event['verb'];hand=event['hand'];b=time_bin(i,len(events))
                if verb not in self.verbs or type(hand) is not int or hand not in (0,1,2):raise ValueError('Unsupported verb/hand')
                self.verb.add(contexts('verb',family,b,pv),verb)
                self.hand.add(contexts('hand',family,b,previous_hand=ph,verb=verb),hand)
                for role in ROLES:
                    value=event[role];support,others,available=role_support(entities,role,registers[role])
                    if type(value) is not int or value not in role_candidates(entities,role)+([0] if role!='object' else []):raise ValueError('Unsupported role pointer')
                    mode='NULL' if role!='object' and value==0 else 'REUSE' if available and value==registers[role] else 'OTHER'
                    assert mode in support
                    ctx=contexts(role,family,b,verb=verb,available=available);self.role_modes[role].add(ctx,mode)
                    if mode=='OTHER':self.role_classes[role].add(ctx,role_class(entities,role,value))
                update_registers(registers,event);pv=verb;ph=hand
            self.fit_worlds+=1
        if not self.fit_worlds:raise ValueError('Empty Fit')
        self.fitted=True;return self
    def role_distribution(self,entities,role,family,b,verb,register):
        support,others,available=role_support(entities,role,register)
        ctx=contexts(role,family,b,verb=verb,available=available)
        modes=self.role_modes[role].probabilities(ctx,support);result={}
        if 'NULL' in modes:result[0]=modes['NULL']
        if 'REUSE' in modes:result[register]=modes['REUSE']
        if 'OTHER' in modes:
            grouped={}
            for value in others:grouped.setdefault(role_class(entities,role,value),[]).append(value)
            classes=self.role_classes[role].probabilities(ctx,sorted(grouped))
            for name,values in grouped.items():
                for value in values:result[value]=modes['OTHER']*classes[name]/len(values)
        assert set(result)==set(role_candidates(entities,role)+([0] if role!='object' else []))
        if abs(sum(result.values())-1)>1e-10:raise ArithmeticError('Role support not normalized')
        return result
    def sample(self,shape,length,seed,family):
        if not self.fitted:raise ValueError('Committed count model required')
        family=require_family(family)
        if len(shape)!=4 or any(type(n) is not int or n<1 for n in shape) or type(length) is not int or length<1:raise ValueError('Frozen nonempty typed shape required')
        rng=random.Random(seed);entities=[]
        for kind,n in zip(KINDS,shape):
            weights=self.nouns.probabilities([(kind,)],self.vocab[kind])
            for noun in draw_nouns_without_replacement(rng,weights,n):entities.append({'kind':kind,'noun':noun})
        pv=START;ph=-1;registers={r:None for r in ROLES};events=[]
        for i in range(length):
            b=time_bin(i,length)
            verb=draw(rng,self.verb.probabilities(contexts('verb',family,b,pv),self.verbs))
            hand=draw(rng,self.hand.probabilities(contexts('hand',family,b,previous_hand=ph,verb=verb),[0,1,2]))
            event={'verb':verb,'hand':hand}
            for role in ROLES:event[role]=draw(rng,self.role_distribution(entities,role,family,b,verb,registers[role]))
            update_registers(registers,event);events.append(event);pv=verb;ph=hand
        return {'entities':entities,'events':events}
    def to_dict(self):
        if not self.fitted:raise ValueError('Unfitted counts')
        return dict(schema='sgqa-factored-markov-register-v1',alpha_total=ALPHA_TOTAL,temporal_bins=BINS,
          vocab=self.vocab,verbs=self.verbs,family_priors=self.family_priors,fit_worlds=self.fit_worlds,
          noun_counts=self.nouns.to_dict(),verb_counts=self.verb.to_dict(),hand_counts=self.hand.to_dict(),
          role_mode_counts={r:self.role_modes[r].to_dict() for r in ROLES},
          role_class_counts={r:self.role_classes[r].to_dict() for r in ROLES})
    @classmethod
    def from_dict(cls,p):
        required={'schema','alpha_total','temporal_bins','vocab','verbs','family_priors','fit_worlds','noun_counts','verb_counts','hand_counts','role_mode_counts','role_class_counts'}
        if set(p)!=required or p['schema']!='sgqa-factored-markov-register-v1' or p['alpha_total']!=1 or p['temporal_bins']!=4:raise ValueError('Frozen count schema changed')
        obj=cls(p['vocab'],p['verbs'],p['family_priors']);obj.nouns=Counts.from_dict(p['noun_counts']);obj.verb=Counts.from_dict(p['verb_counts']);obj.hand=Counts.from_dict(p['hand_counts'])
        obj.role_modes={r:Counts.from_dict(p['role_mode_counts'][r]) for r in ROLES};obj.role_classes={r:Counts.from_dict(p['role_class_counts'][r]) for r in ROLES}
        if len(obj.nouns.tables)!=1 or len(obj.verb.tables)!=4 or len(obj.hand.tables)!=4 or any(len(obj.role_modes[r].tables)!=4 or len(obj.role_classes[r].tables)!=4 for r in ROLES):raise ValueError('CPT hierarchy changed')
        obj.fit_worlds=p['fit_worlds'];assert type(obj.fit_worlds) is int and obj.fit_worlds>0;obj.fitted=True
        return obj

