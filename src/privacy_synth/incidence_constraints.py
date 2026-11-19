"""Final categorical reverse-step constraints; never rename, refill or resample."""
from __future__ import annotations
import math


def final_draw(shape, weights, rng, constrained=True):
    lengths=shape['entity_lengths'];m=shape['entities'];a=sum(shape['assertions_per_frame'])
    if len(lengths)!=m or not m or any(not 1<=n<=32 for n in lengths):
        raise ValueError('Invalid fixed entity lengths')
    if 2*a<m:
        raise ValueError('Infeasible entity endpoint coverage')
    if len(weights)!=sum(lengths)+3*a:
        raise ValueError('Final posterior variable count')
    output=[];events=[];seen=set();unused=set(range(m));pointer_left=2*a

    def draw(row, forbidden, kind, position):
        raw=weights[row]
        if any(not math.isfinite(v) or v<0 for v in raw) or sum(raw)<=0:
            raise ValueError('Invalid actual posterior')
        allowed=[i for i in range(len(raw)) if i not in forbidden]
        live=[v if i in allowed else 0. for i,v in enumerate(raw)]
        mass=sum(live)
        if mass<=0:
            raise ValueError('Zero legal posterior mass; retained, never invent uniform probabilities')
        token=rng.choices(range(len(raw)),weights=live)[0]
        removed=1-mass/sum(raw)
        if forbidden:
            events.append(dict(kind=kind,row=row,position=position,forbidden=len(forbidden),
                               removed_posterior_mass=max(0.,removed),legal_mass=mass/sum(raw)))
        output.append(token)
        return token

    for entity,length in enumerate(lengths):
        name=[]
        for pos in range(length):
            forbidden={n[-1] for n in seen if len(n)==length and n[:-1]==tuple(name)} if constrained and pos==length-1 else set()
            name.append(draw(len(output),forbidden,'lexical_uniqueness',[entity,pos]))
        seen.add(tuple(name))
    for j in range(a):
        draw(len(output),set(),'relation',[j])
        for role in ('subject','object'):
            forbidden=set(range(m))-unused if constrained and pointer_left==len(unused) else set()
            value=draw(len(output),forbidden,'entity_coverage',[j,role])
            unused.discard(value);pointer_left-=1
    return output,dict(events=events,constrained=constrained,
                       unique_names=len(seen),unused_entity_slots=len(unused),
                       interpretation='Sequential final-step conditioning on actual reverse probabilities, not unconstrained diffusion fidelity or learned uniqueness')
