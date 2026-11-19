"""Pure, checked UST Dev transformations; no model, training, or network access.

All variants, including identity, use the same whitespace normalization.
Original records and official labels remain unchanged on disk.
"""
from __future__ import annotations
from pathlib import Path
from decimal import Decimal
import collections, hashlib, itertools, json, re, time

ROOT=Path(__file__).resolve().parents[1]
LETTERS='ABCD'
LOC=re.compile(r'\b(location\s+)([1-4])\b',re.I)
LABEL_REF=re.compile(r'\b((?:option|choice|decision|answer)\s+)([A-D])\b',re.I)
OPTION=re.compile(r'^([A-D])\.\s*(.+?)\s*$',re.M)

def _clean(text):
    return '\n'.join(line.rstrip() for line in text.strip().splitlines())

def sections(text):
    matches=list(re.finditer(r'^## ([^\n]+)\n',text,re.M))
    assert matches and not text[:matches[0].start()].strip(),'Unsupported preamble'
    out={}
    for i,m in enumerate(matches):
        title=m.group(1).strip();assert title not in out,'Duplicate section'
        out[title]=_clean(text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)])
    assert {'Task Description','Data Schema','Data','Question','Note'}<=set(out)
    return out

def render_sections(parts):
    return '\n\n'.join('## '+key+'\n\n'+_clean(value) for key,value in parts.items())

def _location_replace(text,mapping):
    return LOC.sub(lambda m:m.group(1)+str(mapping[int(m.group(2))]),text)

def _letter_replace(text,mapping):
    # A lowercase article "a" is not an option reference. Explicit uppercase
    # A/B/C/D references after one of the known nouns are handled precisely.
    return LABEL_REF.sub(lambda m:m.group(1)+mapping[m.group(2)] if m.group(2) in LETTERS else m.group(0),text)

def _data_blocks(text):
    matches=list(re.finditer(r'^location ([1-4]):\s*$',text,re.I|re.M))
    assert len(matches)==4 and not text[:matches[0].start()].strip()
    out={}
    for i,m in enumerate(matches):
        identity=int(m.group(1));assert identity not in out
        out[identity]=_clean(text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)])
    assert set(out)=={1,2,3,4}
    return out

def _options(text):
    matches=list(OPTION.finditer(text));assert len(matches)==4
    bodies={m.group(1):_clean(m.group(2)) for m in matches}
    assert set(bodies)==set(LETTERS)
    assert not any(text[a.end():b.start()].strip() for a,b in zip(matches,matches[1:]))
    return _clean(text[:matches[0].start()]),bodies,_clean(text[matches[-1].end():])

def _render_question(prefix,options,suffix):
    return '\n\n'.join(s for s in [prefix,'\n'.join(letter+'. '+options[letter] for letter in LETTERS),suffix] if s)

def _rank_replace(body,mapping):
    if ' > ' not in body:return body
    assert re.fullmatch(r'[1-4](?:\s*>\s*[1-4]){3}',body),'Unsupported bare ranking option'
    nums=[int(x) for x in re.findall(r'[1-4]',body)];assert set(nums)=={1,2,3,4}
    return ' > '.join(str(mapping[x]) for x in nums)

def transform(question,location_map,option_order):
    """option_order lists old option letters in their new A/B/C/D positions."""
    loc={int(k):int(v) for k,v in location_map.items()};order=list(option_order)
    assert set(loc)==set(loc.values())=={1,2,3,4}
    assert len(order)==4 and set(order)==set(LETTERS)
    letter_map={old:LETTERS[i] for i,old in enumerate(order)}
    original=sections(question);out=dict(original)
    for key in out:
        if key not in ['Task Description','Data Schema','Note']:
            out[key]=_location_replace(out[key],loc)
    blocks=_data_blocks(out['Data'])
    out['Data']='\n\n'.join('location '+str(i)+':\n'+blocks[i] for i in range(1,5))
    prefix,options,suffix=_options(out['Question'])
    options={letter_map[old]:_rank_replace(body,loc) for old,body in options.items()}
    out['Question']=_render_question(prefix,options,suffix)
    if 'Decision' in out:
        decision=original['Decision'].strip();assert decision in LETTERS and len(decision)==1
        out['Decision']=letter_map[decision]
    if 'Reason' in out:out['Reason']=_letter_replace(out['Reason'],letter_map)
    result=dict(question=render_sections(out),old_to_new_letter=letter_map)
    result['checks']=validate_pair(question,result,loc,order)
    return result

def validate_pair(original,result,location_map,option_order):
    """Independent field comparisons, not re-running the forward transformer."""
    loc={int(k):int(v) for k,v in location_map.items()};inverse={v:k for k,v in loc.items()}
    mapping={old:LETTERS[i] for i,old in enumerate(option_order)};inverse_letter={v:k for k,v in mapping.items()}
    assert result['old_to_new_letter']==mapping
    before=sections(original);after=sections(result['question']);assert list(before)==list(after)
    olddata=_data_blocks(before['Data']);newdata=_data_blocks(after['Data'])
    assert list(newdata)==[1,2,3,4]
    assert {loc[k]:v for k,v in olddata.items()}==newdata,'Data values/units or identity binding changed'
    oldprefix,oldopts,oldsuffix=_options(before['Question']);newprefix,newopts,newsuffix=_options(after['Question'])
    assert _location_replace(newprefix,inverse)==oldprefix and _location_replace(newsuffix,inverse)==oldsuffix
    ranking_options=0
    for oldletter,oldbody in oldopts.items():
        newbody=newopts[mapping[oldletter]]
        if re.fullmatch(r'[1-4](?:\s*>\s*[1-4]){3}',oldbody):
            oldsequence=tuple(map(int,re.findall(r'[1-4]',oldbody)))
            newsequence=tuple(map(int,re.findall(r'[1-4]',newbody)))
            assert tuple(inverse[x] for x in newsequence)==oldsequence
            ranking_options+=1
        else:assert _location_replace(newbody,inverse)==oldbody,'Option referent changed'
    for key in before:
        if key in ['Data','Question']:continue
        if key=='Decision':assert after[key]==mapping[before[key]];continue
        text=after[key]
        if key=='Reason':text=_letter_replace(text,inverse_letter)
        if key not in ['Task Description','Data Schema','Note']:text=_location_replace(text,inverse)
        assert text==before[key],'Changed section: '+key
    oldcase=collections.Counter(m.group(1) for m in LOC.finditer(original))
    newcase=collections.Counter(m.group(1) for m in LOC.finditer(result['question']))
    assert oldcase==newcase,'Location reference capitalization/count changed'
    if 'Reason' in before:
        # A semantic prior decision must select the same physical candidate.
        olddec=before['Decision'];newdec=after['Decision']
        assert _location_replace(newopts[newdec],inverse)==oldopts[olddec]
    return dict(status='PASS',data_values_units_unchanged=True,options_referents_preserved=True,
        analysis_and_reason_bindings_preserved=True,decision_referent_preserved=True,
        environment_changes_preserved=True,case_sensitive_reference_counts_preserved=True,
        ranking_options_verified=ranking_options)

def canonical(question):
    return transform(question,{i:i for i in range(1,5)},LETTERS)['question']

def understanding_answers(question):
    """Independent Decimal ordering over displayed numbers; no NaN convention invented."""
    part=sections(question);intro,options,_=_options(part['Question'])
    if not all(re.fullmatch(r'[1-4](?:\s*>\s*[1-4]){3}',x) for x in options.values()):return None
    metric='distance' if 'average distance' in intro else 'waiting_time' if 'average waiting time' in intro else 'charging_time' if 'average charging time' in intro else None
    assert metric is not None
    values={}
    for number,block in _data_blocks(part['Data']).items():
        match=re.search(r'^- '+metric+r':\s*(.+)$',block,re.M);assert match
        raw=match.group(1)
        if 'nan' in raw.lower():return None
        value=re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:m|h)?',raw);assert value
        values[number]=Decimal(value.group(1))
    answers=[]
    for letter,body in options.items():
        sequence=list(map(int,re.findall(r'[1-4]',body)))
        if all(values[a]>=values[b] for a,b in zip(sequence,sequence[1:])):answers.append(letter)
    return answers

def run_qualification(output_path):
    started=time.monotonic();path=ROOT/'runs/2026-10-25-qwen-transfer-repair/data/USTBench.repair-dev32.requests.jsonl'
    rows=[json.loads(line) for line in path.open()];assert len(rows)==32
    goldpath=path.with_name('USTBench.repair-dev32.gold.local.json');gold=json.loads(goldpath.read_text())
    locperms=list(itertools.permutations(range(1,5)));optionperms=list(itertools.permutations(LETTERS))
    cases=0;inverse_cases=0;decimal_cases=0;identity_cases=0;gold_checked=0;gold_matches=0;unsupported_nan=0;uppercase=0
    for row in rows:
        raw=row['question'];identity=canonical(raw)
        assert re.sub(r'\s+','',identity)==re.sub(r'\s+','',raw),'Identity changed non-whitespace text'
        identity_cases+=1;uppercase+=len(re.findall(r'\bLocation [1-4]\b',raw))
        independent=understanding_answers(raw)
        if row['metadata']['ability']=='st_understanding':
            if independent is None:unsupported_nan+=1
            else:gold_checked+=1;gold_matches+=gold[row['id']] in independent
        for lp in locperms:
            loc=dict(zip(range(1,5),lp));invloc={v:k for k,v in loc.items()}
            for op in optionperms:
                result=transform(raw,loc,op);cases+=1
                reverse=transform(result['question'],invloc,[result['old_to_new_letter'][x] for x in LETTERS])
                assert reverse['question']==identity,'Inverse canonical text mismatch';inverse_cases+=1
                if independent is not None:
                    actual=understanding_answers(result['question'])
                    assert set(actual)=={result['old_to_new_letter'][x] for x in independent}
                    decimal_cases+=1
    proof=dict(status='PASS',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),Dev32_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),Dev32_count=32,
        location_permutations=24,option_permutations=24,full_crossproduct_cases=cases,inverse_roundtrip_cases=inverse_cases,
        identity_whitespace_only_cases=identity_cases,independent_Decimal_mapping_cases=decimal_cases,
        original_understanding_gold_checked=gold_checked,original_understanding_gold_matches=gold_matches,
        NaN_cases_without_claimed_ordering=unsupported_nan,uppercase_Location_references_covered=uppercase,
        no_model_scores=True,no_GPU=True,no_API=True,no_Final_Common_Audit_read=True,seconds=time.monotonic()-started)
    out=Path(output_path);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
    return proof

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('output_path');args=p.parse_args();print(json.dumps(run_qualification(args.output_path)))
