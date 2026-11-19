"""Public CityFM classification contracts; no model, network or sealed evaluation."""
from __future__ import annotations
import hashlib
import json
import math
from collections import Counter

CLASSES = ('RESIDENTIAL', 'INDUSTRIAL', 'COMMERCIAL', 'COMMERCIAL & RESIDENTIAL',
           'EDUCATIONAL', 'CIVIC & COMMUNITY INSTITUTION', 'SPORTS & RECREATION', 'TRANSPORT')
# Concise task definitions follow URA's named broad land-use handbooks and the
# pinned CityFM dismiss/rename map below. Merged classes are stated explicitly;
# these are public class semantics, never facts about a target building.
CLASS_DEFINITIONS = {
 'RESIDENTIAL':'Housing such as houses or apartments; includes the released Residential/Institution merge.',
 'INDUSTRIAL':'Industrial business/production or warehouses; includes agriculture and utility under the released mapping.',
 'COMMERCIAL':'Shops, offices and other commercial uses; includes hotels and business parks under the released mapping.',
 'COMMERCIAL & RESIDENTIAL':'Combined commercial and residential use, including residential with commercial at the first storey.',
 'EDUCATIONAL':'Educational institutions, such as schools, colleges and universities.',
 'CIVIC & COMMUNITY INSTITUTION':'Civic, community or cultural institutions; also worship and health/medical sites under the released mapping.',
 'SPORTS & RECREATION':'Sports or recreational facilities and grounds.',
 'TRANSPORT':'Transport facilities, including rapid-transit sites and port/airport sites under the released mapping.'}
CLASS_DEFINITION_SOURCES = ['https://www.ura.gov.sg/guidelines/development-control/',
 'https://www.ura.gov.sg/guidelines/development-control/development-control-handbooks/non-residential/c-ci/',
 'https://raw.githubusercontent.com/PasqualeTurin/CityFM/1bf4985c7e86665bb38818cd0e443914bfbb7e46/utils/pre_processing.py']
ROLES = ('CityFM-SG-GeoGeneratorFit', 'CityFM-SG-GeoPromptDev32',
         'CityFM-SG-GeoCandidateScreen96', 'CityFM-SG-GeoFinalTestSealed')
DISMISS = frozenset(('BEACH AREA','WATERBODY','CEMETERY','ROAD','WHITE','RESERVE SITE','OPEN SPACE','PARK','SPECIAL USE'))
RENAME = {'EDUCATIONAL INSTITUTION':'EDUCATIONAL', 'RESIDENTIAL WITH COMMERCIAL AT 1ST STOREY':'COMMERCIAL & RESIDENTIAL',
 'BUSINESS 1':'INDUSTRIAL','BUSINESS 2':'INDUSTRIAL','BUSINESS 1 - WHITE':'INDUSTRIAL','BUSINESS 2 - WHITE':'INDUSTRIAL',
 'RESIDENTIAL / INSTITUTION':'RESIDENTIAL','COMMERCIAL / INSTITUTION':'COMMERCIAL','AGRICULTURE':'INDUSTRIAL',
 'TRANSPORT FACILITIES':'TRANSPORT','MASS RAPID TRANSIT':'TRANSPORT','LIGHT RAPID TRANSIT':'TRANSPORT','HOTEL':'COMMERCIAL',
 'UTILITY':'INDUSTRIAL','BUSINESS PARK - WHITE':'COMMERCIAL','BUSINESS PARK':'COMMERCIAL','PLACE OF WORSHIP':'CIVIC & COMMUNITY INSTITUTION',
 'PORT / AIRPORT':'TRANSPORT','HEALTH & MEDICAL CARE':'CIVIC & COMMUNITY INSTITUTION'}
SEMANTIC_KEYS = ('amenity','shop','leisure','landuse','building','office','tourism','public_transport','railway','highway','natural','sport','industrial')
# Fixed categorical vocabulary is deliberately bounded; free text never becomes a category.
SEMANTIC_VALUES = frozenset(('yes','school','university','college','kindergarten','hospital','clinic','doctors','pharmacy',
 'place_of_worship','community_centre','library','police','fire_station','townhall','post_office','bank','atm','restaurant','cafe',
 'fast_food','food_court','bar','pub','parking','parking_entrance','bicycle_parking','fuel','bus_station','bus_stop','station',
 'platform','subway_entrance','stop_position','tram_stop','residential','apartments','house','detached','terrace','dormitory',
 'commercial','retail','industrial','warehouse','supermarket','convenience','mall','hotel','hostel','park','pitch','sports_centre',
 'stadium','swimming_pool','playground','fitness_centre','garden','track','grass','forest','water','wood','office','toilets',
 'shelter','bench','recycling','vending_machine','car_repair','car','bicycle','clothes','hairdresser','beauty','bakery','butcher',
 'electronics','department_store','construction','service','footway','pedestrian','path','residential','primary','secondary',
 'tertiary','motorway','trunk','cycleway','steps','living_street','crossing','traffic_signals','turning_circle','telephone',
 'drinking_water','waste_basket','fountain','memorial','museum','attraction','viewpoint','theatre','cinema','arts_centre'))

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def recompute_labels(rows):
    """Match pandas idxmax first-tie semantics and official dismiss/rename."""
    best = {}
    for i,row in enumerate(rows):
        key = str(int(float(row['id'])))
        score = float(row['PERCENTAGE'])
        if not math.isfinite(score):
            continue
        if key not in best or score > best[key][0]:
            best[key] = (score,row['LU_DESC'],i)
    return {key:RENAME.get(v[1],v[1]) for key,v in best.items() if v[1] not in DISMISS}, best

def semantic_categories(tags):
    out=[]
    for key in SEMANTIC_KEYS:
        if key in tags:
            val=str(tags[key]).lower().strip()
            out.append(key+':'+(val if val in SEMANTIC_VALUES else 'other'))
    return sorted(set(out or ['entity:other']))

def role_allocation(cells, seed=20261022):
    """Exact largest-remainder 60/15/15/10 allocation of geographic groups."""
    cells=sorted(set(cells),key=lambda c:digest([seed,list(c)]))
    raw=[len(cells)*x for x in (.60,.15,.15,.10)]
    counts=[int(x) for x in raw]
    for i in sorted(range(4),key=lambda i:(-(raw[i]-counts[i]),i))[:len(cells)-sum(counts)]: counts[i]+=1
    result={};start=0
    for role,n in zip(ROLES,counts):
        for cell in cells[start:start+n]:result[cell]=role
        start+=n
    return result

def balanced_members(rows, role, per_class, seed=20261022):
    chosen=[];counts={}
    for label in CLASSES:
        candidates=sorted((r for r in rows if r['role']==role and r['label']==label),key=lambda r:digest([seed,r['id']]))
        counts[label]=len(candidates)
        chosen.extend(r['id'] for r in candidates[:per_class])
    return chosen,counts,all(v>=per_class for v in counts.values())

def parse_label(text):
    try:
        obj=json.loads(text)
        if isinstance(obj,dict) and set(obj)=={'class'} and obj['class'] in CLASSES:return obj['class']
    except (ValueError,TypeError):pass
    return None

def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-h,c+h]

def score(gold,outputs):
    if len(gold)!=len(outputs) or not gold:raise ValueError('Fixed nonempty denominator required')
    if any(x not in CLASSES for x in gold):raise ValueError('Unknown gold')
    pred=[parse_label(x) for x in outputs];correct=sum(g==p for g,p in zip(gold,pred));per={}
    for label in CLASSES:
        tp=sum(g==label and p==label for g,p in zip(gold,pred));fn=sum(g==label and p!=label for g,p in zip(gold,pred));fp=sum(g!=label and p==label for g,p in zip(gold,pred))
        per[label]={'support':tp+fn,'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}
    return {'n':len(gold),'correct':correct,'accuracy':correct/len(gold),'valid':sum(p is not None for p in pred),
            'valid_rate':sum(p is not None for p in pred)/len(gold),'macro_f1':sum(v['f1'] for v in per.values())/len(CLASSES),
            'per_class':per,'wilson95':wilson(correct,len(gold)),
            'confusion':{g:{p:sum(a==g and (b or 'INVALID')==p for a,b in zip(gold,pred)) for p in (*CLASSES,'INVALID')} for g in CLASSES}}

def qualification_gate(result,majority,logistic,no_leak):
    return bool(result['n']==96 and result['correct']>=63 and result['accuracy']>=.65 and
                result['accuracy']>=majority+.10 and result['accuracy']>=logistic+.05 and
                result['macro_f1']>=.60 and result['valid_rate']>=.95 and no_leak)

def prompt(payload,variant=0):
    instructions=(
      'Classify the target building planning-use proxy from its physical shape and surrounding public map context.',
      'Infer the most plausible planning-use class. Weigh neighborhood category composition and relative distances alongside geometry.',
      'Select a planning-use class using the supplied structured spatial evidence. Nearby facilities may serve other buildings; do not equate one neighbor with the target.')
    if variant not in range(3):raise ValueError('Only three predeclared prompt skeletons')
    system=instructions[variant]+' Label definitions (URA broad land use with the released CityFM merges): '+json.dumps(CLASS_DEFINITIONS,sort_keys=True)+'. Return exactly one JSON object {"class":"<label>"}, no other keys or explanation.'
    return {'system':system,'user':json.dumps(payload,sort_keys=True,separators=(',',':')),'max_tokens':64,
            'response_contract':{'type':'object','required':['class'],'additionalProperties':False,'properties':{'class':{'enum':list(CLASSES)}}}}

def baseline_features(payload):
    geo=payload['target'];f={k:float(v) for k,v in geo.items() if isinstance(v,(int,float))}
    for category,count in payload['all_neighbor_category_counts'].items(): f['count:'+category]=float(count)
    for n in payload['nearest_neighbors']:
        b=0 if n['distance_m']<25 else 1 if n['distance_m']<50 else 2 if n['distance_m']<75 else 3
        for c in n['categories']:f['bin:'+c+':'+str(b)]=f.get('bin:'+c+':'+str(b),0)+1
    return f
