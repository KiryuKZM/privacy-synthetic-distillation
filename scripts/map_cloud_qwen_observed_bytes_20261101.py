"""CPU-only observed cloud candidate-byte to student-event mapping; no weights or API."""
from pathlib import Path
import hashlib,json,time
R=Path(__file__).resolve().parents[1];P=R/'runs/2026-11-01-cloud-qwen-probability-generic-v1'
def bind(p):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
 from transformers import AutoTokenizer
 settings=R/'runs/2026-10-31-new-generator-three-method-comparison/common-eval-cuda-visible-v2/endpoints/original-Qwen3.5-2B/settings.json';base=Path(json.loads(settings.read_text())['base'])
 tok=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False);backend=json.loads(tok.backend_tokenizer.to_str());assert backend['decoder']['type']=='ByteLevel'
 bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs[:];n=0
 for b in range(256):
  if b not in bs:bs.append(b);cs.append(256+n);n+=1
 byte_of={chr(c):b for b,c in zip(bs,cs)};added=set(tok.get_added_vocab());lookup={}
 for piece,index in tok.get_vocab().items():
  value=piece.encode() if piece in added else bytes(byte_of[ch] for ch in piece)
  lookup.setdefault(value,[]).append(index)
 assert len(tok)==248077
 observed=[]
 for index in range(2):
  result=json.loads((P/f'call-{index}.json').read_text());assert result['status']=='RESPONDED' and not result['thinking_content_present']
  for position in result['probability']['top5']:
   for item in position['top_logprobs']:
    value=bytes(item['bytes']);ids=lookup.get(value,[]);record=dict(call=index,token=item['token'],bytes=item['bytes'],student_ID_candidates=ids,unique_single_student_event=len(ids)==1)
    if len(ids)==1:
     decoded=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False);record['exact_single_ID_decoded_bytes']=(decoded.encode()==value)
    observed.append(record)
 prefix='The sky is';ids=tok.encode(prefix,add_special_tokens=False);roundtrip=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)==prefix
 result=dict(status='OBSERVED_CLOUD_TOP5_BYTE_EVENTS_CPU_MAPPING_COMPLETE',observations=observed,count=len(observed),unique_mapped=sum(x['unique_single_student_event'] and x.get('exact_single_ID_decoded_bytes',False) for x in observed),prefix=dict(text=prefix,student_ids=ids,UTF8_roundtrip_exact=roundtrip),student_semantic_vocab=len(tok),input_metadata=bind(base/'tokenizer.json'),API_observations=[bind(P/f'call-{i}.json') for i in range(2)],model_weights_loaded=0,GPU_calls=0,API_calls=0,global_hosted_Max_tokenizer_equivalence_proven=False,all_possible_mixed_prefixes_proven=False,hidden_Max_template_and_probability_processing_stage_unproven=True,epoch=time.time())
 path=P/'observed-token-byte-map.json'
 with path.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({k:result[k] for k in ('status','count','unique_mapped','prefix','student_semantic_vocab','global_hosted_Max_tokenizer_equivalence_proven')}))
if __name__=='__main__':main()
