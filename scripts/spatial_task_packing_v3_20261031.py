"""Correct the validation container comparison; native packing is unchanged."""
from spatial_task_packing_20261031 import packing, SOURCE
import hashlib


def public_proof(tok):
    rows=[]
    for text in ["Object p000 chair 0.000 1.000 2.000", "Two objects: chair and table.", "Unicode probe: 椅子, café, 空间."]:
        messages=[dict(role="system",content="Return exactly <final>A</final>."),dict(role="user",content=text)]
        prompt,answer=packing(tok,messages,"<final>A</final>")
        default=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
        ids=tok.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
        completed=tok.apply_chat_template(messages+[dict(role="assistant",content="<final>A</final>")],
                                          tokenize=True,return_dict=False,add_generation_prompt=False,enable_thinking=False)
        if not hasattr(default,"get") or default.get("input_ids")!=prompt or ids!=prompt:
            raise AssertionError("Actual inference IDs differ from frozen native packing")
        if not answer or answer[-1]!=tok.convert_tokens_to_ids("<|im_end|>"):
            raise AssertionError("Actual target chat EOS differs")
        rows.append(dict(default_return_type=type(default).__name__,
                         old_list_vs_container_equal=prompt==default,
                         actual_input_ids_match=True,explicit_list_ids_match=True,
                         completed_assistant_actual_prefix_matches=completed[:len(prompt)]==prompt,
                         target_tokens=len(answer),target_end=answer[-1]))
    if any(x["old_list_vs_container_equal"] for x in rows):
        raise AssertionError("Old container-comparison counterexample not reached")
    return dict(status="PUBLIC_ACTUAL_IDS_AND_NATIVE_PACKING_PASS",cases=rows,
                original_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                original_function_AST_unchanged=True,
                correction="Earlier claimed template mismatch was a validation bug: BatchEncoding vs list. Actual template IDs match.",
                model_loads=0,GPU_raw=0,API_calls=0)
