"""Reuse the frozen native response-only packing contract for a new local task."""
from pathlib import Path
import ast
import hashlib

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/probe_overnight_train_fit_20261025.py"
# The enclosing CPU/GPU protocol also binds SOURCE's actual bytes.
node = next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "packing")
namespace = {}
exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(SOURCE), "exec"), namespace)
native_packing = namespace["packing"]


def packing(tok, messages, target):
    return native_packing(tok, dict(messages=messages), dict(text=target), "zero-shot", [])


def public_proof(tok):
    """The completed-assistant template is not the inference prefix on this model."""
    rows = []
    for text in ["Object p000 chair 0.000 1.000 2.000", "Two objects: chair and table.", "Unicode probe: 椅子, café, 空间."]:
        messages = [dict(role="system", content="Return exactly <final>A</final>."), dict(role="user", content=text)]
        prompt, answer = packing(tok, messages, "<final>A</final>")
        actual_inference = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        completed = tok.apply_chat_template(messages+[dict(role="assistant",content="<final>A</final>")],
                                            tokenize=True, add_generation_prompt=False, enable_thinking=False)
        if prompt != actual_inference or not answer or answer[-1] != tok.convert_tokens_to_ids("<|im_end|>"):
            raise AssertionError("Frozen native packing must match inference prefix and chat EOS")
        rows.append(dict(original_completed_template_prefix_matches=completed[:len(prompt)] == prompt,
                         actual_native_packing_matches_inference=True, input_tokens=len(prompt),
                         target_tokens=len(answer), target_end=answer[-1]))
    if all(x["original_completed_template_prefix_matches"] for x in rows):
        raise AssertionError("Expected actual old prefix counterexample before using this fix")
    return dict(status="PUBLIC_NATIVE_PACKING_RED_GREEN_PASS", cases=rows,
                original_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                original_function_AST_unchanged=True, model_loads=0, GPU_raw=0, API_calls=0)
