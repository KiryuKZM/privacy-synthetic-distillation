"""Answer choice semantics, distinct from legacy strict output-format validity.

This prospective parser accepts only a sole A-E or a sole final-tag envelope.
Whitespace/case differences are typography; extra prose/multiple choices fail.
"""
import re
def parse_choice(text):
    if not isinstance(text,str): return None
    normalized="".join(text.split()).casefold()
    match=re.fullmatch(r"([a-e])|<final>([a-e])</final>",normalized)
    return (match.group(1) or match.group(2)).upper() if match else None
def strict_format_valid(text):
    from privacy_synth.spatial_native_qa import parse_answer
    return isinstance(text,str) and parse_answer(text) is not None
