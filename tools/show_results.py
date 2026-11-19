"""Recalculate aggregate result tables without protected data or model access."""
from pathlib import Path
import json, statistics
R = Path(__file__).resolve().parents[1]
d = json.loads((R/'results/matched68.json').read_text())
for arm in ('real','old','current'):
    counts = [d['scores'][f'{arm}_n68_s{s}']['original']['overall']['correct'] for s in d['orders']]
    print(arm, counts, f"mean={statistics.mean(counts)/d['questions']:.4%}")
q = json.loads((R/'results/quantization72.json').read_text())
for arm in ('bf16','int8','nf4'):
    x=q['summaries'][arm]
    print(arm, f"{x['correct']}/{x['n']}", f"weights={x['weights_bytes']/1e9:.3f} GB")
