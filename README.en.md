# Privacy-Constrained Synthetic Distillation

Local data modeling → synthetic supervision → student fine-tuning → compact deployment.

[中文与完整结果](README.md) · [Method](docs/method.md) · [Results](docs/results.md) · [Reproduction](docs/reproduce.md)

This research project investigates whether synthetic training signals can substitute for limited real training signals when protected records remain local. Generator fitting, answer validation, teacher labeling, student learning and deployment are separate, auditable components.

The primary evidence concerns **USTBench POI understanding**, using Qwen3.5-2B. With 68 supervised source questions, 136 updates and three training orders, mean accuracy on the same 1,154 evaluation questions was **88.13% real / 89.63% old generator / 89.80% revised generator**.

These are reference-labelled SFT controls with eight shared real demonstrations, evaluated in a supplemental comparison on a previously exposed Final split. They do not establish formal equivalence, independent final acceptance, general planning ability or differential privacy. The separately collected 1,088 synthetic teacher labels matched the rule targets after one approved recovery call; identical executions are not independent teacher-vs-oracle replicates.

Deployment diagnostics on one synthetic-trained checkpoint achieved **66/72 with BF16 and INT8**, and **61/72 with NF4**. INT8 reduced model size but was slower in the measured setup. Prompt compression and exact-prefix caching were also evaluated. All results remain scoped to their datasets, prompts and hardware.

Run without a model, data download or API:

```bash
python3 examples/local_contract_demo.py
python3 tools/show_results.py
python3 tools/verify_release.py
python3 tests/test_ust_task_contract.py
python3 tests/test_ust_task_contract_independent.py
```

The original research source is preserved with hashes. Historical GPU entrypoints require locally held datasets, run manifests, checkpoints and separately authorized API credentials; cloning this repository alone does not reproduce the full study. No raw protected data, per-question model traces, model weights or credentials are included. Historical public-proxy exceptions are disclosed in the scope document.
