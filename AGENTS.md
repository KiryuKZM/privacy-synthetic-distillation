# Public research archive

Read README.md, docs/scope.md and docs/reproduce.md first. Preserve the distinction between verified results and hypotheses. Do not change claims without evidence.

Never add raw/private records, per-question outputs, credentials, model weights or local run manifests. Do not run historical paid API entrypoints without new explicit authorization and current pricing. Historical prices are not current quotes.

Source snapshots listed in archive/source-manifest.json should remain byte-identical; put intentional future adaptations in a new documented module and record provenance. Offline checks: python3 tools/verify_release.py; python3 tests/test_ust_task_contract.py; python3 tests/test_ust_task_contract_independent.py.
