"""Run the ScienceWorld recovery using the preserved original local worker.

The original code and blocked v1 manifests stay unchanged. This entry point
requires four real native Fit/worker completions in the recovery receipt.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"runs/2026-10-23-llm-task-sweep/agent-preparation/recovery-v2"


def main():
    checks=json.loads((BASE/"scienceworld-native-worker-smoke.json").read_text())
    if len(checks)!=4 or not all(x["success"] and x["native_completed"] and x["native_raw_score"]>=1.0 for x in checks):
        raise RuntimeError("ScienceWorld recovery has no complete native worker evidence")
    spec=importlib.util.spec_from_file_location("preserved_agent_world_worker",ROOT/"scripts/agent_world_sweep_runtime_20261023.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.BASE=BASE
    module.main()


if __name__=="__main__":main()
