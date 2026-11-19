"""Local text-world worker. JSON lines in/out; no model or network client.

Only public observations cross the interface. Native gold/policy state never
appears in the worker protocol. The manifest must already freeze roles/assets.
"""
from __future__ import annotations
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/2026-10-23-llm-task-sweep/agent-preparation"


def model_observation(result):
    """Allowlist teacher-visible fields; native scores stay in the controller."""
    return {key: result[key] for key in ("observation", "look", "inventory", "task") if key in result}


class Session:
    def __init__(self, benchmark, manifest=None):
        self.benchmark = benchmark
        self.manifest = manifest or json.loads((BASE / f"{benchmark}-episodes.json").read_text())
        if self.manifest.get("status") != "READY":
            raise RuntimeError("Native world preparation is not READY; no model workload may start")
        self.episodes = {x["episode_id"]: x for x in self.manifest["episodes"]}
        self.env = None
        self.turn = 0
        self.closed = False
        self.terminated = False

    def start(self, episode_id):
        row = self.episodes[episode_id]
        if row["role"] not in {"Fit", "Dev", "Screen"}:
            raise ValueError("Official final/test episodes are sealed")
        self.close()
        self.closed = False
        self.turn = 0
        self.terminated = False
        self.row = row
        if self.benchmark == "alfworld":
            import textworld
            from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
            # Deliberately no facts/policy_commands/expert_plan/admissible list.
            self.env = textworld.start(str(ROOT / row["game_path"]),
                                       request_infos=textworld.EnvInfos(won=True),
                                       wrappers=[AlfredDemangler(shuffle=False)])
            state = self.env.reset()
            return self._alf_visible(state, 0, False)
        if self.benchmark == "scienceworld":
            from scienceworld import ScienceWorldEnv
            jre = ROOT / self.manifest["java_home"]
            os.environ["JAVA_HOME"] = str(jre)
            os.environ["PATH"] = str(jre / "bin") + os.pathsep + os.environ.get("PATH", "")
            os.environ["_JAVA_OPTIONS"] = "-Xms64m -Xmx512m -XX:ActiveProcessorCount=2"
            self.env = ScienceWorldEnv(envStepLimit=100)
            self.env.load(row["task_name"], row["variation"], "", generateGoldPath=False)
            observation, info = self.env.reset()
            return self._science_visible(observation, 0, False, info)
        raise ValueError(self.benchmark)

    def step(self, action):
        if self.closed or self.env is None or self.terminated:
            raise ValueError("No active episode")
        if not isinstance(action, str) or not action.strip() or "\n" in action or len(action) > 300:
            # The runner should count this as a model turn, without a repair API.
            self.turn += 1
            self.terminated = self.turn >= self.manifest["model_turn_cap"]
            return {"observation": "FORMAT_ERROR: supply exactly one action string.",
                    "turn": self.turn, "done": self.terminated,
                    "success": False, "format_invalid": True}
        self.turn += 1
        if self.benchmark == "alfworld":
            state, reward, done = self.env.step(action.strip())
            return self._alf_visible(state, reward, done)
        observation, reward, done, info = self.env.step(action.strip())
        return self._science_visible(observation, reward, done, info)

    def _alf_visible(self, state, reward, done):
        success = bool(state.get("won", False))
        self.terminated = bool(done or success or self.turn >= self.manifest["model_turn_cap"])
        return {"observation": state.feedback, "reward": reward,
                "turn": self.turn, "done": self.terminated,
                "success": success, "native_won": success}

    def _science_visible(self, observation, reward, done, info):
        score = int(info["score"])
        raw_score = float(self.env.server.getScore())
        native_completed = bool(self.env.server.getCompleted())
        self.terminated = bool(done or self.turn >= self.manifest["model_turn_cap"])
        # done also fires on native negative score and >100 moves. Full success
        # requires the native full score, never done alone.
        return {"observation": observation, "look": info["look"], "inventory": info["inv"],
                "task": info["taskDesc"], "reward": reward, "score": score,
                "moves": info["moves"], "turn": self.turn,
                "done": self.terminated, "native_completed": native_completed,
                "native_raw_score": raw_score,
                "success": raw_score >= 1.0 and native_completed}

    def close(self):
        if self.env is not None:
            if self.benchmark == "scienceworld":
                self.env.close()
            else:
                self.env.close()
        self.env = None
        self.closed = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=["alfworld", "scienceworld"])
    args = parser.parse_args()
    session = Session(args.benchmark)
    try:
        for line in sys.stdin:
            try:
                msg = json.loads(line)
                # Keep upstream diagnostic printouts off the machine channel.
                with contextlib.redirect_stdout(sys.stderr):
                    if msg["op"] == "start":
                        result = session.start(msg["episode_id"])
                    elif msg["op"] == "step":
                        result = session.step(msg["action"])
                    elif msg["op"] == "close":
                        session.close()
                        result = {"closed": True}
                    else:
                        raise ValueError("Unknown operation")
                print(json.dumps({"ok": True, "result": result}), flush=True)
            except Exception as exc:
                print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), flush=True)
    finally:
        session.close()


if __name__ == "__main__":
    main()
