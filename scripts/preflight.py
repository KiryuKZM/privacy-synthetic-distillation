#!/usr/bin/env python3
"""Zero-teacher preflight: git, Adult archive, submodules, GPU, Python, Docker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_ADULT_SHA256 = (
    "7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb"
)
EXPECTED_ADULT_BYTES = 620237
PINNED_SUBMODULES = {
    "vendor/ctgan": "66e5686fbc55a55eb137253dce115946f5785ba2",
    "vendor/tab-ddpm": "b476257dd460b778ba09eb97f7a51d6490fa17f8",
    "vendor/tabsyn": "cb5ac0f74ec36ee88e7a974a393dfbef50d42da7",
}
UNRESOLVED = ("vendor/midst", "vendor/midst-models", "vendor/tabllm")


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gpu_snapshot() -> dict[str, object]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {"nvidia_smi": None}
    query = run(
        [
            smi,
            "--query-gpu=name,compute_cap,memory.total,memory.used,memory.free,driver_version,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    processes = run([smi, "--query-compute-apps=pid,process_name,used_memory", "--format=csv"])
    return {
        "nvidia_smi": smi,
        "query_csv": query.stdout.strip(),
        "query_err": query.stderr.strip(),
        "query_code": query.returncode,
        "processes": processes.stdout.strip(),
    }


def python_snapshot() -> dict[str, object]:
    mods = [
        "numpy",
        "pandas",
        "sklearn",
        "torch",
        "ctgan",
        "rdt",
        "xgboost",
        "transformers",
        "peft",
        "bitsandbytes",
    ]
    found: dict[str, str] = {}
    missing: list[str] = []
    for name in mods:
        try:
            mod = __import__(name)
            found[name] = getattr(mod, "__version__", "present")
        except Exception:
            missing.append(name)
    torch_cuda: dict[str, object] | None = None
    if "torch" in found:
        import torch  # type: ignore

        torch_cuda = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda": getattr(torch.version, "cuda", None),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
    return {
        "executable": sys.executable,
        "version": sys.version,
        "found": found,
        "missing": missing,
        "torch_cuda": torch_cuda,
        "uv": shutil.which("uv"),
        "pip": shutil.which("pip") or shutil.which("pip3"),
        "nvcc": shutil.which("nvcc"),
        "conda": shutil.which("conda"),
    }


def git_snapshot(root: Path) -> dict[str, object]:
    status = run(["git", "status", "--porcelain=v1", "--branch"], cwd=root)
    head = run(["git", "rev-parse", "HEAD"], cwd=root)
    sub = run(["git", "submodule", "status"], cwd=root)
    dirty = [line for line in status.stdout.splitlines() if line and not line.startswith("##")]
    sub_lines = [line for line in sub.stdout.splitlines() if line.strip()]
    pinned_ok = True
    pinned_detail = {}
    for path, expected in PINNED_SUBMODULES.items():
        actual = run(["git", "-C", str(root / path), "rev-parse", "HEAD"], cwd=root)
        sha = actual.stdout.strip()
        pinned_detail[path] = {"expected": expected, "actual": sha, "ok": sha == expected}
        pinned_ok = pinned_ok and sha == expected
        if not (root / path / ".git").exists() and not (root / ".git" / "modules" / path).exists():
            # submodule working tree exists if rev-parse succeeded
            pass
    unresolved = {path: (root / path).exists() for path in UNRESOLVED}
    return {
        "branch_status": status.stdout.strip(),
        "head": head.stdout.strip(),
        "dirty_lines": dirty,
        "clean": not dirty,
        "submodule_status": sub_lines,
        "pinned": pinned_detail,
        "pinned_ok": pinned_ok,
        "unresolved_present": unresolved,
    }


def archive_snapshot(root: Path) -> dict[str, object]:
    archive = root / "data" / "raw" / "adult.zip"
    if not archive.is_file():
        return {"path": str(archive), "exists": False}
    digest = sha256_file(archive)
    inner = {}
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            inner[name] = {
                "bytes": zf.getinfo(name).file_size,
                "sha256": hashlib.sha256(zf.read(name)).hexdigest(),
            }
    return {
        "path": str(archive),
        "exists": True,
        "bytes": archive.stat().st_size,
        "sha256": digest,
        "expected_sha256": EXPECTED_ADULT_SHA256,
        "expected_bytes": EXPECTED_ADULT_BYTES,
        "sha256_ok": digest == EXPECTED_ADULT_SHA256,
        "bytes_ok": archive.stat().st_size == EXPECTED_ADULT_BYTES,
        "inner": inner,
    }


def docker_snapshot() -> dict[str, object]:
    docker = shutil.which("docker")
    info = None
    if docker:
        proc = run([docker, "info"])
        info = {"code": proc.returncode, "stderr_tail": proc.stderr.strip().splitlines()[-8:]}
    return {
        "docker_binary": docker,
        "info": info,
        "not_started_by_this_preflight": True,
    }


def memory_snapshot() -> dict[str, object]:
    mem = {}
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:")):
                key, value, _unit = line.split()
                mem[key[:-1]] = int(value)
    return {
        "uname": os.uname().release,
        "nproc": os.cpu_count(),
        "meminfo_kb": mem,
    }


def build_report(root: Path) -> dict[str, object]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "root": str(root),
        "git": git_snapshot(root),
        "archive": archive_snapshot(root),
        "python": python_snapshot(),
        "gpu": gpu_snapshot(),
        "docker": docker_snapshot(),
        "host": memory_snapshot(),
        "gates": {
            "docker_started": False,
            "external_teacher_calls": False,
            "checkpoint_downloads": False,
            "model_training": False,
        },
    }


def render_markdown(data: dict[str, object]) -> str:
    git = data["git"]
    archive = data["archive"]
    python = data["python"]
    gpu = data["gpu"]
    docker = data["docker"]
    host = data["host"]
    inner_lines = []
    if archive.get("inner"):
        inner_lines = [
            "| Inner file | bytes | sha256 |",
            "| --- | ---: | --- |",
        ]
        for name, meta in archive["inner"].items():
            inner_lines.append(f"| `{name}` | {meta['bytes']} | `{meta['sha256']}` |")
    pinned_lines = ["| Path | pinned SHA | actual | ok |", "| --- | --- | --- | --- |"]
    for path, meta in git["pinned"].items():
        pinned_lines.append(
            f"| `{path}` | `{meta['expected']}` | `{meta['actual']}` | {meta['ok']} |"
        )
    missing = ", ".join(f"`{name}`" for name in python["missing"]) or "(none)"
    found = ", ".join(f"`{k}={v}`" for k, v in python["found"].items()) or "(none)"
    unresolved = ", ".join(
        f"`{path}`={'present' if present else 'absent'}"
        for path, present in git["unresolved_present"].items()
    )
    lines = [
        "# Preflight 2026-10-01",
        "",
        "Zero external teacher calls. Docker was not started. No checkpoint",
        "weights were downloaded and no model was trained.",
        "",
        f"Generated at `{data['generated_at_utc']}` in `{data['root']}`.",
        "",
        "## Git",
        "",
        f"- HEAD: `{git['head']}`",
        f"- Working tree clean: {git['clean']}",
        f"- Branch status: `{git['branch_status'].splitlines()[0] if git['branch_status'] else ''}`",
        "",
        *pinned_lines,
        "",
        f"Unresolved inventory paths (not installed): {unresolved}.",
        "",
        "## Adult archive",
        "",
        f"- Path: `{archive.get('path')}`",
        f"- Exists: {archive.get('exists')}",
        f"- Bytes: {archive.get('bytes')} (expected {archive.get('expected_bytes')}, ok={archive.get('bytes_ok')})",
        f"- SHA-256: `{archive.get('sha256')}`",
        f"- SHA-256 match: {archive.get('sha256_ok')}",
        "",
        *inner_lines,
        "",
        "## GPU / WSL",
        "",
        f"- nvidia-smi: `{gpu.get('nvidia_smi')}`",
        f"- Query: `{gpu.get('query_csv')}`",
        f"- Compute processes visible to WSL: `{gpu.get('processes')}`",
        f"- Host kernel: `{host.get('uname')}`",
        f"- nproc: {host.get('nproc')}",
        f"- MemTotal_kB: {host.get('meminfo_kb', {}).get('MemTotal')}",
        f"- MemAvailable_kB: {host.get('meminfo_kb', {}).get('MemAvailable')}",
        "",
        "RTX 5080 Laptop is Blackwell `sm_120` with 16303 MiB. A CUDA wheel must",
        "include `sm_120` kernels (PyTorch CUDA 12.8+). That check is still open",
        "because `torch` is not installed. WSL `nvidia-smi` may hide Windows-side",
        "GPU users; the used-memory field is the occupancy signal.",
        "",
        "## Python / CUDA tooling",
        "",
        f"- Executable: `{python['executable']}`",
        f"- Version: `{python['version'].splitlines()[0]}`",
        f"- uv: `{python['uv']}`",
        f"- pip: `{python['pip']}`",
        f"- nvcc: `{python['nvcc']}`",
        f"- conda: `{python['conda']}`",
        f"- Present scientific packages: {found}",
        f"- Missing scientific packages: {missing}",
        "",
        "## Docker",
        "",
        f"- Binary on PATH: `{docker.get('docker_binary')}`",
        f"- `docker info` exit: {None if docker.get('info') is None else docker['info'].get('code')}",
        f"- Not started by this preflight: {docker.get('not_started_by_this_preflight')}",
        "",
        "A Docker Desktop binary is visible from WSL. This preflight did not start",
        "a daemon and did not run containers. If a later step truly needs Docker,",
        "stop and wait for the user to enable WSL integration and start it.",
        "",
        "## Verdict",
        "",
        "- Asset hash and pinned generator submodules are usable for local data work.",
        "- MIDST, MIDSTModels, and TabLLM remain inventory-only.",
        "- No local scientific Python environment exists yet; Adult splitting uses",
        "  the standard library only.",
        "- Do not install generator/teacher/student weights until `idea.md` records",
        "  a confirmed component choice.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    parser.add_argument("--write-markdown", type=Path, default=None)
    parser.add_argument("--write-json", type=Path, default=None)
    args = parser.parse_args()
    data = build_report(args.root)
    markdown = render_markdown(data)
    print(markdown)
    if args.write_markdown:
        args.write_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.write_markdown.write_text(markdown, encoding="utf-8")
    if args.write_json:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    archive_ok = bool(data["archive"].get("sha256_ok") and data["archive"].get("bytes_ok"))
    pinned_ok = bool(data["git"]["pinned_ok"])
    return 0 if archive_ok and pinned_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
