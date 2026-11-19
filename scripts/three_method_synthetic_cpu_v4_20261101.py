"""Bounded CPU-only synthetic scenes for the user-authorized three-method run."""
from pathlib import Path
import argparse, csv, datetime, hashlib, json, math, os, signal, sqlite3, subprocess, sys, time, traceback
R = Path(__file__).resolve().parents[1]
O = R / "runs/2026-10-31-new-generator-three-method-comparison"
D = R / "runs/2026-10-31-spatial-joint-composition-inner-v1"
G = O / "synthetic-scenes-v1"
STOP = False
if os.name == 'posix': import resource
sys.path[:0] = [str(R / "src"), str(R / "scripts")]
def load(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):
    with Path(p).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()
def bind(p):
    p = Path(p)
    return dict(path=p.relative_to(R).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
def once(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
def atomic(p, value):
    p = Path(p); tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); tmp.replace(p)
def stopped():
    return STOP or (O / "STOP").exists() or (R / "runs/STOP").exists()
def on_stop(*_):
    global STOP
    STOP = True
def public_conditions():
    counts = [8, 16, 32, 64, 128, 256]
    sizes = [(3., 3., 2.5), (4., 3., 3.), (5., 4., 3.5), (6., 5., 4.)]
    return [dict(public_ordinal=i, count=counts[i % 6],
                 room_size_native=list(sizes[(i // 6) % 4]), source="independent public fixed schedule, no real room/ID/statistics")
            for i in range(96)]
def teacher_body(q):
    return dict(model="deepseek-flash", messages=[dict(role="system", content=q["system"]),
                dict(role="user", content=q["prompt"])], thinking=dict(type="disabled"),
                max_tokens=512, temperature=0)
def public_tests():
    from privacy_synth.spatial_native_qa import FAMILIES, build_question, independent_answer
    c = public_conditions()
    assert len(c) == 96 and len({x["public_ordinal"] for x in c}) == 96
    assert all(set(x) == {"public_ordinal", "count", "room_size_native", "source"} for x in c)
    toy = dict(count=3, objects=[dict(category_id=i, position_native=[float(i), 0., 1.]) for i in [0, 1, 0]])
    for family in FAMILIES:
        q = build_question(toy, ["chair", "table", "lamp", "shelf"], family, 17)
        assert independent_answer(q["prompt"], family, q["anchor"]) == q["answer"]
        dirty = dict(q, record_id="PRIVATE_SENTINEL", building_index=991, answer="SECRET_GOLD", statistics={"private": 3})
        body = teacher_body(dirty)
        assert "PRIVATE_SENTINEL" not in json.dumps(body) and "SECRET_GOLD" not in json.dumps(body)
        assert set(body) == {"model", "messages", "thinking", "max_tokens", "temperature"}
        assert [x["role"] for x in body["messages"]] == ["system", "user"]
    return dict(public_condition_contract=True, three_serialized_oracles=True,
                explicit_outbound_allowlist=True, injected_private_fields_not_serialized=True)
def sources():
    return [Path(__file__), R / "src/privacy_synth/spatial_native_qa.py",
            R / "src/privacy_synth/spatial_room_composition_model.py",
            R / "src/privacy_synth/spatial_joint_composition_reference.py",
            R / "src/privacy_synth/spatial_mixture_reference.py",
            R / "src/privacy_synth/spatial_set_diffusion.py",
            R / "src/privacy_synth/spatial_reverse_variance.py",
            R / "scripts/sg_formal_windows_resources_20261027.py",
            D / "protocol.json", D / "reference-parameters.local.json", D / "finite-composition.local.json",
            D / "condition-normalization.local.json", D / "normalization.local.json",
            D / "seed01/checkpoints/4000.pt", D / "seed02/checkpoints/4000.pt",
            R / "runs/current-execution-permissions.json"]
def prepare():
    G.mkdir(exist_ok=False)
    tests = public_tests()
    with sqlite3.connect("file:" + str(R / "runs/2026-10-26-generator-discovery/budget/new-api-ledger.sqlite") + "?mode=ro", uri=True) as db:
        db.execute("BEGIN")
        cfg = db.execute("SELECT config FROM settings").fetchone()[0]
        assert json.loads(cfg)["global_pico"] == 500 * 10**12
        tables = {}
        for table in ("calls", "qwen_max_calls"):
            n, pico, unknown, inflight = db.execute("SELECT COUNT(*),COALESCE(SUM(pico),0),SUM(1-usage_known),SUM(1-settled) FROM " + table).fetchone()
            tables[table] = dict(rows=n, accounted_pico=pico, unknown_usage=unknown, inflight=inflight)
        triggers = dict(db.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger'"))
        assert all(name in triggers for name in ("sgqa_global500_legacy_insert", "sgqa_global500_qwen_insert"))
        total = sum(x["accounted_pico"] for x in tables.values())
        assert total + 20 * 10**12 <= 500 * 10**12
        once(O / "budget-before-new-calls.json", dict(tables=tables, cumulative_conservative_USD=total / 10**12,
            global_cap_USD=500, new_package_USD_proposed=20, additional_budget=False,
            canonical_settings_sha256=hashlib.sha256(cfg.encode()).hexdigest(), triggers=triggers,
            provider_invoice_reconciled=False, inspected_epoch=time.time(), paid_calls_dispatched=0))
    once(G / "public-conditions96.json", public_conditions())
    p = dict(source=bind(__file__), bindings=[bind(x) for x in sources()[1:]], public_tests=tests,
        conditions=bind(G / "public-conditions96.json"), generator_seeds=["seed01", "seed02"],
        each=96, new_scenes=192, task_families=["category_majority", "nearest_category", "centroid_category"],
        teacher_logical_labels=576, generator_updates=0, sampler="unchanged FP32 fixed_large64 original sample()",
        condition_source="independent fixed public count/shape schedule; no protected real condition, IDs or gold",
        no_refill=True, keep_all_invalid=True, inference_device="cpu", worker_seconds_max=3600,
        protections=dict(CPU=2, own_host_GiB=2, shared_host_GiB=14, noSwap=True,
            Windows_start_GiB=22, Windows_run_GiB=20, disk_GiB=20, output_GiB=3, GPU_calls=0),
        model_claim="New generated training scenes from two already completed frozen generators; not new generator validation",
        outward_contract="Only generated object categories and displayed rounded coordinates, fresh synthetic pNNN aliases, fixed task instructions/options. No real IDs/rows/statistics/gold/retrieval. API dispatch needs separate exact source/body/cumulative pool grant.")
    once(G / "protocol.v4.json", p)
    once(G / "root-decision.v4.json", dict(approved=True, actor="Root", user_intent=bind(O / "user-intent-and-trigger.json"),
        source_sha256=sha(__file__), protocol_sha256=sha(G / "protocol.v4.json"),
        actions=["generate-two-frozen-models-on-CPU-no-optimizer", "freeze576-synthetic-questions-no-API"],
        resources=p["protections"], no_other_app_control=True, epoch=time.time()))
    print(json.dumps(dict(status="CPU_GENERATION192_SOURCE_BOUND", tests=tests, budget=total / 10**12)))
def cgroup():
    name = next(x.split("::", 1)[1] for x in Path("/proc/self/cgroup").read_text().splitlines() if x.startswith("0::"))
    cg = Path("/sys/fs/cgroup") / name.lstrip("/")
    shared = next(x for x in cg.parents if x.name == "sg-formal-20261027.slice")
    def val(path, name): return (path / name).read_text().strip()
    quota, period = val(cg, "cpu.max").split()
    assert quota != "max" and int(quota) / int(period) <= 2
    assert val(cg, "memory.max") != "max" and int(val(cg, "memory.max")) <= 2 * 2**30
    assert int(val(shared, "memory.max")) <= 14 * 2**30
    assert val(cg, "memory.swap.max") == val(shared, "memory.swap.max") == "0"
    assert int(val(cg, "memory.swap.current")) == 0
    return dict(path=str(cg), current_bytes=int(val(cg, "memory.current")),
                peak_bytes=int(val(cg, "memory.peak")), shared_current_bytes=int(val(shared, "memory.current")),
                swap_bytes=int(val(cg, "memory.swap.current")))
LAST_CHECK = 0.
GUARD_SECONDS = 0.
def guard(start=False, force=False):
    global LAST_CHECK, GUARD_SECONDS
    if stopped(): raise InterruptedError("Own user STOP")
    now = time.monotonic()
    if not force and now - LAST_CHECK < 2: return
    begin = time.monotonic()
    path = "\\\\wsl.localhost\\Ubuntu-24.04\\home\\kiryu\\privacy-synthetic-distillation\\scripts\\three_method_synthetic_cpu_v4_20261101.py"
    q = subprocess.run(["/mnt/c/Users/10950/miniconda3/python.exe", "-X", "utf8", path, "windows-probe"],
                       capture_output=True, text=True, timeout=10)
    if q.returncode != 0: raise RuntimeError("Native read-only physical metrics probe failed: " + q.stderr[:300])
    w = json.loads(q.stdout)
    cg = cgroup()
    checks = dict(Windows=w["windows_free_gib"] >= (22 if start else 20), disk=w["c_free_gib"] >= 20,
                  shared=cg["shared_current_bytes"] < 14 * 2**30,
                  output=sum(x.stat().st_size for x in G.rglob("*") if x.is_file()) < 3 * 2**30)
    rec = dict(epoch=time.time(), Windows=w, cgroup=cg, checks=checks, CPU_only=True, GPU_calls=0)
    with (G / "resource-trace.jsonl").open("a") as f: f.write(json.dumps(rec) + "\n")
    atomic(G / "resource-latest.json", rec)
    GUARD_SECONDS += time.monotonic() - begin
    LAST_CHECK = time.monotonic()
    if not all(checks.values()): raise BlockingIOError("Actual CPU physical guard: " + json.dumps(checks))
def model_for(arm, torch):
    from privacy_synth.spatial_room_composition_model import RoomCompositionSetDenoiser
    info, finite = load(D / "reference-parameters.local.json"), load(D / "finite-composition.local.json")
    model = RoomCompositionSetDenoiser(info["parameters"], info["mixture_parameters"], info["prior"], finite)
    data = torch.load(D / arm / "checkpoints/4000.pt", map_location="cpu", weights_only=False)
    assert data["step"] == 4000 and data["protocol_sha"] == sha(D / "protocol.json")
    frozen = {k:v for k,v in model.state_dict().items() if k.startswith("reference.")}
    assert frozen and all(torch.equal(v, data["ema"][k]) for k,v in frozen.items())
    model.load_state_dict(data["ema"], strict=True)
    assert all(torch.equal(v, data["ema"][k]) for k,v in model.state_dict().items())
    return model.eval().requires_grad_(False)
def generate():
    begin = time.monotonic()
    p = load(G / "protocol.v4.json"); a = load(G / "root-decision.v4.json")
    assert a["approved"] and a["source_sha256"] == p["source"]["sha256"] == sha(__file__)
    assert a["protocol_sha256"] == sha(G / "protocol.v4.json")
    for b in p["bindings"]:
        assert bind(R / b["path"]) == b, b["path"]
    once(G / "queue-after-CPU-load-proof.started.json", dict(PID=os.getpid(), epoch=time.time(), source=p["source"]))
    guard(start=True, force=True)
    import torch
    from privacy_synth.spatial_set_diffusion import sample
    from privacy_synth.spatial_reverse_variance import make_schedule
    from privacy_synth.spatial_native_qa import FAMILIES, build_question, independent_answer
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    cn, norm = load(D / "condition-normalization.local.json"), load(D / "normalization.local.json")
    conditions = load(G / "public-conditions96.json")
    rows, local_gold, invalid = [], [], []
    for arm in p["generator_seeds"]:
        tick = time.monotonic(); model = model_for(arm, torch)
        model.register_forward_pre_hook(lambda *_: guard())
        schedule = make_schedule("fixed_large", device="cpu")
        once(G / arm / "load.json", dict(seconds=time.monotonic()-tick, exact_EMA=True, step=4000, new_updates=0))
        for i, c in enumerate(conditions):
            guard(force=True)
            if time.monotonic() - begin > p["worker_seconds_max"] - 30: raise TimeoutError("Bounded CPU generation")
            file = G / arm / "raw" / f"{i:04d}.json"
            seed = 2026110100 + i
            once(file.with_suffix(".started.json"), dict(arm=arm, ordinal=i, seed=seed, condition=c))
            values = [*map(math.log, c["room_size_native"]), math.log1p(c["count"])]
            cond = torch.tensor([[(v-u)/s for v,u,s in zip(values, cn["mean"], cn["std"])]], dtype=torch.float32)
            tick = time.monotonic(); old_guard = GUARD_SECONDS
            z, labels = sample(model, schedule, cond, c["count"], torch.Generator().manual_seed(seed))
            wall = time.monotonic() - tick
            obs, error = [], None
            try:
                for xyz, label in zip(z.tolist(), labels.tolist()):
                    native = [(math.sinh(v)*s+u)*size for v,s,u,size in zip(xyz, norm["IQR"], norm["center"], c["room_size_native"])]
                    if not all(math.isfinite(v) for v in xyz+native) or not 0 <= label < 35: raise ValueError("Invalid raw neural sample")
                    obs.append(dict(category_id=label, position_latent=xyz, position_native=native))
            except (ValueError, OverflowError) as ex: error = str(ex)
            raw = dict(ordinal=i, condition=c, objects=obs, raw_valid=error is None, invalid_reason=error,
                       sample_wall_seconds=wall, guard_seconds_inside_sample=GUARD_SECONDS-old_guard, new_generator_updates=0)
            once(file, raw)
            if error:
                invalid.append(dict(arm=arm, ordinal=i, reason=error))
            else:
                for j, family in enumerate(FAMILIES):
                    q = build_question(raw, norm["category_vocab"], family, 2026111000+3*i+j)
                    assert independent_answer(q["prompt"], family, q["anchor"]) == q["answer"]
                    identifier = f"{arm}-public-{i:04d}-{family}"
                    body = teacher_body(q)
                    utf8_bound = sum(len(x["content"].encode())+32 for x in body["messages"])+128
                    assert utf8_bound <= 32768, "No scene truncation or replacement"
                    rows.append(dict(id=identifier, generator=arm, ordinal=i, family=family,
                        body=body, body_sha256=hashlib.sha256(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
                        input_utf8_upper_bound=utf8_bound, source_scene=bind(file)))
                    local_gold.append(dict(id=identifier, answer=q["answer"], answer_category=q["answer_category"]))
            atomic(G / "progress.json", dict(completed_scenes=96*(p["generator_seeds"].index(arm))+i+1,
                scene_cap=192, questions_prepared=len(rows), invalid_scenes=len(invalid), actual_source="closed raw scene files",
                generator_updates=0, API_calls=0, epoch=time.time()))
            if i % 12 == 0: print("CPU_SYNTHETIC_COMMIT", arm, i+1, flush=True)
        del model
    assert not torch.cuda.is_initialized()
    with (G / "requests576.jsonl").open("x") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False)+"\n")
    once(G / "synthetic-oracle-gold.local.json", local_gold)
    once(G / "terminal.json", dict(status="ALL192_CPU_SCENES_FROZEN_NO_API", raw_scenes=192, invalid=invalid,
         questions=len(rows), expected_questions=576, invalid_never_refilled=True,
         wall_seconds=time.monotonic()-begin, guard_seconds=GUARD_SECONDS, RSS_peak_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         cgroup=cgroup(), CUDA_initialized=False, generator_updates=0, API_calls=0,
         requests=bind(G / "requests576.jsonl"), protected_real_gold_opened=False))
    print("CPU192_TERMINAL", len(rows), flush=True)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=["prepare", "generate", "windows-probe"]); args = parser.parse_args()
    if args.action == "windows-probe":
        from sg_formal_windows_resources_20261027 import sample
        print(json.dumps(sample()))
    else:
        signal.signal(signal.SIGTERM, on_stop); signal.signal(signal.SIGINT, on_stop)
        try:
            if args.action == "prepare": prepare()
            else: generate()
        except BaseException as exc:
            G.mkdir(exist_ok=True)
            once(G / (args.action + ".v4.failure.json"), dict(error_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc(), epoch=time.time()))
            raise
