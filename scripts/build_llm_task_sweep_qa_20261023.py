#!/usr/bin/env python3
"""Offline QA preparation helpers. Public-source retrieval only; no model client."""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/2026-10-23-llm-task-sweep/preparation/text-qa"
SEED = "20261023"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fetch(url, name, revision=None):
    path = OUT / "sources" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "public-benchmark-audit/1.0"})
        with urllib.request.urlopen(req, timeout=25) as response:
            payload = response.read(20_000_001)
        if len(payload) > 20_000_000:
            raise ValueError("bounded source fetch exceeds20MB")
        path.write_bytes(payload)
    payload = path.read_bytes()
    record = {"url": url, "path": str(path.relative_to(ROOT)), "revision": revision,
              "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    dump(path.with_name(path.name + ".provenance.json"), record)
    return record


def fetch_initial():
    routes = [
        ("https://api.github.com/repos/docworlds/tsg-bench/git/trees/master?recursive=1", "tsg-tree.json"),
        ("https://api.github.com/repos/usail-hkust/USTBench/git/trees/main?recursive=1", "ust-tree.json"),
        ("https://huggingface.co/api/datasets/Haruto2099/USTBench-Dataset/tree/main?recursive=false&limit=100", "ust-hf-root.json"),
        ("https://raw.githubusercontent.com/MapEval/MapEval-Textual/4adde14abd8dbeb820eaed694ec528d5ef3d664f/Evaluator.py", "mapeval-Evaluator.py"),
        ("https://raw.githubusercontent.com/usail-hkust/USTBench/main/UST_tasks/question_answering/main.py", "ust-main.py"),
    ]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(fetch, url, name): (url, name) for url, name in routes}
        for future in concurrent.futures.as_completed(futures):
            url, name = futures[future]
            try:
                results.append(future.result())
                print(name, "OK", flush=True)
            except Exception as exc:
                results.append({"url": url, "name": name, "error": repr(exc)})
                print(name, repr(exc), flush=True)
    dump(OUT / "source-initial-receipt.json", results)


def fetch_details():
    tasks = []
    for task, repo, wanted in [
        ("tsg", "docworlds/tsg-bench", ["resource/dataset/understanding/sgqa.jsonl", "resource/prompts/sgqa.txt", "evaluation/understanding/sgqa.py", "README.md"]),
        ("ust", "usail-hkust/USTBench", ["UST_tasks/question_answering/main.py", "README.md"]),
    ]:
        tree = json.loads((OUT / "sources" / f"{task}-tree.json").read_text())
        for member in tree["tree"]:
            if member["path"] in wanted:
                tasks.append((task, repo, member, tree["sha"]))
    def one(task, repo, member, rev):
        rec = fetch(member["url"], f"{task}-blob-{member['sha']}.json", rev)
        blob = json.loads((ROOT / rec["path"]).read_text())
        payload = base64.b64decode(blob["content"])
        name = f"{task}-" + member["path"].replace("/", "__")
        path = OUT / "sources" / name
        path.write_bytes(payload)
        record = {"url": f"https://github.com/{repo}/blob/{rev}/{member['path']}", "revision": rev,
                  "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "path": str(path.relative_to(ROOT))}
        dump(path.with_name(path.name + ".provenance.json"), record)
        return record
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(one, *task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
                print(results[-1]["path"], "OK", flush=True)
            except Exception as exc:
                results.append({"error": repr(exc)})
                print(repr(exc), flush=True)
    dump(OUT / "source-details-receipt.json", results)


def stable_key(value):
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records), encoding="utf-8")


MAP_SYSTEM = """Solve this map-grounded question using only the supplied context. The task is to select a numbered answer, not to classify the question. Option0 means the facts are insufficient; content options start at1. For trips and routes, check each directed leg in the requested travel mode, elapsed travel and visit times, opening hours and all requested constraints. Do not assume reverse-direction travel is equal. For nearby/place questions, apply every requested attribute and compare the stated distances/times; names alone are insufficient. Do not invent missing hours, routes, reviews or facts. Return only JSON {\"option_no\": integer}."""
TSG_SYSTEM = """Answer the question from the complete ordered scene-graph sequence only. Each graph describes one successive action; a triple is [subject, relation, object]. Preserve the chronological graph order and track the same named entity across actions. Resolve before/after, immediately, first/last and action-object/instrument/location distinctions by the supplied triples, not everyday assumptions. Copy the exact entity spelling from the graph; an entity can contain spaces or hyphens. Return only [entity], with no explanation."""
UST_SYSTEM = """Solve the supplied urban spatiotemporal QA from its full provided data, schema and question. Follow the stated objective and units. Compare the actual numeric values, preserve ties and missing/NaN semantics, and distinguish a location label from its rank. For station planning, evaluate the specific requested benefit rather than choosing the largest demand for every objective; do not invent an optimization formula or an unobserved simulation result. For reflection, reassess the prior decision against the requested objective using the supplied environment changes and facts. Preserve the original answer options. For an understanding question return only <Answer>A</Answer> (replace A with the chosen option). For planning/reflection return only JSON {\"summary\":\"brief reasoning\",\"answer\":\"A\"}."""


def render_mapeval(row):
    sys.path.insert(0, str(ROOT / "src"))
    from privacy_synth.spatial_scene import content_options
    options = ["0. Unanswerable"] + [f"{i}. {v}" for i, v in enumerate(content_options(row["options"]), 1)]
    return "Context:\n" + row["context"] + "\nQuestion:\n" + row["question"] + "\nOptions:\n" + "\n".join(options)


def parse_mapeval(text, options):
    from privacy_synth.spatial_scene import legal_labels
    try:
        obj = json.loads(text)
        value = obj["option_no"]
        if set(obj) != {"option_no"} or isinstance(value, bool) or not isinstance(value, int):
            return None
        return str(value) if str(value) in legal_labels(options) else None
    except (ValueError, KeyError, TypeError):
        return None


def score_sgqa(text, gold):
    # Exact author behavior: first bracket capture, or entire response, then lower/strip.
    captures = re.findall(r"\[(.*?)\]", text)
    answer = captures[0] if captures else text
    return answer.lower().strip() == gold.lower().strip()


def parse_ust(text, ability):
    if ability == "st_understanding":
        captures = re.findall(r"<Answer>(.*?)</Answer>", text)
        return captures[-1] if captures else None
    try:
        captures = re.findall(r"```(?:[A-Za-z]*)\s*(.*?)```", text, re.DOTALL)
        obj = json.loads(captures[-1] if captures else text)
        if "summary" not in obj:
            return None
        return obj["answer"]
    except (ValueError, TypeError, KeyError):
        return None


def make_ust():
    rows, parents, seen = [], [], {}
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    audit_indices = []
    for path in sorted((OUT / "sources").glob("ust-data-poi*.json")):
        if path.name.endswith("provenance.json"):
            continue
        ability = path.name.split("__", 1)[1].removesuffix("_QA.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        for index, row in enumerate(data):
            i = len(rows)
            parents.append(i)
            if index == 0:
                audit_indices.append(i)
            text = row["question"].split("## Data\n\n", 1)[1].split("\n\n## ", 1)[0]
            vectors = [block.strip() for block in re.split(r"location \d+:\n", text)[1:]]
            assert len(vectors) == 4
            rows.append({"id": f"poi_placement:{ability}:{index}", "ability": ability,
                         "question": row["question"], "gold": row["answer"], "entity_fingerprints": [hashlib.sha256(v.encode()).hexdigest() for v in vectors]})
            for vector in vectors:
                if vector in seen:
                    parents[root(i)] = root(seen[vector])
                else:
                    seen[vector] = i
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(root(i), []).append(row)
    audit = {root(i) for i in audit_indices}
    ordered = sorted([g for g in groups if g not in audit], key=lambda g: stable_key(min(r["id"] for r in groups[g])))
    assigned = {"Audit": sorted(audit), "FitDemos": ordered[:600], "Dev": ordered[600:900], "Screen": ordered[900:1500], "FinalSealed": ordered[1500:]}
    target = OUT / "USTBench-POI-processQA"
    roles = {}
    role_entities = {}
    abilities = ["st_understanding", "planning", "reflection"]
    for role, gs in assigned.items():
        records = [dict(r, group="g:" + min(v["id"] for v in groups[g])) for g in gs for r in groups[g]]
        records.sort(key=lambda r: stable_key(r["id"]))
        if role in ("Dev", "Screen"):
            quotas = [6, 5, 5] if role == "Dev" else [20, 20, 20]
            selected = [r for a, count in zip(abilities, quotas) for r in [x for x in records if x["ability"] == a][:count]]
            assert len(selected) == sum(quotas)
        else:
            selected = records
        roles[role] = {"members": [r["id"] for r in records], "selected": [r["id"] for r in selected],
                       "groups": sorted({r["group"] for r in records}), "ability_counts_selected": dict(Counter(r["ability"] for r in selected))}
        role_entities[role] = {e for r in records for e in r["entity_fingerprints"]}
        if role != "FinalSealed":
            jsonl(target / (role + ".local.jsonl"), records)
        if role in ("Dev", "Screen"):
            jsonl(target / (role + ".requests.jsonl"), [{"id": r["id"], "group": r["group"], "ability": r["ability"],
                "system": UST_SYSTEM, "user": r["question"]} for r in selected])
    names = list(role_entities)
    assert all(not (role_entities[a] & role_entities[b]) for i, a in enumerate(names) for b in names[i+1:])
    dump(target / "roles.json", {"seed": SEED, "roles": roles, "grouping": "connected components sharing any exact four-feature-location numeric record, across3abilities", "physical_identity_limitation": "source omits stable location IDs/geographic provenance; fingerprint disjointness does not prove disjoint physical properties", "final_contents_rendered": False})
    dump(target / "prompts.json", {"adapted": UST_SYSTEM, "allowed_query": "original question field only; no answer/target/analysis_text/feedbacks metadata", "native_reflection": "prior decision/reason/environment changes already within original question are required allowed public task context"})
    # Forecasting has named target/nearby/example regions. Enforce their connected components.
    forecast = json.loads((OUT / "sources/ust-data-socio_ecomic_prediction__forecasting_QA.json").read_text(encoding="utf-8"))
    fp = list(range(len(forecast)))
    def fr(i):
        while fp[i] != i:
            fp[i] = fp[fp[i]]
            i = fp[i]
        return i
    entities = {}
    for i, record in enumerate(forecast):
        for entity in re.findall(r"(?:Target|Nearby|Example) Region (\d+):", record["question"]):
            if entity in entities:
                fp[fr(i)] = fr(entities[entity])
            else:
                entities[entity] = i
    fc = Counter(fr(i) for i in range(len(forecast)))
    dump(OUT / "USTBench-forecasting-readiness.json", {"records": len(forecast), "named_region_components": len(fc), "component_sizes": sorted(fc.values(), reverse=True), "status": "BLOCKED_GROUP_INDEPENDENCE" if len(fc) < 4 else "REQUIRES_ROLE_PREPARATION", "calls_prepared": 0, "all27_source_QA_files_downloaded": False})
    return {role: {"pool": len(v["members"]), "selected": len(v["selected"]), "groups": len(v["groups"])} for role, v in roles.items()}


def make_sgqa():
    source = OUT / "sources/tsg-resource__dataset__understanding__sgqa.jsonl"
    worlds = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(worlds) == 100 and sum(len(w["qa_pairs"]) for w in worlds) == 500
    # Whole data_id is the available scenario key. Exact duplicate worlds are unioned.
    parent = list(range(len(worlds)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, world in enumerate(worlds):
        for key in ("id:" + world["data_id"], "graphs:" + json.dumps(world["context_graphs"], sort_keys=True)):
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = {}
    for i, world in enumerate(worlds):
        groups.setdefault(find(i), []).append(world)
    audit_group = find(0)  # Only inspected source-record example: quarantine whole world.
    ordered = sorted([g for g in groups if g != audit_group], key=lambda g: stable_key(groups[g][0]["data_id"]))
    assert all(len(groups[g]) == 1 for g in ordered), "revisit capacities for duplicate world groups"
    assigned = {"Audit": [audit_group], "FitDemos": ordered[:20], "Dev": ordered[20:24], "Screen": ordered[24:36], "FinalSealed": ordered[36:]}
    target = OUT / "TSG-SGQA"
    roles = {}
    for role, gs in assigned.items():
        records = []
        for g in gs:
            for world in groups[g]:
                for qi, qa in enumerate(world["qa_pairs"]):
                    identifier = f"{world['data_id']}:{qi}"
                    records.append({"id": identifier, "group": world["data_id"], "source_doc_index": world["doc_index"],
                                    "source_text_part_index": world["text_part_index"], "question": qa["Q"], "gold": qa["A"],
                                    "context_graphs": world["context_graphs"]})
        records.sort(key=lambda r: stable_key(r["id"]))
        selected = records[:16] if role == "Dev" else records
        roles[role] = {"members": [r["id"] for r in records], "selected": [r["id"] for r in selected], "groups": [groups[g][0]["data_id"] for g in gs]}
        if role != "FinalSealed":
            jsonl(target / (role + ".local.jsonl"), records)
        if role in ("Dev", "Screen"):
            jsonl(target / (role + ".requests.jsonl"), [{"id": r["id"], "group": r["group"], "system": TSG_SYSTEM,
                "user": "Ordered scene graphs:\n" + json.dumps(r["context_graphs"], ensure_ascii=False) + "\nQuestion:\n" + r["question"]} for r in selected])
    dump(target / "roles.json", {"seed": SEED, "roles": roles, "grouping": "whole source data_id plus exact graph-sequence duplicates; data_id physical-source scenario equivalence unverified", "final_contents_rendered": False})
    dump(target / "prompts.json", {"author": (OUT / "sources/tsg-resource__prompts__sgqa.txt").read_text(), "adapted": TSG_SYSTEM})
    return {role: {"pool": len(v["members"]), "selected": len(v["selected"]), "groups": len(v["groups"])} for role, v in roles.items()}


def make_mapeval():
    sys.path.insert(0, str(ROOT / "src"))
    from privacy_synth.spatial_scene import gold_label
    base = ROOT / "data/derived/mapeval-textual/v1"
    public = [json.loads(x) for x in (base / "public_calibration.jsonl").read_text().splitlines()]
    local = [json.loads(x) for x in (base / "local_dev.jsonl").read_text().splitlines()]
    digest = json.loads((ROOT / "configs/mapeval_textual_v1.digest.json").read_text())
    audit_group = next(r["group_id"] for r in public if r["id"] == 479)
    groups = sorted({r["group_id"] for r in local}, key=stable_key)
    sizes = {g: sum(r["group_id"] == g for r in local) for g in groups}
    # Deterministic subset-sum gives16 without splitting inherited dependence groups.
    choices = {0: []}
    for g in groups:
        for size, chosen in list(choices.items()):
            if size + sizes[g] <= 16 and size + sizes[g] not in choices:
                choices[size + sizes[g]] = chosen + [g]
    assert 16 in choices
    dev_groups = set(choices[16])
    parts = {"Audit": [r for r in public if r["group_id"] == audit_group],
             "FitDemos": [r for r in public if r["group_id"] != audit_group],
             "Dev": [r for r in local if r["group_id"] in dev_groups],
             "ScreenHistorical": [r for r in local if r["group_id"] not in dev_groups]}
    target = OUT / "MapEval-Textual"
    roles = {}
    for role, rows in parts.items():
        rows.sort(key=lambda r: stable_key(r["id"]))
        assert all(gold_label(r["github_correct"], r["options"]) == r["gold_label"] for r in rows)
        jsonl(target / (role + ".local.jsonl"), rows)
        roles[role] = {"members": [r["id"] for r in rows], "groups": sorted({r["group_id"] for r in rows})}
        if role in ("Dev", "ScreenHistorical"):
            jsonl(target / (role + ".requests.jsonl"), [{"id": str(r["id"]), "group": r["group_id"], "system": MAP_SYSTEM, "user": render_mapeval(r)} for r in rows])
    roles["FinalSealedPreserved"] = {"members": digest["role_ids"]["sealed_evaluation"], "contents_opened": False}
    dump(target / "roles.json", {"seed": SEED, "roles": roles, "historical_exposure": "all128 public_calibration/local_dev already used by prior project; not fresh screening", "source_roles_unchanged": True})
    dump(target / "prompts.json", {"adapted": MAP_SYSTEM, "mapping": "GitHub-1→Option0; GitHub0→Option1; trailing blanks omitted; all128 gold mappings rechecked"})
    return {k: len(v["members"]) for k, v in roles.items()}


def verify_scorers():
    sys.path.insert(0, str(ROOT / "src"))
    from privacy_synth.spatial_scene import gold_label, legal_labels
    options = ["first", "second", ""]
    assert gold_label(-1, options) == "0" and gold_label(0, options) == "1" and gold_label(1, options) == "2"
    assert legal_labels(options) == ("0", "1", "2")
    assert parse_mapeval('{"option_no":0}', options) == "0"
    assert parse_mapeval('{"option_no":1}', options) == "1"
    assert parse_mapeval('{"option_no":true}', options) is None
    assert parse_mapeval('{"option_no":3}', options) is None
    assert parse_mapeval('{"option_no":1,"gold":1}', options) is None
    assert score_sgqa("[mop-stick]", "mop-stick")
    assert score_sgqa(" [HAND1] ", "hand1")
    assert score_sgqa("[wooden table]", "wooden table")
    assert not score_sgqa("[mop stick]", "mop-stick")
    assert not score_sgqa("[hand1] [hand2]", "hand2")
    assert parse_ust("<Answer>D</Answer>", "st_understanding") == "D"
    assert parse_ust("<Answer>A</Answer><Answer>D</Answer>", "st_understanding") == "D"
    assert parse_ust('{"summary":"x","answer":"A"}', "planning") == "A"
    assert parse_ust('{"answer":"A"}', "reflection") is None
    assert parse_ust("<Answer> D </Answer>", "st_understanding") != "D"
    return {"status": "PASS", "checks": 17, "no_model_calls": True}


def prepare():
    result = {"MapEval-Textual": make_mapeval(), "TSG-SGQA": make_sgqa(), "USTBench-POI-processQA": make_ust(), "scorer_checks": verify_scorers(), "model_calls": 0}
    result["fixed8_demos"] = prepare_demos()
    result["export_checks"] = verify_exports()
    dump(OUT / "preparation-summary.json", result)
    freeze_manifest()
    print(json.dumps(result, indent=2))


def prepare_demos():
    result = {}
    for task in ("MapEval-Textual", "TSG-SGQA", "USTBench-POI-processQA"):
        path = OUT / task / "FitDemos.local.jsonl"
        pool = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        # Fit-only diversity is an explicit part of the fixed demonstration strategy.
        category = (lambda r: r["classification"]) if task == "MapEval-Textual" else (lambda r: r.get("ability", "sgqa"))
        chosen, groups = [], set()
        keys = sorted({category(r) for r in pool})
        for key in keys:
            row = next(r for r in pool if category(r) == key and r.get("group", r.get("group_id")) not in groups)
            chosen.append(row)
            groups.add(row.get("group", row.get("group_id")))
        for row in pool:
            group = row.get("group", row.get("group_id"))
            if group not in groups:
                chosen.append(row)
                groups.add(group)
            if len(chosen) == 8:
                break
        assert len(chosen) == 8 and len(groups) == 8
        messages = []
        for row in chosen:
            if task == "MapEval-Textual":
                query, answer = render_mapeval(row), json.dumps({"option_no": int(row["gold_label"])})
            elif task == "TSG-SGQA":
                query = "Ordered scene graphs:\n" + json.dumps(row["context_graphs"], ensure_ascii=False) + "\nQuestion:\n" + row["question"]
                answer = "[" + row["gold"] + "]"
            else:
                query = row["question"]
                answer = "<Answer>" + row["gold"] + "</Answer>" if row["ability"] == "st_understanding" else json.dumps({"summary": "Public reference demonstration.", "answer": row["gold"]})
            messages.append({"id": str(row["id"]), "group": row.get("group", row.get("group_id")), "user": query, "assistant": answer})
        jsonl(OUT / task / "FitDemos.fixed8.jsonl", messages)
        result[task] = {"count": 8, "strategy": "Fit-only seed hash order, category/ability coverage first, distinct source groups", "characters": sum(len(m["user"]) + len(m["assistant"]) for m in messages)}
    return result


def freeze_manifest():
    records = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "artifact-manifest.json":
            payload = path.read_bytes()
            records.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    records.append({"path": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"), "bytes": Path(__file__).stat().st_size, "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    dump(OUT / "artifact-manifest.json", {"files": records, "calls": 0, "phase": "PREPARED_NOT_APPROVED", "FinalSealed_role_contents_exported": False})


def verify_exports():
    counts = {}
    for task in ("MapEval-Textual", "TSG-SGQA", "USTBench-POI-processQA"):
        folder = OUT / task
        roles = json.loads((folder / "roles.json").read_text(encoding="utf-8"))["roles"]
        member_sets = [{str(i) for i in role["members"]} for role in roles.values()]
        assert all(not (a & b) for i, a in enumerate(member_sets) for b in member_sets[i+1:])
        fit_ids = {str(i) for i in roles["FitDemos"]["members"]}
        demos = [json.loads(line) for line in (folder / "FitDemos.fixed8.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(demos) == 8 and all(d["id"] in fit_ids for d in demos)
        assert len({d["group"] for d in demos}) == 8
        request_count = 0
        for request_file in folder.glob("*.requests.jsonl"):
            role = request_file.name.split(".")[0]
            local = {str(r["id"]): r for r in (json.loads(line) for line in (folder / (role + ".local.jsonl")).read_text(encoding="utf-8").splitlines())}
            for line in request_file.read_text(encoding="utf-8").splitlines():
                request = json.loads(line)
                assert not ({"gold", "answer", "gold_label", "analysis_text", "feedbacks"} & set(request))
                source = local[request["id"]]
                if task == "USTBench-POI-processQA":
                    assert request["user"] == source["question"]
                elif task == "MapEval-Textual":
                    assert request["user"] == render_mapeval(source)
                else:
                    assert json.dumps(source["context_graphs"], ensure_ascii=False) in request["user"]
                    assert source["question"] in request["user"]
                request_count += 1
        counts[task] = request_count
        assert not list(folder.glob("Final*.local.jsonl")) and not list(folder.glob("Final*.requests.jsonl"))
    return {"status": "PASS", "query_counts": counts, "query_gold_metadata_absent": True, "full_context_preserved": True, "fixed8_fit_only": True, "role_ids_disjoint": True}


def audit_ust():
    rows, parents, seen = [], [], {}
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for path in sorted((OUT / "sources").glob("ust-data-poi*.json")):
        if path.name.endswith("provenance.json"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        print(path.name, len(data), Counter(x["answer"] for x in data))
        for row in data:
            i = len(rows)
            rows.append(row)
            parents.append(i)
            text = row["question"].split("## Data\n\n", 1)[1].split("\n\n## ", 1)[0]
            blocks = re.split(r"location \d+:\n", text)[1:]
            for block in blocks:
                key = block.strip()
                if key in seen:
                    parents[root(i)] = root(seen[key])
                else:
                    seen[key] = i
    counts = Counter(root(i) for i in range(len(rows)))
    result = {"poi_records": len(rows), "exact_attribute_entity_groups": len(counts), "largest_components": counts.most_common(15), "distinct_entity_vectors": len(seen)}
    path = OUT / "sources/ust-data-socio_ecomic_prediction__forecasting_QA.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    regions = [set(re.findall(r"(?:Target|Nearby|Example) Region (\d+):", r["question"])) for r in data]
    result["socio_samples"] = len(data)
    result["socio_region_count"] = len(set.union(*regions))
    result["socio_target_count"] = len({re.search(r"Target Region (\d+)", r["question"]).group(1) for r in data})
    print(json.dumps(result, indent=2))
    dump(OUT / "ust-group-audit.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["fetch-initial", "fetch-details", "prepare", "audit-ust"])
    args = parser.parse_args()
    if args.action == "fetch-initial":
        fetch_initial()
    elif args.action == "fetch-details":
        fetch_details()
    elif args.action == "prepare":
        prepare()
    elif args.action == "audit-ust":
        audit_ust()


if __name__ == "__main__":
    main()
