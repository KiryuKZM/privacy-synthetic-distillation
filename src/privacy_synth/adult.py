"""Official UCI Adult parse, fingerprinting, and role assignment.

The identity of a record is the SHA-256 of the 15 normalized fields, including
income. Role assignment is a partition of unique records. Exact copies collapse
to one canonical row. Cross-file exact copies and feature-level label conflicts
are excluded from every experimental role.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

COLUMNS: tuple[str, ...] = (
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
)
FEATURE_COLUMNS: tuple[str, ...] = COLUMNS[:-1]
NUMERIC_COLUMNS: tuple[str, ...] = (
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
)
DISCRETE_COLUMNS: tuple[str, ...] = tuple(c for c in COLUMNS if c not in NUMERIC_COLUMNS)
QUASI_ID_COLUMNS: tuple[str, ...] = tuple(c for c in COLUMNS if c != "fnlwgt")
QUASI_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    c for c in COLUMNS if c not in ("fnlwgt", "income")
)

ROLES: tuple[str, ...] = (
    "generator_train",
    "local_dev",
    "public_calibration",
    "audit_nonmember",
    "sealed_test",
)
EXCLUDED_ROLES: tuple[str, ...] = (
    "excluded_cross_file",
    "excluded_label_conflict",
)
ALL_ROLES: tuple[str, ...] = ROLES + EXCLUDED_ROLES
PROTECTED_ROLES: tuple[str, ...] = (
    "generator_train",
    "local_dev",
    "audit_nonmember",
    "sealed_test",
)
STRATA_COLUMNS: tuple[str, ...] = ("income", "sex")

SUBGROUP_CANDIDATES: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("female_gt50k", (("sex", "Female"), ("income", ">50K"))),
    ("amer_indian_eskimo", (("race", "Amer-Indian-Eskimo"),)),
    ("race_other", (("race", "Other"),)),
    ("education_preschool", (("education", "Preschool"),)),
    ("priv_house_serv", (("occupation", "Priv-house-serv"),)),
    ("native_not_us_known", (("native-country", "__not_us_known__"),)),
)

NATIVE_US = "United-States"
MISSING = "?"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(values: Sequence[str]) -> str:
    return sha256_bytes("\t".join(values).encode("utf-8"))


def normalize_income(value: str) -> str:
    text = value.strip()
    if text.endswith("."):
        text = text[:-1].strip()
    return text


def parse_adult_line(line: str) -> dict[str, str] | None:
    text = line.strip()
    if not text or text.startswith("|"):
        return None
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != len(COLUMNS):
        raise ValueError(f"expected {len(COLUMNS)} fields, got {len(parts)}: {text[:80]!r}")
    record = dict(zip(COLUMNS, parts, strict=True))
    record["income"] = normalize_income(record["income"])
    if record["income"] not in {"<=50K", ">50K"}:
        raise ValueError(f"unexpected income {record['income']!r}")
    for column in NUMERIC_COLUMNS:
        int(record[column])
    return record


@dataclass(frozen=True)
class AdultRow:
    values: tuple[str, ...]
    source_file: str
    source_line: int

    @property
    def mapping(self) -> dict[str, str]:
        return dict(zip(COLUMNS, self.values, strict=True))

    @property
    def record_fp(self) -> str:
        return fingerprint(self.values)

    @property
    def feature_fp(self) -> str:
        return fingerprint(self.values[:-1])

    @property
    def quasi_id_fp(self) -> str:
        mapping = self.mapping
        return fingerprint(tuple(mapping[column] for column in QUASI_ID_COLUMNS))

    @property
    def quasi_feature_fp(self) -> str:
        mapping = self.mapping
        return fingerprint(tuple(mapping[column] for column in QUASI_FEATURE_COLUMNS))

    @property
    def stratum(self) -> str:
        mapping = self.mapping
        return f"{mapping['income']}|{mapping['sex']}"


@dataclass
class CanonicalRecord:
    row: AdultRow
    n_copies: int
    sources: tuple[str, ...]
    source_lines: tuple[tuple[str, int], ...]
    role: str = ""
    thinking_probe: bool = False

    @property
    def record_fp(self) -> str:
        return self.row.record_fp

    @property
    def mapping(self) -> dict[str, str]:
        return self.row.mapping


@dataclass
class SplitConfig:
    version: str = "v1"
    seed: int = 20261001
    public_calibration: int = 256
    local_development: int = 4000
    audit_nonmember: int = 6000
    thinking_probe: int = 64
    thinking_probe_seed: int = 20261011
    min_gen_train_subgroup: int = 50
    min_local_dev_subgroup: int = 15
    expected_sha256: str = (
        "7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb"
    )


@dataclass
class SplitResult:
    config: SplitConfig
    archive_sha256: str
    inner_sha256: dict[str, str]
    canonical: list[CanonicalRecord]
    summary: dict[str, object] = field(default_factory=dict)


def parse_adult_zip(archive: Path) -> tuple[list[AdultRow], dict[str, str]]:
    inner_sha256: dict[str, str] = {}
    rows: list[AdultRow] = []
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            inner_sha256[name] = sha256_bytes(zf.read(name))
        for source_file in ("adult.data", "adult.test"):
            text = zf.read(source_file).decode("utf-8")
            for source_line, line in enumerate(text.splitlines(), start=1):
                parsed = parse_adult_line(line)
                if parsed is None:
                    continue
                values = tuple(parsed[column] for column in COLUMNS)
                rows.append(
                    AdultRow(
                        values=values,
                        source_file=source_file,
                        source_line=source_line,
                    )
                )
    return rows, inner_sha256


def hamilton_allocate(counts: Mapping[str, int], n: int) -> dict[str, int]:
    """Largest-remainder allocation. Tie-break: larger stratum, then key."""
    keys = list(counts)
    total = sum(counts[key] for key in keys)
    assigned = {key: 0 for key in keys}
    if n <= 0 or total <= 0:
        return assigned
    if n >= total:
        return {key: counts[key] for key in keys}
    remainders: list[tuple[float, int, str]] = []
    for key in keys:
        exact = n * counts[key] / total
        floor = math.floor(exact)
        assigned[key] = floor
        remainders.append((exact - floor, counts[key], key))
    leftover = n - sum(assigned.values())
    remainders.sort(key=lambda item: (-item[0], -item[1], item[2]))
    for i in range(leftover):
        assigned[remainders[i][2]] += 1
    return assigned


def _fit_role_targets(n: int, cal: int, dev: int, nonmember: int) -> tuple[int, int, int, int]:
    """Fit cal/dev/nonmember into a stratum, keeping rare rows in generator_train."""
    cal = max(0, cal)
    dev = max(0, dev)
    nonmember = max(0, nonmember)
    if n <= 0:
        return 0, 0, 0, 0
    if n <= 2:
        return 0, 0, 0, n
    gen_floor = (n + 1) // 2
    budget = n - gen_floor
    desired = {"cal": cal, "dev": dev, "nonmember": nonmember}
    desired_total = sum(desired.values())
    if desired_total <= budget:
        return cal, dev, nonmember, n - desired_total
    scaled = hamilton_allocate(desired, budget)
    taken = scaled["cal"] + scaled["dev"] + scaled["nonmember"]
    return scaled["cal"], scaled["dev"], scaled["nonmember"], n - taken


def _stratum_seed(seed: int, stratum: str) -> int:
    material = f"{seed}:{stratum}".encode("utf-8")
    return int(hashlib.sha256(material).hexdigest()[:16], 16)


def _shuffle_canonical(records: list[CanonicalRecord], seed: int, stratum: str) -> list[CanonicalRecord]:
    ordered = sorted(records, key=lambda item: item.record_fp)
    rng = random.Random(_stratum_seed(seed, stratum))
    rng.shuffle(ordered)
    return ordered


def collapse_and_tag(rows: Sequence[AdultRow]) -> list[CanonicalRecord]:
    by_record: dict[str, list[AdultRow]] = defaultdict(list)
    for row in rows:
        by_record[row.record_fp].append(row)

    incomes_by_feature: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        incomes_by_feature[row.feature_fp].add(row.values[-1])
    conflict_features = {
        feature_fp
        for feature_fp, incomes in incomes_by_feature.items()
        if len(incomes) > 1
    }

    canonical: list[CanonicalRecord] = []
    for record_fp, group in by_record.items():
        group_sorted = sorted(group, key=lambda row: (row.source_file, row.source_line))
        sources = tuple(sorted({row.source_file for row in group_sorted}))
        source_lines = tuple((row.source_file, row.source_line) for row in group_sorted)
        item = CanonicalRecord(
            row=group_sorted[0],
            n_copies=len(group_sorted),
            sources=sources,
            source_lines=source_lines,
        )
        feature_conflict = item.row.feature_fp in conflict_features
        cross_file = set(sources) == {"adult.data", "adult.test"}
        if feature_conflict:
            item.role = "excluded_label_conflict"
        elif cross_file:
            item.role = "excluded_cross_file"
        canonical.append(item)
    return canonical


def assign_roles(canonical: Sequence[CanonicalRecord], config: SplitConfig) -> list[CanonicalRecord]:
    assigned = list(canonical)
    train_pool = [
        item
        for item in assigned
        if not item.role and item.sources == ("adult.data",)
    ]
    test_pool = [
        item
        for item in assigned
        if not item.role and item.sources == ("adult.test",)
    ]
    for item in test_pool:
        item.role = "sealed_test"

    by_stratum: dict[str, list[CanonicalRecord]] = defaultdict(list)
    for item in train_pool:
        by_stratum[item.row.stratum].append(item)
    stratum_counts = {key: len(value) for key, value in by_stratum.items()}

    cal_targets = hamilton_allocate(stratum_counts, config.public_calibration)
    dev_targets = hamilton_allocate(stratum_counts, config.local_development)
    nonmember_targets = hamilton_allocate(stratum_counts, config.audit_nonmember)

    for stratum, items in by_stratum.items():
        shuffled = _shuffle_canonical(items, config.seed, stratum)
        n_cal, n_dev, n_nonmember, _n_gen = _fit_role_targets(
            len(shuffled),
            cal_targets.get(stratum, 0),
            dev_targets.get(stratum, 0),
            nonmember_targets.get(stratum, 0),
        )
        index = 0
        for item in shuffled[index : index + n_cal]:
            item.role = "public_calibration"
        index += n_cal
        for item in shuffled[index : index + n_dev]:
            item.role = "local_dev"
        index += n_dev
        for item in shuffled[index : index + n_nonmember]:
            item.role = "audit_nonmember"
        index += n_nonmember
        for item in shuffled[index:]:
            item.role = "generator_train"

    _assign_thinking_probe(assigned, config)
    unset = [item.record_fp for item in assigned if not item.role]
    if unset:
        raise RuntimeError(f"{len(unset)} records were not assigned a role")
    return assigned


def _assign_thinking_probe(canonical: Sequence[CanonicalRecord], config: SplitConfig) -> None:
    cal = [item for item in canonical if item.role == "public_calibration"]
    by_stratum: dict[str, list[CanonicalRecord]] = defaultdict(list)
    for item in cal:
        by_stratum[item.row.stratum].append(item)
    counts = {key: len(value) for key, value in by_stratum.items()}
    n_probe = min(config.thinking_probe, len(cal))
    targets = hamilton_allocate(counts, n_probe)
    for stratum, items in by_stratum.items():
        shuffled = _shuffle_canonical(items, config.thinking_probe_seed, f"probe:{stratum}")
        for item in shuffled[: targets.get(stratum, 0)]:
            item.thinking_probe = True


def record_matches(mapping: Mapping[str, str], spec: Sequence[tuple[str, str]]) -> bool:
    for column, expected in spec:
        if expected == "__not_us_known__":
            if mapping["native-country"] in {NATIVE_US, MISSING}:
                return False
            continue
        if mapping[column] != expected:
            return False
    return True


def role_counts(canonical: Sequence[CanonicalRecord]) -> dict[str, int]:
    counts = Counter(item.role for item in canonical)
    return {role: counts.get(role, 0) for role in ALL_ROLES}


def stratum_crosstab(canonical: Sequence[CanonicalRecord], role: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in canonical:
        if item.role == role:
            counts[item.row.stratum] += 1
    return dict(sorted(counts.items()))


def subgroup_counts(
    canonical: Sequence[CanonicalRecord], roles: Sequence[str]
) -> dict[str, dict[str, int]]:
    wanted = set(roles)
    result: dict[str, dict[str, int]] = {}
    for name, spec in SUBGROUP_CANDIDATES:
        per_role = {role: 0 for role in roles}
        for item in canonical:
            if item.role in wanted and record_matches(item.mapping, spec):
                per_role[item.role] += 1
        result[name] = per_role
    return result


def freeze_subgroups(canonical: Sequence[CanonicalRecord], config: SplitConfig) -> list[str]:
    counts = subgroup_counts(canonical, ("generator_train", "local_dev"))
    frozen: list[str] = []
    for name, _spec in SUBGROUP_CANDIDATES:
        gen_n = counts[name]["generator_train"]
        dev_n = counts[name]["local_dev"]
        if gen_n >= config.min_gen_train_subgroup and dev_n >= config.min_local_dev_subgroup:
            frozen.append(name)
    return frozen


def contamination_audit(canonical: Sequence[CanonicalRecord]) -> dict[str, object]:
    fps_by_role: dict[str, set[str]] = defaultdict(set)
    feature_by_role: dict[str, set[str]] = defaultdict(set)
    quasi_by_role: dict[str, set[str]] = defaultdict(set)
    for item in canonical:
        fps_by_role[item.role].add(item.record_fp)
        feature_by_role[item.role].add(item.row.feature_fp)
        quasi_by_role[item.role].add(item.row.quasi_id_fp)

    exact_overlap = {}
    feature_overlap = {}
    quasi_overlap = {}
    roles = list(ALL_ROLES)
    for i, left in enumerate(roles):
        for right in roles[i + 1 :]:
            exact = fps_by_role[left] & fps_by_role[right]
            feat = feature_by_role[left] & feature_by_role[right]
            quasi = quasi_by_role[left] & quasi_by_role[right]
            key = f"{left}∩{right}"
            if exact:
                exact_overlap[key] = len(exact)
            if feat:
                feature_overlap[key] = len(feat)
            if quasi:
                quasi_overlap[key] = len(quasi)

    copies = Counter(item.n_copies for item in canonical)
    return {
        "exact_record_overlap_between_roles": exact_overlap,
        "feature_fp_overlap_between_roles": feature_overlap,
        "quasi_id_overlap_between_roles": quasi_overlap,
        "copy_multiplicity": {str(k): v for k, v in sorted(copies.items())},
        "cross_file_exact_records": role_counts(canonical)["excluded_cross_file"],
        "label_conflict_records": role_counts(canonical)["excluded_label_conflict"],
        "label_conflict_feature_fps": len(
            {
                item.row.feature_fp
                for item in canonical
                if item.role == "excluded_label_conflict"
            }
        ),
        "disjoint_exact_roles": not exact_overlap,
        "disjoint_feature_roles": not feature_overlap,
    }


def assignment_digest(canonical: Sequence[CanonicalRecord]) -> str:
    lines = []
    for item in sorted(canonical, key=lambda row: row.record_fp):
        probe = "1" if item.thinking_probe else "0"
        lines.append(
            f"{item.record_fp}\t{item.role}\t{item.row.source_file}\t"
            f"{item.row.source_line}\t{item.n_copies}\t{probe}"
        )
    return sha256_bytes("\n".join(lines).encode("utf-8") + b"\n")


def numeric_summary(canonical: Sequence[CanonicalRecord], role: str) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[int]] = {column: [] for column in NUMERIC_COLUMNS}
    for item in canonical:
        if item.role != role:
            continue
        mapping = item.mapping
        for column in NUMERIC_COLUMNS:
            buckets[column].append(int(mapping[column]))
    summary: dict[str, dict[str, float]] = {}
    for column, values in buckets.items():
        if not values:
            summary[column] = {"n": 0}
            continue
        ordered = sorted(values)
        n = len(ordered)
        summary[column] = {
            "n": n,
            "min": ordered[0],
            "max": ordered[-1],
            "mean": sum(ordered) / n,
            "p50": ordered[n // 2],
        }
    return summary


def income_rate(canonical: Sequence[CanonicalRecord], role: str) -> dict[str, float | int]:
    counts: Counter[str] = Counter()
    for item in canonical:
        if item.role == role:
            counts[item.mapping["income"]] += 1
    n = sum(counts.values())
    return {
        "n": n,
        "gt50k": counts.get(">50K", 0),
        "lte50k": counts.get("<=50K", 0),
        "gt50k_rate": (counts.get(">50K", 0) / n) if n else 0.0,
    }


def build_splits(archive: Path, config: SplitConfig) -> SplitResult:
    archive_sha256 = sha256_file(archive)
    if archive_sha256 != config.expected_sha256:
        raise ValueError(
            f"adult.zip sha256 {archive_sha256} != expected {config.expected_sha256}"
        )
    rows, inner_sha256 = parse_adult_zip(archive)
    canonical = collapse_and_tag(rows)
    canonical = assign_roles(canonical, config)
    result = SplitResult(
        config=config,
        archive_sha256=archive_sha256,
        inner_sha256=inner_sha256,
        canonical=canonical,
    )
    result.summary = build_summary(result, parsed_rows=len(rows))
    return result


def build_summary(result: SplitResult, parsed_rows: int) -> dict[str, object]:
    canonical = result.canonical
    config = result.config
    frozen = freeze_subgroups(canonical, config)
    frozen_counts = {
        name: subgroup_counts(canonical, ROLES)[name] for name in frozen
    }
    candidate_counts_non_test = subgroup_counts(
        canonical, ("generator_train", "local_dev", "public_calibration", "audit_nonmember")
    )
    return {
        "version": config.version,
        "seed": config.seed,
        "thinking_probe_seed": config.thinking_probe_seed,
        "archive_sha256": result.archive_sha256,
        "inner_sha256": result.inner_sha256,
        "parsed_rows": parsed_rows,
        "unique_records": len(canonical),
        "assignment_digest": assignment_digest(canonical),
        "role_counts": role_counts(canonical),
        "thinking_probe_n": sum(1 for item in canonical if item.thinking_probe),
        "stratum_by_role": {role: stratum_crosstab(canonical, role) for role in ALL_ROLES},
        "income_by_role": {role: income_rate(canonical, role) for role in ROLES},
        "numeric_by_role": {
            role: numeric_summary(canonical, role)
            for role in ("generator_train", "local_dev", "sealed_test")
        },
        "contamination": contamination_audit(canonical),
        "subgroup_candidates_non_test": candidate_counts_non_test,
        "frozen_subgroups": frozen,
        "frozen_subgroup_counts_all_roles": frozen_counts,
        "targets": {
            "public_calibration": config.public_calibration,
            "local_development": config.local_development,
            "audit_nonmember": config.audit_nonmember,
            "thinking_probe": config.thinking_probe,
            "generator_train": "remainder of unique adult.data-only records",
            "sealed_test": "all unique adult.test-only records after exclusions",
        },
    }


def write_csv(path: Path, records: Iterable[CanonicalRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)
        for item in records:
            writer.writerow(item.row.values)
            n += 1
    return n


def write_outputs(result: SplitResult, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    by_role: dict[str, list[CanonicalRecord]] = defaultdict(list)
    for item in result.canonical:
        by_role[item.role].append(item)
    for role in ALL_ROLES:
        items = sorted(by_role[role], key=lambda row: row.record_fp)
        csv_path = output_dir / f"{role}.csv"
        write_csv(csv_path, items)
        written[role] = str(csv_path)
    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in sorted(result.canonical, key=lambda row: row.record_fp):
            payload = {
                "record_fp": item.record_fp,
                "feature_fp": item.row.feature_fp,
                "role": item.role,
                "source_file": item.row.source_file,
                "source_line": item.row.source_line,
                "n_copies": item.n_copies,
                "sources": list(item.sources),
                "thinking_probe": item.thinking_probe,
                "stratum": item.row.stratum,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    written["manifest"] = str(manifest_path)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(result.summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    written["summary"] = str(summary_path)
    return written


def config_from_toml(path: Path) -> SplitConfig:
    import tomllib

    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    block = raw["adult_split"]
    targets = block.get("targets", {})
    probe = block.get("thinking_probe", {})
    thresholds = block.get("subgroup_thresholds", {})
    return SplitConfig(
        version=block.get("version", "v1"),
        seed=int(block["seed"]),
        public_calibration=int(targets.get("public_calibration", 256)),
        local_development=int(targets.get("local_development", 4000)),
        audit_nonmember=int(targets.get("audit_nonmember", 6000)),
        thinking_probe=int(probe.get("n", 64)),
        thinking_probe_seed=int(probe.get("seed", 20261011)),
        min_gen_train_subgroup=int(thresholds.get("min_generator_train", 50)),
        min_local_dev_subgroup=int(thresholds.get("min_local_dev", 15)),
        expected_sha256=block["expected_sha256"],
    )


def render_split_report(summary: Mapping[str, object]) -> str:
    roles = summary["role_counts"]
    contamination = summary["contamination"]
    frozen = summary["frozen_subgroups"]
    lines = [
        "# Adult split v1",
        "",
        "Derived CSVs stay in `data/derived/adult/v1/` (Git-ignored). This report",
        "records counts, seeds, hashes, and contamination checks only.",
        "",
        "## Provenance",
        "",
        f"- Archive SHA-256: `{summary['archive_sha256']}`",
        f"- Assignment digest: `{summary['assignment_digest']}`",
        f"- Split seed: `{summary['seed']}`",
        f"- Thinking-probe seed: `{summary['thinking_probe_seed']}`",
        f"- Parsed official rows: {summary['parsed_rows']}",
        f"- Unique exact records: {summary['unique_records']}",
        "",
        "## Role counts",
        "",
        "| Role | n |",
        "| --- | ---: |",
    ]
    for role in ALL_ROLES:
        lines.append(f"| `{role}` | {roles[role]} |")
    lines.append(f"| thinking_probe subset of public_calibration | {summary['thinking_probe_n']} |")
    lines.extend(
        [
            "",
            "## Income rate by experimental role",
            "",
            "| Role | n | >50K | rate |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for role, stats in summary["income_by_role"].items():
        lines.append(
            f"| `{role}` | {stats['n']} | {stats['gt50k']} | {stats['gt50k_rate']:.4f} |"
        )
    lines.extend(["", "## Strata (`income|sex`) on generator_train", "", "| Stratum | n |", "| --- | ---: |"])
    for stratum, n in summary["stratum_by_role"]["generator_train"].items():
        lines.append(f"| `{stratum}` | {n} |")
    lines.extend(
        [
            "",
            "## Contamination",
            "",
            f"- Exact-record overlap between roles: `{contamination['exact_record_overlap_between_roles']}`",
            f"- Feature-fingerprint overlap between roles: `{contamination['feature_fp_overlap_between_roles']}`",
            f"- Quasi-id (all fields except `fnlwgt`) overlap between roles: `{contamination['quasi_id_overlap_between_roles']}`",
            f"- Cross-file exact records excluded: {contamination['cross_file_exact_records']}",
            f"- Label-conflict records excluded: {contamination['label_conflict_records']} "
            f"({contamination['label_conflict_feature_fps']} feature fingerprints)",
            f"- Exact roles disjoint: {contamination['disjoint_exact_roles']}",
            f"- Feature roles disjoint: {contamination['disjoint_feature_roles']}",
            "",
            "Quasi-id overlap is a near-copy screen. Those rows differ in `fnlwgt` and",
            "remain distinct official records; they are not merged and are not a privacy proof.",
            "",
            "## Frozen rare subgroups",
            "",
            "Candidates were counted on `generator_train` and `local_dev` only. A candidate",
            "is frozen when generator_train ≥ 50 and local_dev ≥ 15. Sealed-test counts",
            "are reported only for frozen names.",
            "",
            f"Frozen: {', '.join(f'`{name}`' for name in frozen) if frozen else '(none)'}",
            "",
        ]
    )
    if frozen:
        lines.extend(
            [
                "| Subgroup | generator_train | local_dev | public_calibration | audit_nonmember | sealed_test |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name in frozen:
            counts = summary["frozen_subgroup_counts_all_roles"][name]
            lines.append(
                f"| `{name}` | {counts['generator_train']} | {counts['local_dev']} | "
                f"{counts['public_calibration']} | {counts['audit_nonmember']} | "
                f"{counts['sealed_test']} |"
            )
    lines.extend(
        [
            "",
            "## Non-test candidate counts used for the freeze decision",
            "",
            "| Candidate | generator_train | local_dev | public_calibration | audit_nonmember |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, counts in summary["subgroup_candidates_non_test"].items():
        lines.append(
            f"| `{name}` | {counts['generator_train']} | {counts['local_dev']} | "
            f"{counts['public_calibration']} | {counts['audit_nonmember']} |"
        )
    lines.extend(
        [
            "",
            "## Regeneration",
            "",
            "```bash",
            "PYTHONPATH=src python3 scripts/build_adult_splits.py --config configs/adult_split.toml",
            "```",
            "",
            "The assignment digest must match this report.",
            "",
        ]
    )
    return "\n".join(lines)
