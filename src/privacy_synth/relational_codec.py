"""Lossless retained-field codec for public relational generator diagnostics.

The codec learns vocabulary and fill values from fit rows only. Numeric NULL
masks are modeled alongside values; decode never applies a business rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

import numpy as np
import pandas as pd


def _date_format(series: pd.Series):
    values = series.dropna().astype(str)
    if values.empty:
        return None
    if values.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        return {"format": "%Y-%m-%d", "unit": "D", "fraction_digits": 0}
    match = values.str.extract(r"^\d{4}-\d{2}-\d{2}([ T])\d{2}:\d{2}:\d{2}(?:\.(\d{1,6}))?$")
    if match[0].isna().any() or match[0].nunique() != 1:
        return None
    digits = match[1].fillna("").str.len()
    if digits.nunique() != 1:
        return None
    n = int(digits.iloc[0])
    return {"format": f"%Y-%m-%d{match[0].iloc[0]}%H:%M:%S" + (".%f" if n else ""),
            "unit": "us", "fraction_digits": n}


def _python_value(value):
    return value.item() if isinstance(value, np.generic) else value


@dataclass
class TableCodec:
    """A JSON-serializable, fit-only contract for one relational table."""

    columns: list[dict[str, Any]]
    id_columns: list[str]
    version: int = 2

    @classmethod
    def fit(cls, frame: pd.DataFrame, id_columns=()):
        ids = list(id_columns)
        if not set(ids).issubset(frame.columns):
            raise ValueError("ID columns absent from frame")
        columns = []
        for i, name in enumerate(frame.columns):
            s = frame[name]
            feature = f"feature{i:04d}"
            info = {"name": name, "feature": feature, "dtype": str(s.dtype)}
            if s.isna().any():
                marker = s[s.isna()].iloc[0]
                info['null_marker'] = 'pd.NA' if marker is pd.NA else ('None' if marker is None else 'nan')
            if name in ids:
                info.update(kind="id", feature=name)
            else:
                date = _date_format(s) if not pd.api.types.is_numeric_dtype(s) else None
                if date:
                    parsed = pd.to_datetime(s, format=date["format"], errors="raise")
                    origin = pd.Timestamp("1970-01-01")
                    factor = pd.Timedelta(days=1) if date["unit"] == "D" else pd.Timedelta(microseconds=1)
                    values = (parsed - origin) / factor
                    info.update(kind="date", date=date, fill=float(values.median()) if values.notna().any() else 0.0,
                                mask=feature + "missing")
                elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                    values = pd.to_numeric(s, errors="raise")
                    valid = values.dropna()
                    if pd.api.types.is_integer_dtype(s) and not valid.empty and int(valid.abs().max()) > 2**53:
                        info.update(kind="category", values=[_python_value(x) for x in pd.unique(s.dropna())])
                    else:
                        info.update(kind="number", fill=float(values.median()) if valid.size else 0.0,
                                    integer=bool(pd.api.types.is_integer_dtype(s)), mask=feature + "missing")
                else:
                    info.update(kind="category", values=[_python_value(x) for x in pd.unique(s.dropna())])
                if info["kind"] == "category" and isinstance(s.dtype, pd.CategoricalDtype):
                    info.update(categories=[_python_value(x) for x in s.cat.categories], ordered=bool(s.cat.ordered))
            columns.append(info)
        return cls(columns=columns, id_columns=ids)

    def to_dict(self):
        return {"version": self.version, "id_columns": self.id_columns, "columns": self.columns}

    @classmethod
    def from_dict(cls, value):
        if value.get("version") != 2:
            raise ValueError("Unsupported codec version")
        return cls(columns=value["columns"], id_columns=value["id_columns"])

    def encode(self, frame: pd.DataFrame):
        result, domain = {}, {}
        for info in self.columns:
            s = frame[info["name"]]
            f = info["feature"]
            if info["kind"] == "id":
                result[f] = s.copy()
                continue
            if info["kind"] == "category":
                mapping = {x: f"value{i:06d}" for i, x in enumerate(info["values"])}
                nonnull = s.notna()
                mapped = s.astype(object).map(mapping)
                if mapped[nonnull].isna().any():
                    raise ValueError(f"Unseen vocabulary in {info['name']}; fit vocabulary cannot be expanded")
                result[f] = mapped.where(nonnull, "missing").astype(str)
                domain[f] = {"type": "discrete", "size": int(result[f].nunique(dropna=True))}
            else:
                if info["kind"] == "date":
                    fmt = info["date"]
                    parsed = pd.to_datetime(s, format=fmt["format"], errors="raise")
                    unit = pd.Timedelta(days=1) if fmt["unit"] == "D" else pd.Timedelta(microseconds=1)
                    numbers = (parsed - pd.Timestamp("1970-01-01")) / unit
                else:
                    numbers = pd.to_numeric(s, errors="raise")
                result[f] = numbers.astype(float).fillna(info["fill"])
                result[info["mask"]] = s.isna().astype(int)
                domain[f] = {"type": "continuous", "size": int(s.nunique(dropna=True))}
                domain[info["mask"]] = {"type": "discrete", "size": int(result[info["mask"]].nunique())}
        return pd.DataFrame(result, index=frame.index), domain

    def decode(self, frame: pd.DataFrame):
        result = {}
        for info in self.columns:
            f, name = info["feature"], info["name"]
            if f not in frame:
                raise ValueError(f"Missing generated feature {f}")
            s = frame[f]
            if info["kind"] == "id":
                result[name] = pd.to_numeric(s, errors='raise').astype(info['dtype']) if info['dtype'].startswith(('int', 'uint', 'float', 'Int', 'UInt')) else s.astype(info['dtype'])
                continue
            if info["kind"] == "category":
                mapping = {f"value{i:06d}": x for i, x in enumerate(info["values"])}
                tokens = s.astype(str)
                unknown = ~tokens.isin(list(mapping) + ["missing"])
                if unknown.any():
                    raise ValueError(f"Unknown category tokens for {name}: {int(unknown.sum())}")
                decoded = pd.Series([None if token == "missing" else mapping[token] for token in tokens],
                                    index=frame.index, dtype=object)
            else:
                if info["mask"] not in frame:
                    raise ValueError(f"Missing NULL mask for {name}")
                mask_numbers = pd.to_numeric(frame[info["mask"]], errors="raise")
                if not mask_numbers.isin([0, 1]).all():
                    raise ValueError(f"Non-binary NULL mask for {name}")
                missing = mask_numbers.eq(1)
                numbers = pd.to_numeric(s, errors="raise").astype(float)
                if not np.isfinite(numbers[~missing].to_numpy()).all():
                    raise ValueError(f"Non-finite retained values for {name}")
                numbers = numbers.mask(missing, info["fill"])
                if info["kind"] == "date":
                    meta = info["date"]
                    parsed = pd.to_datetime(np.rint(numbers), unit=meta["unit"], origin="1970-01-01", errors="raise")
                    decoded = pd.Series(parsed, index=frame.index).dt.strftime(meta["format"]).astype(object)
                    digits = meta["fraction_digits"]
                    if digits and digits < 6:
                        decoded = decoded.str.slice(stop=-(6 - digits))
                    decoded.loc[missing] = None
                else:
                    decoded = numbers.mask(missing)
                    if info["integer"]:
                        decoded = decoded.round()
            marker = {'pd.NA': pd.NA, 'None': None, 'nan': np.nan}.get(info.get('null_marker'), None)
            if info['dtype'] == 'object' and decoded.isna().any():
                decoded.loc[decoded.isna()] = marker
            dtype = info["dtype"]
            # pandas3 records its nullable string dtype as "str". In pandas2
            # the same spelling stringifies NULL (None -> "None"). Preserve
            # SQL NULL with object dtype when this runtime lacks that dtype.
            if dtype == 'str' and pd.Series([None], dtype=object).astype('str').notna().any():
                dtype = 'object'
            if dtype == "category":
                decoded = pd.Series(pd.Categorical(decoded, categories=info["categories"], ordered=info["ordered"]), index=frame.index)
            else:
                try:
                    decoded = decoded.astype(dtype)
                except (ValueError, TypeError):
                    # A generated nullable integer retains SQL integer semantics.
                    if dtype.startswith(("int", "uint")):
                        decoded = decoded.astype("Int64")
                    else:
                        raise
            result[name] = decoded
        return pd.DataFrame(result, index=frame.index)


@dataclass
class JointChildCounts:
    """Fit-only vocabulary of complete child-degree vectors, including zero.

    A root generator must learn/sample this feature. This representation is
    not a business rule, a generator-quality pass or a privacy guarantee.
    """

    root: str
    root_key: str
    children: dict[str, str]
    states: list[list[int]]
    feature: str = 'joint_child_counts'

    @staticmethod
    def count_vectors(frames, root, root_key, children):
        parent = frames[root][root_key]
        if parent.isna().any() or parent.duplicated().any():
            raise ValueError('Parent keys must be unique and nonnull')
        out = {}
        for table, fk in children.items():
            values = frames[table][fk]
            if values.isna().any() or not set(values).issubset(set(parent)):
                raise ValueError(f'Dangling or null foreign key: {table}.{fk}')
            out[table] = parent.map(values.value_counts()).fillna(0).astype('int64').to_numpy()
        return pd.DataFrame(out)

    @classmethod
    def fit(cls, frames, root, root_key, children, feature='joint_child_counts'):
        if not children:
            raise ValueError('Joint counts require child tables')
        counts = cls.count_vectors(frames, root, root_key, children)
        states = [list(x) for x in sorted(set(map(tuple, counts.to_numpy().tolist())))]
        return cls(root, root_key, dict(children), states, feature)

    def encode(self, frames):
        counts = self.count_vectors(frames, self.root, self.root_key, self.children)
        mapping = {tuple(v): f'degree{i:06d}' for i, v in enumerate(self.states)}
        try:
            return pd.Series([mapping[tuple(v)] for v in counts.to_numpy().tolist()], name=self.feature)
        except KeyError as exc:
            raise ValueError('Unseen joint degree; fit vocabulary cannot expand') from exc

    def decode(self, tokens):
        mapping = {f'degree{i:06d}': v for i, v in enumerate(self.states)}
        try:
            return pd.DataFrame([mapping[str(t)] for t in tokens], columns=list(self.children), dtype='int64')
        except KeyError as exc:
            raise ValueError('Unknown generated joint-degree token') from exc

    def to_dict(self):
        return dict(version=1, root=self.root, root_key=self.root_key,
                    children=self.children, states=self.states, feature=self.feature)

    @classmethod
    def from_dict(cls, value):
        if value.get('version') != 1:
            raise ValueError('Unknown joint-degree contract version')
        out = cls(value['root'], value['root_key'], dict(value['children']), value['states'], value['feature'])
        if not out.states or len(set(map(tuple, out.states))) != len(out.states):
            raise ValueError('Empty or duplicate joint states')
        if any(len(s) != len(out.children) or any(type(n) is not int or n < 0 for n in s) for s in out.states):
            raise ValueError('Invalid joint-degree vector')
        return out
