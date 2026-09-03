"""Configuration loading and canonical project paths.

A single YAML file (``configs/default.yaml``) defines the primary
specification. Robustness runs override individual keys via
:meth:`Config.override` so that every persisted estimate can be traced back to
an explicit, fully-specified parameter set.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class _Paths:
    root: Path = ROOT
    configs: Path = ROOT / "configs"
    data_raw: Path = ROOT / "data" / "raw"
    data_processed: Path = ROOT / "data" / "processed"
    results: Path = ROOT / "results"
    figures: Path = ROOT / "results" / "figures"
    tables: Path = ROOT / "results" / "tables"
    metrics: Path = ROOT / "results" / "metrics"

    def ensure(self) -> "_Paths":
        for p in (
            self.data_raw,
            self.data_processed,
            self.results,
            self.figures,
            self.tables,
            self.metrics,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self


PATHS = _Paths()


class Config(dict):
    """Dict-like config with dotted access and immutable-style overrides."""

    def __getattr__(self, item: str) -> Any:
        try:
            val = self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc
        return Config(val) if isinstance(val, dict) else val

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def override(self, **dotted_values: Any) -> "Config":
        """Return a deep copy with ``a.b.c=value`` style overrides applied."""
        new = Config(copy.deepcopy(dict(self)))
        for dotted, value in dotted_values.items():
            parts = dotted.split(".")
            node: dict = new
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return new

    @property
    def fingerprint(self) -> str:
        """Stable short hash of the full config - stamped onto persisted output."""
        blob = json.dumps(dict(self), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else PATHS.configs / "default.yaml"
    with open(path) as fh:
        return Config(yaml.safe_load(fh))
