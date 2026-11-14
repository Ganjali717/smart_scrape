"""
Declarative extraction schema.

A schema lists the fields of a record and the constraints the record must
satisfy. It is written in YAML (see configs/schemas/*.yaml) and compiled by
`solver.ConstraintSolver` into an integer linear program.

Constraint kinds (field names refer to schema fields):

  mutual_exclusion                        a node fills at most one field
  format          field                   node text must parse as the field type
  exclude_regions regions, fields         nodes inside e.g. footer/nav cannot fill the fields
  position_band   field, max (min)        relative vertical position of the field in [min, max]
  proximity       a, b, max_distance      a may be chosen only if some chosen b is near it
  ordering        before, after           `before` must appear above `after` on the page

Each constraint has a `rank`: when the program is infeasible, constraints are
relaxed from the lowest rank upward (geometry first, cardinality never).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

FIELD_TYPES = {"text", "price", "number", "date", "url", "isbn"}
CARDINALITIES = {"one", "optional"}
CONSTRAINT_KINDS = {"mutual_exclusion", "format", "exclude_regions", "position_band", "proximity", "ordering"}
NODE_LOCAL_KINDS = {"format", "exclude_regions", "position_band"}


@dataclass
class FieldSpec:
    name: str
    type: str = "text"
    cardinality: str = "one"

    def __post_init__(self):
        if self.type not in FIELD_TYPES:
            raise ValueError(f"field {self.name}: unknown type {self.type!r}")
        if self.cardinality not in CARDINALITIES:
            raise ValueError(f"field {self.name}: unknown cardinality {self.cardinality!r}")


@dataclass
class ConstraintSpec:
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)
    rank: int = 5
    name: str = ""

    def __post_init__(self):
        if self.kind not in CONSTRAINT_KINDS:
            raise ValueError(f"unknown constraint kind {self.kind!r}")
        if not self.name:
            extra = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
            self.name = f"{self.kind}({extra})" if extra else self.kind

    @property
    def node_local(self) -> bool:
        return self.kind in NODE_LOCAL_KINDS


@dataclass
class Schema:
    name: str
    fields: List[FieldSpec]
    constraints: List[ConstraintSpec] = field(default_factory=list)

    @property
    def field_names(self) -> List[str]:
        return [f.name for f in self.fields]

    def field(self, name: str) -> FieldSpec:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)

    def fields_for(self, spec_fields: Optional[Any]) -> List[str]:
        if spec_fields in (None, "all"):
            return self.field_names
        if isinstance(spec_fields, str):
            return [spec_fields]
        return list(spec_fields)

    def subset(self, kinds: Optional[set] = None, node_local_only: bool = False) -> "Schema":
        """Copy of the schema keeping only some constraint kinds (used for ablations/baselines)."""
        cs = [c for c in self.constraints
              if (kinds is None or c.kind in kinds) and (not node_local_only or c.node_local)]
        return Schema(self.name, list(self.fields), cs)

    # -- I/O ----------------------------------------------------------------
    @staticmethod
    def from_dict(d: dict) -> "Schema":
        fields = [FieldSpec(name=k, **(v or {})) for k, v in d["fields"].items()]
        cons = []
        for c in d.get("constraints", []) or []:
            c = dict(c)
            kind = c.pop("kind")
            rank = c.pop("rank", 5)
            name = c.pop("name", "")
            cons.append(ConstraintSpec(kind=kind, params=c, rank=rank, name=name))
        s = Schema(name=d.get("name", "schema"), fields=fields, constraints=cons)
        s.validate()
        return s

    @staticmethod
    def load(path: str) -> "Schema":
        with open(path, encoding="utf-8") as f:
            return Schema.from_dict(yaml.safe_load(f))

    def validate(self) -> None:
        names = set(self.field_names)
        for c in self.constraints:
            for key in ("field", "a", "b", "before", "after"):
                if key in c.params and c.params[key] not in names:
                    raise ValueError(f"constraint {c.name}: unknown field {c.params[key]!r}")


def schema_to_dict(schema: Schema) -> dict:
    return {
        "name": schema.name,
        "fields": {f.name: {"type": f.type, "cardinality": f.cardinality} for f in schema.fields},
        "constraints": [{"kind": c.kind, "rank": c.rank, **c.params} for c in schema.constraints],
    }


def save_schema(schema: Schema, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(schema_to_dict(schema), f, sort_keys=False, allow_unicode=True)
