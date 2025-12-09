from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import re


# ---------------------------------------------------------------------
# Core data structures: NodeCandidate, ExtractionResult
# ---------------------------------------------------------------------


@dataclass
class NodeCandidate:
    """
    Единица выбора для селектора: DOM/layout-узел с модельным скором.

    Это минимальный объект, с которым работают селекторы и ограничения.
    """

    node_id: str
    text: str
    score: float
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x, y, w, h)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """
    Финальный результат извлечения для одного поля.

    Поле proof содержит:
      - "node_ids": список ID DOM-узлов, на которые опирается значение;
      - "constraints": список имен/ID ограничений Γ, которые были соблюдены
                       для данного поля при выбранном присваивании;
      - "violated": (опционально) список нарушенных ограничений (для дебага).
    """

    field_name: str
    value: Any
    confidence: float
    proof: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Constraints (Γ)
# ---------------------------------------------------------------------


class Constraint(ABC):
    """
    Абстрактный базовый класс для всех ограничений Γ.

    Ограничение проверяется на *присваивании*:
        assignment: dict[field_name -> NodeCandidate | List[NodeCandidate] | None]

    и опциональном page_context (метаданные страницы, например высота).
    """

    def __init__(self, name: str, field_names: Optional[List[str]] = None) -> None:
        self.name = name
        # None -> глобальное ограничение, иначе применимо только к перечисленным полям.
        self.field_names = field_names

    def applies_to(self, field_name: str) -> bool:
        return self.field_names is None or field_name in self.field_names

    @abstractmethod
    def check(
        self,
        assignment: Dict[str, Any],
        page_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        True, если ограничение удовлетворено данным присваиванием.
        """
        raise NotImplementedError


class UniqueConstraint(Constraint):
    """
    Уникальность: не более max_count значений для данного поля на странице.

    Типичный кейс: "один title на страницу".
    """

    def __init__(
        self,
        field_name: str,
        max_count: int = 1,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or f"unique[{field_name}]", field_names=[field_name])
        self.field_name = field_name
        self.max_count = max_count

    def check(
        self,
        assignment: Dict[str, Any],
        page_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        value = assignment.get(self.field_name)
        if value is None:
            return True
        if isinstance(value, list):
            return len(value) <= self.max_count
        # Одна value – уникальность тривиально соблюдается.
        return True


class FormatConstraint(Constraint):
    """
    Простейшее форматное ограничение на основе regex.

    Пример: цена должна содержать хотя бы одну цифру.
    """

    def __init__(
        self,
        field_name: str,
        pattern: str,
        name: Optional[str] = None,
        flags: int = 0,
    ) -> None:
        super().__init__(name=name or f"format[{field_name}]", field_names=[field_name])
        self.field_name = field_name
        self._pattern_str = pattern
        self.pattern = re.compile(pattern, flags)

    def check(
        self,
        assignment: Dict[str, Any],
        page_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        cand = assignment.get(self.field_name)
        if cand is None:
            # Отсутствие candidate’а не ломает формат, обязательность поля – в Field.required.
            return True

        candidates: List[NodeCandidate]
        if isinstance(cand, list):
            candidates = cand
        else:
            candidates = [cand]

        for c in candidates:
            text = getattr(c, "text", None)
            if text is None:
                return False
            if not self.pattern.search(str(text)):
                return False
        return True


class VisualConstraint(Constraint):
    """
    Визуальное ограничение: например, исключение footer-зоны.

    Упрощённый вариант "footer exclusion" из прототипа:
        y / page_height <= max_relative_y

    где y – верхняя координата выбранного узла.
    """

    def __init__(
        self,
        field_name: str,
        max_relative_y: float = 0.75,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or f"visual[{field_name}]", field_names=[field_name])
        self.field_name = field_name
        self.max_relative_y = max_relative_y

    def check(
        self,
        assignment: Dict[str, Any],
        page_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        cand = assignment.get(self.field_name)
        if cand is None:
            return True

        candidates: List[NodeCandidate]
        if isinstance(cand, list):
            candidates = cand
        else:
            candidates = [cand]

        page_height = None
        if page_context is not None:
            page_height = page_context.get("page_height")

        if not page_height:
            # Не знаем высоту страницы — лучше не заваливать constraint.
            return True

        for c in candidates:
            bbox = getattr(c, "bbox", None)
            if not bbox:
                continue
            _, y, _, _ = bbox
            rel_y = float(y) / float(page_height)
            if rel_y > self.max_relative_y:
                return False

        return True


# ---------------------------------------------------------------------
# Constraint factory for declarative schemas
# ---------------------------------------------------------------------


def build_constraint_from_spec(
    field_name: Optional[str],
    spec: Dict[str, Any],
) -> Constraint:
    """
    Построить Constraint из декларативного описания.

    Примеры:
      {"type": "unique"}
      {"type": "format", "pattern": "\\d+"}
      {"type": "visual", "max_relative_y": 0.75}
    """
    ctype = spec.get("type")
    name = spec.get("name")

    if ctype == "unique":
        target_field = spec.get("field_name") or field_name
        if not target_field:
            raise ValueError("UniqueConstraint requires a 'field_name'.")
        max_count = int(spec.get("max_count", 1))
        return UniqueConstraint(
            field_name=target_field,
            max_count=max_count,
            name=name,
        )

    if ctype == "format":
        target_field = spec.get("field_name") or field_name
        if not target_field:
            raise ValueError("FormatConstraint requires a 'field_name'.")
        pattern = spec["pattern"]
        flags = spec.get("flags", 0)
        return FormatConstraint(
            field_name=target_field,
            pattern=pattern,
            name=name,
            flags=flags,
        )

    if ctype == "visual":
        target_field = spec.get("field_name") or field_name
        if not target_field:
            raise ValueError("VisualConstraint requires a 'field_name'.")
        max_rel_y = float(spec.get("max_relative_y", 0.75))
        return VisualConstraint(
            field_name=target_field,
            max_relative_y=max_rel_y,
            name=name,
        )

    raise ValueError(f"Unknown constraint type in spec: {ctype!r}")


# ---------------------------------------------------------------------
# Fields & Schema
# ---------------------------------------------------------------------


@dataclass
class Field:
    """
    Логическое поле целевой схемы, например "title", "price".
    """

    name: str
    dtype: str = "string"
    required: bool = True
    constraints: List[Constraint] = field(default_factory=list)
    selector: Optional["Selector"] = None  # биндинг селектора для этого поля


@dataclass
class Schema:
    """
    Декларативное описание схемы страницы (Section III.A).
    """

    name: str
    fields: Dict[str, Field]
    constraints: List[Constraint] = field(default_factory=list)

    # ---------- Declarative loading ----------

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Schema":
        """
        Построить Schema из простого декларативного dict.

        Пример:

        schema_spec = {
            "name": "ProductPage",
            "fields": [
                {
                    "name": "title",
                    "dtype": "string",
                    "required": True,
                    "constraints": [
                        {"type": "unique"},
                    ],
                },
                {
                    "name": "price",
                    "dtype": "string",
                    "constraints": [
                        {"type": "format", "pattern": r"\\d"},
                        {"type": "visual", "max_relative_y": 0.75},
                    ],
                },
            ],
            "constraints": [],
        }
        """
        name = data.get("name", "UnnamedSchema")
        field_specs = data.get("fields", [])
        schema_constraints_specs = data.get("constraints", [])

        fields: Dict[str, Field] = {}
        schema_constraints: List[Constraint] = []

        # Field-level constraints
        for f_spec in field_specs:
            fname = f_spec["name"]
            fdtype = f_spec.get("dtype", "string")
            required = f_spec.get("required", True)
            constraint_specs = f_spec.get("constraints", [])

            f_constraints: List[Constraint] = []
            for c_spec in constraint_specs:
                f_constraints.append(build_constraint_from_spec(fname, c_spec))

            fields[fname] = Field(
                name=fname,
                dtype=fdtype,
                required=required,
                constraints=f_constraints,
            )

        # Schema-level constraints (могут ссылаться на field_name в spec)
        for c_spec in schema_constraints_specs:
            schema_constraints.append(build_constraint_from_spec(None, c_spec))

        return cls(name=name, fields=fields, constraints=schema_constraints)

    # ---------- Constraint evaluation helper ----------

    def check_all_constraints(
        self,
        assignment: Dict[str, Any],
        page_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Проверить все field-level и schema-level ограничения Γ на присваивании.
        """
        # Field-level
        for fname, field in self.fields.items():
            for c in field.constraints:
                if not c.check(assignment, page_context):
                    return False

        # Schema-level
        for c in self.constraints:
            if not c.check(assignment, page_context):
                return False

        return True


# ---------------------------------------------------------------------
# Selector abstraction
# ---------------------------------------------------------------------


class Selector(ABC):
    """
    Абстракция селектора f (Section III.B):
    отображает страницу (граф) в множество кандидатов для одного поля.
    """

    def __init__(self, field_name: str) -> None:
        self.field_name = field_name

    @abstractmethod
    def propose(self, page_graph: Any) -> List[NodeCandidate]:
        """
        Вернуть список NodeCandidate, отсортированный по score по убыванию.
        """
        raise NotImplementedError


class DefaultSelector(Selector):
    """
    Простейший селектор по умолчанию.

    Ожидает, что page_graph имеет атрибут:
        page_graph.candidates_by_field[field_name] -> List[NodeCandidate]

    Вся логика получения кандидатов (GNN, эвристики и т.д.) лежит на внешнем
    пайплайне – здесь только абстракция.
    """

    def propose(self, page_graph: Any) -> List[NodeCandidate]:
        candidates_by_field = getattr(page_graph, "candidates_by_field", {})
        candidates = list(candidates_by_field.get(self.field_name, []))
        candidates.sort(key=lambda c: getattr(c, "score", 0.0), reverse=True)
        return candidates
