"""Strict ordered discovery schemas with lossless column access."""
from dataclasses import dataclass

from .research_record import ResearchRecord
from .discovery_projection import DiscoveryRow
from .discovery_context_projection import project_research_record_with_context


def _schema(row):
    return tuple(tuple(value.field_id for value in getattr(row, group))
                 for group in ("identity", "predictors", "targets"))


@dataclass(frozen=True)
class DiscoveryDataset:
    rows: tuple[DiscoveryRow, ...]
    identity_field_ids: tuple[str, ...]
    predictor_field_ids: tuple[str, ...]
    target_field_ids: tuple[str, ...]

    def __post_init__(self):
        rows = tuple(self.rows)
        if any(not isinstance(row, DiscoveryRow) for row in rows):
            raise TypeError("rows must contain DiscoveryRow")
        expected = _schema(rows[0]) if rows else ((), (), ())
        for row in rows[1:]:
            for role, actual, required in zip(("identity", "predictor", "target"), _schema(row), expected):
                if actual != required:
                    raise ValueError(f"{role} schema mismatch")
        for name, required in zip(("identity_field_ids", "predictor_field_ids", "target_field_ids"), expected):
            supplied = tuple(getattr(self, name))
            if supplied != required:
                raise ValueError(f"inconsistent {name}")
            object.__setattr__(self, name, supplied)
        object.__setattr__(self, "rows", rows)


def assemble_discovery_dataset(rows) -> DiscoveryDataset:
    rows = tuple(rows)
    if rows and not isinstance(rows[0], DiscoveryRow):
        raise TypeError("rows must contain DiscoveryRow")
    return DiscoveryDataset(rows, *(_schema(rows[0]) if rows else ((), (), ())))


def build_discovery_dataset(records) -> DiscoveryDataset:
    records = tuple(records)
    if any(not isinstance(record, ResearchRecord) for record in records):
        raise TypeError("records must contain ResearchRecord")
    return assemble_discovery_dataset(tuple(project_research_record_with_context(record) for record in records))


def _locate(dataset, field_id):
    if not isinstance(dataset, DiscoveryDataset):
        raise TypeError("dataset must be DiscoveryDataset")
    if not isinstance(field_id, str):
        raise TypeError("field_id must be string")
    matches = []
    for role, group, schema in (("identity", "identity", dataset.identity_field_ids),
        ("predictor", "predictors", dataset.predictor_field_ids), ("target", "targets", dataset.target_field_ids)):
        matches.extend((role, group, index) for index, name in enumerate(schema) if name == field_id)
    if len(matches) != 1:
        raise KeyError(field_id)
    return matches[0]


def column_values(dataset: DiscoveryDataset, field_id: str) -> tuple[object, ...]:
    _, group, index = _locate(dataset, field_id)
    return tuple(getattr(row, group)[index].value for row in dataset.rows)


def field_role(dataset: DiscoveryDataset, field_id: str) -> str:
    return _locate(dataset, field_id)[0]
