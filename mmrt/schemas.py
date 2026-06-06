"""Declarative Tardis CSV schemas for the current MMRT raw inputs."""

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

from mmrt.contracts import TardisDataType


COMMON_TARDIS_COLUMNS = (
    "exchange",
    "symbol",
    "timestamp",
    "local_timestamp",
)

BOOK_SNAPSHOT_25_DEPTH = 25

CURRENT_REQUIRED_BOOK_DEPTH_LEVELS = (1, 3, 5, 10, 20)
CURRENT_MAX_REQUIRED_BOOK_DEPTH = 20
CURRENT_REQUIRED_BOOK_SNAPSHOT_DEPTH = 25

SUPPORTED_TARDIS_SCHEMA_TYPES = (
    TardisDataType.INCREMENTAL_BOOK_L2,
    TardisDataType.BOOK_SNAPSHOT_25,
    TardisDataType.TRADES,
)


class _StrEnum(str, Enum):
    pass


class ColumnKind(_StrEnum):
    STRING = "string"
    INT_US = "int_us"
    FLOAT = "float"
    BOOL = "bool"
    SIDE = "side"
    BOOK_SIDE = "book_side"


def _require_nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _coerce_data_type(value: TardisDataType | str, name: str) -> TardisDataType:
    if isinstance(value, TardisDataType):
        return value
    if isinstance(value, str):
        try:
            return TardisDataType(value)
        except ValueError as exc:
            raise ValueError(f"{name} has invalid value {value!r}") from exc
    raise ValueError(f"{name} must be TardisDataType or str")


def _coerce_column_kind(value: ColumnKind | str, name: str) -> ColumnKind:
    if isinstance(value, ColumnKind):
        return value
    if isinstance(value, str):
        try:
            return ColumnKind(value)
        except ValueError as exc:
            raise ValueError(f"{name} has invalid value {value!r}") from exc
    raise ValueError(f"{name} must be ColumnKind or str")


def _require_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


def _tuple_of_str(values: Iterable[str], name: str) -> tuple[str, ...]:
    out: list[str] = []
    for i, value in enumerate(tuple(values)):
        out.append(_require_nonempty_str(value, f"{name}[{i}]"))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    name: str
    kind: ColumnKind
    nullable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_nonempty_str(self.name, "name"))
        object.__setattr__(self, "kind", _coerce_column_kind(self.kind, "kind"))
        object.__setattr__(self, "nullable", _require_bool(self.nullable, "nullable"))


@dataclass(frozen=True, slots=True)
class TardisCSVSchema:
    data_type: TardisDataType
    columns: tuple[ColumnSpec, ...]
    depth_limit: int | None = None

    def __post_init__(self) -> None:
        data_type = _coerce_data_type(self.data_type, "data_type")
        object.__setattr__(self, "data_type", data_type)

        if not isinstance(self.columns, tuple) or not self.columns:
            raise ValueError("columns must be a non-empty tuple[ColumnSpec, ...]")
        for idx, col in enumerate(self.columns):
            if not isinstance(col, ColumnSpec):
                raise ValueError(f"columns[{idx}] must be ColumnSpec")

        names = self.column_names
        if len(set(names)) != len(names):
            raise ValueError("column names must be unique")
        if names[:4] != COMMON_TARDIS_COLUMNS:
            raise ValueError(f"first four columns must be {COMMON_TARDIS_COLUMNS!r}")

        if data_type not in SUPPORTED_TARDIS_SCHEMA_TYPES:
            raise ValueError(f"unsupported data_type for schema registry: {data_type.value}")

        if data_type == TardisDataType.BOOK_SNAPSHOT_25:
            if self.depth_limit != BOOK_SNAPSHOT_25_DEPTH:
                raise ValueError("depth_limit must be 25 for BOOK_SNAPSHOT_25")
        elif self.depth_limit is not None:
            raise ValueError("depth_limit must be None for non-snapshot schemas")

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(col.name for col in self.columns)

    @property
    def required_column_names(self) -> tuple[str, ...]:
        return tuple(col.name for col in self.columns if not col.nullable)

    def column_index(self, name: str) -> int:
        target = _require_nonempty_str(name, "name")
        try:
            return self.column_names.index(target)
        except ValueError as exc:
            raise ValueError(f"column {target!r} not found in {self.data_type.value} schema") from exc

    def validate_header(self, header: Sequence[str], *, exact: bool = True) -> None:
        header_names = _tuple_of_str(header, "header")
        schema_names = self.column_names
        if exact:
            if header_names != schema_names:
                raise ValueError(
                    f"header mismatch for {self.data_type.value}: expected exact columns {schema_names!r}, got {header_names!r}"
                )
            return
        missing = [name for name in schema_names if name not in header_names]
        if missing:
            raise ValueError(
                f"header missing required schema columns for {self.data_type.value}: {tuple(missing)!r}"
            )


def _common_columns() -> tuple[ColumnSpec, ...]:
    return (
        ColumnSpec("exchange", ColumnKind.STRING),
        ColumnSpec("symbol", ColumnKind.STRING),
        ColumnSpec("timestamp", ColumnKind.INT_US),
        ColumnSpec("local_timestamp", ColumnKind.INT_US),
    )


def book_snapshot_25_columns() -> tuple[ColumnSpec, ...]:
    cols = list(_common_columns())
    for i in range(BOOK_SNAPSHOT_25_DEPTH):
        cols.extend(
            (
                ColumnSpec(f"asks[{i}].price", ColumnKind.FLOAT, nullable=True),
                ColumnSpec(f"asks[{i}].amount", ColumnKind.FLOAT, nullable=True),
                ColumnSpec(f"bids[{i}].price", ColumnKind.FLOAT, nullable=True),
                ColumnSpec(f"bids[{i}].amount", ColumnKind.FLOAT, nullable=True),
            )
        )
    return tuple(cols)


def incremental_book_l2_columns() -> tuple[ColumnSpec, ...]:
    return _common_columns() + (
        ColumnSpec("is_snapshot", ColumnKind.BOOL),
        ColumnSpec("side", ColumnKind.BOOK_SIDE),
        ColumnSpec("price", ColumnKind.FLOAT),
        ColumnSpec("amount", ColumnKind.FLOAT),
    )


def trades_columns() -> tuple[ColumnSpec, ...]:
    return _common_columns() + (
        ColumnSpec("id", ColumnKind.STRING, nullable=True),
        ColumnSpec("side", ColumnKind.SIDE),
        ColumnSpec("price", ColumnKind.FLOAT),
        ColumnSpec("amount", ColumnKind.FLOAT),
    )


INCREMENTAL_BOOK_L2_SCHEMA = TardisCSVSchema(
    data_type=TardisDataType.INCREMENTAL_BOOK_L2,
    columns=incremental_book_l2_columns(),
)
BOOK_SNAPSHOT_25_SCHEMA = TardisCSVSchema(
    data_type=TardisDataType.BOOK_SNAPSHOT_25,
    columns=book_snapshot_25_columns(),
    depth_limit=BOOK_SNAPSHOT_25_DEPTH,
)
TRADES_SCHEMA = TardisCSVSchema(
    data_type=TardisDataType.TRADES,
    columns=trades_columns(),
)

TARDIS_CSV_SCHEMAS = {
    TardisDataType.INCREMENTAL_BOOK_L2: INCREMENTAL_BOOK_L2_SCHEMA,
    TardisDataType.BOOK_SNAPSHOT_25: BOOK_SNAPSHOT_25_SCHEMA,
    TardisDataType.TRADES: TRADES_SCHEMA,
}


def tardis_csv_schema(data_type: TardisDataType | str) -> TardisCSVSchema:
    dtype = _coerce_data_type(data_type, "data_type")
    try:
        return TARDIS_CSV_SCHEMAS[dtype]
    except KeyError as exc:
        raise ValueError(f"no schema registered for data_type={dtype.value!r}") from exc


def supported_tardis_schema_types() -> tuple[TardisDataType, ...]:
    return tuple(TARDIS_CSV_SCHEMAS.keys())


__all__ = [
    "ColumnKind",
    "ColumnSpec",
    "TardisCSVSchema",
    "COMMON_TARDIS_COLUMNS",
    "BOOK_SNAPSHOT_25_DEPTH",
    "CURRENT_REQUIRED_BOOK_DEPTH_LEVELS",
    "CURRENT_MAX_REQUIRED_BOOK_DEPTH",
    "CURRENT_REQUIRED_BOOK_SNAPSHOT_DEPTH",
    "SUPPORTED_TARDIS_SCHEMA_TYPES",
    "book_snapshot_25_columns",
    "incremental_book_l2_columns",
    "trades_columns",
    "INCREMENTAL_BOOK_L2_SCHEMA",
    "BOOK_SNAPSHOT_25_SCHEMA",
    "TRADES_SCHEMA",
    "TARDIS_CSV_SCHEMAS",
    "tardis_csv_schema",
    "supported_tardis_schema_types",
]
