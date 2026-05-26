"""This module contains declarative Tardis CSV and feature-vector schemas for the microsecond-native MMRT pipeline. It is intentionally IO-free: it validates schema metadata and header shapes but does not parse rows, convert timestamps, or compute features."""

from dataclasses import dataclass
from enum import Enum
import hashlib
import re
from typing import Iterable, Sequence

from mmrt.contracts import TardisDataType


COMMON_TARDIS_COLUMNS = (
    "exchange",
    "symbol",
    "timestamp",
    "local_timestamp",
)

BOOK_SNAPSHOT_25_DEPTH = 25
BOOK_SNAPSHOT_5_DEPTH = 5

CURRENT_REQUIRED_BOOK_DEPTH_LEVELS = (1, 3, 5, 10, 20)
CURRENT_MAX_REQUIRED_BOOK_DEPTH = 20
CURRENT_REQUIRED_BOOK_SNAPSHOT_DEPTH = 25

SUPPORTED_TARDIS_SCHEMA_TYPES = (
    TardisDataType.INCREMENTAL_BOOK_L2,
    TardisDataType.BOOK_SNAPSHOT_25,
    TardisDataType.BOOK_SNAPSHOT_5,
    TardisDataType.TRADES,
    TardisDataType.BOOK_TICKER,
    TardisDataType.DERIVATIVE_TICKER,
    TardisDataType.LIQUIDATIONS,
)

UNSUPPORTED_TARDIS_SCHEMA_TYPES_V1 = (
    TardisDataType.OPTIONS_CHAIN,
    TardisDataType.QUOTES,
)

_FEATURE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


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


def _validate_feature_name(name: str) -> str:
    _require_nonempty_str(name, "name")
    if _FEATURE_NAME_RE.fullmatch(name) is None:
        raise ValueError("feature name must match ^[a-z][a-z0-9_]*$")
    return name


def _stable_hash_strings(values: Sequence[str]) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(value.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


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
            if data_type in UNSUPPORTED_TARDIS_SCHEMA_TYPES_V1:
                raise ValueError(f"{data_type.value} schema is unsupported in v1")
            raise ValueError(f"unsupported data_type for schema registry: {data_type.value}")

        if data_type == TardisDataType.BOOK_SNAPSHOT_25:
            if self.depth_limit != 25:
                raise ValueError("depth_limit must be 25 for BOOK_SNAPSHOT_25")
        elif data_type == TardisDataType.BOOK_SNAPSHOT_5:
            if self.depth_limit != 5:
                raise ValueError("depth_limit must be 5 for BOOK_SNAPSHOT_5")
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


def book_snapshot_columns(depth: int) -> tuple[ColumnSpec, ...]:
    if depth not in (5, 25):
        raise ValueError("depth must be 5 or 25")
    cols = list(_common_columns())
    for i in range(depth):
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


def book_ticker_columns() -> tuple[ColumnSpec, ...]:
    return _common_columns() + (
        ColumnSpec("ask_amount", ColumnKind.FLOAT, nullable=True),
        ColumnSpec("ask_price", ColumnKind.FLOAT, nullable=True),
        ColumnSpec("bid_amount", ColumnKind.FLOAT, nullable=True),
        ColumnSpec("bid_price", ColumnKind.FLOAT, nullable=True),
    )


def derivative_ticker_columns() -> tuple[ColumnSpec, ...]:
    return _common_columns() + (
        ColumnSpec("funding_timestamp", ColumnKind.INT_US, nullable=True),
        ColumnSpec("funding_rate", ColumnKind.FLOAT, nullable=True),
        ColumnSpec("predicted_funding_rate", ColumnKind.FLOAT, nullable=True),
        ColumnSpec("open_interest", ColumnKind.FLOAT, nullable=True),
        ColumnSpec("last_price", ColumnKind.FLOAT, nullable=True),
        ColumnSpec("index_price", ColumnKind.FLOAT, nullable=True),
        ColumnSpec("mark_price", ColumnKind.FLOAT, nullable=True),
    )


def liquidations_columns() -> tuple[ColumnSpec, ...]:
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
    columns=book_snapshot_columns(25),
    depth_limit=25,
)
BOOK_SNAPSHOT_5_SCHEMA = TardisCSVSchema(
    data_type=TardisDataType.BOOK_SNAPSHOT_5,
    columns=book_snapshot_columns(5),
    depth_limit=5,
)
TRADES_SCHEMA = TardisCSVSchema(
    data_type=TardisDataType.TRADES,
    columns=trades_columns(),
)
BOOK_TICKER_SCHEMA = TardisCSVSchema(
    data_type=TardisDataType.BOOK_TICKER,
    columns=book_ticker_columns(),
)
DERIVATIVE_TICKER_SCHEMA = TardisCSVSchema(
    data_type=TardisDataType.DERIVATIVE_TICKER,
    columns=derivative_ticker_columns(),
)
LIQUIDATIONS_SCHEMA = TardisCSVSchema(
    data_type=TardisDataType.LIQUIDATIONS,
    columns=liquidations_columns(),
)

TARDIS_CSV_SCHEMAS = {
    TardisDataType.INCREMENTAL_BOOK_L2: INCREMENTAL_BOOK_L2_SCHEMA,
    TardisDataType.BOOK_SNAPSHOT_25: BOOK_SNAPSHOT_25_SCHEMA,
    TardisDataType.BOOK_SNAPSHOT_5: BOOK_SNAPSHOT_5_SCHEMA,
    TardisDataType.TRADES: TRADES_SCHEMA,
    TardisDataType.BOOK_TICKER: BOOK_TICKER_SCHEMA,
    TardisDataType.DERIVATIVE_TICKER: DERIVATIVE_TICKER_SCHEMA,
    TardisDataType.LIQUIDATIONS: LIQUIDATIONS_SCHEMA,
}


def tardis_csv_schema(data_type: TardisDataType | str) -> TardisCSVSchema:
    dtype = _coerce_data_type(data_type, "data_type")
    if dtype in UNSUPPORTED_TARDIS_SCHEMA_TYPES_V1:
        raise ValueError(f"{dtype.value} schema is explicitly unsupported in v1")
    try:
        return TARDIS_CSV_SCHEMAS[dtype]
    except KeyError as exc:
        raise ValueError(f"no schema registered for data_type={dtype.value!r}") from exc


def supported_tardis_schema_types() -> tuple[TardisDataType, ...]:
    return tuple(TARDIS_CSV_SCHEMAS.keys())


@dataclass(frozen=True, slots=True)
class FeatureField:
    name: str
    family: str
    dtype: str = "float32"
    unit: str = "unitless"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_feature_name(self.name))
        object.__setattr__(self, "family", _require_nonempty_str(self.family, "family"))
        if self.dtype != "float32":
            raise ValueError("dtype must be 'float32' in v1")
        object.__setattr__(self, "unit", _require_nonempty_str(self.unit, "unit"))


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    version: str
    fields: tuple[FeatureField, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _require_nonempty_str(self.version, "version"))
        fields = tuple(self.fields)
        if not fields:
            raise ValueError("fields must be non-empty")
        for idx, field in enumerate(fields):
            if not isinstance(field, FeatureField):
                raise ValueError(f"fields[{idx}] must be FeatureField")
        names = tuple(field.name for field in fields)
        if len(set(names)) != len(names):
            raise ValueError("feature names must be unique")
        object.__setattr__(self, "fields", fields)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    @property
    def dim(self) -> int:
        return len(self.fields)

    @property
    def names_hash(self) -> str:
        return feature_names_hash(self.names)

    def index(self, name: str) -> int:
        target = _validate_feature_name(name)
        try:
            return self.names.index(target)
        except ValueError as exc:
            raise ValueError(f"feature name {target!r} not found") from exc

    def select_by_family(self, family: str) -> tuple[FeatureField, ...]:
        target = _require_nonempty_str(family, "family")
        return tuple(field for field in self.fields if field.family == target)


def feature_names_hash(names: Iterable[str]) -> str:
    normalized = tuple(_validate_feature_name(name) for name in tuple(names))
    if not normalized:
        raise ValueError("names must be non-empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("names must be unique")
    return _stable_hash_strings(normalized)


DECISION_ROW_FIXED_COLUMNS = (
    "ts_us",
    "local_ts_us",
    "source_row",
    "raw_mid",
    "dt_us",
)

LABEL_ROW_FIXED_COLUMNS = (
    "decision_ts_us",
    "entry_ts_us",
)

__all__ = [
    "ColumnKind",
    "ColumnSpec",
    "TardisCSVSchema",
    "COMMON_TARDIS_COLUMNS",
    "BOOK_SNAPSHOT_25_DEPTH",
    "BOOK_SNAPSHOT_5_DEPTH",
    "CURRENT_REQUIRED_BOOK_DEPTH_LEVELS",
    "CURRENT_MAX_REQUIRED_BOOK_DEPTH",
    "CURRENT_REQUIRED_BOOK_SNAPSHOT_DEPTH",
    "SUPPORTED_TARDIS_SCHEMA_TYPES",
    "UNSUPPORTED_TARDIS_SCHEMA_TYPES_V1",
    "book_snapshot_columns",
    "incremental_book_l2_columns",
    "trades_columns",
    "book_ticker_columns",
    "derivative_ticker_columns",
    "liquidations_columns",
    "INCREMENTAL_BOOK_L2_SCHEMA",
    "BOOK_SNAPSHOT_25_SCHEMA",
    "BOOK_SNAPSHOT_5_SCHEMA",
    "TRADES_SCHEMA",
    "BOOK_TICKER_SCHEMA",
    "DERIVATIVE_TICKER_SCHEMA",
    "LIQUIDATIONS_SCHEMA",
    "TARDIS_CSV_SCHEMAS",
    "tardis_csv_schema",
    "supported_tardis_schema_types",
    "FeatureField",
    "FeatureSchema",
    "feature_names_hash",
    "DECISION_ROW_FIXED_COLUMNS",
    "LABEL_ROW_FIXED_COLUMNS",
]
