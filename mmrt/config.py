"""This module contains immutable default configuration objects for the new microsecond-native Tardis/Binance linear market-making pipeline. It is intentionally IO-free and does not read environment variables."""

from dataclasses import dataclass, field
from typing import Iterable

from mmrt.contracts import (
    AsOfPolicy,
    DecisionReason,
    LabelSpec,
    PriceReference,
    StorageFormat,
    TardisDataType,
    TimeUnit,
)
from mmrt.features.schedule import DecisionScheduleConfig
from mmrt.features.specs import FEATURE_SCHEMA as DEFAULT_FEATURE_SCHEMA

DEFAULT_EXCHANGE = "binance-futures"
DEFAULT_SYMBOL = "BTCUSDT"

DEFAULT_SOURCE_DATA_TYPES = (
    TardisDataType.INCREMENTAL_BOOK_L2,
    TardisDataType.TRADES,
)


DEFAULT_HORIZONS_US = (
    200_000,
    500_000,
    1_000_000,
)
DEFAULT_ENTRY_DELAY_US = 1_000

DEFAULT_LOOKBACK_ROWS = 10
DEFAULT_DECISION_REASON = DecisionReason.EVENT_SCHEDULE
DEFAULT_DECISION_POLICY = "event_schedule"

DEFAULT_PIPELINE_SCHEMA = "mmrt_pipeline_config"
DEFAULT_STORAGE_FORMAT = StorageFormat.FLAT_DECISION_ROWS_US
DEFAULT_TIME_UNIT = TimeUnit.MICROSECOND


DEFAULT_FEATURE_DTYPE = "float32"
DEFAULT_LABEL_DTYPE = "float32"
DEFAULT_TIMESTAMP_DTYPE = "int64"


def _require_nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value



def _tuple_of_unique_data_types(values: Iterable[TardisDataType | str], name: str) -> tuple[TardisDataType, ...]:
    seq = tuple(values)
    out: list[TardisDataType] = []
    for idx, value in enumerate(seq):
        try:
            out.append(TardisDataType(value))
        except ValueError as exc:
            raise ValueError(f"{name}[{idx}] has invalid data type {value!r}") from exc
    if len(set(out)) != len(out):
        raise ValueError(f"{name} must contain unique data types")
    return tuple(out)


def _tuple_of_positive_ints(values: Iterable[int], name: str, *, sort: bool = False) -> tuple[int, ...]:
    seq = tuple(values)
    out: list[int] = []
    for idx, value in enumerate(seq):
        out.append(_require_positive_int(value, f"{name}[{idx}]"))
    if len(set(out)) != len(out):
        raise ValueError(f"{name} must contain unique values")
    return tuple(sorted(out) if sort else out)


@dataclass(frozen=True, slots=True)
class MarketConfig:
    exchange: str = DEFAULT_EXCHANGE
    symbol: str = DEFAULT_SYMBOL

    def __post_init__(self) -> None:
        _require_nonempty_str(self.exchange, "exchange")
        _require_nonempty_str(self.symbol, "symbol")


@dataclass(frozen=True, slots=True)
class DataConfig:
    source_data_types: tuple[TardisDataType, ...] = DEFAULT_SOURCE_DATA_TYPES

    def __post_init__(self) -> None:
        source = _tuple_of_unique_data_types(self.source_data_types, "source_data_types")
        if set(source) != {TardisDataType.INCREMENTAL_BOOK_L2, TardisDataType.TRADES}:
            raise ValueError("source_data_types must be exactly INCREMENTAL_BOOK_L2 and TRADES")
        object.__setattr__(self, "source_data_types", source)


@dataclass(frozen=True, slots=True)
class DecisionConfig:
    policy: str = DEFAULT_DECISION_POLICY
    reason: DecisionReason = DEFAULT_DECISION_REASON
    schedule: DecisionScheduleConfig = field(default_factory=DecisionScheduleConfig)

    def __post_init__(self) -> None:
        if _require_nonempty_str(self.policy, "policy") != DEFAULT_DECISION_POLICY:
            raise ValueError("policy must be 'event_schedule'")
        try:
            reason = DecisionReason(self.reason)
        except ValueError as exc:
            raise ValueError("reason must be DecisionReason.EVENT_SCHEDULE") from exc
        if reason != DecisionReason.EVENT_SCHEDULE:
            raise ValueError("reason must be DecisionReason.EVENT_SCHEDULE")
        if not isinstance(self.schedule, DecisionScheduleConfig):
            raise ValueError("schedule must be DecisionScheduleConfig")
        object.__setattr__(self, "reason", reason)


@dataclass(frozen=True, slots=True)
class LabelConfig:
    horizons_us: tuple[int, ...] = DEFAULT_HORIZONS_US
    entry_delay_us: int = DEFAULT_ENTRY_DELAY_US
    price_reference: PriceReference = PriceReference.MID
    asof_policy: AsOfPolicy = AsOfPolicy.LAST_OBSERVATION

    def __post_init__(self) -> None:
        horizons = _tuple_of_positive_ints(self.horizons_us, "horizons_us", sort=True)
        if not horizons:
            raise ValueError("horizons_us must be non-empty")
        entry_delay = _require_nonnegative_int(self.entry_delay_us, "entry_delay_us")
        try:
            price_reference = PriceReference(self.price_reference)
        except ValueError as exc:
            raise ValueError("price_reference has invalid value") from exc
        if price_reference != PriceReference.MID:
            raise ValueError("price_reference must be PriceReference.MID")
        try:
            asof_policy = AsOfPolicy(self.asof_policy)
        except ValueError as exc:
            raise ValueError("asof_policy has invalid value") from exc
        if asof_policy != AsOfPolicy.LAST_OBSERVATION:
            raise ValueError("asof_policy must be AsOfPolicy.LAST_OBSERVATION")
        object.__setattr__(self, "horizons_us", horizons)
        object.__setattr__(self, "entry_delay_us", entry_delay)
        object.__setattr__(self, "price_reference", price_reference)
        object.__setattr__(self, "asof_policy", asof_policy)

    def to_label_spec(self) -> LabelSpec:
        return LabelSpec(
            horizons_us=self.horizons_us,
            entry_delay_us=self.entry_delay_us,
            price_reference=self.price_reference,
            asof_policy=self.asof_policy,
        )

    @property
    def label_context_us(self) -> int:
        return self.entry_delay_us + max(self.horizons_us)


@dataclass(frozen=True, slots=True)
class FeatureRuntimeConfig:
    lookback_rows: int = DEFAULT_LOOKBACK_ROWS
    feature_dtype: str = DEFAULT_FEATURE_DTYPE
    label_dtype: str = DEFAULT_LABEL_DTYPE
    timestamp_dtype: str = DEFAULT_TIMESTAMP_DTYPE

    def __post_init__(self) -> None:
        _require_positive_int(self.lookback_rows, "lookback_rows")
        if self.feature_dtype != "float32":
            raise ValueError("feature_dtype must be 'float32'")
        if self.label_dtype != "float32":
            raise ValueError("label_dtype must be 'float32'")
        if self.timestamp_dtype != "int64":
            raise ValueError("timestamp_dtype must be 'int64'")


@dataclass(frozen=True, slots=True)
class StorageConfig:
    storage_format: StorageFormat = DEFAULT_STORAGE_FORMAT
    time_unit: TimeUnit = DEFAULT_TIME_UNIT
    pipeline_schema: str = DEFAULT_PIPELINE_SCHEMA
    feature_schema: str = DEFAULT_FEATURE_SCHEMA

    def __post_init__(self) -> None:
        try:
            storage_format = StorageFormat(self.storage_format)
        except ValueError as exc:
            raise ValueError("storage_format has invalid value") from exc
        if storage_format != StorageFormat.FLAT_DECISION_ROWS_US:
            raise ValueError("storage_format must be StorageFormat.FLAT_DECISION_ROWS_US")
        try:
            time_unit = TimeUnit(self.time_unit)
        except ValueError as exc:
            raise ValueError("time_unit has invalid value") from exc
        if time_unit != TimeUnit.MICROSECOND:
            raise ValueError("time_unit must be TimeUnit.MICROSECOND")
        _require_nonempty_str(self.pipeline_schema, "pipeline_schema")
        _require_nonempty_str(self.feature_schema, "feature_schema")
        object.__setattr__(self, "storage_format", storage_format)
        object.__setattr__(self, "time_unit", time_unit)


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    market: MarketConfig = field(default_factory=MarketConfig)
    data: DataConfig = field(default_factory=DataConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    runtime: FeatureRuntimeConfig = field(default_factory=FeatureRuntimeConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.market, MarketConfig):
            raise ValueError("market must be MarketConfig")
        if not isinstance(self.data, DataConfig):
            raise ValueError("data must be DataConfig")
        if not isinstance(self.decision, DecisionConfig):
            raise ValueError("decision must be DecisionConfig")
        if not isinstance(self.labels, LabelConfig):
            raise ValueError("labels must be LabelConfig")
        if not isinstance(self.runtime, FeatureRuntimeConfig):
            raise ValueError("runtime must be FeatureRuntimeConfig")
        if not isinstance(self.storage, StorageConfig):
            raise ValueError("storage must be StorageConfig")


    @property
    def label_spec(self) -> LabelSpec:
        return self.labels.to_label_spec()

    @property
    def source_data_type_values(self) -> tuple[str, ...]:
        return tuple(dtype.value for dtype in self.data.source_data_types)


def default_config() -> PipelineConfig:
    return PipelineConfig()


def default_label_spec() -> LabelSpec:
    return default_config().label_spec


__all__ = [
    "DEFAULT_EXCHANGE",
    "DEFAULT_SYMBOL",
    "DEFAULT_SOURCE_DATA_TYPES",
    "DEFAULT_HORIZONS_US",
    "DEFAULT_ENTRY_DELAY_US",
    "DEFAULT_LOOKBACK_ROWS",
    "DEFAULT_DECISION_POLICY",
    "DEFAULT_PIPELINE_SCHEMA",
    "DEFAULT_FEATURE_SCHEMA",
    "DEFAULT_FEATURE_DTYPE",
    "DEFAULT_LABEL_DTYPE",
    "DEFAULT_TIMESTAMP_DTYPE",
    "MarketConfig",
    "DataConfig",
    "DecisionConfig",
    "LabelConfig",
    "FeatureRuntimeConfig",
    "StorageConfig",
    "PipelineConfig",
    "default_config",
    "default_label_spec",
]
