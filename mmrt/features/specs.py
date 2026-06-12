"""Feature registry for the microsecond-native MMRT feature pipeline.

This module defines the canonical feature names, ordering, ownership, and metadata
for the new Tardis/Binance linear pipeline. It does not compute features, transform
features, build labels, read market data, or import heavy numeric/data libraries.
"""

from dataclasses import dataclass
from enum import Enum
import hashlib
import re
from typing import Iterable, Mapping, Sequence

FEATURE_SCHEMA = "mmrt_features_tape25_trades_active44_ctx4_corr90"

CORE_FEATURE_COUNT = 44
EVENT_CONTEXT_FEATURE_COUNT = 4
FEATURE_COUNT = 48

REQUIRED_BOOK_SNAPSHOT_DEPTH = 25
MAX_REQUIRED_BOOK_FEATURE_DEPTH = 20

SUPPORTED_WINDOWS_US = (
    100_000,
    200_000,
    500_000,
    1_000_000,
    3_000_000,
)

DEFAULT_FEATURE_DTYPE = "float32"

CORE_FEATURE_NAMES = (
    "mid_slope_bps_per_sec_1000000us",
    "time_since_mid_change_us",
    "bid_l1_notional_usd",
    "ask_l1_notional_usd",
    "total_depth_notional_5bps",
    "obi_l1",
    "ofi_l10_sum_over_depth_1000000us",
    "micro_l10_minus_mid_bps",
    "ask_depth_within_1bps",
    "depth_imbalance_within_1bps",
    "ask_l1_depletion_over_depth_200000us",
    "ask_l1_depletion_500000us",
    "bid_price_change_rate_1000000us",
    "bid_l1_depletion_1000000us",
    "bid_l1_depletion_over_depth_1000000us",
    "ask_l1_depletion_over_depth_1000000us",
    "ob_update_rate_200000us",
    "ob_update_rate_500000us",
    "bid_l1_rem_rate_over_depth_200000us",
    "trade_count_per_second_200000us",
    "trade_imbalance_notional_500000us",
    "trade_count_per_second_500000us",
    "zero_tick_fraction_1000000us",
    "trade_count_per_second_1000000us",
    "time_since_last_buy_trade_us",
    "time_since_last_sell_trade_us",
    "max_signed_trade_notional_usd_1000000us",
    "absorption_bid_1000000us",
    "absorption_ask_1000000us",
    "depth_imbalance_5bps_slope_1000000us",
    "depth_imbalance_5bps_slope_3000000us",
    "ofi_l1_pressure_over_realized_vol_1000000us",
    "microprice_zero_cross_rate_1000000us",
    "l1_churn_over_depth_1000000us",
    "same_side_trade_cluster_notional_1000000us",
    "touch_flicker_score_3000000us",
    "spread_state_transition_rate_3000000us",
    "max_trade_silence_gap_3000000us",
    "microprice_realized_vol_1000000us",
    "trade_sign_entropy_3000000us",
    "best_bid_size_age_us",
    "best_ask_size_age_us",
    "trade_side_quote_response_asymmetry_500000us",
    "near_touch_depth_drop_asymmetry",
)
assert len(CORE_FEATURE_NAMES) == 44
assert len(set(CORE_FEATURE_NAMES)) == 44

EVENT_CONTEXT_FEATURE_NAMES = (
    "log_events_200000us",
    "log_events_500000us",
    "log_events_1000000us",
    "log_events_3000000us",
)
assert len(EVENT_CONTEXT_FEATURE_NAMES) == 4
assert not set(EVENT_CONTEXT_FEATURE_NAMES).intersection(CORE_FEATURE_NAMES)


class FeatureSource(str, Enum):
    BOOK = "book"
    TRADE = "trade"
    CROSS = "cross"
    EVENT_CONTEXT = "event_context"


class FeatureOwner(str, Enum):
    BOOK_STATE = "book_state"
    TRADE_STATE = "trade_state"
    ENGINE = "engine"


class FeatureFamily(str, Enum):
    PRICE = "price"
    BOOK_LEVEL = "book_level"
    DEPTH = "depth"
    OFI_OBI = "ofi_obi"
    BOOK_DYNAMICS = "book_dynamics"
    TRADE_FLOW = "trade_flow"
    CVD = "cvd"
    LARGE_TRADE = "large_trade"
    ABSORPTION = "absorption"
    REGIME = "regime"
    CROSS_SIGNAL = "cross_signal"
    EVENT_CONTEXT = "event_context"


class FeatureUnit(str, Enum):
    BPS = "bps"
    USD = "usd"
    NOTIONAL_USD = "notional_usd"
    COUNT = "count"
    RATIO = "ratio"
    RATE_PER_SECOND = "rate_per_second"
    MICROSECONDS = "microseconds"
    LOG1P = "log1p"
    SIGN = "sign"
    SCORE = "score"


class TransformKey(str, Enum):
    IDENTITY_EWMA_FAST = "identity_ewma_fast"
    IDENTITY_EWMA_MEDIUM = "identity_ewma_medium"
    IDENTITY_EWMA_SLOW = "identity_ewma_slow"
    IDENTITY_NO_EWMA = "identity_no_ewma"
    LOG1P_POS_NO_EWMA = "log1p_pos_no_ewma"
    LOG1P_POS_EWMA_FAST = "log1p_pos_ewma_fast"
    LOG1P_POS_EWMA_MEDIUM = "log1p_pos_ewma_medium"
    LOG1P_POS_EWMA_SLOW = "log1p_pos_ewma_slow"
    SIGNED_LOG1P_EWMA_FAST = "signed_log1p_ewma_fast"
    SIGNED_LOG1P_EWMA_MEDIUM = "signed_log1p_ewma_medium"
    SIGNED_LOG1P_EWMA_SLOW = "signed_log1p_ewma_slow"
    RATIO_BOUNDED = "ratio_bounded"
    SIGN_NO_EWMA = "sign_no_ewma"
    TIME_LOG1P_NO_EWMA = "time_log1p_no_ewma"


def _require_nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty str")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be non-negative int")
    return value


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be positive int")
    return value


def _coerce_enum(enum_cls: type[Enum], value: object, name: str):
    try:
        return value if isinstance(value, enum_cls) else enum_cls(value)
    except Exception as exc:
        raise ValueError(f"{name} must be {enum_cls.__name__}") from exc


def _tuple_positive_ints(values: Iterable[int], name: str) -> tuple[int, ...]:
    out = tuple(_require_positive_int(v, name) for v in values)
    return out


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    index: int
    name: str
    source: FeatureSource
    owner: FeatureOwner
    family: FeatureFamily
    unit: FeatureUnit
    transform_key: TransformKey
    windows_us: tuple[int, ...] = ()
    required_book_depth: int = 0
    formula_group: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _require_nonnegative_int(self.index, "index"))
        object.__setattr__(self, "name", _require_nonempty_str(self.name, "name"))
        if "ms" in self.name:
            raise ValueError("name must not contain ms")
        object.__setattr__(self, "source", _coerce_enum(FeatureSource, self.source, "source"))
        object.__setattr__(self, "owner", _coerce_enum(FeatureOwner, self.owner, "owner"))
        object.__setattr__(self, "family", _coerce_enum(FeatureFamily, self.family, "family"))
        object.__setattr__(self, "unit", _coerce_enum(FeatureUnit, self.unit, "unit"))
        object.__setattr__(self, "transform_key", _coerce_enum(TransformKey, self.transform_key, "transform_key"))
        windows_us = _tuple_positive_ints(self.windows_us, "windows_us")
        for win in windows_us:
            if win not in SUPPORTED_WINDOWS_US:
                raise ValueError("unsupported window")
        object.__setattr__(self, "windows_us", windows_us)
        required_book_depth = _require_nonnegative_int(self.required_book_depth, "required_book_depth")
        if required_book_depth > REQUIRED_BOOK_SNAPSHOT_DEPTH:
            raise ValueError("required_book_depth exceeds snapshot depth")
        object.__setattr__(self, "required_book_depth", required_book_depth)
        object.__setattr__(self, "formula_group", _require_nonempty_str(self.formula_group, "formula_group"))
        if self.source == FeatureSource.EVENT_CONTEXT and (
            self.owner != FeatureOwner.ENGINE or self.family != FeatureFamily.EVENT_CONTEXT
        ):
            raise ValueError("event context must be engine/event_context")
        if self.source == FeatureSource.TRADE and self.owner != FeatureOwner.TRADE_STATE:
            raise ValueError("trade source must be trade_state owner")
        if self.source == FeatureSource.BOOK and self.owner != FeatureOwner.BOOK_STATE:
            raise ValueError("book source must be book_state owner")
        if self.source == FeatureSource.CROSS and self.owner != FeatureOwner.ENGINE:
            raise ValueError("cross source must be engine owner")


_US_WINDOW_RE = re.compile(r"(?P<num>\d+)us")
_ACCEL_US_RE = re.compile(r"(?P<fast>\d+)us_minus_(?P<slow>\d+)us")


def _require_known_feature_name(name: str) -> str:
    name = _require_nonempty_str(name, "name")
    if name not in CORE_FEATURE_NAMES and name not in EVENT_CONTEXT_FEATURE_NAMES:
        raise ValueError(f"unknown active feature name: {name}")
    return name

def infer_windows_us_from_name(name: str) -> tuple[int, ...]:
    name = _require_known_feature_name(name)
    if "ms" in name:
        raise ValueError("name must not contain ms")
    match = _ACCEL_US_RE.search(name)
    wins: list[int] = []
    if match:
        wins.extend((int(match.group("fast")), int(match.group("slow"))))
    else:
        wins.extend(int(m.group("num")) for m in _US_WINDOW_RE.finditer(name))
    dedup: list[int] = []
    seen: set[int] = set()
    for win in wins:
        if win not in seen:
            if win not in SUPPORTED_WINDOWS_US:
                raise ValueError("unsupported inferred window")
            seen.add(win)
            dedup.append(win)
    return tuple(dedup)


def infer_source(name: str) -> FeatureSource:
    name = _require_known_feature_name(name)
    if name in EVENT_CONTEXT_FEATURE_NAMES:
        return FeatureSource.EVENT_CONTEXT
    cross_exact_names = {
        "trade_side_quote_response_asymmetry_500000us",
    }
    if name in cross_exact_names:
        return FeatureSource.CROSS
    cross_prefixes = (
        "absorption_",
        "ofi_l1_pressure_",
        "post_buy_trade_",
        "post_sell_trade_",
        "opposite_side_replenishment_",
        "same_side_replenishment_",
    )
    if name.startswith(cross_prefixes):
        return FeatureSource.CROSS
    trade_prefixes = (
        "signed_",
        "trade_",
        "zero_tick_",
        "tick_sign_",
        "last_trade_",
        "last_tick_",
        "time_since_last_buy_trade",
        "time_since_last_sell_trade",
        "cvd_",
        "consecutive_buy_trade",
        "consecutive_sell_trade",
        "top5_trade",
        "max_signed_trade",
        "buy_trade_",
        "sell_trade_",
    )
    if name.startswith(trade_prefixes) or name in {
        "max_trade_silence_gap_3000000us",
        "trade_sign_entropy_3000000us",
        "same_side_trade_cluster_notional_1000000us",
    }:
        return FeatureSource.TRADE
    return FeatureSource.BOOK


def infer_owner(source: FeatureSource) -> FeatureOwner:
    return {
        FeatureSource.BOOK: FeatureOwner.BOOK_STATE,
        FeatureSource.TRADE: FeatureOwner.TRADE_STATE,
        FeatureSource.CROSS: FeatureOwner.ENGINE,
        FeatureSource.EVENT_CONTEXT: FeatureOwner.ENGINE,
    }[source]


def infer_family(name: str, source: FeatureSource) -> FeatureFamily:
    name = _require_known_feature_name(name)
    if source == FeatureSource.EVENT_CONTEXT:
        return FeatureFamily.EVENT_CONTEXT
    if name.startswith("micro_ret") or name.startswith("mid_") or "microprice" in name:
        return FeatureFamily.PRICE
    if name.startswith("obi_") or name.startswith("ofi_") or "_ofi_" in name:
        return FeatureFamily.OFI_OBI
    if "depth" in name or "liquidity_void" in name or "centroid" in name:
        return FeatureFamily.DEPTH
    if name.startswith(("bid_l1", "ask_l1")) or any(k in name for k in ("depletion", "replenishment", "churn", "flicker", "spread_state_transition", "ob_arrival")):
        return FeatureFamily.BOOK_DYNAMICS
    if name.startswith("cvd_"):
        return FeatureFamily.CVD
    if any(k in name for k in ("top5_trade", "max_signed_trade", "p90_over_median")):
        return FeatureFamily.LARGE_TRADE
    if name.startswith("absorption_"):
        return FeatureFamily.ABSORPTION
    if "realized_vol" in name:
        return FeatureFamily.REGIME
    if source == FeatureSource.TRADE:
        return FeatureFamily.TRADE_FLOW
    if source == FeatureSource.CROSS:
        return FeatureFamily.CROSS_SIGNAL
    return FeatureFamily.BOOK_LEVEL


def infer_unit(name: str) -> FeatureUnit:
    name = _require_known_feature_name(name)
    if name.startswith("log_"):
        return FeatureUnit.LOG1P
    if "time_since" in name or "age_us" in name or "silence_gap" in name:
        return FeatureUnit.MICROSECONDS
    if "bps" in name or "realized_vol" in name:
        return FeatureUnit.BPS
    if "usd" in name or "notional" in name:
        return FeatureUnit.NOTIONAL_USD
    if any(k in name for k in ("imbalance", "fraction", "share", "entropy", "asymmetry", "proxy", "score")):
        return FeatureUnit.RATIO
    if "count" in name or "run_length" in name:
        return FeatureUnit.COUNT
    if "per_second" in name or "rate" in name or "slope" in name:
        return FeatureUnit.RATE_PER_SECOND
    if "sign" in name or name in {"last_trade_side_sign", "last_tick_sign"}:
        return FeatureUnit.SIGN
    return FeatureUnit.SCORE


def infer_required_book_depth(name: str, source: FeatureSource) -> int:
    name = _require_known_feature_name(name)
    if source == FeatureSource.EVENT_CONTEXT:
        return 0
    if source == FeatureSource.TRADE and not any(k in name for k in ("depth", "book", "l1", "l3", "l5", "l10", "vamp", "micro_l", "replenishment")):
        return 0
    if source == FeatureSource.CROSS:
        return 20
    if "depth_5bps" in name or any(k in name for k in ("within_1bps", "centroid", "liquidity_void")):
        return 20
    if "_l10" in name or "l10_" in name:
        return 10
    if any(k in name for k in ("_l5", "l5_", "top5")):
        return 5
    if "_l3" in name or "l3_" in name:
        return 3
    if "_l1" in name or "l1_" in name:
        return 1
    if any(k in name for k in ("depth", "vamp", "micro_l", "liquidity_void", "centroid")):
        return 20
    return 1 if source == FeatureSource.BOOK else 0


def _has_window_at_most(name: str, max_window_us: int) -> bool:
    windows_us = infer_windows_us_from_name(name)
    return bool(windows_us) and min(windows_us) <= max_window_us


def _has_window_at_least(name: str, min_window_us: int) -> bool:
    windows_us = infer_windows_us_from_name(name)
    return bool(windows_us) and max(windows_us) >= min_window_us


def _is_ratio_or_bounded_feature(name: str, unit: FeatureUnit) -> bool:
    if unit == FeatureUnit.RATIO:
        return True
    bounded_terms = (
        "imbalance",
        "fraction",
        "share",
        "entropy",
        "asymmetry",
        "proxy",
        "score",
        "over_depth",
    )
    return any(term in name for term in bounded_terms)


def _is_slow_regime_feature(name: str) -> bool:
    return name == "microprice_realized_vol_1000000us"


def _is_fast_microstructure_feature(name: str, source: FeatureSource, family: FeatureFamily) -> bool:
    if _is_slow_regime_feature(name) or not _has_window_at_most(name, 1_000_000):
        return name in {"ofi_l3", "obi_l1"}
    return source in {FeatureSource.BOOK, FeatureSource.TRADE, FeatureSource.CROSS} and family in {
        FeatureFamily.PRICE,
        FeatureFamily.OFI_OBI,
        FeatureFamily.BOOK_DYNAMICS,
        FeatureFamily.TRADE_FLOW,
        FeatureFamily.ABSORPTION,
        FeatureFamily.CROSS_SIGNAL,
    }


def _signed_log_ewma_key(name: str, source: FeatureSource, family: FeatureFamily) -> TransformKey:
    if _is_slow_regime_feature(name):
        return TransformKey.SIGNED_LOG1P_EWMA_SLOW
    if _is_fast_microstructure_feature(name, source, family) and not name.startswith("cvd_"):
        return TransformKey.SIGNED_LOG1P_EWMA_FAST
    return TransformKey.SIGNED_LOG1P_EWMA_MEDIUM


def _positive_log_ewma_key(name: str, source: FeatureSource, family: FeatureFamily) -> TransformKey:
    if _is_slow_regime_feature(name):
        return TransformKey.LOG1P_POS_EWMA_SLOW
    if _is_fast_microstructure_feature(name, source, family) and not any(
        term in name for term in ("depth_notional", "depth_within", "top5_trade_notional", "max_trade_silence_gap", "p90_over_median")
    ):
        return TransformKey.LOG1P_POS_EWMA_FAST
    return TransformKey.LOG1P_POS_EWMA_MEDIUM


def infer_transform_key(name: str, unit: FeatureUnit, source: FeatureSource, family: FeatureFamily) -> TransformKey:
    name = _require_known_feature_name(name)
    if source == FeatureSource.EVENT_CONTEXT:
        return TransformKey.LOG1P_POS_NO_EWMA
    if unit == FeatureUnit.MICROSECONDS:
        return TransformKey.TIME_LOG1P_NO_EWMA
    if unit == FeatureUnit.SIGN:
        return TransformKey.SIGN_NO_EWMA
    if _is_ratio_or_bounded_feature(name, unit):
        return TransformKey.RATIO_BOUNDED

    signed_heavy_tailed = (
        name.startswith("signed_notional_flow_usd_")
        or name.startswith("cvd_")
        or name.startswith("max_signed_trade_notional_usd_")
    )
    if signed_heavy_tailed:
        return _signed_log_ewma_key(name, source, family)

    positive_heavy_tailed = (
"notional" in name
        or "usd" in name
        or unit in {FeatureUnit.COUNT, FeatureUnit.RATE_PER_SECOND}
    )
    if positive_heavy_tailed:
        return _positive_log_ewma_key(name, source, family)

    if unit == FeatureUnit.BPS:
        if _is_slow_regime_feature(name):
            return TransformKey.IDENTITY_EWMA_SLOW
        return TransformKey.IDENTITY_EWMA_FAST

    if _is_slow_regime_feature(name):
        return TransformKey.IDENTITY_EWMA_SLOW
    if _is_fast_microstructure_feature(name, source, family):
        return TransformKey.IDENTITY_EWMA_FAST
    if _has_window_at_least(name, 3_000_000):
        return TransformKey.IDENTITY_EWMA_MEDIUM
    return TransformKey.IDENTITY_EWMA_MEDIUM

def infer_formula_group(name: str, source: FeatureSource, family: FeatureFamily) -> str:
    name = _require_known_feature_name(name)
    if source == FeatureSource.EVENT_CONTEXT:
        return "event_context"
    if family == FeatureFamily.PRICE:
        return "price_history"
    if family == FeatureFamily.DEPTH:
        return "depth_curve"
    if family == FeatureFamily.OFI_OBI:
        return "ofi_obi"
    if family == FeatureFamily.BOOK_DYNAMICS:
        return "book_dynamics"
    if family == FeatureFamily.CVD:
        return "cvd"
    if family == FeatureFamily.LARGE_TRADE:
        return "large_trade"
    if family == FeatureFamily.ABSORPTION:
        return "absorption"
    if family == FeatureFamily.REGIME:
        return "regime"
    if source == FeatureSource.CROSS:
        return "cross_signal"
    if source == FeatureSource.TRADE:
        return "trade_window"
    return "book_dynamics"


def _build_feature_specs() -> tuple[FeatureSpec, ...]:
    feature_names = CORE_FEATURE_NAMES + EVENT_CONTEXT_FEATURE_NAMES
    assert len(feature_names) == FEATURE_COUNT
    assert len(set(feature_names)) == FEATURE_COUNT

    out: list[FeatureSpec] = []
    for index, name in enumerate(feature_names):
        source = infer_source(name)
        owner = infer_owner(source)
        family = infer_family(name, source)
        unit = infer_unit(name)
        transform_key = infer_transform_key(name, unit, source, family)
        windows_us = infer_windows_us_from_name(name)
        required_book_depth = infer_required_book_depth(name, source)
        formula_group = infer_formula_group(name, source, family)
        out.append(
            FeatureSpec(
                index=index,
                name=name,
                source=source,
                owner=owner,
                family=family,
                unit=unit,
                transform_key=transform_key,
                windows_us=windows_us,
                required_book_depth=required_book_depth,
                formula_group=formula_group,
            )
        )
    return tuple(out)


FEATURE_SPECS = _build_feature_specs()
FEATURE_NAMES = tuple(spec.name for spec in FEATURE_SPECS)
FEATURE_NAME_TO_INDEX = {spec.name: spec.index for spec in FEATURE_SPECS}

assert len(FEATURE_SPECS) == FEATURE_COUNT
assert FEATURE_NAMES[-4:] == (
    "log_events_200000us",
    "log_events_500000us",
    "log_events_1000000us",
    "log_events_3000000us",
)
assert max(spec.required_book_depth for spec in FEATURE_SPECS) == MAX_REQUIRED_BOOK_FEATURE_DEPTH




def feature_names_hash(names: Sequence[str] = FEATURE_NAMES) -> str:
    if not names or len(set(names)) != len(names):
        raise ValueError("names must be non-empty and unique")
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()[:12]


FEATURE_NAMES_HASH = feature_names_hash(FEATURE_NAMES)


def feature_specs_hash(specs: Sequence[FeatureSpec] = FEATURE_SPECS) -> str:
    rows = []
    for spec in specs:
        windows_us_csv = ",".join(str(v) for v in spec.windows_us)
        rows.append(
            f"{spec.index}|{spec.name}|{spec.source.value}|{spec.owner.value}|{spec.family.value}|"
            f"{spec.unit.value}|{spec.transform_key.value}|{windows_us_csv}|"
            f"{spec.required_book_depth}|{spec.formula_group}"
        )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()[:12]


FEATURE_SPECS_HASH = feature_specs_hash(FEATURE_SPECS)

def all_feature_names() -> tuple[str, ...]:
    return FEATURE_NAMES


def feature_count() -> int:
    return FEATURE_COUNT

def feature_specs() -> tuple[FeatureSpec, ...]:
    return FEATURE_SPECS

def feature_spec_by_name(name: str) -> FeatureSpec:
    return FEATURE_SPECS[FEATURE_NAME_TO_INDEX[name]]

def feature_index(name: str) -> int:
    return FEATURE_NAME_TO_INDEX[name]

def feature_name(index: int) -> str:
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("index must be int")
    if index < 0 or index >= FEATURE_COUNT:
        raise IndexError("feature index out of range")
    return FEATURE_NAMES[index]


def feature_indices_by_source(source: FeatureSource | str) -> tuple[int, ...]:
    source_enum = _coerce_enum(FeatureSource, source, "source")
    return tuple(spec.index for spec in FEATURE_SPECS if spec.source == source_enum)

def feature_indices_by_owner(owner: FeatureOwner | str) -> tuple[int, ...]:
    owner_enum = _coerce_enum(FeatureOwner, owner, "owner")
    return tuple(spec.index for spec in FEATURE_SPECS if spec.owner == owner_enum)

def feature_indices_by_family(family: FeatureFamily | str) -> tuple[int, ...]:
    family_enum = _coerce_enum(FeatureFamily, family, "family")
    return tuple(spec.index for spec in FEATURE_SPECS if spec.family == family_enum)

def required_windows_us() -> tuple[int, ...]:
    return tuple(sorted({win for spec in FEATURE_SPECS for win in spec.windows_us}))

def max_required_book_depth() -> int:
    return max(spec.required_book_depth for spec in FEATURE_SPECS)

assert max_required_book_depth() <= REQUIRED_BOOK_SNAPSHOT_DEPTH

def schema_record() -> Mapping[str, object]:
    return {
      "schema": FEATURE_SCHEMA,
      "feature_count": FEATURE_COUNT,
      "feature_names_hash": FEATURE_NAMES_HASH,
      "feature_specs_hash": FEATURE_SPECS_HASH,
      "feature_dtype": DEFAULT_FEATURE_DTYPE,
      "time_unit": "us",
      "required_book_snapshot_depth": REQUIRED_BOOK_SNAPSHOT_DEPTH,
      "max_required_book_feature_depth": MAX_REQUIRED_BOOK_FEATURE_DEPTH,
      "source_counts": {s.value: len(feature_indices_by_source(s)) for s in FeatureSource},
      "owner_counts": {o.value: len(feature_indices_by_owner(o)) for o in FeatureOwner},
      "family_counts": {f.value: len(feature_indices_by_family(f)) for f in FeatureFamily},
    }

__all__ = [
    "FEATURE_SCHEMA", "CORE_FEATURE_COUNT", "EVENT_CONTEXT_FEATURE_COUNT", "FEATURE_COUNT",
    "REQUIRED_BOOK_SNAPSHOT_DEPTH", "MAX_REQUIRED_BOOK_FEATURE_DEPTH", "SUPPORTED_WINDOWS_US",
    "DEFAULT_FEATURE_DTYPE", "CORE_FEATURE_NAMES", "EVENT_CONTEXT_FEATURE_NAMES", "FeatureSource",
    "FeatureOwner", "FeatureFamily", "FeatureUnit", "TransformKey", "FeatureSpec", "FEATURE_SPECS",
    "FEATURE_NAMES", "FEATURE_NAME_TO_INDEX", "FEATURE_NAMES_HASH", "FEATURE_SPECS_HASH",
    "infer_windows_us_from_name", "all_feature_names", "feature_count", "feature_specs",
    "feature_spec_by_name", "feature_index", "feature_name", "feature_indices_by_source",
    "feature_indices_by_owner", "feature_indices_by_family", "required_windows_us", "max_required_book_depth",
    "schema_record",
]
