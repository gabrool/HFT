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

FEATURE_SCHEMA_VERSION = "mmrt_feature_schema_v3_snapshot25_trades_active44_ctx4_feature_subset_corr90"

CORE_FEATURE_COUNT = 44
EVENT_CONTEXT_FEATURE_COUNT = 4
FEATURE_COUNT = 48

REQUIRED_TARDIS_BOOK_SNAPSHOT_DEPTH = 25
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
    "mid_slope_bps_per_sec_1000ms",
    "time_since_mid_change_ms",
    "bid_l1_notional_usd",
    "ask_l1_notional_usd",
    "total_depth_notional_5bps",
    "obi_l1",
    "ofi_l10_sum_over_depth_1000ms",
    "micro_l10_minus_mid_bps",
    "ask_depth_within_1bps",
    "depth_imbalance_within_1bps",
    "ask_l1_depletion_over_depth_200ms",
    "ask_l1_depletion_500ms",
    "bid_price_change_rate_1000ms",
    "bid_l1_depletion_1000ms",
    "bid_l1_depletion_over_depth_1000ms",
    "ask_l1_depletion_over_depth_1000ms",
    "ob_update_rate_200ms",
    "ob_update_rate_500ms",
    "bid_l1_rem_rate_over_depth_200ms",
    "trade_count_per_second_200ms",
    "trade_imbalance_notional_500ms",
    "trade_count_per_second_500ms",
    "zero_tick_fraction_1000ms",
    "trade_count_per_second_1000ms",
    "time_since_last_buy_trade_ms",
    "time_since_last_sell_trade_ms",
    "max_signed_trade_notional_usd_1000ms",
    "absorption_bid_1000ms",
    "absorption_ask_1000ms",
    "depth_imbalance_5bps_slope_1000ms",
    "depth_imbalance_5bps_slope_3000ms",
    "ofi_l1_pressure_over_realized_vol_1000ms",
    "microprice_zero_cross_rate_1000ms",
    "l1_churn_over_depth_1000ms",
    "same_side_trade_cluster_notional_1000ms",
    "touch_flicker_score_3000ms",
    "spread_state_transition_rate_3000ms",
    "max_trade_silence_gap_3000ms",
    "microprice_realized_vol_1000ms",
    "trade_sign_entropy_3000ms",
    "best_bid_size_age_ms",
    "best_ask_size_age_ms",
    "trade_side_quote_response_asymmetry_500ms",
    "near_touch_depth_drop_asymmetry",
)
assert len(CORE_FEATURE_NAMES) == 44
assert len(set(CORE_FEATURE_NAMES)) == 44

EVENT_CONTEXT_FEATURE_NAMES = (
    "log_events_200ms",
    "log_events_500ms",
    "log_events_1000ms",
    "log_events_3000ms",
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
    legacy_name: str
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
        object.__setattr__(self, "legacy_name", _require_nonempty_str(self.legacy_name, "legacy_name"))
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
        if required_book_depth > REQUIRED_TARDIS_BOOK_SNAPSHOT_DEPTH:
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


_MS_WINDOW_RE = re.compile(r"(?P<num>\d+)ms")
_ACCEL_RE = re.compile(r"(?P<fast>\d+)_minus_(?P<slow>\d+)ms")

def legacy_name_to_canonical_name(legacy_name: str) -> str:
    name = _require_nonempty_str(legacy_name, "legacy_name")
    name = _ACCEL_RE.sub(lambda m: f"{int(m.group('fast')) * 1000}us_minus_{int(m.group('slow')) * 1000}us", name)
    name = _MS_WINDOW_RE.sub(lambda m: f"{int(m.group('num')) * 1000}us", name)
    if name.endswith("_ms"):
        name = name[:-3] + "_us"
    assert "ms" not in name
    return name


def infer_windows_us_from_legacy_name(legacy_name: str) -> tuple[int, ...]:
    name = _require_nonempty_str(legacy_name, "legacy_name")
    match = _ACCEL_RE.search(name)
    wins: list[int] = []
    if match:
        wins.extend((int(match.group("fast")) * 1000, int(match.group("slow")) * 1000))
    else:
        wins.extend(int(m.group("num")) * 1000 for m in _MS_WINDOW_RE.finditer(name))
    dedup: list[int] = []
    seen: set[int] = set()
    for win in wins:
        if win not in seen:
            if win not in SUPPORTED_WINDOWS_US:
                raise ValueError("unsupported inferred window")
            seen.add(win)
            dedup.append(win)
    return tuple(dedup)


def infer_source(legacy_name: str) -> FeatureSource:
    if legacy_name in EVENT_CONTEXT_FEATURE_NAMES:
        return FeatureSource.EVENT_CONTEXT
    trade_exact_names = {
        "time_since_trade_ms",
        "regime_volume_ewma_500ms",
        "regime_volume_ewma_3000ms",
    }
    cross_exact_names = {
        "trade_side_quote_response_asymmetry_500ms",
        "trade_impact_half_life_proxy",
        "vwap_vs_mid_bps_200ms",
        "vwap_vs_mid_bps_500ms",
    }
    if legacy_name in trade_exact_names:
        return FeatureSource.TRADE
    if legacy_name in cross_exact_names:
        return FeatureSource.CROSS
    cross_prefixes = (
        "absorption_",
        "ofi_l1_pressure_",
        "post_buy_trade_",
        "post_sell_trade_",
        "opposite_side_replenishment_",
        "same_side_replenishment_",
    )
    if legacy_name.startswith(cross_prefixes):
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
    if legacy_name.startswith(trade_prefixes) or legacy_name in {
        "max_trade_silence_gap_3000ms",
        "trade_sign_entropy_3000ms",
        "same_side_trade_cluster_notional_1000ms",
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


def infer_family(legacy_name: str, source: FeatureSource) -> FeatureFamily:
    if source == FeatureSource.EVENT_CONTEXT:
        return FeatureFamily.EVENT_CONTEXT
    if legacy_name in {"vwap_vs_mid_bps_200ms", "vwap_vs_mid_bps_500ms"}:
        return FeatureFamily.CROSS_SIGNAL
    if legacy_name in {"regime_volume_ewma_500ms", "regime_volume_ewma_3000ms"}:
        return FeatureFamily.REGIME
    if legacy_name == "time_since_trade_ms":
        return FeatureFamily.TRADE_FLOW
    if legacy_name.startswith("micro_ret") or legacy_name.startswith("mid_") or "microprice" in legacy_name:
        return FeatureFamily.PRICE
    if legacy_name.startswith("obi_") or legacy_name.startswith("ofi_") or "_ofi_" in legacy_name:
        return FeatureFamily.OFI_OBI
    if "depth" in legacy_name or "liquidity_void" in legacy_name or "centroid" in legacy_name:
        return FeatureFamily.DEPTH
    if legacy_name.startswith(("bid_l1", "ask_l1")) or any(k in legacy_name for k in ("depletion", "replenishment", "churn", "flicker", "spread_state_transition", "ob_arrival")):
        return FeatureFamily.BOOK_DYNAMICS
    if legacy_name.startswith("cvd_"):
        return FeatureFamily.CVD
    if any(k in legacy_name for k in ("top5_trade", "max_signed_trade", "p90_over_median")):
        return FeatureFamily.LARGE_TRADE
    if legacy_name.startswith("absorption_"):
        return FeatureFamily.ABSORPTION
    if any(k in legacy_name for k in ("regime", "down_up_vol", "return_std", "realized_vol", "max_abs_return")) or legacy_name.startswith("spread_z"):
        return FeatureFamily.REGIME
    if source == FeatureSource.TRADE:
        return FeatureFamily.TRADE_FLOW
    if source == FeatureSource.CROSS:
        return FeatureFamily.CROSS_SIGNAL
    return FeatureFamily.BOOK_LEVEL


def infer_unit(legacy_name: str) -> FeatureUnit:
    if legacy_name.startswith("log_"):
        return FeatureUnit.LOG1P
    if legacy_name.endswith("_ms") or "time_since" in legacy_name or "age_ms" in legacy_name or "silence_gap" in legacy_name:
        return FeatureUnit.MICROSECONDS
    if "bps" in legacy_name or "spread_z" in legacy_name or "return_std" in legacy_name or "realized_vol" in legacy_name:
        return FeatureUnit.BPS
    if "usd" in legacy_name or "notional" in legacy_name:
        return FeatureUnit.NOTIONAL_USD
    if any(k in legacy_name for k in ("imbalance", "fraction", "share", "entropy", "asymmetry", "proxy", "score")):
        return FeatureUnit.RATIO
    if "count" in legacy_name or "run_length" in legacy_name:
        return FeatureUnit.COUNT
    if "per_second" in legacy_name or "rate" in legacy_name or "slope" in legacy_name:
        return FeatureUnit.RATE_PER_SECOND
    if "sign" in legacy_name or legacy_name in {"last_trade_side_sign", "last_tick_sign"}:
        return FeatureUnit.SIGN
    return FeatureUnit.SCORE


def infer_required_book_depth(legacy_name: str, source: FeatureSource) -> int:
    trade_exact_names = {
        "time_since_trade_ms",
        "regime_volume_ewma_500ms",
        "regime_volume_ewma_3000ms",
    }
    if source == FeatureSource.EVENT_CONTEXT:
        return 0
    if legacy_name in {"vwap_vs_mid_bps_200ms", "vwap_vs_mid_bps_500ms"}:
        return 1
    if source == FeatureSource.TRADE and legacy_name in trade_exact_names:
        return 0
    if source == FeatureSource.TRADE and not any(k in legacy_name for k in ("depth", "book", "l1", "l3", "l5", "l10", "vamp", "micro_l", "replenishment")):
        return 0
    if source == FeatureSource.CROSS:
        return 20
    if "depth_5bps" in legacy_name or any(k in legacy_name for k in ("within_1bps", "centroid", "liquidity_void")):
        return 20
    if "_l10" in legacy_name or "l10_" in legacy_name:
        return 10
    if any(k in legacy_name for k in ("_l5", "l5_", "top5")):
        return 5
    if "_l3" in legacy_name or "l3_" in legacy_name:
        return 3
    if "_l1" in legacy_name or "l1_" in legacy_name or legacy_name in {"bsz1", "asz1", "gap_b_bps"}:
        return 1
    if any(k in legacy_name for k in ("depth", "vamp", "micro_l", "liquidity_void", "centroid")):
        return 20
    return 1 if source == FeatureSource.BOOK else 0


def _has_window_at_most(legacy_name: str, max_window_us: int) -> bool:
    windows_us = infer_windows_us_from_legacy_name(legacy_name)
    return bool(windows_us) and min(windows_us) <= max_window_us


def _has_window_at_least(legacy_name: str, min_window_us: int) -> bool:
    windows_us = infer_windows_us_from_legacy_name(legacy_name)
    return bool(windows_us) and max(windows_us) >= min_window_us


def _is_ratio_or_bounded_feature(legacy_name: str, unit: FeatureUnit) -> bool:
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
    return any(term in legacy_name for term in bounded_terms)


def _is_slow_regime_feature(legacy_name: str) -> bool:
    slow_exact = {
        "regime_volume_ewma_500ms",
        "regime_volume_ewma_3000ms",
        "return_std_bps_200ms",
        "microprice_realized_vol_1000ms",
        "max_abs_return_bps_500ms",
        "depth_imbalance_realized_vol_1000ms",
    }
    return legacy_name in slow_exact or legacy_name.startswith("spread_z_") or legacy_name.startswith("depth_5bps_z_")


def _is_fast_microstructure_feature(legacy_name: str, source: FeatureSource, family: FeatureFamily) -> bool:
    if _is_slow_regime_feature(legacy_name) or not _has_window_at_most(legacy_name, 1_000_000):
        return legacy_name in {"spread_bps", "gap_b_bps", "ofi_l3", "obi_l1"}
    return source in {FeatureSource.BOOK, FeatureSource.TRADE, FeatureSource.CROSS} and family in {
        FeatureFamily.PRICE,
        FeatureFamily.OFI_OBI,
        FeatureFamily.BOOK_DYNAMICS,
        FeatureFamily.TRADE_FLOW,
        FeatureFamily.ABSORPTION,
        FeatureFamily.CROSS_SIGNAL,
    }


def _signed_log_ewma_key(legacy_name: str, source: FeatureSource, family: FeatureFamily) -> TransformKey:
    if _is_slow_regime_feature(legacy_name):
        return TransformKey.SIGNED_LOG1P_EWMA_SLOW
    if _is_fast_microstructure_feature(legacy_name, source, family) and not legacy_name.startswith("cvd_"):
        return TransformKey.SIGNED_LOG1P_EWMA_FAST
    return TransformKey.SIGNED_LOG1P_EWMA_MEDIUM


def _positive_log_ewma_key(legacy_name: str, source: FeatureSource, family: FeatureFamily) -> TransformKey:
    if _is_slow_regime_feature(legacy_name):
        return TransformKey.LOG1P_POS_EWMA_SLOW
    if _is_fast_microstructure_feature(legacy_name, source, family) and not any(
        term in legacy_name for term in ("depth_notional", "depth_within", "top5_trade_notional", "max_trade_silence_gap", "p90_over_median")
    ):
        return TransformKey.LOG1P_POS_EWMA_FAST
    return TransformKey.LOG1P_POS_EWMA_MEDIUM


def infer_transform_key(legacy_name: str, unit: FeatureUnit, source: FeatureSource, family: FeatureFamily) -> TransformKey:
    if source == FeatureSource.EVENT_CONTEXT:
        return TransformKey.LOG1P_POS_NO_EWMA
    if unit == FeatureUnit.MICROSECONDS:
        return TransformKey.TIME_LOG1P_NO_EWMA
    if unit == FeatureUnit.SIGN:
        return TransformKey.SIGN_NO_EWMA
    if _is_ratio_or_bounded_feature(legacy_name, unit):
        return TransformKey.RATIO_BOUNDED

    signed_heavy_tailed = (
        legacy_name.startswith("signed_notional_flow_usd_")
        or legacy_name.startswith("cvd_")
        or legacy_name.startswith("max_signed_trade_notional_usd_")
    )
    if signed_heavy_tailed:
        return _signed_log_ewma_key(legacy_name, source, family)

    positive_heavy_tailed = (
        legacy_name.startswith("regime_volume_ewma_")
        or "notional" in legacy_name
        or "usd" in legacy_name
        or unit in {FeatureUnit.COUNT, FeatureUnit.RATE_PER_SECOND}
    )
    if positive_heavy_tailed:
        return _positive_log_ewma_key(legacy_name, source, family)

    if unit == FeatureUnit.BPS:
        if _is_slow_regime_feature(legacy_name):
            return TransformKey.IDENTITY_EWMA_SLOW
        return TransformKey.IDENTITY_EWMA_FAST

    if _is_slow_regime_feature(legacy_name):
        return TransformKey.IDENTITY_EWMA_SLOW
    if _is_fast_microstructure_feature(legacy_name, source, family):
        return TransformKey.IDENTITY_EWMA_FAST
    if _has_window_at_least(legacy_name, 3_000_000):
        return TransformKey.IDENTITY_EWMA_MEDIUM
    return TransformKey.IDENTITY_EWMA_MEDIUM

def infer_formula_group(legacy_name: str, source: FeatureSource, family: FeatureFamily) -> str:
    if source == FeatureSource.EVENT_CONTEXT:
        return "event_context"
    if legacy_name in {"vwap_vs_mid_bps_200ms", "vwap_vs_mid_bps_500ms"}:
        return "cross_signal"
    if legacy_name in {"regime_volume_ewma_500ms", "regime_volume_ewma_3000ms"}:
        return "regime"
    if legacy_name == "time_since_trade_ms":
        return "trade_window"
    if family == FeatureFamily.PRICE:
        return "price_history"
    if legacy_name in {"spread_bps", "gap_b_bps", "bsz1", "asz1"}:
        return "top_of_book"
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
    canonical_core_names = tuple(legacy_name_to_canonical_name(n) for n in CORE_FEATURE_NAMES)
    canonical_context_names = tuple(legacy_name_to_canonical_name(n) for n in EVENT_CONTEXT_FEATURE_NAMES)
    feature_names = canonical_core_names + canonical_context_names
    assert len(feature_names) == FEATURE_COUNT
    assert len(set(feature_names)) == FEATURE_COUNT
    specs = []
    legacy_names = CORE_FEATURE_NAMES + EVENT_CONTEXT_FEATURE_NAMES
    for index, (name, legacy_name) in enumerate(zip(feature_names, legacy_names)):
        source = infer_source(legacy_name)
        owner = infer_owner(source)
        family = infer_family(legacy_name, source)
        unit = infer_unit(legacy_name)
        transform_key = infer_transform_key(legacy_name, unit, source, family)
        windows_us = infer_windows_us_from_legacy_name(legacy_name)
        required_book_depth = infer_required_book_depth(legacy_name, source)
        formula_group = infer_formula_group(legacy_name, source, family)
        specs.append(FeatureSpec(index, name, legacy_name, source, owner, family, unit, transform_key, windows_us, required_book_depth, formula_group))
    return tuple(specs)


FEATURE_SPECS = _build_feature_specs()
FEATURE_NAMES = tuple(spec.name for spec in FEATURE_SPECS)
SOURCE_TO_CANONICAL_FEATURE_NAME = {spec.legacy_name: spec.name for spec in FEATURE_SPECS}
CANONICAL_TO_SOURCE_FEATURE_NAME = {spec.name: spec.legacy_name for spec in FEATURE_SPECS}
FEATURE_NAME_TO_INDEX = {spec.name: spec.index for spec in FEATURE_SPECS}

assert len(FEATURE_SPECS) == FEATURE_COUNT
assert FEATURE_NAMES[-4:] == (
    "log_events_200000us",
    "log_events_500000us",
    "log_events_1000000us",
    "log_events_3000000us",
)
assert max(spec.required_book_depth for spec in FEATURE_SPECS) == MAX_REQUIRED_BOOK_FEATURE_DEPTH


def canonical_name_to_legacy_name(name: str) -> str:
    return CANONICAL_TO_SOURCE_FEATURE_NAME[name]


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
            f"{spec.index}|{spec.name}|{spec.legacy_name}|{spec.source.value}|{spec.owner.value}|{spec.family.value}|"
            f"{spec.unit.value}|{spec.transform_key.value}|{windows_us_csv}|{spec.required_book_depth}|{spec.formula_group}"
        )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()[:12]


FEATURE_SPECS_HASH = feature_specs_hash(FEATURE_SPECS)

def all_feature_names() -> tuple[str, ...]:
    return FEATURE_NAMES

def legacy_feature_names() -> tuple[str, ...]:
    return CORE_FEATURE_NAMES + EVENT_CONTEXT_FEATURE_NAMES

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

def legacy_name_for_feature(name: str) -> str:
    return CANONICAL_TO_SOURCE_FEATURE_NAME[name]

def canonical_name_for_legacy_feature(legacy_name: str) -> str:
    return SOURCE_TO_CANONICAL_FEATURE_NAME[legacy_name]

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

assert max_required_book_depth() <= REQUIRED_TARDIS_BOOK_SNAPSHOT_DEPTH

def schema_record() -> Mapping[str, object]:
    return {
      "feature_schema_version": FEATURE_SCHEMA_VERSION,
      "feature_count": FEATURE_COUNT,
      "feature_names_hash": FEATURE_NAMES_HASH,
      "feature_specs_hash": FEATURE_SPECS_HASH,
      "feature_dtype": DEFAULT_FEATURE_DTYPE,
      "time_unit": "us",
      "required_tardis_book_snapshot_depth": REQUIRED_TARDIS_BOOK_SNAPSHOT_DEPTH,
      "max_required_book_feature_depth": MAX_REQUIRED_BOOK_FEATURE_DEPTH,
      "source_counts": {s.value: len(feature_indices_by_source(s)) for s in FeatureSource},
      "owner_counts": {o.value: len(feature_indices_by_owner(o)) for o in FeatureOwner},
      "family_counts": {f.value: len(feature_indices_by_family(f)) for f in FeatureFamily},
    }

__all__ = [
    "FEATURE_SCHEMA_VERSION", "CORE_FEATURE_COUNT", "EVENT_CONTEXT_FEATURE_COUNT", "FEATURE_COUNT",
    "REQUIRED_TARDIS_BOOK_SNAPSHOT_DEPTH", "MAX_REQUIRED_BOOK_FEATURE_DEPTH", "SUPPORTED_WINDOWS_US",
    "DEFAULT_FEATURE_DTYPE", "CORE_FEATURE_NAMES", "EVENT_CONTEXT_FEATURE_NAMES", "FeatureSource",
    "FeatureOwner", "FeatureFamily", "FeatureUnit", "TransformKey", "FeatureSpec", "FEATURE_SPECS",
    "FEATURE_NAMES", "SOURCE_TO_CANONICAL_FEATURE_NAME", "CANONICAL_TO_SOURCE_FEATURE_NAME", "FEATURE_NAME_TO_INDEX",
    "FEATURE_NAMES_HASH", "FEATURE_SPECS_HASH", "legacy_name_to_canonical_name", "infer_windows_us_from_legacy_name",
    "all_feature_names", "legacy_feature_names", "feature_count", "feature_specs", "feature_spec_by_name",
    "feature_index", "feature_name", "legacy_name_for_feature", "canonical_name_for_legacy_feature",
    "feature_indices_by_source", "feature_indices_by_owner", "feature_indices_by_family", "required_windows_us",
    "max_required_book_depth", "schema_record",
]
