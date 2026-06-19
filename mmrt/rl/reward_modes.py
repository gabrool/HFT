"""Projected training reward modes for execution PPO rollouts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Mapping, Sequence

import numpy as np

from mmrt.execution.contracts import Fill, OrderSide


class TrainingRewardMode(str, Enum):
    EQUITY_DELTA = "equity_delta"
    HORIZON_PATH_EQUITY = "horizon_path_equity"
    FILL_MARKOUT_HORIZON = "fill_markout_horizon"
    HORIZON_BLEND = "horizon_blend"
    HORIZON_POTENTIAL_SHAPED = "horizon_potential_shaped"
    REALIZED_LOT_HORIZON = "realized_lot_horizon"
    MULTI_HORIZON_PATH = "multi_horizon_path"


TRAINING_REWARD_MODES = tuple(mode.value for mode in TrainingRewardMode)


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive int")
    if value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return int(value)


def _require_nonnegative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative float")
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative float")
    return out


def _require_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite float")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be a finite float")
    return out


def _coerce_reward_mode(value: TrainingRewardMode | str) -> TrainingRewardMode:
    if isinstance(value, TrainingRewardMode):
        return value
    if isinstance(value, str):
        try:
            return TrainingRewardMode(value.strip())
        except ValueError as exc:
            raise ValueError(f"training_reward_mode must be one of {TRAINING_REWARD_MODES}") from exc
    raise ValueError("training_reward_mode must be str or TrainingRewardMode")


def _tuple_positive_ints(values: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of positive ints")
    try:
        out = tuple(int(item) for item in values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of positive ints") from exc
    out = tuple(_require_positive_int(item, f"{name}[{idx}]") for idx, item in enumerate(out))
    if tuple(sorted(out)) != out:
        raise ValueError(f"{name} must be sorted ascending")
    if len(set(out)) != len(out):
        raise ValueError(f"{name} must be unique")
    return out


def _tuple_nonnegative_floats(values: Sequence[float], name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of finite nonnegative floats")
    try:
        out = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of finite nonnegative floats") from exc
    return tuple(_require_nonnegative_float(item, f"{name}[{idx}]") for idx, item in enumerate(out))


@dataclass(frozen=True, slots=True)
class TrainingRewardConfig:
    training_reward_mode: TrainingRewardMode | str = TrainingRewardMode.EQUITY_DELTA
    reward_horizon_us: int = 1_000_000
    reward_scale: float = 1.0
    horizon_path_weight: float = 1.0
    fill_markout_weight: float = 0.25
    horizon_potential_weight: float = 1.0
    realized_lot_weight: float = 1.0
    unrealized_horizon_weight: float = 1.0
    multi_horizon_us: tuple[int, ...] = (250_000, 500_000, 1_000_000)
    multi_horizon_weights: tuple[float, ...] = (0.25, 0.25, 1.0)

    def __post_init__(self) -> None:
        mode = _coerce_reward_mode(self.training_reward_mode)
        object.__setattr__(self, "training_reward_mode", mode)
        object.__setattr__(
            self,
            "reward_horizon_us",
            _require_positive_int(self.reward_horizon_us, "reward_horizon_us"),
        )
        for name in (
            "reward_scale",
            "horizon_path_weight",
            "fill_markout_weight",
            "horizon_potential_weight",
            "realized_lot_weight",
            "unrealized_horizon_weight",
        ):
            object.__setattr__(self, name, _require_nonnegative_float(getattr(self, name), name))
        object.__setattr__(
            self,
            "multi_horizon_us",
            _tuple_positive_ints(self.multi_horizon_us, "multi_horizon_us"),
        )
        object.__setattr__(
            self,
            "multi_horizon_weights",
            _tuple_nonnegative_floats(self.multi_horizon_weights, "multi_horizon_weights"),
        )
        if len(self.multi_horizon_weights) != len(self.multi_horizon_us):
            raise ValueError("multi_horizon_weights length must match multi_horizon_us")
        if sum(self.multi_horizon_weights) <= 0.0:
            raise ValueError("multi_horizon_weights sum must be > 0")
        if mode is TrainingRewardMode.HORIZON_BLEND:
            if self.horizon_path_weight + self.fill_markout_weight <= 0.0:
                raise ValueError("horizon_path_weight + fill_markout_weight must be > 0")
        if mode is TrainingRewardMode.REALIZED_LOT_HORIZON:
            total = self.realized_lot_weight + self.unrealized_horizon_weight + self.fill_markout_weight
            if total <= 0.0:
                raise ValueError(
                    "realized_lot_weight + unrealized_horizon_weight + fill_markout_weight must be > 0"
                )

    @property
    def mode(self) -> TrainingRewardMode:
        return self.training_reward_mode  # type: ignore[return-value]

    @property
    def is_equity_delta(self) -> bool:
        return self.mode is TrainingRewardMode.EQUITY_DELTA

    @property
    def max_required_horizon_us(self) -> int:
        if self.mode is TrainingRewardMode.MULTI_HORIZON_PATH:
            return max(self.multi_horizon_us)
        return int(self.reward_horizon_us)

    def as_dict(self) -> dict[str, object]:
        return {
            "training_reward_mode": self.mode.value,
            "reward_horizon_us": int(self.reward_horizon_us),
            "reward_scale": float(self.reward_scale),
            "horizon_path_weight": float(self.horizon_path_weight),
            "fill_markout_weight": float(self.fill_markout_weight),
            "horizon_potential_weight": float(self.horizon_potential_weight),
            "realized_lot_weight": float(self.realized_lot_weight),
            "unrealized_horizon_weight": float(self.unrealized_horizon_weight),
            "multi_horizon_us": [int(item) for item in self.multi_horizon_us],
            "multi_horizon_weights": [float(item) for item in self.multi_horizon_weights],
        }


@dataclass(frozen=True, slots=True)
class RewardAnchor:
    t_index: int
    env_index: int
    episode_id: int
    decision_row: int
    next_decision_row: int
    decision_local_ts_us: int
    next_decision_local_ts_us: int
    range_end_row: int
    previous_equity: float
    current_equity: float
    env_reward: float
    fills: tuple[Fill, ...] = ()
    inventory_after_step: float = 0.0
    current_mid_after_step: float = 0.0
    realized_lot_pnl: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "t_index",
            "env_index",
            "episode_id",
            "decision_row",
            "next_decision_row",
            "decision_local_ts_us",
            "next_decision_local_ts_us",
            "range_end_row",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be int")
            if name != "episode_id" and value < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in (
            "previous_equity",
            "current_equity",
            "env_reward",
            "inventory_after_step",
            "current_mid_after_step",
            "realized_lot_pnl",
        ):
            object.__setattr__(self, name, _require_finite_float(getattr(self, name), name))
        fills = tuple(self.fills)
        if not all(isinstance(fill, Fill) for fill in fills):
            raise ValueError("fills must contain Fill values")
        object.__setattr__(self, "fills", fills)


@dataclass(frozen=True, slots=True)
class RewardProjectionResult:
    projected_rewards: np.ndarray
    valid_mask: np.ndarray
    components: dict[str, np.ndarray]
    stats: dict[str, object]


@dataclass(slots=True)
class _Lot:
    side: int
    qty_remaining: float
    entry_price: float
    entry_fee_remaining: float


class FifoLotLedger:
    """Deterministic FIFO realized-PnL ledger for execution fills."""

    def __init__(
        self,
        *,
        tick_size: float = 1.0,
        contract_size: float = 1.0,
        qty_epsilon: float = 1e-12,
    ) -> None:
        self.tick_size = _require_finite_float(tick_size, "tick_size")
        if self.tick_size <= 0.0:
            raise ValueError("tick_size must be > 0")
        self.contract_size = _require_nonnegative_float(contract_size, "contract_size")
        if self.contract_size <= 0.0:
            raise ValueError("contract_size must be > 0")
        self.qty_epsilon = _require_nonnegative_float(qty_epsilon, "qty_epsilon")
        self._lots: list[_Lot] = []

    @property
    def lots(self) -> tuple[tuple[str, float, float, float], ...]:
        return tuple(
            ("long" if lot.side > 0 else "short", lot.qty_remaining, lot.entry_price, lot.entry_fee_remaining)
            for lot in self._lots
        )

    def reset(self, *, initial_inventory_qty: float = 0.0, entry_price: float = 0.0) -> None:
        qty = _require_finite_float(initial_inventory_qty, "initial_inventory_qty")
        price = _require_finite_float(entry_price, "entry_price")
        self._lots.clear()
        if abs(qty) <= self.qty_epsilon:
            return
        if price <= 0.0:
            raise ValueError("entry_price must be > 0 for nonzero initial inventory")
        self._lots.append(
            _Lot(
                side=1 if qty > 0.0 else -1,
                qty_remaining=abs(qty),
                entry_price=price,
                entry_fee_remaining=0.0,
            )
        )

    def apply_fills(self, fills: Sequence[Fill]) -> float:
        realized = 0.0
        for fill in fills:
            realized += self.apply_fill(fill)
        return realized

    def apply_fill(self, fill: Fill) -> float:
        if not isinstance(fill, Fill):
            raise ValueError("fill must be Fill")
        fill_qty = _require_finite_float(fill.qty, "fill.qty")
        if fill_qty <= 0.0:
            raise ValueError("fill.qty must be > 0")
        fill_price = _require_finite_float(float(fill.price_tick) * self.tick_size, "fill.price")
        fill_fee = _require_finite_float(fill.fee, "fill.fee")
        incoming_side = 1 if fill.side == OrderSide.BUY else -1
        remaining_qty = fill_qty
        realized = 0.0

        while remaining_qty > self.qty_epsilon and self._lots and self._lots[0].side != incoming_side:
            lot = self._lots[0]
            lot_qty_before = lot.qty_remaining
            close_qty = min(remaining_qty, lot.qty_remaining)
            entry_fee_alloc = lot.entry_fee_remaining * (close_qty / lot_qty_before)
            exit_fee_alloc = fill_fee * (close_qty / fill_qty)
            if lot.side > 0:
                realized += close_qty * self.contract_size * (fill_price - lot.entry_price)
            else:
                realized += close_qty * self.contract_size * (lot.entry_price - fill_price)
            realized -= entry_fee_alloc + exit_fee_alloc
            lot.qty_remaining -= close_qty
            lot.entry_fee_remaining -= entry_fee_alloc
            remaining_qty -= close_qty
            if lot.qty_remaining <= self.qty_epsilon:
                self._lots.pop(0)

        if remaining_qty > self.qty_epsilon:
            entry_fee = fill_fee * (remaining_qty / fill_qty)
            self._lots.append(
                _Lot(
                    side=incoming_side,
                    qty_remaining=remaining_qty,
                    entry_price=fill_price,
                    entry_fee_remaining=entry_fee,
                )
            )

        return float(realized)


def future_row_for_ts(
    decision_local_ts_us: Sequence[int] | np.ndarray,
    anchor_ts_us: int,
    horizon_us: int,
    range_end_row: int,
) -> int | None:
    timestamps = np.asarray(decision_local_ts_us, dtype=np.int64)
    if timestamps.ndim != 1:
        raise ValueError("decision_local_ts_us must be 1D")
    horizon_us = _require_positive_int(horizon_us, "horizon_us")
    if isinstance(range_end_row, bool) or not isinstance(range_end_row, int):
        raise ValueError("range_end_row must be int")
    target = int(anchor_ts_us) + int(horizon_us)
    row = int(np.searchsorted(timestamps, target, side="left"))
    if row >= int(range_end_row) or row >= timestamps.shape[0]:
        return None
    return row


def mid_prices_from_execution(*, decision_grid: object, tape: object) -> np.ndarray:
    book_ptr = np.asarray(getattr(decision_grid, "book_ptr"), dtype=np.int64)
    l2_events = getattr(getattr(tape, "arrays"), "l2_events")
    bid_ticks = np.asarray(l2_events["best_bid_tick"][book_ptr], dtype=np.float64)
    ask_ticks = np.asarray(l2_events["best_ask_tick"][book_ptr], dtype=np.float64)
    symbol_spec = getattr(getattr(tape, "manifest"), "symbol_spec")
    tick_size = float(symbol_spec.tick_size)
    return (bid_ticks + ask_ticks) * tick_size * 0.5


class HorizonRewardProjector:
    def __init__(
        self,
        *,
        decision_local_ts_us: Sequence[int] | np.ndarray,
        mid_prices: Sequence[float] | np.ndarray,
        tick_size: float = 1.0,
        contract_size: float = 1.0,
        config: TrainingRewardConfig = TrainingRewardConfig(),
    ) -> None:
        if not isinstance(config, TrainingRewardConfig):
            raise ValueError("config must be TrainingRewardConfig")
        self.config = config
        self.decision_local_ts_us = np.asarray(decision_local_ts_us, dtype=np.int64)
        self.mid_prices = np.asarray(mid_prices, dtype=np.float64)
        if self.decision_local_ts_us.ndim != 1 or self.mid_prices.ndim != 1:
            raise ValueError("decision_local_ts_us and mid_prices must be 1D")
        if self.decision_local_ts_us.shape != self.mid_prices.shape:
            raise ValueError("decision_local_ts_us and mid_prices must have matching shapes")
        if self.decision_local_ts_us.shape[0] == 0:
            raise ValueError("decision rows must be non-empty")
        if np.diff(self.decision_local_ts_us).min(initial=0) < 0:
            raise ValueError("decision_local_ts_us must be sorted ascending")
        if not np.isfinite(self.mid_prices).all() or (self.mid_prices <= 0.0).any():
            raise ValueError("mid_prices must be finite and positive")
        self.tick_size = _require_finite_float(tick_size, "tick_size")
        if self.tick_size <= 0.0:
            raise ValueError("tick_size must be > 0")
        self.contract_size = _require_finite_float(contract_size, "contract_size")
        if self.contract_size <= 0.0:
            raise ValueError("contract_size must be > 0")

    @classmethod
    def from_execution(
        cls,
        *,
        decision_grid: object,
        tape: object,
        config: TrainingRewardConfig = TrainingRewardConfig(),
    ) -> "HorizonRewardProjector":
        symbol_spec = getattr(getattr(tape, "manifest"), "symbol_spec")
        return cls(
            decision_local_ts_us=getattr(decision_grid, "decision_local_ts_us"),
            mid_prices=mid_prices_from_execution(decision_grid=decision_grid, tape=tape),
            tick_size=float(symbol_spec.tick_size),
            contract_size=float(symbol_spec.contract_size),
            config=config,
        )

    def required_rows(self, anchor: RewardAnchor) -> tuple[int, ...]:
        mode = self.config.mode
        rows: list[int] = []

        def add(ts_us: int, horizon_us: int) -> None:
            row = future_row_for_ts(
                self.decision_local_ts_us,
                int(ts_us),
                int(horizon_us),
                int(anchor.range_end_row),
            )
            if row is not None:
                rows.append(row)

        if mode in (
            TrainingRewardMode.HORIZON_PATH_EQUITY,
            TrainingRewardMode.HORIZON_BLEND,
            TrainingRewardMode.HORIZON_POTENTIAL_SHAPED,
            TrainingRewardMode.REALIZED_LOT_HORIZON,
        ):
            add(anchor.decision_local_ts_us, self.config.reward_horizon_us)
        if mode is TrainingRewardMode.HORIZON_POTENTIAL_SHAPED:
            add(anchor.next_decision_local_ts_us, self.config.reward_horizon_us)
        if mode is TrainingRewardMode.MULTI_HORIZON_PATH:
            for horizon_us in self.config.multi_horizon_us:
                add(anchor.decision_local_ts_us, horizon_us)
        if mode in (
            TrainingRewardMode.FILL_MARKOUT_HORIZON,
            TrainingRewardMode.HORIZON_BLEND,
            TrainingRewardMode.REALIZED_LOT_HORIZON,
        ):
            for fill in anchor.fills:
                add(fill.local_ts_us, self.config.reward_horizon_us)
        return tuple(sorted(set(rows)))

    def project(
        self,
        anchors: Sequence[RewardAnchor],
        equity_by_episode: Mapping[tuple[int, int], Mapping[int, float]],
        *,
        shape: tuple[int, int] | None = None,
    ) -> RewardProjectionResult:
        anchors = tuple(anchors)
        if shape is None:
            max_t = max((anchor.t_index for anchor in anchors), default=-1)
            max_env = max((anchor.env_index for anchor in anchors), default=-1)
            shape = (max_t + 1, max_env + 1)
        rewards = np.zeros(shape, dtype=np.float64)
        env_rewards = np.zeros(shape, dtype=np.float64)
        valid = np.zeros(shape, dtype=np.bool_)
        component_values: dict[str, np.ndarray] = {}
        reasons: Counter[str] = Counter()

        def component(name: str) -> np.ndarray:
            return component_values.setdefault(name, np.zeros(shape, dtype=np.float64))

        for anchor in anchors:
            env_rewards[anchor.t_index, anchor.env_index] = anchor.env_reward
            projected, is_valid, values, reason = self._project_one(anchor, equity_by_episode)
            rewards[anchor.t_index, anchor.env_index] = projected if is_valid else 0.0
            valid[anchor.t_index, anchor.env_index] = is_valid
            for name, value in values.items():
                component(name)[anchor.t_index, anchor.env_index] = float(value)
            if reason is not None:
                reasons[reason] += 1

        components = {name: array for name, array in sorted(component_values.items())}
        valid_values = rewards[valid]
        valid_env_rewards = env_rewards[valid]
        stats = {
            "train_anchor_count": int(len(anchors)),
            "valid_anchor_count": int(valid.sum()),
            "invalid_anchor_count": int(len(anchors) - int(valid.sum())),
            "valid_fraction": float(valid.sum() / max(len(anchors), 1)),
            "env_reward_mean": float(valid_env_rewards.mean()) if valid_env_rewards.size else None,
            "env_reward_sum": float(valid_env_rewards.sum()) if valid_env_rewards.size else 0.0,
            "projected_reward_mean": float(valid_values.mean()) if valid_values.size else None,
            "projected_reward_std": float(valid_values.std(ddof=0)) if valid_values.size else None,
            "projected_reward_sum": float(valid_values.sum()) if valid_values.size else 0.0,
            "component_means": {
                name: (float(array[valid].mean()) if valid.any() else None)
                for name, array in components.items()
            },
            "component_sums": {
                name: (float(array[valid].sum()) if valid.any() else 0.0)
                for name, array in components.items()
            },
            "unavailable_reason_counts": dict(sorted(reasons.items())),
        }
        return RewardProjectionResult(
            projected_rewards=rewards,
            valid_mask=valid,
            components=components,
            stats=stats,
        )

    def _path_component(
        self,
        anchor: RewardAnchor,
        equity_by_episode: Mapping[tuple[int, int], Mapping[int, float]],
        *,
        horizon_us: int,
        ts_us: int | None = None,
        base_equity: float | None = None,
    ) -> tuple[float | None, str | None]:
        future_row = future_row_for_ts(
            self.decision_local_ts_us,
            anchor.decision_local_ts_us if ts_us is None else int(ts_us),
            horizon_us,
            anchor.range_end_row,
        )
        if future_row is None:
            return None, "future_row_unavailable"
        episode_equity = equity_by_episode.get((anchor.env_index, anchor.episode_id), {})
        if future_row not in episode_equity:
            return None, "future_equity_unavailable"
        return float(episode_equity[future_row]) - (
            anchor.previous_equity if base_equity is None else float(base_equity)
        ), None

    def _future_mid_for_ts(
        self,
        *,
        ts_us: int,
        horizon_us: int,
        range_end_row: int,
    ) -> tuple[float | None, str | None]:
        future_row = future_row_for_ts(
            self.decision_local_ts_us,
            int(ts_us),
            int(horizon_us),
            int(range_end_row),
        )
        if future_row is None:
            return None, "future_mid_unavailable"
        return float(self.mid_prices[future_row]), None

    def _fill_markout_component(
        self,
        anchor: RewardAnchor,
        *,
        horizon_us: int,
    ) -> tuple[float | None, dict[str, float], str | None]:
        components = {
            "fill_markout_pnl_H": 0.0,
            "fill_count_scored": 0.0,
            "fill_count_unavailable": 0.0,
        }
        if not anchor.fills:
            return 0.0, components, None
        total = 0.0
        unavailable = 0
        for fill in anchor.fills:
            future_mid, reason = self._future_mid_for_ts(
                ts_us=fill.local_ts_us,
                horizon_us=horizon_us,
                range_end_row=anchor.range_end_row,
            )
            if future_mid is None:
                unavailable += 1
                continue
            fill_price = float(fill.price_tick) * self.tick_size
            if fill.side == OrderSide.BUY:
                gross = float(fill.qty) * self.contract_size * (future_mid - fill_price)
            elif fill.side == OrderSide.SELL:
                gross = float(fill.qty) * self.contract_size * (fill_price - future_mid)
            else:
                unavailable += 1
                continue
            total += gross - float(fill.fee)
            components["fill_count_scored"] += 1.0
        components["fill_markout_pnl_H"] = total
        components["fill_count_unavailable"] = float(unavailable)
        if unavailable:
            return None, components, "fill_horizon_unavailable"
        return total, components, None

    def _project_one(
        self,
        anchor: RewardAnchor,
        equity_by_episode: Mapping[tuple[int, int], Mapping[int, float]],
    ) -> tuple[float, bool, dict[str, float], str | None]:
        mode = self.config.mode
        scale = float(self.config.reward_scale)
        if mode is TrainingRewardMode.EQUITY_DELTA:
            return anchor.env_reward, True, {"env_reward": anchor.env_reward}, None

        if mode is TrainingRewardMode.HORIZON_PATH_EQUITY:
            path, reason = self._path_component(
                anchor,
                equity_by_episode,
                horizon_us=self.config.reward_horizon_us,
            )
            if path is None:
                return 0.0, False, {}, reason
            return scale * path, True, {"path_equity_delta_H": path}, None

        if mode is TrainingRewardMode.FILL_MARKOUT_HORIZON:
            fill, components, reason = self._fill_markout_component(
                anchor,
                horizon_us=self.config.reward_horizon_us,
            )
            if fill is None:
                return 0.0, False, components, reason
            return scale * fill, True, components, None

        if mode is TrainingRewardMode.HORIZON_BLEND:
            path, reason = self._path_component(
                anchor,
                equity_by_episode,
                horizon_us=self.config.reward_horizon_us,
            )
            fill, components, fill_reason = self._fill_markout_component(
                anchor,
                horizon_us=self.config.reward_horizon_us,
            )
            if path is None:
                return 0.0, False, components, reason
            if fill is None:
                return 0.0, False, components, fill_reason
            raw = self.config.horizon_path_weight * path + self.config.fill_markout_weight * fill
            components.update({"path_equity_delta_H": path, "blended_reward_raw": raw})
            return scale * raw, True, components, None

        if mode is TrainingRewardMode.HORIZON_POTENTIAL_SHAPED:
            phi_t, reason = self._path_component(
                anchor,
                equity_by_episode,
                horizon_us=self.config.reward_horizon_us,
            )
            phi_next, next_reason = self._path_component(
                anchor,
                equity_by_episode,
                horizon_us=self.config.reward_horizon_us,
                ts_us=anchor.next_decision_local_ts_us,
                base_equity=anchor.current_equity,
            )
            if phi_t is None:
                return 0.0, False, {"env_reward": anchor.env_reward}, reason
            if phi_next is None:
                return 0.0, False, {"env_reward": anchor.env_reward, "horizon_phi_t": phi_t}, next_reason
            delta = phi_next - phi_t
            projected = anchor.env_reward + scale * self.config.horizon_potential_weight * delta
            return projected, True, {
                "env_reward": anchor.env_reward,
                "horizon_phi_t": phi_t,
                "horizon_phi_next": phi_next,
                "horizon_potential_delta": delta,
            }, None

        if mode is TrainingRewardMode.REALIZED_LOT_HORIZON:
            future_mid, reason = self._future_mid_for_ts(
                ts_us=anchor.decision_local_ts_us,
                horizon_us=self.config.reward_horizon_us,
                range_end_row=anchor.range_end_row,
            )
            fill, components, fill_reason = self._fill_markout_component(
                anchor,
                horizon_us=self.config.reward_horizon_us,
            )
            if future_mid is None:
                return 0.0, False, components, reason
            if fill is None:
                return 0.0, False, components, fill_reason
            unrealized = (
                anchor.inventory_after_step
                * self.contract_size
                * (future_mid - anchor.current_mid_after_step)
            )
            raw = (
                self.config.realized_lot_weight * anchor.realized_lot_pnl
                + self.config.unrealized_horizon_weight * unrealized
                + self.config.fill_markout_weight * fill
            )
            components.update(
                {
                    "realized_lot_pnl": anchor.realized_lot_pnl,
                    "unrealized_horizon_carry_pnl": unrealized,
                    "open_inventory_qty_after_step": anchor.inventory_after_step,
                }
            )
            return scale * raw, True, components, None

        if mode is TrainingRewardMode.MULTI_HORIZON_PATH:
            raw = 0.0
            components: dict[str, float] = {}
            for horizon_us, weight in zip(self.config.multi_horizon_us, self.config.multi_horizon_weights):
                path, reason = self._path_component(anchor, equity_by_episode, horizon_us=horizon_us)
                if path is None:
                    return 0.0, False, components, reason
                components[f"path_equity_delta_{int(horizon_us)}"] = path
                raw += float(weight) * path
            components["multi_horizon_path_reward_raw"] = raw
            return scale * raw, True, components, None

        raise RuntimeError("unhandled training reward mode")


__all__ = [
    "TRAINING_REWARD_MODES",
    "TrainingRewardMode",
    "TrainingRewardConfig",
    "RewardAnchor",
    "RewardProjectionResult",
    "FifoLotLedger",
    "future_row_for_ts",
    "mid_prices_from_execution",
    "HorizonRewardProjector",
]
