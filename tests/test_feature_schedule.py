import pytest

from mmrt.features.schedule import (
    DecisionSchedule,
    DecisionScheduleConfig,
    decision_schedule_config_from_dict,
)


def _book(schedule, ts, *, bid=100.0, ask=100.1, bid_sz=1.0, ask_sz=1.0):
    schedule.observe_book(ts, best_bid=bid, best_ask=ask, bid_l1_size=bid_sz, ask_l1_size=ask_sz)


def test_schedule_config_validation():
    with pytest.raises(ValueError):
        DecisionScheduleConfig(min_decision_interval_us=0)
    with pytest.raises(ValueError):
        DecisionScheduleConfig(min_decision_interval_us=200, max_decision_interval_us=100)
    with pytest.raises(ValueError):
        DecisionScheduleConfig(l1_size_change_fraction=0.0)
    cfg = DecisionScheduleConfig()
    assert cfg.min_decision_interval_us == 100_000
    assert cfg.max_decision_interval_us == 500_000


def test_schedule_payload_round_trip_and_strictness():
    cfg = DecisionScheduleConfig(min_decision_interval_us=50_000, wake_on_trade=False)
    assert decision_schedule_config_from_dict(cfg.as_dict()) == cfg
    payload = cfg.as_dict()
    payload.pop("wake_on_trade")
    with pytest.raises(ValueError, match="missing fields"):
        decision_schedule_config_from_dict(payload)
    payload = cfg.as_dict()
    payload["surprise"] = 1
    with pytest.raises(ValueError, match="unknown fields"):
        decision_schedule_config_from_dict(payload)


def test_first_book_event_always_fires():
    schedule = DecisionSchedule(DecisionScheduleConfig())
    _book(schedule, 1_000_000)
    assert schedule.should_fire(1_000_000)
    schedule.mark_decision(1_000_000)
    assert schedule.last_decision_local_ts_us == 1_000_000


def test_throttle_blocks_decisions_within_min_interval():
    cfg = DecisionScheduleConfig(min_decision_interval_us=100_000, max_decision_interval_us=500_000)
    schedule = DecisionSchedule(cfg)
    _book(schedule, 1_000_000)
    schedule.mark_decision(1_000_000)
    # Top-of-book change arms the trigger, but the throttle window blocks it.
    _book(schedule, 1_050_000, bid=100.1, ask=100.2)
    assert schedule.is_armed
    assert not schedule.should_fire(1_050_000)
    # Once the throttle elapses, the armed trigger fires.
    _book(schedule, 1_100_000, bid=100.1, ask=100.2)
    assert schedule.should_fire(1_100_000)


def test_unarmed_schedule_waits_for_heartbeat():
    cfg = DecisionScheduleConfig(min_decision_interval_us=100_000, max_decision_interval_us=500_000)
    schedule = DecisionSchedule(cfg)
    _book(schedule, 1_000_000)
    schedule.mark_decision(1_000_000)
    # Identical book updates do not arm; nothing fires until the heartbeat.
    _book(schedule, 1_200_000)
    assert not schedule.is_armed
    assert not schedule.should_fire(1_200_000)
    _book(schedule, 1_400_000)
    assert not schedule.should_fire(1_400_000)
    _book(schedule, 1_500_000)
    assert schedule.should_fire(1_500_000)


def test_trade_wake_arms_trigger():
    cfg = DecisionScheduleConfig(min_decision_interval_us=100_000, max_decision_interval_us=500_000)
    schedule = DecisionSchedule(cfg)
    _book(schedule, 1_000_000)
    schedule.mark_decision(1_000_000)
    schedule.observe_trade(1_150_000)
    assert schedule.is_armed
    _book(schedule, 1_200_000)
    assert schedule.should_fire(1_200_000)


def test_wake_on_trade_disabled():
    cfg = DecisionScheduleConfig(wake_on_trade=False)
    schedule = DecisionSchedule(cfg)
    _book(schedule, 1_000_000)
    schedule.mark_decision(1_000_000)
    schedule.observe_trade(1_150_000)
    assert not schedule.is_armed


def test_l1_size_change_fraction_threshold():
    cfg = DecisionScheduleConfig(l1_size_change_fraction=0.5)
    schedule = DecisionSchedule(cfg)
    _book(schedule, 1_000_000, bid_sz=2.0, ask_sz=2.0)
    schedule.mark_decision(1_000_000)
    # 25% size change: below the fraction, prices unchanged -> not armed.
    _book(schedule, 1_200_000, bid_sz=2.5, ask_sz=2.0)
    assert not schedule.is_armed
    # 50%+ size change arms.
    _book(schedule, 1_300_000, bid_sz=3.75, ask_sz=2.0)
    assert schedule.is_armed


def test_wake_on_top_of_book_disabled_still_heartbeats():
    cfg = DecisionScheduleConfig(
        min_decision_interval_us=100_000,
        max_decision_interval_us=300_000,
        wake_on_trade=False,
        wake_on_top_of_book=False,
    )
    schedule = DecisionSchedule(cfg)
    _book(schedule, 1_000_000)
    schedule.mark_decision(1_000_000)
    _book(schedule, 1_150_000, bid=101.0, ask=101.1)
    assert not schedule.should_fire(1_150_000)
    _book(schedule, 1_300_000, bid=102.0, ask=102.1)
    assert schedule.should_fire(1_300_000)


def test_mark_decision_requires_strictly_increasing_ts():
    schedule = DecisionSchedule()
    schedule.mark_decision(1_000_000)
    with pytest.raises(ValueError):
        schedule.mark_decision(1_000_000)


def test_reset_restores_initial_state():
    schedule = DecisionSchedule()
    _book(schedule, 1_000_000)
    schedule.mark_decision(1_000_000)
    schedule.reset()
    assert schedule.last_decision_local_ts_us is None
    assert not schedule.is_armed
    assert schedule.should_fire(2_000_000)
