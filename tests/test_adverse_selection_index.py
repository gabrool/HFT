import json

import numpy as np
import pytest

from mmrt.execution.adverse_selection import KyleLambdaConfig, _future_mid_and_key_at_or_after_key, _valid_l2_view_from_tape
from mmrt.execution.adverse_selection_index import (
    ADVERSE_SELECTION_INDEX_SCHEMA,
    AdverseSelectionIndexConfig,
    ValidL2Index,
    build_adverse_selection_index,
    build_or_load_adverse_selection_index,
)
from mmrt.time_key import EventKey, MAX_EVENT_SEQ
from test_adverse_selection import _l2, _tape, _trade
from mmrt.contracts import AggressorSide


def _direct_index() -> ValidL2Index:
    return ValidL2Index(
        local_ts_us=np.array([1000, 1000, 1000, 2000], dtype=np.int64),
        event_seq=np.array([1, 3, 8, 0], dtype=np.int64),
        mid_tick=np.array([10, 11, 12, 20], dtype=np.float32),
    )


def test_future_mid_same_timestamp_max_event_seq_returns_last_same_timestamp_row():
    assert _direct_index().future_mid_and_key_at_or_after(EventKey(1000, MAX_EVENT_SEQ)) == (12.0, EventKey(1000, 8))


def test_future_mid_same_timestamp_intermediate_event_sequence():
    assert _direct_index().future_mid_and_key_at_or_after(EventKey(1000, 4)) == (11.0, EventKey(1000, 3))


def test_future_mid_same_timestamp_no_prior_or_equal_event_sequence_skips_group():
    assert _direct_index().future_mid_and_key_at_or_after(EventKey(1000, 0)) == (20.0, EventKey(2000, 0))


def test_future_mid_no_same_timestamp_future_fallback():
    assert _direct_index().future_mid_and_key_at_or_after(EventKey(1500, MAX_EVENT_SEQ)) == (20.0, EventKey(2000, 0))


def test_future_mid_after_last_timestamp_returns_none():
    assert _direct_index().future_mid_and_key_at_or_after(EventKey(3000, MAX_EVENT_SEQ)) is None


def test_disk_index_future_mid_matches_in_memory_helper_for_same_timestamp_groups(tmp_path):
    tape = _tape([
        _l2(seq=0, local_ts_us=1000, bid_ticks=(100, 99), ask_ticks=(102, 103)),
        _l2(seq=1, local_ts_us=1000, bid_ticks=(101, 100), ask_ticks=(103, 104)),
        _l2(seq=2, local_ts_us=1000, bid_ticks=(102, 101), ask_ticks=(104, 105)),
        _l2(seq=3, local_ts_us=2000, bid_ticks=(200, 199), ask_ticks=(202, 203)),
    ], [])
    in_memory = _valid_l2_view_from_tape(tape)
    index = build_adverse_selection_index(tape, config=_cfg(tmp_path / "idx", overwrite=True))
    keys = [EventKey(1000, MAX_EVENT_SEQ), EventKey(1000, 1), EventKey(1000, 0), EventKey(1500, MAX_EVENT_SEQ), EventKey(3000, MAX_EVENT_SEQ)]
    for key in keys:
        assert index.valid_l2.future_mid_and_key_at_or_after(key) == _future_mid_and_key_at_or_after_key(in_memory, key)


def test_kyle_sample_response_uses_last_same_timestamp_l2_for_max_event_seq(tmp_path):
    tape = _tape([
        _l2(seq=0, local_ts_us=1000, bid_ticks=(100, 99), ask_ticks=(102, 103)),
        _l2(seq=1, local_ts_us=2000, bid_ticks=(110, 109), ask_ticks=(112, 113)),
        _l2(seq=2, local_ts_us=2000, bid_ticks=(120, 119), ask_ticks=(122, 123)),
        _l2(seq=3, local_ts_us=3000, bid_ticks=(200, 199), ask_ticks=(202, 203)),
    ], [_trade(local_ts_us=1500, side=AggressorSide.BUY, price_tick=111, amount=1.0, source_row=0)])
    index = build_adverse_selection_index(tape, config=_cfg(tmp_path / "idx", overwrite=True, sample_interval=1000, response_horizon=1000))
    assert index.kyle_samples.count >= 1
    # start mid 101, response at timestamp 2000 should use last same-timestamp mid 121, not next timestamp mid 201.
    assert float(index.kyle_samples.y_mid_bps[0]) == pytest.approx((121.0 - 101.0) / 101.0 * 10_000.0)
    assert int(index.kyle_samples.end_local_ts_us[0]) == 2000


def _cfg(root, *, overwrite=False, sample_interval=500, response_horizon=500, windows=(1000,), use_notional=False, tick_size=0.1):
    return AdverseSelectionIndexConfig(
        output_root=str(root),
        kyle=KyleLambdaConfig(sample_interval_us=sample_interval, response_horizon_us=response_horizon, windows_us=windows, min_samples=1, use_notional_flow=use_notional),
        use_notional_flow=use_notional,
        tick_size=tick_size,
        chunk_rows=2,
        overwrite=overwrite,
    )


def _reuse_tape():
    return _tape([_l2(seq=0, local_ts_us=1000), _l2(seq=1, local_ts_us=2000)], [])


def test_adverse_selection_index_reuses_matching_manifest(tmp_path):
    tape = _reuse_tape()
    cfg = _cfg(tmp_path / "idx", overwrite=True)
    first = build_or_load_adverse_selection_index(tape, config=cfg)
    second = build_or_load_adverse_selection_index(tape, config=_cfg(tmp_path / "idx"))
    assert second.manifest.created_at_utc == first.manifest.created_at_utc


@pytest.mark.parametrize("mutator", [
    lambda raw: raw.update(tape_num_events=raw["tape_num_events"] + 1),
    lambda raw: raw["kyle"].update(sample_interval_us=12345),
    lambda raw: raw["kyle"].update(response_horizon_us=12345),
    lambda raw: raw["kyle"].update(windows_us=[12345]),
    lambda raw: raw["trade_flow"].update(use_notional_flow=True),
    lambda raw: raw.update(tick_size=999.0),
])
def test_adverse_selection_index_rejects_stale_manifest_without_overwrite(tmp_path, mutator):
    tape = _reuse_tape()
    root = tmp_path / "idx"
    build_or_load_adverse_selection_index(tape, config=_cfg(root, overwrite=True))
    path = root / "index_manifest.json"
    raw = json.loads(path.read_text())
    mutator(raw)
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="stale adverse-selection index manifest"):
        build_or_load_adverse_selection_index(tape, config=_cfg(root))


def test_adverse_selection_index_rebuilds_stale_manifest_with_overwrite(tmp_path):
    tape = _reuse_tape()
    root = tmp_path / "idx"
    build_or_load_adverse_selection_index(tape, config=_cfg(root, overwrite=True))
    path = root / "index_manifest.json"
    raw = json.loads(path.read_text())
    raw["tape_num_events"] += 1
    path.write_text(json.dumps(raw), encoding="utf-8")
    rebuilt = build_or_load_adverse_selection_index(tape, config=_cfg(root, overwrite=True))
    assert rebuilt.manifest.schema == ADVERSE_SELECTION_INDEX_SCHEMA
    assert rebuilt.manifest.tape_num_events == tape.manifest.num_events


def test_adverse_selection_index_schema_one_is_rejected(tmp_path):
    tape = _reuse_tape()
    root = tmp_path / "idx"
    build_or_load_adverse_selection_index(tape, config=_cfg(root, overwrite=True))
    path = root / "index_manifest.json"
    raw = json.loads(path.read_text())
    raw["schema"] = "mmrt_adverse_selection_index_v1"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="stale adverse-selection index manifest"):
        build_or_load_adverse_selection_index(tape, config=_cfg(root))
    rebuilt = build_or_load_adverse_selection_index(tape, config=_cfg(root, overwrite=True))
    assert rebuilt.manifest.schema == ADVERSE_SELECTION_INDEX_SCHEMA
