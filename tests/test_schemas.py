import pytest

from mmrt.config import DEFAULT_SOURCE_DATA_TYPES
from mmrt.contracts import TardisDataType
from mmrt.schemas import (
    BOOK_TICKER_SCHEMA,
    COMMON_TARDIS_COLUMNS,
    DECISION_ROW_FIXED_COLUMNS,
    DERIVATIVE_TICKER_SCHEMA,
    FeatureField,
    FeatureSchema,
    INCREMENTAL_BOOK_L2_SCHEMA,
    LABEL_ROW_FIXED_COLUMNS,
    LIQUIDATIONS_SCHEMA,
    TardisCSVSchema,
    TRADES_SCHEMA,
    ColumnKind,
    feature_names_hash,
    supported_tardis_schema_types,
    tardis_csv_schema,
)


def test_common_columns_constant():
    assert COMMON_TARDIS_COLUMNS == ("exchange", "symbol", "timestamp", "local_timestamp")


def test_book_snapshot_25_columns_layout_and_nullable_levels():
    schema = tardis_csv_schema(TardisDataType.BOOK_SNAPSHOT_25)
    assert schema.depth_limit == 25
    assert len(schema.column_names) == 4 + 25 * 4
    assert schema.column_names[:8] == (
        "exchange",
        "symbol",
        "timestamp",
        "local_timestamp",
        "asks[0].price",
        "asks[0].amount",
        "bids[0].price",
        "bids[0].amount",
    )
    assert schema.column_names[-4:] == ("asks[24].price", "asks[24].amount", "bids[24].price", "bids[24].amount")
    assert all(col.nullable for col in schema.columns[4:])


def test_book_snapshot_5_columns_layout():
    schema = tardis_csv_schema(TardisDataType.BOOK_SNAPSHOT_5)
    assert len(schema.column_names) == 4 + 5 * 4
    assert schema.depth_limit == 5
    assert schema.column_names[-4:] == ("asks[4].price", "asks[4].amount", "bids[4].price", "bids[4].amount")


def test_incremental_book_l2_columns():
    schema = INCREMENTAL_BOOK_L2_SCHEMA
    assert schema.column_names == (
        "exchange",
        "symbol",
        "timestamp",
        "local_timestamp",
        "is_snapshot",
        "side",
        "price",
        "amount",
    )
    assert schema.columns[4].kind == ColumnKind.BOOL
    assert schema.columns[5].kind == ColumnKind.BOOK_SIDE
    assert schema.columns[7].nullable is False


def test_trades_columns():
    schema = TRADES_SCHEMA
    assert schema.column_names == (
        "exchange",
        "symbol",
        "timestamp",
        "local_timestamp",
        "id",
        "side",
        "price",
        "amount",
    )
    assert schema.columns[4].nullable is True
    assert schema.columns[5].kind == ColumnKind.SIDE


def test_book_ticker_columns_order_and_nullable():
    schema = BOOK_TICKER_SCHEMA
    assert schema.column_names == (
        "exchange",
        "symbol",
        "timestamp",
        "local_timestamp",
        "ask_amount",
        "ask_price",
        "bid_price",
        "bid_amount",
    )
    assert all(col.nullable for col in schema.columns[4:])


def test_derivative_ticker_columns_order_and_nullable():
    schema = DERIVATIVE_TICKER_SCHEMA
    assert schema.column_names == (
        "exchange",
        "symbol",
        "timestamp",
        "local_timestamp",
        "funding_timestamp",
        "funding_rate",
        "predicted_funding_rate",
        "open_interest",
        "last_price",
        "index_price",
        "mark_price",
    )
    assert all(col.nullable for col in schema.columns[4:])


def test_liquidations_columns():
    schema = LIQUIDATIONS_SCHEMA
    assert schema.column_names == (
        "exchange",
        "symbol",
        "timestamp",
        "local_timestamp",
        "id",
        "side",
        "price",
        "amount",
    )
    assert schema.columns[4].nullable is True
    assert schema.columns[5].kind == ColumnKind.SIDE


def test_unsupported_schemas_raise():
    with pytest.raises(ValueError):
        tardis_csv_schema(TardisDataType.OPTIONS_CHAIN)
    with pytest.raises(ValueError):
        tardis_csv_schema(TardisDataType.QUOTES)


def test_supported_tardis_schema_types():
    types = supported_tardis_schema_types()
    assert TardisDataType.OPTIONS_CHAIN not in types
    assert TardisDataType.QUOTES not in types
    assert types == (
        TardisDataType.INCREMENTAL_BOOK_L2,
        TardisDataType.BOOK_SNAPSHOT_25,
        TardisDataType.BOOK_SNAPSHOT_5,
        TardisDataType.TRADES,
        TardisDataType.BOOK_TICKER,
        TardisDataType.DERIVATIVE_TICKER,
        TardisDataType.LIQUIDATIONS,
    )


def test_header_validation_modes():
    schema = tardis_csv_schema(TardisDataType.TRADES)
    schema.validate_header(schema.column_names)
    with pytest.raises(ValueError):
        schema.validate_header(("bad",), exact=True)
    with pytest.raises(ValueError):
        schema.validate_header(tuple(reversed(schema.column_names)), exact=True)
    schema.validate_header(tuple(reversed(schema.column_names)), exact=False)
    with pytest.raises(ValueError):
        schema.validate_header(schema.column_names[:-1], exact=False)


def test_column_index():
    schema = tardis_csv_schema(TardisDataType.TRADES)
    assert schema.column_index("timestamp") == 2
    with pytest.raises(ValueError):
        schema.column_index("missing")


def test_feature_field_validation():
    FeatureField("mid_ret_1000000us", "price")
    with pytest.raises(ValueError):
        FeatureField("Mid_ret_1000000us", "price")
    with pytest.raises(ValueError):
        FeatureField("mid-ret", "price")
    with pytest.raises(ValueError):
        FeatureField("mid_ret_1000000us", "price", dtype="float64")
    with pytest.raises(ValueError):
        FeatureField("mid_ret_1000000us", "")


def test_feature_schema_validation_and_helpers():
    f1 = FeatureField("mid_ret_200000us", "price", unit="bps")
    f2 = FeatureField("spread_bps", "book", unit="bps")
    schema = FeatureSchema("v1", (f1, f2))
    assert schema.dim == 2
    assert schema.names == ("mid_ret_200000us", "spread_bps")
    assert schema.index("spread_bps") == 1
    assert schema.select_by_family("price") == (f1,)

    with pytest.raises(ValueError):
        FeatureSchema("v1", (f1, FeatureField("mid_ret_200000us", "book")))
    with pytest.raises(ValueError):
        FeatureSchema("v1", (f1, "not-a-feature"))


def test_feature_names_hash_behavior():
    names = ("a", "b", "c")
    assert feature_names_hash(names) == feature_names_hash(names)
    assert feature_names_hash(("a", "b")) != feature_names_hash(("b", "a"))
    with pytest.raises(ValueError):
        feature_names_hash(("dup", "dup"))
    with pytest.raises(ValueError):
        feature_names_hash(())


def test_fixed_storage_columns():
    assert DECISION_ROW_FIXED_COLUMNS == ("ts_us", "local_ts_us", "source_row", "raw_mid", "dt_us")
    assert all(("_m" + "s") not in name for name in DECISION_ROW_FIXED_COLUMNS)
    assert all("aux" not in name and "core" not in name for name in DECISION_ROW_FIXED_COLUMNS)
    assert LABEL_ROW_FIXED_COLUMNS == ("decision_ts_us", "entry_ts_us")


def test_no_config_dependency_defaults_are_schema_covered():
    for dtype in DEFAULT_SOURCE_DATA_TYPES:
        assert tardis_csv_schema(dtype).data_type == dtype


def test_tardis_csv_schema_constructor_validates_unsupported_type():
    with pytest.raises(ValueError):
        TardisCSVSchema(TardisDataType.OPTIONS_CHAIN, INCREMENTAL_BOOK_L2_SCHEMA.columns)
