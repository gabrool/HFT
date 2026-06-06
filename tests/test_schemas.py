import pytest

from mmrt.config import DEFAULT_SOURCE_DATA_TYPES
from mmrt.contracts import TardisDataType
from mmrt.schemas import (
    BOOK_SNAPSHOT_25_SCHEMA,
    COMMON_TARDIS_COLUMNS,
    INCREMENTAL_BOOK_L2_SCHEMA,
    TRADES_SCHEMA,
    ColumnKind,
    TardisCSVSchema,
    supported_tardis_schema_types,
    tardis_csv_schema,
)


def test_common_columns_constant():
    assert COMMON_TARDIS_COLUMNS == ("exchange", "symbol", "timestamp", "local_timestamp")


def test_book_snapshot_25_columns_layout_and_nullable_levels():
    schema = tardis_csv_schema(TardisDataType.BOOK_SNAPSHOT_25)
    assert schema is BOOK_SNAPSHOT_25_SCHEMA
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


def test_supported_tardis_schema_types_are_current_only():
    assert supported_tardis_schema_types() == (
        TardisDataType.INCREMENTAL_BOOK_L2,
        TardisDataType.BOOK_SNAPSHOT_25,
        TardisDataType.TRADES,
    )


def test_tardis_csv_schema_current_types_only():
    assert tardis_csv_schema(TardisDataType.INCREMENTAL_BOOK_L2) is INCREMENTAL_BOOK_L2_SCHEMA
    assert tardis_csv_schema(TardisDataType.BOOK_SNAPSHOT_25) is BOOK_SNAPSHOT_25_SCHEMA
    assert tardis_csv_schema(TardisDataType.TRADES) is TRADES_SCHEMA


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


def test_no_config_dependency_defaults_are_schema_covered():
    for dtype in DEFAULT_SOURCE_DATA_TYPES:
        assert tardis_csv_schema(dtype).data_type == dtype


def test_tardis_csv_schema_constructor_validates_current_types_only():
    with pytest.raises(ValueError):
        TardisCSVSchema("not_a_current_type", INCREMENTAL_BOOK_L2_SCHEMA.columns)


def test_no_unused_tardis_schema_symbols():
    import mmrt.schemas as schemas

    forbidden = (
        "BOOK" + "_SNAPSHOT_5_SCHEMA",
        "BOOK" + "_TICKER_SCHEMA",
        "DERIVATIVE" + "_TICKER_SCHEMA",
        "LIQ" + "UIDATIONS" + "_SCHEMA",
        "Feature" + "Field",
        "Feature" + "Schema",
        "DECISION" + "_ROW_FIXED_COLUMNS",
        "LABEL" + "_ROW_FIXED_COLUMNS",
    )
    for name in forbidden:
        assert not hasattr(schemas, name)
