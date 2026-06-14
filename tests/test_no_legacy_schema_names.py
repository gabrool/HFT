from pathlib import Path
import ast

ROOTS = (Path("mmrt"), Path("tests"))

FORBIDDEN_SUBSTRINGS = (
    "schema" + "_" + "version",
    "SCHEMA" + "_" + "VERSION",
    "_" + "V1",
    "_" + "V2",
    "_" + "V3",
    "_" + "v1",
    "_" + "v2",
    "_" + "v3",
    " " + "v1",
    " " + "v2",
    " " + "v3",
)

ALLOWED_SUBSTRINGS = (
    "parquet_version",
    "DEFAULT_PARQUET_VERSION",
    "mmrt_execution_decision_grid_v1",
    "mmrt_execution_linear_signals_grid_v1",
    "mmrt_linear_training_result_tape25_grid_v1",
    "mmrt_adverse_selection_dataset_grid_v1",
    "mmrt_adverse_selection_feature_dataset_grid_v1",
    "mmrt_adverse_selection_ridge_grid_v1",
    "mmrt_adverse_selection_signals_grid_v1",
    "mmrt_adverse_selection_index_grid_v1",
    "event_schedule_reason_v1",
)

PRODUCTION_ROOT = Path("mmrt")

FORBIDDEN_HISTORY_WORDS = (
    "leg" + "acy",
    "compat" + "ibility",
    "back" + "ward " + "compat" + "ibility",
    "migration",
    "deprecated",
    "no longer " + "supported",
    "old " + "schema",
    "old " + "format",
)

ALLOWED_HISTORY_SUBSTRINGS_BY_FILE = {
    Path("mmrt/metadata/__init__.py"): (
        "RuleCompatibility",
        "rule_compatibility",
    ),
    Path("mmrt/metadata/rule_compatibility.py"): (
        "Diagnostic grid compatibility checks for market data and symbol rules.",
        "RuleCompatibility",
        "compatibility report must be a mapping",
        "symbol rule compatibility strict mode failed",
    ),
    Path("mmrt/execution/contracts.py"): (
        "RuleCompatibilityReport",
        "symbol_rule_compatibility",
    ),
    Path("mmrt/execution/execution_tape.py"): (
        "RuleCompatibilityReport",
        "symbol_rule_compatibility",
    ),
    Path("mmrt/execution/execution_tape_writer.py"): (
        "RuleCompatibilityReport",
        "symbol_rule_compatibility",
    ),
    Path("mmrt/cli/build_execution_tape.py"): (
        "RuleCompatibility",
        "rule_compatibility",
        "symbol_rule_compatibility",
        "compatibility_report",
        "_coerce_compatibility_mode",
        "compatibility=compatibility",
        "compatibility=compat",
        "compatibility: RuleCompatibilityAccumulator",
        "compatibility is not None",
        "compatibility.observe_price_array",
        "compatibility.observe_qty_array",
    ),
}


def test_no_history_language_in_production_mmrt():
    offenders = []
    for path in PRODUCTION_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_HISTORY_WORDS:
            for line_no, line in enumerate(text.splitlines(), 1):
                if forbidden in line.lower():
                    rel = path
                    allowed = ALLOWED_HISTORY_SUBSTRINGS_BY_FILE.get(rel, ())
                    if any(substring in line for substring in allowed):
                        continue
                    offenders.append(f"{path}:{line_no}: {line.strip()}")
    assert offenders == []


def test_no_feature_retired_name_layer_symbols():
    text = Path("mmrt/features/specs.py").read_text(encoding="utf-8")
    forbidden = (
        "leg" + "acy_name",
        "leg" + "acy_feature_names",
        "canonical_name_for_leg" + "acy",
        "SOURCE_TO_" + "CANONICAL",
        "CANONICAL_TO_" + "SOURCE",
    )
    offenders = [s for s in forbidden if s in text]
    assert offenders == []


def _iter_py_files():
    for root in ROOTS:
        for path in root.rglob("*.py"):
            yield path


def test_no_internal_schema_release_names():
    offenders = []
    for path in _iter_py_files():
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_SUBSTRINGS:
            if forbidden not in text:
                continue
            for line_no, line in enumerate(text.splitlines(), 1):
                if forbidden in line and not any(allowed in line for allowed in ALLOWED_SUBSTRINGS):
                    offenders.append(f"{path}:{line_no}: {line.strip()}")
    assert offenders == []


def test_no_dead_generic_contract_layer_symbols():
    text = Path("mmrt/contracts.py").read_text(encoding="utf-8")
    forbidden = (
        "Event" + "Type",
        "Event" + "Meta",
        "Price" + "Level",
        "Book" + "SnapshotEvent",
        "Book" + "DeltaEvent",
        "Trade" + "Event",
        "Book" + "TickerEvent",
        "Derivative" + "TickerEvent",
        "Liquidation" + "Event",
        "Market" + "Event",
        "Feature" + "BuildResult",
        "Decision" + "RowRef",
        "Segment" + "Spec",
        "Split" + "Entry",
        "Split" + "Plan",
        "Dataset" + "Manifest",
    )
    offenders = [name for name in forbidden if name in text]
    assert offenders == []


def test_ingest_has_no_numeric_zero_coercion_or_skip_counters():
    text = Path("mmrt/cli/ingest.py").read_text(encoding="utf-8")
    forbidden = (
        "_to_float" + "_or_zero",
        "skipped_bad" + "_trade_events",
        "skipped_empty" + "_book_events",
    )
    offenders = [name for name in forbidden if name in text]
    assert offenders == []
    replay_text = Path("mmrt/execution/feature_replay.py").read_text(encoding="utf-8")
    assert "def book_snapshot_input_from_tape_row" in replay_text
    assert "return None" not in replay_text.split("def book_snapshot_input_from_tape_row", 1)[1].split("def trade_input_from_tape_row", 1)[0]
    assert "return None" not in replay_text.split("def trade_input_from_tape_row", 1)[1].split("def _l2_event_is_two_sided", 1)[0]


def test_ingest_has_no_fake_data_type_or_validation_opt_out_flags():
    text = Path("mmrt/cli/ingest.py").read_text(encoding="utf-8")
    forbidden = (
        "--book" + "-data-type",
        "--no" + "-validate-output",
        "--validate" + "-output",
        "args.validate" + "_output",
    )
    offenders = [s for s in forbidden if s in text]
    assert offenders == []


def test_no_allow_nan_true_in_mmrt():
    offenders = []
    for path in Path("mmrt").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "allow_nan" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        offenders.append(f"{path}:{node.lineno}")
    assert offenders == []
