import json
from pathlib import Path

import pytest

import mmrt.cli.audit_preprocess as cli
from mmrt.storage import manifest as mf
from tests.test_analysis_preprocess_audit import _write_ds


def test_cli_writes_artifacts(tmp_path: Path, capsys):
    ds = tmp_path / "ds"
    _write_ds(ds)
    out = tmp_path / "out"

    rc = cli.main(["--dataset-root", str(ds), "--output-dir", str(out)])
    assert rc == 0
    assert (out / "preprocess_audit_summary.json").exists()
    assert (out / "preprocess_audit_features.csv").exists()

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert "warnings" in payload


def test_feature_columns_option(tmp_path: Path, capsys):
    ds = tmp_path / "ds"
    _write_ds(ds)
    out = tmp_path / "out"

    cols = mf.feature_columns()[:2]
    cli.main(
        [
            "--dataset-root",
            str(ds),
            "--output-dir",
            str(out),
            "--feature-columns",
            ",".join(cols),
        ]
    )
    _ = json.loads(capsys.readouterr().out)

    text = (out / "preprocess_audit_features.csv").read_text()
    assert cols[0] in text
    assert cols[1] in text
    assert mf.feature_columns()[2] not in text


def test_cli_invalid_args():
    parser = cli.build_arg_parser()
    invalid = [
        ["--dataset-root", "d", "--output-dir", "o", "--clip-z", "0"],
        ["--dataset-root", "d", "--output-dir", "o", "--clip-z", "nan"],
        ["--dataset-root", "d", "--output-dir", "o", "--clip-z", "inf"],
        ["--dataset-root", "d", "--output-dir", "o", "--variance-floor", "0"],
        ["--dataset-root", "d", "--output-dir", "o", "--variance-floor", "nan"],
        ["--dataset-root", "d", "--output-dir", "o", "--extractor-dtype", "bad"],
        ["--dataset-root", "d", "--output-dir", "o", "--preprocess-dtype", "bad"],
        ["--output-dir", "o"],
    ]
    for args in invalid:
        with pytest.raises(SystemExit):
            parser.parse_args(args)


def test_cli_empty_feature_columns(tmp_path: Path):
    with pytest.raises(SystemExit):
        cli.main(["--dataset-root", "d", "--output-dir", "o", "--feature-columns", ",,,"])
