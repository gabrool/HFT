import json
from pathlib import Path

import pytest

from mmrt.cli import audit_features as cli
from mmrt.storage import manifest as mf
from tests.test_analysis_feature_audit import _write_feature_audit_ds


def test_cli_writes_all_artifacts(tmp_path: Path, capsys):
    root = tmp_path / "ds"
    outdir = tmp_path / "out"
    _write_feature_audit_ds(root)

    rc = cli.main(["--dataset-root", str(root), "--output-dir", str(outdir)])
    assert rc == 0

    for n in [
        "feature_audit_summary.json",
        "feature_health.csv",
        "feature_train_val_drift.csv",
        "feature_family_summary.csv",
        "feature_corr_top_pairs.csv",
        "feature_clusters.csv",
        "feature_cluster_summary.json",
    ]:
        assert (outdir / n).exists()

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "ok"
    assert "summary_json" in payload
    assert "health_csv" in payload
    assert "drift_csv" in payload
    assert "family_csv" in payload
    assert "corr_pairs_csv" in payload
    assert "clusters_csv" in payload
    assert "cluster_summary_json" in payload


def test_feature_columns_option(tmp_path: Path):
    root = tmp_path / "ds2"
    outdir = tmp_path / "out2"
    _write_feature_audit_ds(root)
    cols = mf.feature_columns()[:2]

    cli.main([
        "--dataset-root",
        str(root),
        "--output-dir",
        str(outdir),
        "--feature-columns",
        f"{cols[0]},{cols[1]}",
    ])
    txt = (outdir / "feature_health.csv").read_text()
    assert cols[0] in txt and cols[1] in txt and mf.feature_columns()[2] not in txt


def test_invalid_args():
    p = cli.build_arg_parser()
    bad = [
        ["--dataset-root", "x", "--output-dir", "y", "--high-corr-threshold", "0"],
        ["--dataset-root", "x", "--output-dir", "y", "--high-corr-threshold", "1"],
        ["--dataset-root", "x", "--output-dir", "y", "--min-corr-output-threshold", "0"],
        ["--dataset-root", "x", "--output-dir", "y", "--min-corr-output-threshold", "1"],
        ["--dataset-root", "x", "--output-dir", "y", "--low-variance-std-threshold", "0"],
        ["--dataset-root", "x", "--output-dir", "y", "--drift-mean-z-threshold", "nan"],
        ["--dataset-root", "x", "--output-dir", "y", "--drift-std-ratio-low", "0"],
        ["--dataset-root", "x", "--output-dir", "y", "--drift-std-ratio-low", "1"],
        ["--dataset-root", "x", "--output-dir", "y", "--drift-std-ratio-high", "1"],
        ["--dataset-root", "x", "--output-dir", "y", "--drift-std-ratio-high", "nan"],
        ["--dataset-root", "x", "--output-dir", "y", "--max-corr-pairs", "0"],
        ["--dataset-root", "x", "--output-dir", "y", "--extractor-dtype", "bad"],
        ["--output-dir", "y"],
    ]
    for argv in bad:
        with pytest.raises(SystemExit):
            p.parse_args(argv)

    with pytest.raises(SystemExit):
        cli.main(["--dataset-root", "x", "--output-dir", "y", "--feature-columns", ",,,"])

    cols = mf.feature_columns()[:1]
    with pytest.raises(SystemExit):
        cli.main([
            "--dataset-root",
            "x",
            "--output-dir",
            "y",
            "--feature-columns",
            f"{cols[0]},{cols[0]}",
        ])
