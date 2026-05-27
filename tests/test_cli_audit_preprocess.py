import json
from pathlib import Path
import mmrt.cli.audit_preprocess as cli
from tests.test_analysis_preprocess_audit import _write_ds

def test_cli_writes_artifacts(tmp_path:Path, capsys):
    ds=tmp_path/'ds'; _write_ds(ds)
    out=tmp_path/'out'
    rc=cli.main(['--dataset-root',str(ds),'--output-dir',str(out)])
    assert rc==0
    assert (out/'preprocess_audit_summary.json').exists()
    assert (out/'preprocess_audit_features.csv').exists()
    payload=json.loads(capsys.readouterr().out)
    assert payload['status']=='ok' and 'warnings' in payload

def test_feature_columns_option(tmp_path:Path, capsys):
    ds=tmp_path/'ds'; _write_ds(ds)
    out=tmp_path/'out'
    cli.main(['--dataset-root',str(ds),'--output-dir',str(out),'--feature-columns','x_0,x_1'])
    _=json.loads(capsys.readouterr().out)
    txt=(out/'preprocess_audit_features.csv').read_text()
    assert 'x_0' in txt and 'x_1' in txt

def test_cli_invalid_args(tmp_path:Path):
    p=cli.build_arg_parser()
    for args in (["--dataset-root","d","--output-dir","o","--clip-z","0"],["--dataset-root","d","--output-dir","o","--variance-floor","0"],["--output-dir","o"]):
        try:
            p.parse_args(args)
            assert False
        except SystemExit:
            pass
