"""CLI for storage-backed MMRT preprocessing audits.

This command audits train-only z-score, variance-floor, and clipping behavior
for an already-written storage dataset. It does not ingest raw Tardis CSV,
compute features or labels, create splits, train models, or mutate storage.
"""
import argparse, json
from mmrt.analysis.preprocess_audit import PreprocessAuditConfig, run_preprocess_audit, write_preprocess_audit_artifacts, DEFAULT_PREPROCESS_AUDIT_MAX_SAMPLE_ROWS
from mmrt.linear import extractors as ex
from mmrt.linear import preprocess as pp
from mmrt.storage import reader as rd

def _positive_int(v:str)->int:
    x=int(v)
    if x<=0: raise argparse.ArgumentTypeError("must be positive")
    return x

def _nonnegative_int(v:str)->int:
    x=int(v)
    if x<0: raise argparse.ArgumentTypeError("must be nonnegative")
    return x

def _positive_float(v:str)->float:
    x=float(v)
    if x<=0: raise argparse.ArgumentTypeError("must be positive")
    return x

def build_arg_parser():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset-root",required=True); p.add_argument("--output-dir",required=True)
    p.add_argument("--batch-size",type=_positive_int,default=rd.DEFAULT_BATCH_SIZE)
    p.add_argument("--max-sample-rows-per-split",type=_nonnegative_int,default=DEFAULT_PREPROCESS_AUDIT_MAX_SAMPLE_ROWS)
    p.add_argument("--clip-z",type=_positive_float,default=pp.DEFAULT_CLIP_Z); p.add_argument("--variance-floor",type=_positive_float,default=pp.DEFAULT_VARIANCE_FLOOR)
    p.add_argument("--extractor-dtype",choices=ex.ALLOWED_EXTRACTOR_DTYPES,default="float32"); p.add_argument("--preprocess-dtype",choices=pp.ALLOWED_PREPROCESS_DTYPES,default="float32")
    p.add_argument("--feature-columns",default=None); p.add_argument("--no-validate-on-open",action="store_true")
    return p

def main(argv=None):
    a=build_arg_parser().parse_args(argv)
    features=None if not a.feature_columns else tuple([x.strip() for x in a.feature_columns.split(",") if x.strip()])
    cfg=PreprocessAuditConfig(batch_size=a.batch_size,validate_dataset_on_open=(not a.no_validate_on_open),max_sample_rows_per_split=a.max_sample_rows_per_split,extractor_config=ex.LinearFeatureExtractorConfig(feature_columns=features,output_dtype=a.extractor_dtype),preprocess_config=pp.LinearPreprocessConfig(variance_floor=a.variance_floor,clip_z=a.clip_z,output_dtype=a.preprocess_dtype))
    result=run_preprocess_audit(a.dataset_root,config=cfg); paths=write_preprocess_audit_artifacts(result,a.output_dir)
    print(json.dumps({"status":"ok","summary_json":paths["summary_json"],"features_csv":paths["features_csv"],"warnings":list(result.warnings),"splits":{k:v.__dict__ for k,v in result.splits.items()}},sort_keys=True,separators=(",",":"),allow_nan=True))
    return 0

__all__=["build_arg_parser","main"]
if __name__=="__main__":
    raise SystemExit(main())
