def test_analysis_package_exports_feature_lab_api():
    import mmrt.analysis as analysis

    expected = {
        "FEATURE_LAB_REPORT_TYPE",
        "DEFAULT_FEATURE_LAB_BATCH_SIZE",
        "DEFAULT_FEATURE_LAB_MAX_SAMPLE_ROWS_TRAIN",
        "DEFAULT_FEATURE_LAB_MAX_SAMPLE_ROWS_VAL",
        "DEFAULT_FEATURE_LAB_SEED",
        "DEFAULT_FEATURE_LAB_SUMMARY_FILENAME",
        "DEFAULT_CANDIDATE_HEALTH_FILENAME",
        "DEFAULT_CANDIDATE_EXISTING_CORR_FILENAME",
        "DEFAULT_CANDIDATE_REDUNDANCY_FILENAME",
        "DEFAULT_CANDIDATE_HEAD_METRICS_FILENAME",
        "DEFAULT_CANDIDATE_RECOMMENDATIONS_FILENAME",
        "FeatureLabConfig",
        "CandidateHealthRecord",
        "CandidateExistingCorrelationRecord",
        "CandidateRedundancyRecord",
        "CandidateHeadMetricRecord",
        "CandidateRecommendationRecord",
        "FeatureLabResult",
        "run_feature_lab",
        "write_feature_lab_artifacts",
    }

    assert expected.issubset(set(analysis.__all__))

    for name in expected:
        assert hasattr(analysis, name)


def test_analysis_package_does_not_export_feature_lab_cli():
    import mmrt.analysis as analysis

    assert "feature_lab" not in analysis.__all__
    assert "main" not in analysis.__all__
    assert "build_arg_parser" not in analysis.__all__
