import preprocess
import utils
import classification

rule remove_nan:
    input:
        "outputs/batch_profiles/{batch}/raw.parquet"
    output:
        "outputs/batch_profiles/{batch}/filtered.parquet"
    run:
        preprocess.filter_nan(*input, *output)

rule wellpos:
    input:
        "outputs/batch_profiles/{batch}/filtered.parquet"
    output:
        "outputs/batch_profiles/{batch}/filtered_wellpos.parquet"
    params:
        parallel = config['parallel']
    run:
        preprocess.subtract_well_mean_polar(*input, *output)

rule plate_stats:
    input:
        "outputs/batch_profiles/{batch}/filtered.parquet"
    output:
        "outputs/batch_profiles/{batch}/plate_stats.parquet"
    run:
        preprocess.compute_norm_stats_polar(*input, *output)

rule select_variant_feats:
    input:
        "outputs/batch_profiles/{batch}/{pipeline}.parquet",
        "outputs/batch_profiles/{batch}/plate_stats.parquet"
    output:
        "outputs/batch_profiles/{batch}/{pipeline}_var.parquet",
    run:
        preprocess.select_variant_features_polars(*input, *output)

rule mad:
    input:
        "outputs/batch_profiles/{batch}/{pipeline}.parquet",
        "outputs/batch_profiles/{batch}/plate_stats.parquet"
    output:
        "outputs/batch_profiles/{batch}/{pipeline}_mad.parquet"
    run:
        preprocess.robustmad(input[0], input[1], *output)

rule outlier_removal:
    input: 
        "outputs/batch_profiles/{batch}/{pipeline}.parquet",
    output:
        "outputs/batch_profiles/{batch}/{pipeline}_outlier.parquet",
    run:
        preprocess.clean.outlier_removal_polars(*input, *output)

rule feat_select:
    input:
        "outputs/batch_profiles/{batch}/{pipeline}.parquet"
    output:
        "outputs/batch_profiles/{batch}/{pipeline}_featselect.parquet"
    run:
        preprocess.feat_select(*input, *output)

rule classify:
    input:
        "outputs/batch_profiles/{batch}/{pipeline}.parquet"
    output:
        "outputs/results/{batch}/{pipeline}/feat_importance.csv",
        "outputs/results/{batch}/{pipeline}/result.csv",
    run:
        classification.run_classify_workflow(*input, *output)