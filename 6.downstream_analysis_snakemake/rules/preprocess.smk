import preprocess
import utils
import classification

rule remove_nan:
    input:
        "outputs/batch_profiles/{batch}/{pipeline}.parquet"
    output:
        "outputs/batch_profiles/{batch}/{pipeline}_filtered.parquet"
    benchmark:
        "outputs/benchmarks/{batch}/{pipeline}_filtered.bwa.benchmark.txt"
    params:
        drop_threshold = 100
    run:
        preprocess.drop_nan_features(
            *input, 
            *output, 
            cell_threshold=params.drop_threshold
        )

rule drop_empty_wells:
    input: 
        "outputs/batch_profiles/{batch}/profiles.parquet",
    output: 
        "outputs/batch_profiles/{batch}/profiles_tcdropped.parquet",
    benchmark:
        "outputs/benchmarks/{batch}/profiles_tcdropped.bwa.benchmark.txt"
    run:
        preprocess.drop_empty_wells(
            *input, 
            *output, 
            pert_col=config["transfection_col"], 
            pert_name=config["trasfection_pert"]
        )

rule wellpos:
    input:
        "outputs/batch_profiles/{batch}/filtered.parquet"
    output:
        "outputs/batch_profiles/{batch}/filtered_wellpos.parquet"
    benchmark:
        "benchmarks/wellpos_{batch}.bwa.benchmark.txt"
    params:
        parallel = config['parallel']
    run:
        preprocess.subtract_well_mean_polar(*input, *output)

rule plate_stats:
    input:
        "outputs/batch_profiles/{batch}/profiles_tcdropped_filtered.parquet"
    output:
        "outputs/batch_profiles/{batch}/plate_stats.parquet"
    benchmark:
        "benchmarks/plate_stats_{batch}.bwa.benchmark.txt"
    run:
        preprocess.compute_norm_stats_polar(*input, *output)

rule select_variant_feats:
    input:
        "outputs/batch_profiles/{batch}/{pipeline}.parquet",
        "outputs/batch_profiles/{batch}/plate_stats.parquet"
    output:
        "outputs/batch_profiles/{batch}/{pipeline}_var.parquet",
    benchmark:
        "benchmarks/{pipeline}_var_{batch}.bwa.benchmark.txt"
    run:
        preprocess.select_variant_features_polars(*input, *output)

rule mad:
    input:
        "outputs/batch_profiles/{batch}/{pipeline}.parquet",
        "outputs/batch_profiles/{batch}/plate_stats.parquet"
    output:
        "outputs/batch_profiles/{batch}/{pipeline}_mad.parquet"
    benchmark:
        "benchmarks/{pipeline}_mad_{batch}.bwa.benchmark.txt"
    run:
        preprocess.robustmad(input[0], input[1], *output)

rule outlier_removal:
    input: 
        "outputs/batch_profiles/{batch}/{pipeline}.parquet",
    output:
        "outputs/batch_profiles/{batch}/{pipeline}_outlier.parquet",
    benchmark:
        "benchmarks/{pipeline}_outlier_{batch}.bwa.benchmark.txt"
    run:
        preprocess.clean.outlier_removal_polars(*input, *output)

rule feat_select:
    input:
        "outputs/batch_profiles/{batch}/{pipeline}.parquet"
    output:
        "outputs/batch_profiles/{batch}/{pipeline}_featselect.parquet"
    benchmark:
        "benchmarks/{pipeline}_feat_select_{batch}.bwa.benchmark.txt"
    run:
        preprocess.select_features(*input, *output)

rule classify:
    input:
        "outputs/batch_profiles/{batch}/{pipeline}.parquet"
    output:
        "outputs/results/{batch}/{pipeline}/feat_importance.csv",
        "outputs/results/{batch}/{pipeline}/result.csv",
    benchmark:
        "benchmarks/{pipeline}_classify_{batch}.bwa.benchmark.txt"
    run:
        classification.run_classify_workflow(*input, *output)