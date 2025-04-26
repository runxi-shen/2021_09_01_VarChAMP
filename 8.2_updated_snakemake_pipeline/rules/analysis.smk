import classification


rule classify:
    input:
        "outputs/batch_profiles/{batch}/{pipeline}.parquet",
    output:
        "outputs/classification_results/{batch}/{pipeline}/feat_importance.csv",
        "outputs/classification_results/{batch}/{pipeline}/classifier_info.csv",
        "outputs/classification_results/{batch}/{pipeline}/predictions.parquet"
    benchmark:
        "benchmarks/{pipeline}_classify_{batch}.bwa.benchmark.txt"
    run:
        classification.run_classify_workflow(*input, *output, config["cc_threshold"])


rule calculate_metrics:
    input:
        "outputs/classification_results/{batch}/{pipeline}/classifier_info.csv",
        "outputs/classification_results/{batch}/{pipeline}/predictions.parquet"
    output:
        "outputs/classification_analyses/{batch}/{pipeline}/metrics.csv"
    benchmark:
        "outputs/benchmarks/{batch}/{pipeline}_calc_metrics.bwa.benchmark.txt"
    run:
        classification.calculate_class_metrics(
            *input, 
            *output
        )


rule compute_hits:
    input:
        "outputs/classification_analyses/{batch}/{pipeline}/metrics.csv"
    output:
        "outputs/classification_analyses/{batch}/{pipeline}/metrics_summary.csv"
    benchmark:
        "outputs/benchmarks/{batch}/{pipeline}_comp_hits.bwa.benchmark.txt"
    run:
        classification.compute_hits(
            *input, 
            *output,
            trn_imbal_thres = config["trn_imbal_thres"],
            min_num_classifier = config["min_num_classifier"]
        )