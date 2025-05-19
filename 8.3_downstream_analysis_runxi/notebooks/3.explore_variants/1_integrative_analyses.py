import marimo

__generated_with = "0.8.22"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(r"""# Integrative Analysis""")
    return


@app.cell
def __():
    # imports
    import os
    import polars as pl
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from pathlib import Path
    import matplotlib.pyplot as plt
    import seaborn as sns
    import umap
    from functools import reduce
    import operator
    from adjustText import adjust_text

    CELL_CHANGES = ["localization", "morphology"]
    BIO_PLATE_MAP_BATCHES = ["2024_01_23_Batch_7", "2024_12_09_Batch_11", "2025_01_27_Batch_13", "2025_03_17_Batch_15"]
    BIO_BATCHES = ["2024_02_Batch_7-8", "2024_12_Batch_11-12", "2025_01_Batch_13-14", "2025_03_Batch_15-16"]

    # pl.Config.set_tbl_rows(20)  # Sets the number of rows displayed
    # BIO_REP_BATCHES = ["2024_01_23_Batch_7", "2024_02_06_Batch_8"]
    # COMBINED_BIO_REP_DIR = "2024_02_Batch_7-8"
    # BIO_REP_BATCHES = ["2024_12_09_Batch_11", "2024_12_09_Batch_12"]
    # COMBINED_BIO_REP_DIR = "2024_12_Batch_11-12"
    # BIO_REP_BATCHES = ["2024_12_09_Batch_11_widefield", "2024_12_09_Batch_12_widefield"]
    # COMBINED_BIO_REP_DIR = "2024_12_Batch_11-12_widefield"
    # BIO_REP_BATCHES = ["2025_01_27_Batch_13", "2025_01_28_Batch_14"]
    # COMBINED_BIO_REP_DIR = "2025_01_Batch_13-14"
    # BIO_REP_BATCHES = ["2025_03_17_Batch_15", "2025_03_17_Batch_16"]
    return (
        BIO_BATCHES,
        BIO_PLATE_MAP_BATCHES,
        CELL_CHANGES,
        Path,
        adjust_text,
        np,
        operator,
        os,
        pd,
        pl,
        plt,
        reduce,
        sns,
        tqdm,
        umap,
    )


@app.cell
def __(mo):
    mo.md(r"""## 1. Loading variants and ClinVar annotations""")
    return


@app.cell
def __(mo):
    mo.md(r"""### 1.1 Variant info""")
    return


@app.cell
def __(BIO_BATCHES, BIO_PLATE_MAP_BATCHES, operator, os, pl, reduce):
    profiled_variants_df, profiled_variants_pass_qc_df = pl.DataFrame(), pl.DataFrame()

    for batch, batch_biorep in zip(BIO_PLATE_MAP_BATCHES, BIO_BATCHES):
        profiled_variants_df_batch = pl.DataFrame()
        platemap_dir = f"../../../8.2_updated_snakemake_pipeline/inputs/metadata/platemaps/{batch}/platemap"
        for platemap in os.listdir(platemap_dir):
            platemap_df = pl.read_csv(os.path.join(platemap_dir, platemap), separator="\t")
            profiled_variants_df_batch = pl.concat([profiled_variants_df_batch, 
                                                    platemap_df.filter((~pl.col("node_type").is_in(["TC","NC","PC"]))&(pl.col("node_type").is_not_null()))], 
                                                    how="diagonal_relaxed")

        ## load the img qc results
        allele_qc_df = pl.read_csv(f"../../outputs/{batch_biorep}/well-level_img_qc_gfp_sum.csv")
        # Step 1: Filter only rows with PASS
        allele_qc_df_pass = allele_qc_df.filter(pl.col("QC_Flag")=="PASS")
        # Step 2: Group by gene_allele and Batch, count occurrences
        df_grouped = allele_qc_df_pass.select([
            pl.col("gene_allele"),
            pl.col("Batch"),
            pl.col("QC_Flag")
        ]).unique()
        # Step 3: Pivot to wide format
        df_pivot = df_grouped.pivot(
            values="QC_Flag",
            index="gene_allele",
            on="Batch"
        )
        # Step 4: Filter alleles with PASS in both batches
        # (They will have non-null values in both columns)
        # Step 4: Build filter expression dynamically for non-null in all batches
        batches = allele_qc_df_pass.select("Batch").unique().get_column("Batch").to_list()
        # Step 4: Dynamically build filter expression to check non-null in all batches
        filter_expr = reduce(
            operator.and_,
            [pl.col(batch).is_not_null() for batch in batches]
        )
        # Step 5: Apply filter to get only alleles with PASS in all batches
        df_result = df_pivot.filter(filter_expr)
        # print(df_result)
        profiled_variants_pass_qc_df_batch = profiled_variants_df_batch.filter(
            (pl.col("gene_allele").is_in(df_result["gene_allele"]))
        )

        profiled_variants_df = pl.concat([profiled_variants_df, profiled_variants_df_batch], 
                                          how="diagonal_relaxed")
        profiled_variants_pass_qc_df = pl.concat([profiled_variants_pass_qc_df, profiled_variants_pass_qc_df_batch], 
                                          how="diagonal_relaxed")

    profiled_variants_df = profiled_variants_df.drop(["control_type", "allele_set", "imaging_plate", "batch1", "batch2", "clone_ID", "hgmd_ID"])
    profiled_variants_pass_qc_df = profiled_variants_pass_qc_df.drop(["control_type", "allele_set", "imaging_plate", "batch1", "batch2", "clone_ID", "hgmd_ID"])
    return (
        allele_qc_df,
        allele_qc_df_pass,
        batch,
        batch_biorep,
        batches,
        df_grouped,
        df_pivot,
        df_result,
        filter_expr,
        platemap,
        platemap_df,
        platemap_dir,
        profiled_variants_df,
        profiled_variants_df_batch,
        profiled_variants_pass_qc_df,
        profiled_variants_pass_qc_df_batch,
    )


@app.cell
def __():
    # profiled_variants_pass_qc_df.filter(pl.col("symbol").str.contains("LIG4"))
    return


@app.cell
def __():
    # profiled_variants_df["node_type"].unique()
    ## All variants have a reference, yeah!
    # [gene for gene in profiled_variants_pass_qc_df["gene_allele"].str.split("_").list.get(0) if gene not in profiled_variants_pass_qc_df["symbol"].unique()]
    return


@app.cell
def __(pl, profiled_variants_df, profiled_variants_pass_qc_df):
    print(profiled_variants_df.unique(subset="gene_allele").filter((pl.col("node_type")!="disease_wt") & (pl.col("gene_allele")!=pl.col("symbol"))).shape)
    print(profiled_variants_pass_qc_df.unique(subset="gene_allele").filter((pl.col("node_type")!="disease_wt") & (pl.col("gene_allele")!=pl.col("symbol"))).shape)

    print(profiled_variants_df.unique(subset="symbol").shape)
    print(profiled_variants_pass_qc_df.unique(subset="symbol").shape)
    return


@app.cell
def __():
    1477 / 1578, 336 / 349
    return


@app.cell
def __(BIO_BATCHES, operator, pl, reduce):
    change_sum_local, change_sum_morph = pl.DataFrame(), pl.DataFrame()
    change_wtvar = {}

    for bio_batch in BIO_BATCHES:
        ## load the img qc results
        allele_qc_df = pl.read_csv(f"../../outputs/{bio_batch}/well-level_img_qc_gfp_sum.csv")
        # Step 1: Filter only rows with PASS
        allele_qc_df_pass = allele_qc_df.filter(pl.col("QC_Flag")=="PASS")
        # Step 2: Group by gene_allele and Batch, count occurrences
        df_grouped = allele_qc_df_pass.select([
            pl.col("gene_allele"),
            pl.col("Batch"),
            pl.col("QC_Flag")
        ]).unique()
        # Step 3: Pivot to wide format
        df_pivot = df_grouped.pivot(
            values="QC_Flag",
            index="gene_allele",
            on="Batch"
        )
        # Step 4: Filter alleles with PASS in both batches
        # (They will have non-null values in both columns)
        # Step 4: Build filter expression dynamically for non-null in all batches
        batches = allele_qc_df_pass.select("Batch").unique().get_column("Batch").to_list()
        # Step 4: Dynamically build filter expression to check non-null in all batches
        filter_expr = reduce(
            operator.and_,
            [pl.col(batch).is_not_null() for batch in batches]
        )
        # Step 5: Apply filter to get only alleles with PASS in all batches
        df_result = df_pivot.filter(filter_expr)
        # print(df_result)
        ## Get the mislocalization hits
        change_sum_local_batch = pl.read_csv(f"../../outputs/{bio_batch}/altered_local_summary_auroc.csv")
        change_sum_local_batch = change_sum_local_batch.filter(
            (pl.col("allele_0").is_in(df_result["gene_allele"]))
        )
        change_sum_local = pl.concat([change_sum_local, change_sum_local_batch], how="diagonal_relaxed")
        ## Get the altered morphology hits
        if bio_batch != "2024_02_Batch_7-8":
            change_sum_morph_batch = pl.read_csv(f"../../outputs/{bio_batch}/altered_morph_summary_auroc.csv")
            change_sum_morph_batch = change_sum_morph_batch.filter(
                (pl.col("allele_0").is_in(df_result["gene_allele"]))
            )
            change_sum_morph = pl.concat([change_sum_morph, change_sum_morph_batch], how="diagonal_relaxed")

    change_wtvar["localization"] = change_sum_local
    change_wtvar["morphology"] = change_sum_morph
    return (
        allele_qc_df,
        allele_qc_df_pass,
        batches,
        bio_batch,
        change_sum_local,
        change_sum_local_batch,
        change_sum_morph,
        change_sum_morph_batch,
        change_wtvar,
        df_grouped,
        df_pivot,
        df_result,
        filter_expr,
    )


@app.cell
def __(change_wtvar, pl):
    misloc_vars = change_wtvar["localization"].filter((pl.col("Altered_local_both_batches"))).unique(subset="allele_0")["allele_0"]
    misloc_genes = change_wtvar["localization"].filter((pl.col("Altered_local_both_batches"))).with_columns(
        pl.col("allele_0").str.split("_").list.get(0).alias("gene_symbol")
    ).unique(subset="gene_symbol")["gene_symbol"]
    return misloc_genes, misloc_vars


@app.cell
def __(mo):
    mo.md(r"""### 1.2 ClinVar Info""")
    return


@app.cell
def __(pl, profiled_variants_pass_qc_df, sns):
    # clin_var_scores = pl.read_csv("/home/shenrunx/igvf/varchamp/2025_laval_submitted/3_integrated_assay_results/1_inputs/ai_pred_struc_scores.tsv", 
    #                          separator="\t", infer_schema_length=10000)
    # clin_var_scores = clin_var_scores.with_columns(
    #     pl.concat_str([pl.col("symbol"), pl.col("aa_change")], separator="_").alias("allele_0")
    # )
    # clin_var_scores.head()

    clin_var_scores = pl.scan_parquet(
        "/home/shenrunx/igvf/varchamp/2025_laval_submitted/4_compare_AI_scores/3_outputs/processed_data/dbnsfp/dbNSFP5.0a_variant.clin_var_re-annot_pdb_variants_plddt_rsa.parquet"
    )
    clin_var_scores = clin_var_scores.with_columns(
        pl.concat_str([pl.col("genename"), pl.col("clinvar_aa_change")], separator="_").alias("allele_0"),
        pl.col("clinvar_clnsig_clean").cast(pl.String).alias("clinvar_clnsig_clean_old")
    ).with_columns(
        pl.when(pl.col("clinvar_clnsig_clean_old")=="2_Benign")
        .then(pl.lit("4_Benign"))
        .when(pl.col("clinvar_clnsig_clean_old")=="4_VUS")
        .then(pl.lit("2_VUS"))
        # .when(pl.col("clinvar_clnsig_clean_old")=="3_Conflicting")
        # .then(pl.lit("2_Conflicting"))
        .otherwise(pl.col("clinvar_clnsig_clean_old"))
        .alias("clinvar_clnsig_clean")
    )
    clin_var_scores = clin_var_scores.filter(pl.col("allele_0").is_in(profiled_variants_pass_qc_df["gene_allele"])).collect()
    # clin_var_scores.collect()
    # clin_var_scores

    clinvar_palette = sns.color_palette("Set2")
    clinvar_palette[3], clinvar_palette[0] = clinvar_palette[0], clinvar_palette[3]
    clinvar_palette_old = sns.color_palette("Set2")
    clinvar_palette_old[0], clinvar_palette_old[1], clinvar_palette_old[3] = clinvar_palette_old[3], clinvar_palette_old[0], clinvar_palette_old[1]
    return clin_var_scores, clinvar_palette, clinvar_palette_old


@app.cell
def __(
    clin_var_scores,
    clinvar_palette_old,
    pl,
    plt,
    profiled_variants_pass_qc_df,
):
    profiled_variants_df_clinvar = profiled_variants_pass_qc_df.join(
        clin_var_scores.select(
            pl.col(["allele_0","plddt","rsa","clinvar_clnsig_clean","clinvar_clnsig_clean_old"])
        ), 
        left_on="gene_allele", right_on="allele_0", how="left"
    ).filter(
        (pl.col("node_type")!="disease_wt") & (pl.col("gene_allele")!=pl.col("symbol")) 
    ).with_columns(
        pl.col("clinvar_clnsig_clean").fill_null("6_No_ClinVar"),
        pl.col("clinvar_clnsig_clean_old").fill_null("6_No_ClinVar")
    ).unique(subset="gene_allele") ## get the unique alleles only

    df = profiled_variants_df_clinvar.to_pandas()
    df["symbol"] = df["gene_allele"].apply(lambda x: x.split("_")[0])

    # Count category frequencies
    counts = df["clinvar_clnsig_clean_old"].value_counts()

    # Sort categories if you want specific order
    category_order = sorted(df["clinvar_clnsig_clean_old"].unique())
    counts = counts.reindex([cat for cat in category_order if cat in counts.index])

    # Plot pie chart
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        counts,
        labels=counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=clinvar_palette_old
    )
    ax.set_title(f"ClinVar Categories of\n{profiled_variants_df_clinvar.shape[0]} Profiled Variants",y=.94)
    plt.tight_layout()
    plt.show()
    return ax, category_order, counts, df, fig, profiled_variants_df_clinvar


@app.cell
def __(category_order, clinvar_palette_old, df, plt):
    # 2. Count per (gene, category)
    top_num = 40
    counts = (
        df.groupby(["symbol", "clinvar_clnsig_clean_old"])
        .size()
        .reset_index(name="count")
    )
    # 3. Pivot
    pivot = counts.pivot(index="symbol", columns="clinvar_clnsig_clean_old", values="count").fillna(0)

    # Add total count column and sort by it (descending)
    pivot["total_count"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total_count", ascending=False).head(top_num)
    pivot = pivot.drop(columns=["total_count"])

    # 4. Ensure category order
    for cat in category_order:
        if cat not in pivot.columns:
            pivot[cat] = 0
    pivot = pivot[category_order]

    # 5. Plotting
    fig, ax = plt.subplots(figsize=(6, 7))
    # Get Set2 colors
    bottom = None
    for cat, color in zip(category_order, clinvar_palette_old):
        ax.barh(pivot.index, pivot[cat], left=bottom, label=cat, color=color)
        bottom = pivot[cat] if bottom is None else bottom + pivot[cat]
    # Axis formatting
    ax.set_xlabel("Variant Count")
    ax.set_ylim(-1, pivot.shape[0] - .5)  # Tighten y-axis to remove top/bottom gaps
    ax.set_title(f"ClinVar Categories per Gene (Top {top_num} with Most Variants)", fontsize=14)
    ax.invert_yaxis()
    ax.legend(title="ClinVar Label", fontsize=9, title_fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.show()
    return ax, bottom, cat, color, counts, fig, pivot, top_num


@app.cell
def __(mo):
    mo.md(r"""### 1.3 Called hits""")
    return


@app.cell
def __(change_wtvar, display, pl, profiled_variants_pass_qc_df):
    # pl.Config.set_tbl_rows(20)  # Sets the number of rows displayed
    print(profiled_variants_pass_qc_df.filter((pl.col("node_type")!="disease_wt") & (pl.col("gene_allele")!=pl.col("symbol"))).unique(subset="gene_allele").shape)
    print(profiled_variants_pass_qc_df.filter((pl.col("node_type")!="disease_wt") & (pl.col("gene_allele")!=pl.col("symbol"))).unique(subset="symbol").shape)

    display(change_wtvar["localization"].unique(subset="allele_0"))
    # change_wtvar["localization"].filter((pl.col("allele_0").str.contains("FARS2"))).sort(by="AUROC_Mean", descending=True) #(pl.col(f"Altered_local_both_batches"))
    display(change_wtvar["localization"].filter(pl.col("Altered_local_both_batches")).sort(by=["allele_0","AUROC_Mean"]).unique(subset="allele_0"))
    display(change_wtvar["localization"].with_columns(
        pl.col("allele_0").str.split("_").list.get(0).alias("gene")
    ).sort(by=["allele_0","AUROC_Mean"]).unique(subset="gene"))

    change_wtvar["localization"].filter(pl.col("Altered_local_both_batches")).with_columns(
        pl.col("allele_0").str.split("_").list.get(0).alias("gene")
    ).sort(by=["allele_0","AUROC_Mean"]).unique(subset="gene")
    return


@app.cell
def __():
    74/162, 279/1273
    return


@app.cell
def __(change_wtvar, display, pl, profiled_variants_pass_qc_df):
    print(profiled_variants_pass_qc_df.filter((~pl.col("plate_map_name").str.contains("B7A"))&(~pl.col("plate_map_name").str.contains("B8A"))&\
                                        (pl.col("node_type")!="disease_wt")&(pl.col("gene_allele")!=pl.col("symbol"))).unique(subset="gene_allele").shape)
    print(profiled_variants_pass_qc_df.filter((~pl.col("plate_map_name").str.contains("B7A"))&(~pl.col("plate_map_name").str.contains("B8A"))&\
                                        (pl.col("node_type")!="disease_wt")&(pl.col("gene_allele")!=pl.col("symbol"))).unique(subset="symbol").shape)
    display(change_wtvar["morphology"].unique(subset="allele_0"))
    display(change_wtvar["morphology"].filter(pl.col(f"Altered_morph_both_batches")).sort(by=["allele_0","AUROC_Mean"]).unique(subset="allele_0"))
    display(change_wtvar["morphology"].with_columns(
        pl.col("allele_0").str.split("_").list.get(0).alias("gene")
    ).sort(by=["allele_0","AUROC_Mean"]).unique(subset="gene"))
    change_wtvar["morphology"].filter(pl.col("Altered_morph_both_batches")).with_columns(
        pl.col("allele_0").str.split("_").list.get(0).alias("gene")
    ).sort(by=["allele_0","AUROC_Mean"]).unique(subset="gene")
    return


@app.cell
def __():
    5/59, 20/730
    return


@app.cell
def __(CELL_CHANGES, change_wtvar, pl, plt, profiled_variants_df):
    # import plotly.express as px
    # def interactive_gene_summary_barplot(df, cell_change):
    #     df = df.copy()
    #     df["pct_hits"] = df["len_hits"] / df["len"] * 100
    #     df = df.sort_values(by=["len", "pct_hits"], ascending=False).reset_index(drop=True)

    #     fig = px.bar(
    #         df,
    #         x="len",
    #         y="by",
    #         orientation="h",
    #         color="len_hits",
    #         hover_data=["len", "len_hits", "pct_hits"],
    #         color_continuous_scale="RdBu",
    #         labels={"by": "Gene", "len": "Total Variants", "len_hits": "Hits"},
    #         title=f"Altered {cell_change.capitalize()} Hits per Gene (Interactive)"
    #     )
    #     fig.update_layout(
    #         yaxis=dict(autorange="reversed"),
    #         height=max(800, len(df)*10),
    #         margin=dict(l=120, r=20, t=40, b=20),
    #     )
    #     fig.show()


    def plot_gene_level_summary_horizontal(total_allele_hit_sum_df, cell_change):
        # 1. Fill nulls with 0
        total_allele_hit_sum_df["len_hits"] = total_allele_hit_sum_df["len_hits"].fillna(0).astype(int)

        # 2. Compute percentage of hits
        total_allele_hit_sum_df["pct_hits"] = (
            total_allele_hit_sum_df["len_hits"] / total_allele_hit_sum_df["len"] * 100
        )

        # 3. Sort by percentage descending
        total_allele_hit_sum_df = total_allele_hit_sum_df.sort_values(
            by=["pct_hits","len"], ascending=False
        ).reset_index(drop=True)

        # 4. Keep the hits gene only
        total_allele_hit_sum_df = total_allele_hit_sum_df[total_allele_hit_sum_df["pct_hits"]>0] #.reset_index(drop=True)
        # display(total_allele_hit_sum_df)

        y = range(len(total_allele_hit_sum_df))
        n_genes = len(y)

        # 4. Dynamically scale figure size
        bar_height = 0.5
        fig_height = max(10, min(n_genes * bar_height, 20))  # cap to prevent overgrowth
        fig_width = 6
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Horizontal bars
        ax.barh(y, total_allele_hit_sum_df["len"], color="skyblue", label="# Variants")
        ax.barh(y, total_allele_hit_sum_df["len_hits"], color="tomato", label="# of Hits")

        # Annotate percent
        for i, pct in enumerate(total_allele_hit_sum_df["pct_hits"]):
            ax.text(
                total_allele_hit_sum_df.loc[i, "len"] + 0.5,
                i,
                f"{pct:.1f}%",
                va="center",
                ha="left",
                fontsize=8,
            )

        # Axis formatting
        ax.set_yticks(y)
        ax.set_xlim(None, max(total_allele_hit_sum_df["len"]+15))  # Tighten y-axis to remove top/bottom gaps
        ax.set_ylim(-1, len(y) - .5)  # Tighten y-axis to remove top/bottom gaps
        ax.set_yticklabels(total_allele_hit_sum_df["by"], fontsize=9)
        ax.set_xlabel("Count", fontsize=11)
        ax.set_title(
            f"Hits for Altered {cell_change.capitalize()} per Gene (sorted by % hits)",
            fontsize=13
        )
        ax.invert_yaxis()  # Highest hit on top
        ax.legend(fontsize=9, loc="upper right")
        # Improve spacing
        plt.tight_layout()
        plt.show()

    plt.clf()
    for cell_change in CELL_CHANGES:
        total_allele_hit_sum_df = profiled_variants_df.filter(pl.col("node_type")=="allele").unique(subset="gene_allele").group_by(by="symbol").len().sort(by="len", descending=True).join(
            change_wtvar[cell_change].filter(pl.col(f"Altered_{cell_change[:5]}_both_batches")).with_columns(
                pl.col("allele_0").str.extract(r"^([^_]+)", 1).alias("symbol")
            ).group_by(by="symbol").len().sort(by="len", descending=True), on="by", suffix="_hits", how="left"
        ).to_pandas()
        plot_gene_level_summary_horizontal(total_allele_hit_sum_df, cell_change)
    return (
        cell_change,
        plot_gene_level_summary_horizontal,
        total_allele_hit_sum_df,
    )


@app.cell
def __(change_wtvar, clin_var_scores, clinvar_palette, plt, sns):
    plt.clf()
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    sns.boxenplot(data=change_wtvar["localization"].join(clin_var_scores, on="allele_0").to_pandas(), x="clinvar_clnsig_clean", y="AUROC_Mean", saturation=.8, ax=axes[0],
                  palette=clinvar_palette[:-1],
                  order=sorted(change_wtvar["localization"].join(clin_var_scores, on="allele_0").to_pandas()["clinvar_clnsig_clean"].unique()))
    # sns.swarmplot(data=change_wtvar["localization"].join(clin_var_scores, on="allele_0").to_pandas(), x="clinvar_clnsig_clean", y="AUROC_Mean", ax=axes[0],
    #               palette=clinvar_palette[:-1],
    #               order=sorted(change_wtvar["localization"].join(clin_var_scores, on="allele_0").to_pandas()["clinvar_clnsig_clean"].unique()))
    axes[0].set_title("Protein-Mislocalization Score")
    sns.boxenplot(data=change_wtvar["morphology"].join(clin_var_scores, on="allele_0").to_pandas(), x="clinvar_clnsig_clean", y="AUROC_Mean", saturation=.8, ax=axes[1],
                  palette=clinvar_palette[:-1],
                  order=sorted(change_wtvar["morphology"].join(clin_var_scores, on="allele_0").to_pandas()["clinvar_clnsig_clean"].unique()))
    # sns.swarmplot(data=change_wtvar["morphology"].join(clin_var_scores, on="allele_0").to_pandas(), x="clinvar_clnsig_clean", y="AUROC_Mean", ax=axes[1],
    #               palette=clinvar_palette[:-1],
    #               order=sorted(change_wtvar["morphology"].join(clin_var_scores, on="allele_0").to_pandas()["clinvar_clnsig_clean"].unique()))
    axes[1].set_title("Altered-Morphology Scores")
    plt.show()
    return axes, fig


@app.cell
def __():
    # change_wtvar["localization"].filter((pl.col("allele_0").str.contains("STXBP1"))).sort(by="AUROC_Mean", descending=True) #(pl.col(f"Altered_local_both_batches"))
    # change_wtvar["localization"].filter((pl.col("allele_0").str.contains("MLH1"))).sort(by="AUROC_Mean", descending=True) #(pl.col(f"Altered_local_both_batches"))
    return


@app.cell
def __(mo):
    mo.md(r"""## 2. Loading cell profiles""")
    return


@app.cell
def __():
    BATCH_LIST_DICT = {
        "2024_01_23_Batch_7": "2024_02_Batch_7-8", 
        "2024_02_06_Batch_8": "2024_02_Batch_7-8",
        "2024_12_09_Batch_11": "2024_12_Batch_11-12", 
        "2024_12_09_Batch_12": "2024_12_Batch_11-12",
        "2025_01_27_Batch_13": "2025_01_Batch_13-14", 
        "2025_01_28_Batch_14": "2025_01_Batch_13-14",
        "2025_03_17_Batch_15": "2025_03_Batch_15-16", 
        "2025_03_17_Batch_16": "2025_03_Batch_15-16"
    }
    return (BATCH_LIST_DICT,)


@app.cell
def __(BATCH_LIST_DICT, pl, profiled_variants_pass_qc_df):
    # Paths
    pass_qc_prof_pq_path = "../../../8.2_updated_snakemake_pipeline/outputs/batch_profiles/{batch_id}/profiles_tcdropped_filtered_var_mad_outlier_featselect_filtcells.parquet"
    ref_var_cell_qc_profile_df = pl.DataFrame()

    # Get meta features
    for batch_id in BATCH_LIST_DICT.keys():
        batch_alleles = (
            pl.scan_parquet(
                pass_qc_prof_pq_path.format(batch_id=batch_id)
            )
            .filter(pl.col("Metadata_gene_allele").is_in(profiled_variants_pass_qc_df["gene_allele"].unique()))
            .with_columns(
                pl.concat_str(
                    [
                        "Metadata_Plate",
                        "Metadata_Well",
                        "Metadata_ImageNumber",
                        "Metadata_ObjectNumber",
                    ],
                    separator="_",
                ).alias("Metadata_CellID")
            )
            .select([
                "Metadata_CellID",
                "Metadata_gene_allele",
                "Metadata_Well",
                "Metadata_Plate",
                "Metadata_node_type"
            ])
        )
        ref_var_cell_qc_profile_df = pl.concat([ref_var_cell_qc_profile_df, batch_alleles.collect()])
    return (
        batch_alleles,
        batch_id,
        pass_qc_prof_pq_path,
        ref_var_cell_qc_profile_df,
    )


@app.cell
def __(BATCH_LIST_DICT, pl, profiled_variants_pass_qc_df):
    # Paths
    prof_pq_path = "../../../8.2_updated_snakemake_pipeline/outputs/batch_profiles/{batch_id}/profiles.parquet"
    ref_var_cell_all_profiles = {}

    # Get meta features
    for batch_id in BATCH_LIST_DICT.keys():
        batch_alleles = (
            pl.scan_parquet(
                prof_pq_path.format(batch_id=batch_id)
            )
            .filter(pl.col("Metadata_gene_allele").is_in(profiled_variants_pass_qc_df["gene_allele"].unique()))
            .with_columns(
                pl.concat_str(
                    [
                        "Metadata_Plate",
                        "Metadata_Well",
                        "Metadata_ImageNumber",
                        "Metadata_ObjectNumber",
                    ],
                    separator="_",
                ).alias("Metadata_CellID")
            )
        )
        ref_var_cell_all_profiles[batch_id] = batch_alleles
    return batch_alleles, batch_id, prof_pq_path, ref_var_cell_all_profiles


@app.cell
def __(pl, ref_var_cell_all_profiles, ref_var_cell_qc_profile_df):
    gfp_feats = [i for i in list(ref_var_cell_all_profiles.values())[0].collect_schema() if "GFP" in i and "Brightfield" not in i]
    gfp_feats = ["Metadata_CellID"] + gfp_feats

    ref_var_cell_all_gfp_profiles_df = pl.concat([batch_alleles[1].select(gfp_feats).collect() for batch_alleles in ref_var_cell_all_profiles.items()])
    # This method of joining ensures that only cells past filters are included in the abundance analysis
    ref_var_cell_gfp_profile_df = ref_var_cell_qc_profile_df.join(ref_var_cell_all_gfp_profiles_df, on="Metadata_CellID", how="left")

    morph_feats = [i for i in list(ref_var_cell_all_profiles.values())[2].collect_schema() if ("DNA" in i or "Mito" in i or "AGP" in i) and "Brightfield" not in i]
    morph_feats = ["Metadata_CellID"] + morph_feats

    ref_var_cell_all_morph_profiles_df = pl.concat([batch_alleles[1].select(morph_feats).collect() for batch_alleles in ref_var_cell_all_profiles.items() if "Batch_7" not in batch_alleles[0] and "Batch_8" not in batch_alleles[0]])
    # This method of joining ensures that only cells past filters are included in the abundance analysis
    ref_var_cell_morph_profile_df = ref_var_cell_qc_profile_df.join(ref_var_cell_all_morph_profiles_df, on="Metadata_CellID", how="inner")
    return (
        gfp_feats,
        morph_feats,
        ref_var_cell_all_gfp_profiles_df,
        ref_var_cell_all_morph_profiles_df,
        ref_var_cell_gfp_profile_df,
        ref_var_cell_morph_profile_df,
    )


@app.cell
def __(np, pd, pl, plt, sns, umap):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from scipy.stats import rankdata, norm


    def inverse_normal_transform(x):
        ranks = rankdata(x, method="average")
        return norm.ppf((ranks - 0.5) / len(ranks))


    # Define INT function for a single column
    def inverse_normal_transform_series(series: pl.Series) -> pl.Series:
        values = series #.to_numpy()
        ranks = rankdata(values, method="average")
        n = len(ranks)
        transformed = norm.ppf((ranks - 0.5) / n)
        return pl.Series(series.name, transformed)


    def int_num_cols(df):
        # 2) pick the columns you want to INT (here all numeric columns)
        num_cols = [
            name
            for name, dtype in df.schema.items()
            if "Metadata_" not in name
        ]

        # 3) for each numeric column, pull out a numpy array, compute the INT, and wrap back
        transforms = []
        for col in num_cols:
            arr = df[col].to_numpy()
            ranks = rankdata(arr, method="average")
            quantiles = (ranks - 0.5) / len(arr)
            transformed = norm.ppf(quantiles)
            transforms.append(pl.Series(col, transformed))

        # 4) build a new DataFrame of the transformed columns
        df_int = pl.DataFrame(transforms)
        # print(df_int)

        # 5) you can then concat non-numeric columns back if needed:
        non_num = df.select(pl.exclude(num_cols))
        df_norm = non_num.hstack(df_int)
        return df_norm


    def remove_corr_feats(df, corr_thres=.8):
        # 2. Identify numeric columns
        all_num_feats = [
            name
            for name, dtype in df.schema.items()
            if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64)
        ]
        # 3. Bring that slice into pandas
        pdf = df.select(all_num_feats).to_pandas()
        # 4. Compute absolute correlation matrix
        corr = pdf.corr().abs()
        # 5. Mask out the lower triangle (we only need one side)
        upper = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        )
        # 6. Pick columns to drop: any column with correlation > threshold to *any* other
        to_drop = [col for col in upper.columns if any(upper[col] > corr_thres)]
        # 7. Drop them in your original Polars DF
        df_drop_corr_cols = df.drop(to_drop)

        return df_drop_corr_cols


    def plot_dr(df, dr_method, standardize=True, ax=None, dr_args={}, plot_args={}):
        df_val = df.select(pl.col([col for col in df.columns if not col.startswith("Metadata_")]))#.drop_nulls()
        if standardize:
            # Standardize the features before PCA
            scaler = StandardScaler()
            df_val = pd.DataFrame(scaler.fit_transform(df_val), columns=df_val.columns)

        if dr_method=="pca":
            pca = PCA(**dr_args) #n_components=10
            dr_result = pca.fit_transform(df_val)
        if dr_method=="umap":
            # Perform UMAP
            umap_model = umap.UMAP(**dr_args) ## n_neighbors=15, min_dist=0.1, n_components=2, random_state=42, n_jobs=1
            dr_result = umap_model.fit_transform(df_val.to_numpy())

        dr_res_df = pd.DataFrame(dr_result, columns=[f"Comp{i+1}" for i in range(dr_result.shape[1])])
        for key in ["hue", "style", "size"]:
            if key in plot_args and plot_args[key] in df.columns:
                dr_res_df[plot_args[key]] = df[plot_args[key]].to_numpy()

        if ax is None:
            # Plot UMAP
            fig, ax = plt.subplots(1, 1, figsize=(6, 7))

        # print(len(dr_res_df[plot_args["hue"]].unique()))
        sns.scatterplot(data=dr_res_df.sort_values(by=plot_args["hue"]), alpha=.5,
                        x="Comp1", y="Comp2",
                        palette=sns.color_palette("husl", len(dr_res_df.dropna(subset=plot_args["hue"])[plot_args["hue"]].unique())),#, 
                        s=40, ax=ax, **plot_args)
        # plt.show()
        # ax[1].legend(ncol=2, title="Metadata_Batch")
        return dr_res_df
    return (
        PCA,
        StandardScaler,
        int_num_cols,
        inverse_normal_transform,
        inverse_normal_transform_series,
        norm,
        plot_dr,
        rankdata,
        remove_corr_feats,
    )


@app.cell
def __(ref_var_cell_gfp_profile_df, ref_var_cell_morph_profile_df):
    print(ref_var_cell_morph_profile_df.shape)
    print(ref_var_cell_gfp_profile_df.shape)
    return


@app.cell
def __(mo):
    mo.md(r"""### 2.1 Aggregate to well-level""")
    return


@app.cell
def __(pl, ref_var_cell_gfp_profile_df, remove_corr_feats):
    ref_var_well_gfp_profile_df = (
        ref_var_cell_gfp_profile_df.group_by(["Metadata_Plate", "Metadata_Well", "Metadata_gene_allele"])
        .agg(
            pl.col(col).median().alias(col)
            for col in ref_var_cell_gfp_profile_df.columns
            if not col.startswith("Metadata_")
        )
        .unique()
        .drop_nulls("Metadata_gene_allele")
    )

    ref_var_well_gfp_profile_df = ref_var_well_gfp_profile_df.with_columns(
        pl.col("Metadata_Plate")
          # capture a “B” followed by any non-underscore chars up to the first “A”
          .str.extract(r"(B[^_]*?A)", 1)
          .alias("Metadata_Batch")
    ).drop_nulls()

    ref_var_well_gfp_profile_df = remove_corr_feats(ref_var_well_gfp_profile_df)
    ref_var_well_gfp_profile_df
    return (ref_var_well_gfp_profile_df,)


@app.cell
def __(pl, ref_var_well_gfp_profile_df):
    ref_var_well_gfp_profile_df.filter(pl.col("Metadata_gene_allele").str.contains("GFAP"))
    return


@app.cell
def __(int_num_cols, ref_var_well_gfp_profile_df):
    ref_var_well_gfp_profile_int_df = int_num_cols(ref_var_well_gfp_profile_df)
    return (ref_var_well_gfp_profile_int_df,)


@app.cell
def __(mo):
    mo.md(r"""## 3. Loading subcell compartment per gene""")
    return


@app.cell
def __(pd, pl):
    hpa_subcell_loc = pl.read_csv("/home/shenrunx/data_storage/igvf/varchamp/subcellular_localization/hpa_subcellular_location_annotations/subcellular_location_data.tsv", separator="\t")
    hpa_subcell_loc = hpa_subcell_loc.with_columns(
        pl.when(pl.col("Main location").str.contains(";"))
        .then(pl.lit("Multiple"))
        .otherwise(pl.col("Main location"))
        .alias("Subcell_Loc")
    )
    hpa_subcell_loc_main = hpa_subcell_loc.filter(
        (pl.col("Subcell_Loc")!="Multiple")&(pl.col("Additional location").is_null()&(pl.col("Reliability")=="Approved"))
    )

    hpa_genes = hpa_subcell_loc["Gene name"]
    # hpa_genes

    # Read a JSON file into a pandas DataFrame
    cd_code_json_file_path = "/home/shenrunx/data_storage/igvf/varchamp/subcellular_localization/cd-code/dataset.json"
    cd_code_df = pd.read_json(cd_code_json_file_path)
    # display(cd_code_df.head())
    # 1. normalize into a new DataFrame
    cell_compart_labels = pd.json_normalize(cd_code_df["labels"])#.dropna(how="all")
    # display(cell_compart_labels)
    ## 2. drop the labels
    cd_code_df = pd.concat([cd_code_df, cell_compart_labels], axis=1)
    cd_code_df = cd_code_df.dropna(subset=["gene_names"]).drop(columns=["labels"])
    cd_code_df = cd_code_df[cd_code_df["gene_names"]!=""]
    cd_code_df.sort_values(by="gene_names")
    return (
        cd_code_df,
        cd_code_json_file_path,
        cell_compart_labels,
        hpa_genes,
        hpa_subcell_loc,
        hpa_subcell_loc_main,
    )


@app.cell
def __(hpa_subcell_loc_main, pl, ref_var_well_gfp_profile_int_df):
    ref_var_well_gfp_profile_df_hpa = ref_var_well_gfp_profile_int_df.join(
        hpa_subcell_loc_main.with_columns(
            pl.col("Gene name").alias("Metadata_genename_hpa"),
            pl.col("Subcell_Loc").alias("Metadata_Subcell_Loc"),
        ).select(pl.col(["Metadata_genename_hpa", "Metadata_Subcell_Loc"])), 
        left_on="Metadata_gene_allele", right_on="Metadata_genename_hpa",
        how="left"
    )

    ref_var_well_gfp_profile_df_hpa
    # ref_var_well_gfp_profile_df_hpa.drop_nulls(subset="Metadata_Subcell_Loc")
    # ref_allele_well_profile_df_hpa.filter(pl.col("Metadata_Plate").str.contains("B11"))
    return (ref_var_well_gfp_profile_df_hpa,)


@app.cell
def __(plot_dr, plt, ref_var_well_gfp_profile_df_hpa):
    plt.clf()
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    plot_dr(ref_var_well_gfp_profile_df_hpa, "pca", dr_args={"n_components": 10}, plot_args={"hue": "Metadata_Batch"}, ax=axes[0])
    plot_dr(ref_var_well_gfp_profile_df_hpa, "pca", dr_args={"n_components": 10}, plot_args={"hue": "Metadata_Subcell_Loc"}, ax=axes[1])
    ax=axes[1].legend(ncol=1, title="Metadata_Subcell_Loc",
                 loc="upper left",
                 bbox_to_anchor=(1.02, 1.0),
                 borderaxespad=0)
    plt.show()
    return ax, axes, fig


@app.cell
def __(StandardScaler, pd, pl, ref_var_well_gfp_profile_df_hpa, umap):
    ref_var_well_gfp_profile_hpa_values = ref_var_well_gfp_profile_df_hpa.select(pl.col([col for col in ref_var_well_gfp_profile_df_hpa.columns if not col.startswith("Metadata_")]))#.drop_nulls()
    # Standardize the features before PCA
    scaler = StandardScaler()
    df_val = pd.DataFrame(scaler.fit_transform(ref_var_well_gfp_profile_hpa_values), columns=ref_var_well_gfp_profile_hpa_values.columns)

    # Perform UMAP
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42, n_jobs=1) ## 
    dr_result = umap_model.fit_transform(df_val.to_numpy())
    dr_res_df = pd.DataFrame(dr_result, columns=[f"Comp{i+1}" for i in range(dr_result.shape[1])])
    dr_res_df["allele"] = ref_var_well_gfp_profile_df_hpa["Metadata_gene_allele"]
    dr_res_df["Metadata_Batch"] = ref_var_well_gfp_profile_df_hpa["Metadata_Batch"]
    return (
        df_val,
        dr_res_df,
        dr_result,
        ref_var_well_gfp_profile_hpa_values,
        scaler,
        umap_model,
    )


@app.cell
def __(dr_res_df, plt, sns):
    plt.clf()
    dr_res_df["gene"] = dr_res_df["allele"].apply(lambda x: x.split("_")[0])
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sns.scatterplot(data=dr_res_df, alpha=.5, hue="Metadata_Batch",
                    x="Comp1", y="Comp2", palette="Set2", s=40, ax=ax[0]) ## hue=variant_type, 
    sns.scatterplot(data=dr_res_df[dr_res_df["allele"].str.contains("_")], 
                    x="Comp1", y="Comp2", hue="gene", s=30, ax=ax[1], alpha=.2,
                    palette=sns.color_palette("husl", len(dr_res_df[dr_res_df["allele"].str.contains("_")]["gene"].unique()))) ## hue=variant_type, 
    # ax[1].get_legend().remove()
    ax[1].legend(ncol=10, title="Gene",
                 loc="upper right",
                 bbox_to_anchor=(1.08, -.11),
                 borderaxespad=0)
    plt.subplots_adjust(wspace=0.1)
    plt.show()
    return ax, fig


@app.cell
def __(mo):
    mo.md(r"""## 4. Localization""")
    return


@app.cell
def __(pl, ref_var_cell_gfp_profile_df):
    ref_var_allele_gfp_profile_df = (
        ref_var_cell_gfp_profile_df.group_by(["Metadata_gene_allele"])
        .agg(
            pl.col(col).median().alias(col)
            for col in ref_var_cell_gfp_profile_df.columns
            if not col.startswith("Metadata_")
        )
        .unique()
        .drop_nulls("Metadata_gene_allele")
    )
    ref_var_allele_gfp_profile_corr_df = ref_var_allele_gfp_profile_df.select(["Metadata_gene_allele"]+[col for col in ref_var_allele_gfp_profile_df.columns if "Correlation_Correlation" in col])
    return ref_var_allele_gfp_profile_corr_df, ref_var_allele_gfp_profile_df


@app.cell
def __(ref_var_allele_gfp_profile_corr_df):
    ref_var_allele_gfp_profile_corr_df.describe()
    return


@app.cell
def __(
    change_wtvar,
    misloc_genes,
    pd,
    pl,
    ref_var_allele_gfp_profile_corr_df,
):
    misloc_variant_corr_diff_df = pd.DataFrame()
    for misloc_gene in misloc_genes:
        # fig, axes = plt.subplots(1,1,figsize=(6,5))
        # sns.boxenplot(data=change_wtvar["localization"].join(clin_var_scores, on="allele_0").filter(pl.col("allele_0").str.contains(misloc_gene)).to_pandas(), 
        #               x="clinvar_clnsig_clean", y="AUROC_Mean", saturation=.3, ax=axes,
        #             order=sorted(change_wtvar["localization"].join(clin_var_scores, on="allele_0").filter(pl.col("allele_0").str.contains(misloc_gene)).to_pandas()["clinvar_clnsig_clean"].unique()))
        # sns.swarmplot(data=change_wtvar["localization"].join(clin_var_scores, on="allele_0").filter(pl.col("allele_0").str.contains(misloc_gene)).to_pandas(), 
        #               x="clinvar_clnsig_clean", y="AUROC_Mean", ax=axes,
        #             order=sorted(change_wtvar["localization"].join(clin_var_scores, on="allele_0").filter(pl.col("allele_0").str.contains(misloc_gene)).to_pandas()["clinvar_clnsig_clean"].unique()))
        # axes.set_title(misloc_gene)

        ref_var_allele_gfp_profile_corr_df_gene = ref_var_allele_gfp_profile_corr_df.filter(pl.col("Metadata_gene_allele").str.contains(misloc_gene)).to_pandas()
        # print(misloc_gene)
        # display(ref_var_allele_gfp_profile_corr_df_gene)

        misloc_allele_profile_df_nocorr_wt = ref_var_allele_gfp_profile_corr_df_gene[~ref_var_allele_gfp_profile_corr_df_gene["Metadata_gene_allele"].str.contains("_")].copy().set_index("Metadata_gene_allele")
        misloc_allele_profile_df_nocorr_var = ref_var_allele_gfp_profile_corr_df_gene[ref_var_allele_gfp_profile_corr_df_gene["Metadata_gene_allele"].str.contains("_")].copy()
        misloc_allele_profile_df_nocorr_var = ref_var_allele_gfp_profile_corr_df_gene[(ref_var_allele_gfp_profile_corr_df_gene["Metadata_gene_allele"].str.contains("_"))&\
                                                                          ref_var_allele_gfp_profile_corr_df_gene["Metadata_gene_allele"].isin(
                                                                              change_wtvar["localization"].filter((pl.col(f"Altered_local_both_batches")))["allele_0"]
                                                                          )].copy()
        misloc_allele_profile_df_nocorr_var["Gene"] = misloc_allele_profile_df_nocorr_var["Metadata_gene_allele"].apply(lambda x: x.split("_")[0])

        misloc_allele_var_minus_wt = pd.DataFrame()
        try:
            for gene, group in misloc_allele_profile_df_nocorr_var.groupby("Gene"):
                # print(gene)
                # display(misloc_allele_profile_df_nocorr_wt.loc[gene])
                group = group.set_index("Metadata_gene_allele").drop("Gene", axis=1).copy()
                g_col = group.columns
                # display(group)
                group = group - misloc_allele_profile_df_nocorr_wt.loc[gene]
                # display(misloc_allele_profile_df_nocorr_wt.loc[[gene]])
                misloc_allele_var_minus_wt = pd.concat([misloc_allele_var_minus_wt, group], axis=0)
                # display(misloc_allele_var_minus_wt)
        except:
            continue
        # display(misloc_allele_var_minus_wt)
        # print(misloc_allele_var_minus_wt.shape[0])
        # print(misloc_allele_var_minus_wt.shape[0])
        misloc_variant_corr_diff_df = pd.concat([misloc_variant_corr_diff_df, misloc_allele_var_minus_wt])
    return (
        g_col,
        gene,
        group,
        misloc_allele_profile_df_nocorr_var,
        misloc_allele_profile_df_nocorr_wt,
        misloc_allele_var_minus_wt,
        misloc_gene,
        misloc_variant_corr_diff_df,
        ref_var_allele_gfp_profile_corr_df_gene,
    )


@app.cell
def __():
    # ref_var_allele_gfp_profile_corr_df = ref_var_allele_gfp_profile_corr_df.filter(pl.col("Metadata_gene_allele").is_in(misloc_vars)).to_pandas()
    # misloc_variant_corr_diff_df
    return


@app.cell
def __(misloc_variant_corr_diff_df, pl, plot_dr, plt):
    misloc_variant_corr_diff_df["Metadata_gene"] = [idx.split("_")[0] for idx in misloc_variant_corr_diff_df.index]
    plt.clf()
    fig, ax = plt.subplots(1,1,figsize=(8,7))
    umap_misloc_var_corr_diff = plot_dr(pl.DataFrame(misloc_variant_corr_diff_df), "umap", plot_args={"hue": "Metadata_gene"}, ax=ax)
    ax.legend(ncol=3, title="Gene",
                 loc="upper left",
                 bbox_to_anchor=(1.02, 1.0),
                 borderaxespad=0)
    plt.show()
    return ax, fig, umap_misloc_var_corr_diff


@app.cell
def __():
    # display(umap_misloc_var_corr_diff[umap_misloc_var_corr_diff["Comp1"]<(-2)])
    # display(umap_misloc_var_corr_diff[umap_misloc_var_corr_diff["Comp1"]>10])
    # display(umap_misloc_var_corr_diff[umap_misloc_var_corr_diff["Metadata_gene"].str.contains("RHO")])
    # display(umap_misloc_var_corr_diff[umap_misloc_var_corr_diff["Metadata_gene"].str.contains("ACSF3")])
    return


@app.cell
def __(misloc_variant_corr_diff_df, plt, sns):
    plt.clf()
    g = sns.clustermap(misloc_variant_corr_diff_df.drop("Metadata_gene",axis=1), # .loc[pd.Index(gfap_alleles),col_stds_sig.index].T 
                       cmap="vlag", col_cluster=True, row_cluster=True, figsize=(6, 20), cbar_pos=(1., .32, .03, .2))
    # Change tick label font sizes
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=8, rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=8)

    # Optional: Change colorbar font size
    g.cax.tick_params(labelsize=10)
    plt.show()
    return (g,)


@app.cell
def __(misloc_variant_corr_diff_df, plt, sns):
    plt.clf()
    g = sns.clustermap(misloc_variant_corr_diff_df[sorted(misloc_variant_corr_diff_df.columns, key=lambda x: x.split("_")[-2])].drop("Metadata_gene",axis=1), # .loc[pd.Index(gfap_alleles),col_stds_sig.index].T 
                       cmap="vlag", col_cluster=False, row_cluster=True, figsize=(6, 20), cbar_pos=(1., .32, .03, .2), yticklabels=0)
    # Change tick label font sizes
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=8, rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=8)
    rows_to_label = [idx for idx in misloc_variant_corr_diff_df.index if "RHO" in idx]

    # # Get y-axis tick positions after clustering
    # tick_labels = g.ax_heatmap.get_yticklabels()
    # tick_pos = g.ax_heatmap.get_yticks()
    # # Map label to y-tick position
    # label_pos_dict = {
    #     tick.get_text(): pos 
    #     for tick, pos in zip(tick_labels, tick_pos)
    # }
    # print(label_pos_dict)
    # # Add labels only for selected rows
    # texts = []
    # for label in rows_to_label:
    #     if label in label_pos_dict:
    #         y = label_pos_dict[label]
    #         texts.append(g.ax_heatmap.text(
    #             x=misloc_variant_corr_diff_df[sorted(misloc_variant_corr_diff_df.columns, key=lambda x: x.split("_")[-2])].drop("Metadata_gene",axis=1).shape[1] + 0.2,  # to the right of the heatmap
    #             y=y,
    #             s=label,
    #             va='center',
    #             fontsize=10,
    #             color='black'
    #         ))
    # Adjust text to prevent overlap
    # adjust_text(texts, ax=g.ax_heatmap, only_move={'points': 'y', 'texts': 'y'})

    # Optional: Change colorbar font size
    # g.cax.tick_params(labelsize=10)

    plt.show()
    return g, rows_to_label


@app.cell
def __(misloc_variant_corr_diff_df, plt, sns):
    plt.clf()
    rows_to_label = [idx for idx in misloc_variant_corr_diff_df.index if "RHO" in idx]
    g = sns.clustermap(misloc_variant_corr_diff_df.loc[rows_to_label, sorted(misloc_variant_corr_diff_df.columns, key=lambda x: x.split("_")[-2])].drop("Metadata_gene",axis=1).T, # .loc[pd.Index(gfap_alleles),col_stds_sig.index].T 
                       cmap="vlag", col_cluster=True, row_cluster=False, figsize=(5, 5), cbar_pos=(.85, .1, .03, .2), yticklabels=1)
    # Change tick label font sizes
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=9, rotation=30, ha="right")
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=9)
    g.ax_heatmap.set_xlabel("Gene Variant", fontsize=10)
    g.ax_heatmap.set_ylabel("CP Colocalization Feature", fontsize=10)
    plt.show()
    return g, rows_to_label


@app.cell
def __(
    change_wtvar,
    int_num_cols,
    misloc_genes,
    pd,
    pl,
    ref_var_allele_gfp_profile_df,
):
    ref_var_allele_gfp_profile_df = int_num_cols(ref_var_allele_gfp_profile_df)

    misloc_variant_gfp_feat_diff_df = pd.DataFrame()
    for misloc_gene in misloc_genes:
        # fig, axes = plt.subplots(1,1,figsize=(6,5))
        # sns.boxenplot(data=change_wtvar["localization"].join(clin_var_scores, on="allele_0").filter(pl.col("allele_0").str.contains(misloc_gene)).to_pandas(), 
        #               x="clinvar_clnsig_clean", y="AUROC_Mean", saturation=.3, ax=axes,
        #             order=sorted(change_wtvar["localization"].join(clin_var_scores, on="allele_0").filter(pl.col("allele_0").str.contains(misloc_gene)).to_pandas()["clinvar_clnsig_clean"].unique()))
        # sns.swarmplot(data=change_wtvar["localization"].join(clin_var_scores, on="allele_0").filter(pl.col("allele_0").str.contains(misloc_gene)).to_pandas(), 
        #               x="clinvar_clnsig_clean", y="AUROC_Mean", ax=axes,
        #             order=sorted(change_wtvar["localization"].join(clin_var_scores, on="allele_0").filter(pl.col("allele_0").str.contains(misloc_gene)).to_pandas()["clinvar_clnsig_clean"].unique()))
        # axes.set_title(misloc_gene)

        ref_var_allele_gfp_profile_df_gene = ref_var_allele_gfp_profile_df.filter(pl.col("Metadata_gene_allele").str.contains(misloc_gene)).to_pandas()

        misloc_allele_gfp_profile_wt = ref_var_allele_gfp_profile_df_gene[~ref_var_allele_gfp_profile_df_gene["Metadata_gene_allele"].str.contains("_")].copy().set_index("Metadata_gene_allele")
        # gene_allele_gfp_profile_var = ref_var_allele_gfp_profile_df_gene[ref_var_allele_gfp_profile_df_gene["Metadata_gene_allele"].str.contains("_")].copy()
        misloc_allele_gfp_profile_var = ref_var_allele_gfp_profile_df_gene[(ref_var_allele_gfp_profile_df_gene["Metadata_gene_allele"].str.contains("_"))&\
                                                                            ref_var_allele_gfp_profile_df_gene["Metadata_gene_allele"].isin(
                                                                              change_wtvar["localization"].filter((pl.col(f"Altered_local_both_batches")))["allele_0"]
                                                                            )].copy().set_index("Metadata_gene_allele")
        # display(misloc_allele_gfp_profile_wt)
        # display(misloc_allele_gfp_profile_var)
        if (misloc_allele_gfp_profile_wt.shape[0]>0):
            misloc_allele_var_minus_wt = misloc_allele_gfp_profile_var - misloc_allele_gfp_profile_wt.loc[misloc_gene]
        else:
            # display(misloc_allele_gfp_profile_wt)
            # display(misloc_allele_gfp_profile_var)
            continue
        # misloc_allele_gfp_profile_var_diff = misloc_allele_gfp_profile_var.subtract(misloc_allele_gfp_profile_wt, axis=1)
        # display(misloc_allele_gfp_profile_var_diff)
        # break
        # misloc_allele_gfp_profile_var["Gene"] = misloc_allele_gfp_profile_var["Metadata_gene_allele"].apply(lambda x: x.split("_")[0])

        # misloc_allele_var_minus_wt = pd.DataFrame()
        # try:
        #     for gene, group in misloc_allele_gfp_profile_var.groupby("Gene"):
        #         # print(gene)
        #         # display(misloc_allele_profile_df_nocorr_wt.loc[gene])
        #         group = group.set_index("Metadata_gene_allele").drop("Gene", axis=1).copy()
        #         # display(group)
        #         group = group - misloc_allele_profile_df_nocorr_wt.loc[gene]
        #         # display(misloc_allele_profile_df_nocorr_wt.loc[[gene]])
        # misloc_allele_var_minus_wt = pd.concat([misloc_allele_var_minus_wt, group], axis=0)
        #         # display(misloc_allele_var_minus_wt)
        # except:
        #     continue
        # # display(misloc_allele_var_minus_wt)
        # # print(misloc_allele_var_minus_wt.shape[0])
        # # print(misloc_allele_var_minus_wt.shape[0])
        misloc_variant_gfp_feat_diff_df = pd.concat([misloc_variant_gfp_feat_diff_df, misloc_allele_var_minus_wt])
    return (
        misloc_allele_gfp_profile_var,
        misloc_allele_gfp_profile_wt,
        misloc_allele_var_minus_wt,
        misloc_gene,
        misloc_variant_gfp_feat_diff_df,
        ref_var_allele_gfp_profile_df,
        ref_var_allele_gfp_profile_df_gene,
    )


@app.cell
def __(misloc_variant_gfp_feat_diff_df, pl, plot_dr, plt):
    misloc_variant_gfp_feat_diff_df["Metadata_gene"] = [idx.split("_")[0] for idx in misloc_variant_gfp_feat_diff_df.index]
    plt.clf()
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    misloc_variant_gfp_diff_umap = plot_dr(pl.DataFrame(misloc_variant_gfp_feat_diff_df), "umap", 
                                           dr_args={"random_state":42, "n_jobs":1}, plot_args={"hue": "Metadata_gene"}, ax=ax)
    ax.legend(ncol=3, title="Gene",
                 loc="upper left",
                 bbox_to_anchor=(1.02, 1.0),
                 borderaxespad=0)
    plt.show()
    return ax, fig, misloc_variant_gfp_diff_umap


@app.cell
def __():
    # misloc_variant_gfp_diff_umap[(misloc_variant_gfp_diff_umap["Comp1"]>8)&(misloc_variant_gfp_diff_umap["Comp2"]<4)]
    # misloc_variant_gfp_diff_umap[misloc_variant_gfp_diff_umap["Comp1"]>12]
    # misloc_variant_gfp_diff_umap[(misloc_variant_gfp_diff_umap["Comp1"]>4)&(misloc_variant_gfp_diff_umap["Comp2"]<4.5)]
    return


@app.cell
def __(misloc_variant_gfp_feat_diff_df, pd, plt, sns):
    # sns.clustermap(misloc_variant_no_corr_diff_df)
    gfap_alleles = [x for x in misloc_variant_gfp_feat_diff_df.index if "KCNJ2" in x]
    # col_stds = misloc_variant_no_corr_diff_df.loc[pd.Index(gfap_alleles),].drop("Metadata_gene",axis=1).std()
    # col_stds_sig = col_stds[col_stds>10]

    plt.clf()
    g = sns.clustermap(misloc_variant_gfp_feat_diff_df.loc[pd.Index(gfap_alleles),].drop(["Metadata_gene"],axis=1).T, 
                       cmap="vlag", col_cluster=True, row_cluster=True, figsize=(8, 8), cbar_pos=(1., .32, .03, .2))
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=8, rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=8)
    plt.show()
    return g, gfap_alleles


@app.cell
def __(misloc_variant_gfp_feat_diff_df, pd, plt, sns):
    # sns.clustermap(misloc_variant_no_corr_diff_df)
    gfap_alleles = [x for x in misloc_variant_gfp_feat_diff_df.index if "GFAP" in x]
    col_stds = misloc_variant_gfp_feat_diff_df.loc[pd.Index(gfap_alleles),].drop(["Metadata_gene"],axis=1).std()
    col_stds_sig = col_stds[col_stds>.8]
    plt.clf()
    g = sns.clustermap(misloc_variant_gfp_feat_diff_df.loc[pd.Index(gfap_alleles),col_stds_sig.index].T, 
                       cmap="vlag", col_cluster=True, row_cluster=True, figsize=(8, 8), cbar_pos=(.95, .39, .02, .2))
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=9, rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=9)
    plt.show()
    return col_stds, col_stds_sig, g, gfap_alleles


@app.cell
def __(misloc_variant_gfp_feat_diff_df):
    misloc_variant_gfp_feat_diff_df["Metadata_gene"] = [x.split("_")[0] for x in misloc_variant_gfp_feat_diff_df.index]
    misloc_variant_gfp_feat_diff_df["Metadata_gene_allele"] = [x for x in misloc_variant_gfp_feat_diff_df.index]
    misloc_variant_gfp_feat_diff_df
    return


@app.cell
def __(misloc_variant_gfp_feat_diff_df, umap_no_corr_df):
    umap_no_corr_df
    misloc_variant_gfp_feat_diff_df
    return


@app.cell
def __(umap_no_corr_df):
    umap_no_corr_df
    return


@app.cell
def __(adjust_text, misloc_variant_gfp_feat_diff_df, pl, plot_dr, plt):
    plt.clf()
    fig, ax = plt.subplots(1,1,figsize=(9,9))
    umap_no_corr_df = plot_dr(pl.DataFrame(misloc_variant_gfp_feat_diff_df), "umap", dr_args={"random_state":42, "n_jobs": 1},
                              plot_args={"hue": "Metadata_gene"}, ax=ax)
    umap_no_corr_df["Metadata_gene_allele"] = misloc_variant_gfp_feat_diff_df["Metadata_gene_allele"].values
    ax.legend(ncol=6, title="Gene",
                 loc="upper right",
                 bbox_to_anchor=(1.01, -.08),
                 borderaxespad=0)
    # Add text labels with adjustText
    texts = [
        plt.text(row['Comp1'], row['Comp2'], "-".join(row['Metadata_gene_allele'].split("_")), fontsize=9)
        for _, row in umap_no_corr_df.iterrows() if "GFAP" in row['Metadata_gene_allele'] or "RHO" in row['Metadata_gene_allele']
    ]
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))
    plt.show()
    return ax, fig, texts, umap_no_corr_df


@app.cell
def __(mo):
    mo.md(r"""## 5. Morphology""")
    return


@app.cell
def __(int_num_cols, pl, ref_var_cell_morph_profile_df):
    ref_var_allele_morph_prof_df = (
        ref_var_cell_morph_profile_df.group_by(["Metadata_gene_allele"])
        .agg(
            pl.col(col).median().alias(col)
            for col in ref_var_cell_morph_profile_df.columns
            if not col.startswith("Metadata_")
        )
        .unique()
        .drop_nulls("Metadata_gene_allele")
    )
    ref_var_allele_morph_prof_df = int_num_cols(ref_var_allele_morph_prof_df)
    return (ref_var_allele_morph_prof_df,)


@app.cell
def __(ref_var_allele_morph_prof_df):
    ref_var_allele_morph_prof_df
    return


@app.cell
def __(change_wtvar, misloc_genes, pd, pl, ref_var_allele_morph_prof_df):
    misloc_variant_morph_diff_df = pd.DataFrame()
    for misloc_gene in list(misloc_genes):
        # print(misloc_gene)
        ref_var_allele_morph_prof_df_gene = ref_var_allele_morph_prof_df.filter(pl.col("Metadata_gene_allele").str.contains(misloc_gene)).to_pandas()

        misloc_allele_morph_prof_wt = ref_var_allele_morph_prof_df_gene[~ref_var_allele_morph_prof_df_gene["Metadata_gene_allele"].str.contains("_")].copy().set_index("Metadata_gene_allele")
        misloc_allele_morph_prof_var = ref_var_allele_morph_prof_df_gene[(ref_var_allele_morph_prof_df_gene["Metadata_gene_allele"].str.contains("_"))&\
                                                                          ref_var_allele_morph_prof_df_gene["Metadata_gene_allele"].isin(
                                                                              change_wtvar["morphology"].filter((pl.col(f"Altered_morph_both_batches")))["allele_0"]
                                                                          )].copy().set_index("Metadata_gene_allele")
        if (misloc_allele_morph_prof_wt.shape[0]>0):
            misloc_allele_morph_var_minus_wt = misloc_allele_morph_prof_var - misloc_allele_morph_prof_wt.loc[misloc_gene]
        else:
            # display(misloc_allele_gfp_profile_wt)
            # display(misloc_allele_gfp_profile_var)
            continue
        misloc_variant_morph_diff_df = pd.concat([misloc_variant_morph_diff_df, misloc_allele_morph_var_minus_wt])
    return (
        misloc_allele_morph_prof_var,
        misloc_allele_morph_prof_wt,
        misloc_allele_morph_var_minus_wt,
        misloc_gene,
        misloc_variant_morph_diff_df,
        ref_var_allele_morph_prof_df_gene,
    )


@app.cell
def __(ref_var_allele_morph_prof_df_gene):
    ref_var_allele_morph_prof_df_gene
    return


@app.cell
def __(misloc_allele_morph_prof_wt):
    misloc_allele_morph_prof_wt
    return


if __name__ == "__main__":
    app.run()
