"""Filter cells and wells"""

import polars as pl


def filter_cells(input_path: str, output_path: str) -> None:
    # filter cells based on nuclear:cellular area
    profiles_path = input_path.replace(
        "profiles_tcdropped_filtered_var_mad_outlier_featselect", "profiles"
    )

    cell_IDs = (
        pl.scan_parquet(profiles_path)
        .select([
            "Metadata_well_position",
            "Metadata_Plate",
            "Metadata_ImageNumber",
            "Metadata_ObjectNumber",
            "Nuclei_AreaShape_Area",
            "Cells_AreaShape_Area",
        ])
        .with_columns(
            (pl.col("Nuclei_AreaShape_Area") / pl.col("Cells_AreaShape_Area")).alias(
                "Nucleus_Cell_Area"
            ),
            pl.concat_str(
                [
                    "Metadata_Plate",
                    "Metadata_well_position",
                    "Metadata_ImageNumber",
                    "Metadata_ObjectNumber",
                ],
                separator="_",
            ).alias("Metadata_CellID"),
        )
        .filter(
            (pl.col("Nucleus_Cell_Area") > 0.1) & (pl.col("Nucleus_Cell_Area") < 0.4)
        )
        .collect()
    )
    cell_IDs = cell_IDs.select("Metadata_CellID").to_series().to_list()

    # read in input data
    dframe = (
        pl.scan_parquet(input_path)
        .with_columns(
            pl.concat_str(
                [
                    "Metadata_Plate",
                    "Metadata_well_position",
                    "Metadata_ImageNumber",
                    "Metadata_ObjectNumber",
                ],
                separator="_",
            ).alias("Metadata_CellID")
        )
        .filter(pl.col("Metadata_CellID").is_in(cell_IDs))
        .collect()
    )
    dframe.write_parquet(output_path, compression="gzip")


def correct_metadata(input_data: str, meta_path: str, output_data: str) -> None:
    meta = pl.read_csv(meta_path)
    profiles = pl.read_parquet(input_data)

    # Add new symbol and allele annotations
    meta = (
        meta.with_columns([
            pl.col("imaging_plate_R1").str.replace("B7A1R1_P0", "B7A1R1_P4"),
            pl.col("imaging_plate_R2").str.replace("B8A1R2_P0", "B8A1R2_P4"),
        ])
        .select([
            "imaging_well",
            "imaging_plate_R1",
            "imaging_plate_R2",
            "final_symbol",
            "final_gene_allele",
        ])
        .rename({
            "imaging_well": "Metadata_imaging_well",
            "imaging_plate_R1": "Metadata_imaging_plate_R1",
            "imaging_plate_R2": "Metadata_imaging_plate_R2",
            "final_symbol": "Metadata_symbol",
            "final_gene_allele": "Metadata_gene_allele",
        })
    )

    # join with old profiles
    profiles = profiles.drop(["Metadata_symbol", "Metadata_gene_allele"]).join(
        meta,
        on=[
            "Metadata_imaging_well",
            "Metadata_imaging_plate_R1",
            "Metadata_imaging_plate_R2",
        ],
    )

    # filter wells with NA (allele mixture or empty)
    profiles = profiles.filter(~pl.col("Metadata_symbol").is_null())

    # save results
    profiles.write_parquet(output_data, compression="gzip")
