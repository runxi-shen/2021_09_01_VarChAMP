'''
Get data points with map scores
'''

import pathlib
import polars as pl

# write function that gets the sc profiles for all cells with map scores
def getMapCells(map_path: str, data_path: str, output_path: str):
    # read in map data
    map_df = pl.read_parquet(map_path)
    map_cells = map_df[["Metadata_CellID"]]

    # get only rows with map scores
    df = pl.scan_parquet(data_path)
    df = df.filter(pl.col("Metadata_CellID").is_in(map_cells['Metadata_CellID']))
    df = df.collect()

    # join map info to data
    map_data = map_df.join(df, on="Metadata_CellID", how="inner")

    map_data.write_parquet(output_path, compression="gzip")

    return


def main():

    batch_name = 'B1A1R1'
    data_dir = pathlib.Path("/dgx1nas1/storage/data/jess/varchamp/sc_data/processed_profiles").resolve(strict=True)
    map_dir = pathlib.Path("/dgx1nas1/storage/data/jess/varchamp/sc_data/map_results/").resolve(strict=True)
    map_data_dir = pathlib.Path("/dgx1nas1/storage/data/jess/varchamp/sc_data/map_data/").resolve(strict=True)

    # map file paths
    bl_map = pathlib.Path(f'{map_dir}/baseline_map.parquet')
    well_map = pathlib.Path(f'{map_dir}/well_corrected_map.parquet')
    norm_map = pathlib.Path(f'{map_dir}/mad_normalized_map.parquet')
    cc_map = pathlib.Path(f'{map_dir}/cc_regression_map.parquet')

    # data file paths
    bl_file = pathlib.Path(data_dir / f"{batch_name}_annotated_cellID.parquet")
    well_file = pathlib.Path(data_dir / f"{batch_name}_annotated_corrected_wellmean.parquet")
    norm_file = pathlib.Path(data_dir / f"{batch_name}_annotated_corrected_normalized.parquet")
    cc_file = pathlib.Path(data_dir / f"{batch_name}_annotated_corrected_cc.parquet")

    # output file paths
    bl_out = pathlib.Path(f'{map_data_dir}/bl_map_data.parquet')
    well_out = pathlib.Path(f'{map_data_dir}/well_map_data.parquet')
    norm_out = pathlib.Path(f'{map_data_dir}/norm_map_data.parquet')
    cc_out = pathlib.Path(f'{map_data_dir}/cc_map_data.parquet')

    # subset data to only include those with map
    getMapCells(bl_map, bl_file, bl_out)
    getMapCells(well_map, well_file, well_out)
    getMapCells(norm_map, norm_file, norm_out)
    getMapCells(cc_map, cc_file, cc_out)





if __name__ == '__main__':
    main()