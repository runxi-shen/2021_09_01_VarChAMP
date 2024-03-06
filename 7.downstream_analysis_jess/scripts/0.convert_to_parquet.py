# Download all plates in this batch: 
# $ aws s3 cp s3://cellpainting-gallery/cpg0020-varchamp/broad/workspace/backend/2023_05_30_B1A1R1/ ./  --exclude "*.csv" --recursive
# Download single cell profiles from one plates 2023-05-24_B1A1R1_P1T2 from this batch: 
# $ aws s3 cp s3://cellpainting-gallery/cpg0020-varchamp/broad/workspace/backend/2023_05_30_B1A1R1/2023-05-24_B1A1R1_P1T2/2023-05-24_B1A1R1_P1T2.sqlite
# Examine column names:
# $ sqlite3 /dgx1nas1/storage/data/sam/varchamp/2023-05-24_B1A1R1_P1T2/2023-05-24_B1A1R1_P1T2.sqlite .schema|grep -v AreaShape|grep -vE "Texture|RadialDistribution|Granularity|Location|Neighbors|Correlation|Intensity|ObjectSkeleton|ExecutionTime|ImageQuality|ModuleError|PathName|MD5Digest|Scaling|Width|FileName|URL|Threshold|Height|Count|CREATE INDEX|IncludingEdges"
# Output:
# CREATE TABLE IF NOT EXISTS "Image" (
#         "TableNumber" BIGINT, 
#         "AreaOccupied_AreaOccupied_Cells" FLOAT, 
#         "AreaOccupied_AreaOccupied_Cytoplasm" FLOAT, 
#         "AreaOccupied_AreaOccupied_Nuclei" FLOAT, 
#         "AreaOccupied_Perimeter_Cells" FLOAT, 
#         "AreaOccupied_Perimeter_Cytoplasm" FLOAT, 
#         "AreaOccupied_Perimeter_Nuclei" FLOAT, 
#         "AreaOccupied_TotalArea_Cells" FLOAT, 
#         "AreaOccupied_TotalArea_Cytoplasm" FLOAT, 
#         "AreaOccupied_TotalArea_Nuclei" FLOAT, 
#         "Group_Index" BIGINT, 
#         "Group_Length" BIGINT, 
#         "Group_Number" BIGINT, 
#         "ImageNumber" BIGINT, 
#         "Metadata_AbsPositionZ" FLOAT, 
#         "Metadata_AbsTime" TEXT, 
#         "Metadata_BinningX" BIGINT, 
#         "Metadata_BinningY" BIGINT, 
#         "Metadata_ChannelID" BIGINT, 
#         "Metadata_ChannelName" TEXT, 
#         "Metadata_Col" BIGINT, 
#         "Metadata_ExposureTime" FLOAT, 
#         "Metadata_FieldID" BIGINT, 
#         "Metadata_ImageResolutionX" FLOAT, 
#         "Metadata_ImageResolutionY" FLOAT, 
#         "Metadata_ImageSizeX" BIGINT, 
#         "Metadata_ImageSizeY" BIGINT, 
#         "Metadata_MainEmissionWavelength" BIGINT, 
#         "Metadata_MainExcitationWavelength" BIGINT, 
#         "Metadata_ObjectiveMagnification" BIGINT, 
#         "Metadata_ObjectiveNA" BIGINT, 
#         "Metadata_PlaneID" BIGINT, 
#         "Metadata_Plate" TEXT, 
#         "Metadata_PositionX" FLOAT, 
#         "Metadata_PositionY" FLOAT, 
#         "Metadata_PositionZ" FLOAT, 
#         "Metadata_Row" BIGINT, 
#         "Metadata_Site" BIGINT, 
#         "Metadata_Well" TEXT, 
# );
# CREATE TABLE IF NOT EXISTS "Cytoplasm" (
#         "TableNumber" BIGINT, 
#         "ImageNumber" BIGINT, 
#         "ObjectNumber" BIGINT, 
#         "Cytoplasm_Number_Object_Number" BIGINT, 
#         "Cytoplasm_Parent_Cells" BIGINT, 
#         "Cytoplasm_Parent_Nuclei" BIGINT, 
# );
# CREATE TABLE IF NOT EXISTS "Nuclei" (
#         "TableNumber" BIGINT, 
#         "ImageNumber" BIGINT, 
#         "ObjectNumber" BIGINT, 
#         "Nuclei_Number_Object_Number" BIGINT, 
# );
# CREATE TABLE IF NOT EXISTS "Cells" (
#         "TableNumber" BIGINT, 
#         "ImageNumber" BIGINT, 
#         "ObjectNumber" BIGINT, 
#         "Cells_Number_Object_Number" BIGINT, 
#         "Cells_Parent_Nuclei" BIGINT, 
# );

from cytotable import convert
import os
from tqdm import tqdm 

batch_full = "2023-12-22_B6A4R2"
batch_short = "B6A4R2"

data_dir = f'/dgx1nas1/storage/data/jess/varchamp/sc_data/sqlite/{batch_full}'

file_list = [os.path.join(path, name) for path, subdirs, files in os.walk(data_dir) for name in files ]
dest_path = f'/dgx1nas1/storage/data/jess/varchamp/sc_data/raw_profiles/{batch_short}/'
dest_file_list = [dest_path + i.split('/')[-1].split('.')[0] + '.parquet' for i in file_list]

identifying_cols = (
                    "TableNumber",
                    "ImageNumber",
                    "ObjectNumber",
                    "Metadata_Well",
                    "Metadata_Plate",
                    "Parent_Cells",
                    "Parent_Nuclei",
                    "Cytoplasm_Parent_Cells",
                    "Cytoplasm_Parent_Nuclei"
                )

join_command = """
                    WITH Image_Filtered AS (
                        SELECT
                            Metadata_TableNumber,
                            Metadata_ImageNumber,
                            Metadata_Well,
                            Metadata_Plate
                        FROM
                            read_parquet('image.parquet')
                        )
                    SELECT
                        *
                    FROM
                        Image_Filtered AS image
                    LEFT JOIN read_parquet('cytoplasm.parquet') AS cytoplasm ON
                        cytoplasm.Metadata_TableNumber = image.Metadata_TableNumber
                        AND cytoplasm.Metadata_ImageNumber = image.Metadata_ImageNumber
                    LEFT JOIN read_parquet('cells.parquet') AS cells ON
                        cells.Metadata_TableNumber = cytoplasm.Metadata_TableNumber
                        AND cells.Metadata_ImageNumber = cytoplasm.Metadata_ImageNumber
                        AND cells.Metadata_ObjectNumber = cytoplasm.Metadata_Cytoplasm_Parent_Cells
                    LEFT JOIN read_parquet('nuclei.parquet') AS nuclei ON
                        nuclei.Metadata_TableNumber = cytoplasm.Metadata_TableNumber
                        AND nuclei.Metadata_ImageNumber = cytoplasm.Metadata_ImageNumber
                        AND nuclei.Metadata_ObjectNumber = cytoplasm.Metadata_Cytoplasm_Parent_Nuclei
                """

print('Starting conversion.')
for i in tqdm(range(len(file_list))):
    convert(
            source_path=file_list[i],
            dest_path=dest_file_list[i],
            identifying_columns = identifying_cols,
            dest_datatype="parquet",
            chunk_size=150000,
            preset='cell-health-cellprofiler-to-cytominer-database',
            joins=join_command
    )
    print(f'Parquet file saved at {dest_file_list[i]}') 

print('Job Finished.')