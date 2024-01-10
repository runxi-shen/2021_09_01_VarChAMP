import pandas as pd

def get_features(df: pd.DataFrame) -> list:
    return df.filter(regex="^(?!Metadata_)").columns.to_list()

def get_metadata(df: pd.DataFrame) -> list:
    return df.filter(regex="^(Metadata_)").columns.to_list()