import pandas as pd


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """ Node for basic feature engineering and normalization """
    data = data.fillna(0)
    data["mother_married"] = data["mother_married"].astype(int)
    return data
