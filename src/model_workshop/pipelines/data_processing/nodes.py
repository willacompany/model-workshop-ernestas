import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split

from model import WorkshopModel


def transform_input(data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    data = data[data['created_at'] >= parameters['partner_created_at_min']]

    target = parameters['target']
    data[target] = data[target].astype(int)

    return data


def preprocess(data: pd.DataFrame, model_features: List) -> pd.DataFrame:
    return WorkshopModel.preprocess(data, model_features)


def split_data(data: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple:
    target = parameters['target']

    X = data.drop(columns=target)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=parameters["test_size"], 
        random_state=parameters["random_state"],
        stratify=y
    )
    return X_train, X_test, y_train, y_test