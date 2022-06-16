import logging
from typing import Dict, Tuple

import mlflow
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    y = data["weight_pounds"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, fit_intercept: bool) -> LinearRegression:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    mlflow.autolog()
    regressor = LinearRegression(fit_intercept=fit_intercept)
    regressor.fit(X_train, y_train)
    return regressor


def evaluate_model(
        regressor: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination.
       Uses SHAP explainer to log feature importance.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for weight.
    """
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)

    mlflow.shap.log_explanation(regressor.predict, X_test)

    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)
