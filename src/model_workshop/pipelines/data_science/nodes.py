import logging
from typing import Dict, Tuple, List, Any

import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlflow.models import ModelSignature
from mlflow.models.signature import infer_signature
from mlflow.types import ColSpec, Schema, DataType
import matplotlib.pyplot as plt

from model import WorkshopModel


def infer_model_signature(df: pd.DataFrame) -> ModelSignature:
    return ModelSignature(
        infer_signature(model_input=df).inputs,
        Schema(
            [
                ColSpec(type=DataType.float, name="probability"),
                ColSpec(type=DataType.boolean, name="decision")
            ]
        )
    )

def train_model(X_train: pd.DataFrame, y_train: pd.Series, fit_intercept: bool, model_features: List) -> WorkshopModel:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    model = WorkshopModel(fit_intercept=fit_intercept, model_features=model_features)
    model.fit_preprocessed(X_train, y_train)

    return model

def log_model(model: WorkshopModel, model_signature: ModelSignature) -> None:
    conda_env = mlflow.sklearn.get_default_conda_env()

    mlflow.pyfunc.log_model(
        artifact_path="model",
        code_path=["src/model.py"],
        python_model=model,
        signature=model_signature,
        artifacts=[],
        conda_env=conda_env
    )

def evaluate_model(
        model: WorkshopModel, X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the coefficient of determination.
       Uses SHAP explainer to log feature importance.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for weight.
    """
    y_pred = model.predict_preprocessed(X_test)['decision']
    score = accuracy_score(y_test, y_pred)

    # mlflow.shap.log_explanation(model.predict, X_test)

    logger = logging.getLogger(__name__)
    logger.info("Score: %.3f", score)


def plot_counts(data: pd.DataFrame, name: str) -> None:
    plt.bar('dataset', len(data))
    
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), f"counts/{name}.png")
    plt.close()