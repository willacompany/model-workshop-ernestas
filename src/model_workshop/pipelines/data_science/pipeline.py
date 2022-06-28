
from kedro.pipeline import Pipeline, node

from .nodes import evaluate_model, split_data, train_model, plot_counts




def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["model_input_table", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", 'params:fit_intercept'],
                outputs="regressor",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["regressor", "X_test", "y_test"],
                outputs=None,
                name="evaluate_model_node",
            ),
            node(
                func=lambda data: plot_counts(data, 'X_train'),
                inputs=["X_train"],
                outputs=None,
                name="plot_counts_X_train",
            ),
        ]
    )
