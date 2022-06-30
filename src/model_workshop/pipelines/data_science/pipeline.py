from kedro.pipeline import Pipeline, node

from .nodes import evaluate_model, train_model, plot_counts, infer_model_signature, log_model


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                infer_model_signature,
                ["X_train"],
                "model_signature",
                name="infer_signature"
            ),
            node(
                func=train_model,
                inputs=["X_train_preprocessed", "y_train", 'params:fit_intercept', 'params:features'],
                outputs="model",
                name="train_model",
            ),
            node(
                log_model,
                ["model", 'model_signature'],
                None,
                name="log_model"
            ),
            node(
                func=evaluate_model,
                inputs=["model", "X_test_preprocessed", "y_test"],
                outputs=None,
                name="evaluate_model",
            ),    
            node(
                func=lambda data: plot_counts(data, 'X_train_preprocessed'),
                inputs=['X_train_preprocessed'],
                outputs=None,
                name="plot_counts_X_train",
            ),
            node(
                func=lambda data: plot_counts(data, 'X_test_preprocessed'),
                inputs=['X_test_preprocessed'],
                outputs=None,
                name="plot_counts_X_test",
            ),
            node(
                func=lambda data: plot_counts(data, 'y_train'),
                inputs=['y_train'],
                outputs=None,
                name="plot_counts_y_train",
            ),
            node(
                func=lambda data: plot_counts(data, 'y_test'),
                inputs=['y_test'],
                outputs=None,
                name="plot_counts_y_test",
            ),
        ]
    )