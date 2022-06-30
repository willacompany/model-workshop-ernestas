from kedro.pipeline import Pipeline, node

from .nodes import transform_input, preprocess, split_data


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=transform_input,
                inputs=["input_raw", 'parameters'],
                outputs="data_transformed",
                name="transform_input",
            ),
            node(
                func=split_data,
                inputs=["data_transformed", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data",
            ),
            node(
                func=preprocess,
                inputs=["X_train", 'params:features'],
                outputs="X_train_preprocessed",
                name="preprocess_X_train",
            ),
            node(
                func=preprocess,
                inputs=["X_test", 'params:features'],
                outputs="X_test_preprocessed",
                name="preprocess_X_test",
            )
        ]
    )