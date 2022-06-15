from kedro.pipeline import Pipeline, node

from .nodes import preprocess


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess,
                inputs="natality",
                outputs="model_input_table",
                name="preprocess_node",
            )
        ]
    )
