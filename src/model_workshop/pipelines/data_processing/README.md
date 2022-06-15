# Data Processing pipeline
## Overview

This modular pipeline preprocesses the raw data and creates the model input table.

## Pipeline inputs

### `natality`

|      |                                          |
| ---- |------------------------------------------|
| Type | `pandas.DataFrame`                       |
| Description | Raw natality data from BQ public dataset |

## Pipeline outputs

### `model_input_table`

|      |                                                       |
| ---- |-------------------------------------------------------|
| Type | `pandas.DataFrame`                                    |
| Description | A complete and normalized data for training purposes. |
