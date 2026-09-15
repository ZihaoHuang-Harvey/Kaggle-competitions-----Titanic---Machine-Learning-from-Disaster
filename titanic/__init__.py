"""Titanic Kaggle 机器学习项目"""
from .config import (
    ROOT_DIR, DATA_DIR, OUTPUT_DIR,
    TRAIN_PATH, TEST_PATH, CLEANED_TRAIN_PATH, SUBMISSION_PATH,
    RANDOM_STATE, TARGET, REQUIRED_COLUMNS, TRAIN_FEATURE,
    NUMERIC_FEATURE, CATEGORICAL_FEATURE, DROP_COLUMNS,
)
from .data import (
    read_csv, validate_csv, normalize_column_names,
    normalize_numeric_columns, mark_and_modify_outliers,
    clean_data, append_new_features,
)
from .modeling import MODELS, build_pipeline
from .evaluation import k_fold_evaluate, evaluate_all_models

__all__ = [
    # config
    "ROOT_DIR", "DATA_DIR", "OUTPUT_DIR",
    "TRAIN_PATH", "TEST_PATH", "CLEANED_TRAIN_PATH", "SUBMISSION_PATH",
    "RANDOM_STATE", "TARGET", "REQUIRED_COLUMNS", "TRAIN_FEATURE",
    "NUMERIC_FEATURE", "CATEGORICAL_FEATURE", "DROP_COLUMNS",
    # data
    "read_csv", "validate_csv", "normalize_column_names",
    "normalize_numeric_columns", "mark_and_modify_outliers",
    "clean_data", "append_new_features",
    # modeling
    "MODELS", "build_pipeline",
    # evaluation
    "k_fold_evaluate", "evaluate_all_models",
]
