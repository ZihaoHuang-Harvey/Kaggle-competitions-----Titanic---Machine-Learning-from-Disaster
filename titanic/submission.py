"""用最佳模型预测测试集，输出Kaggle提交格式CSV"""
import pandas as pd

from .config import (
    CLEANED_TRAIN_PATH, TEST_PATH, SUBMISSION_PATH,
    NUMERIC_FEATURE, CATEGORICAL_FEATURE, TARGET,
)
from .data import (
    read_csv,
    normalize_column_names,
    mark_and_modify_outliers,
    normalize_numeric_columns,
    append_new_features,
)
from .modeling import MODELS, build_pipeline


def clean_test_data(file_path):
    """清洗测试数据（test.csv无survived列，添加临时列绕过校验）"""
    df = read_csv(file_path)
    df = normalize_column_names(df)

    # 添加临时survived列，使校验和类型转换能正常运行
    df["survived"] = 0
    df = mark_and_modify_outliers(df)
    df = normalize_numeric_columns(df)
    df = df.drop(columns=["survived"])

    # 添加新特征
    df = append_new_features(df)
    return df


def main():
    """主函数：训练最佳模型 → 预测测试集 → 导出Kaggle提交CSV"""
    # 1. 读取训练数据
    train_df = pd.read_csv(CLEANED_TRAIN_PATH)
    X_train = train_df[NUMERIC_FEATURE + CATEGORICAL_FEATURE].copy()
    y_train = train_df[TARGET].astype(int)

    # 2. 读取并清洗测试数据
    test_df = clean_test_data(TEST_PATH)
    X_test = test_df[NUMERIC_FEATURE + CATEGORICAL_FEATURE].copy()
    passenger_ids = test_df["passengerid"].astype(int)

    # 3. 用最佳模型在全量训练数据上训练
    best_model, scale_numeric = MODELS["GradientBoosting"]
    pipeline = build_pipeline(best_model, scale_numeric=scale_numeric)
    pipeline.fit(X_train, y_train)

    # 4. 预测测试集
    predictions = pipeline.predict(X_test)

    # 5. 构造Kaggle提交格式
    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": predictions.astype(int),
    })

    # 6. 导出CSV
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"提交文件已保存到：{SUBMISSION_PATH}")
    print(f"共 {len(submission)} 条预测")
    print("\n前10条预测：")
    print(submission.head(10))


if __name__ == "__main__":
    main()
