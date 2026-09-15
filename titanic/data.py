"""读CSV、校验、统一规则清洗数据"""
from pathlib import Path
import pandas as pd
import numpy as np

from .config import (
    REQUIRED_COLUMNS, TRAIN_FEATURE, TARGET,
    TRAIN_PATH, CLEANED_TRAIN_PATH,
)

def read_csv(file_path):
    """读取CSV文件"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise ValueError("文件路径不存在")
    if not file_path.is_file():
        raise ValueError("文件路径不是文件")
    if not file_path.suffix == ".csv":
        raise ValueError("文件路径不是CSV文件")
    df = pd.read_csv(file_path)
    return df

def validate_csv(df):
    """校验CSV文件"""
    # 计算实际列名与预期列名的差异。
    actual_columns = set(df.columns)
    missing_columns = REQUIRED_COLUMNS - actual_columns
    extra_columns = actual_columns - REQUIRED_COLUMNS
    if missing_columns or extra_columns:
        missing_text = ", ".join(sorted(missing_columns)) if missing_columns else "无"
        extra_text = ", ".join(sorted(extra_columns)) if extra_columns else "无"
        raise ValueError(
            f"CSV文件列名与预期不匹配。缺失列：{missing_text}；多余列：{extra_text}"
        )
    return True

def normalize_column_names(frame):
    """将列名转换为小写"""
    normalized = frame.copy()
    normalized.columns = normalized.columns.str.lower()
    normalized.columns = normalized.columns.str.replace(
        r"[^0-9a-zA-Z]+",
        "_",
        regex=True,
    )
    # 去除字段名前后可能残留的下划线。
    normalized.columns = normalized.columns.str.strip("_")
    # 返回字段名规范化后的数据。

    return normalized

def normalize_numeric_columns(frame):
    """将数值列转换为浮点数"""
    numeric_columns = [
        "passengerid",
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
    ]
    for column in numeric_columns:
        frame[column] = frame[column].astype(float)
    return frame

def mark_and_modify_outliers(frame):
    """标记并修改异常值"""
    cleaned = frame.copy()
    # 删除完全相同的重复行。
    cleaned = cleaned.drop_duplicates().copy()

    # 将数值字段先转成数值类型，非法字符串变为NaN。
    numeric_columns = [
        "passengerid",
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
    ]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    # 字符串字段统一使用Pandas的string类型保留缺失值。
    for column in ["name", "ticket", "cabin"]:
        cleaned[column] = cleaned[column].astype("string").str.strip()
    cleaned["sex"] = cleaned["sex"].astype("string").str.strip().str.lower()
    cleaned["embarked"] = cleaned["embarked"].astype("string").str.strip().str.upper()

    # 将不属于1、2、3的船舱等级标记为缺失。
    cleaned["pclass"] = cleaned["pclass"].where(cleaned["pclass"].isin([1, 2, 3]), np.nan)
    # 将不在合理范围内的年龄标记为缺失。
    cleaned["age"] = cleaned["age"].where(cleaned["age"].between(0, 120), np.nan)
    # 填充缺失年龄为缺失年龄中位数。
    cleaned["age"] = cleaned["age"].fillna(cleaned["age"].median())
    # 将负数亲属数量标记为缺失。
    cleaned["sibsp"] = cleaned["sibsp"].where(cleaned["sibsp"] >= 0, np.nan)
    # 将负数父母子女数量标记为缺失。
    cleaned["parch"] = cleaned["parch"].where(cleaned["parch"] >= 0, np.nan)
    # 将负数票价标记为缺失。
    cleaned["fare"] = cleaned["fare"].where(cleaned["fare"] >= 0, np.nan)
    # 填充缺失票价为缺失票价中位数。
    cleaned["fare"] = cleaned["fare"].fillna(cleaned["fare"].median())
    # 将非法性别值标记为缺失。
    cleaned["sex"] = cleaned["sex"].where(cleaned["sex"].isin(["male", "female"]), pd.NA)
    # 将非法登船港口标记为缺失。
    cleaned["embarked"] = cleaned["embarked"].where(
        cleaned["embarked"].isin(["S", "C", "Q"]),
        pd.NA,
    )
    # 填充缺失登船港口为缺失登船港口的众数。
    cleaned["embarked"] = cleaned["embarked"].fillna(cleaned["embarked"].mode()[0])
    # 填充缺失船舱为U
    cleaned["cabin"] = cleaned["cabin"].fillna("U")

    if cleaned["passengerid"].isna().any():
        # 无法追踪的样本不应静默进入项目。
        raise ValueError("passengerid中存在缺失值或非法字符串。")
    # 检查乘客编号是否重复。
    if cleaned["passengerid"].duplicated().any():
        # 重复ID可能表示数据拼接错误或同一乘客被重复记录。
        raise ValueError("passengerid中存在重复值，请先确认数据来源。")

    valid_target = cleaned[TARGET].isin([0, 1])
    # 只要存在非法或缺失目标，就停止训练数据处理。
    if not valid_target.all():
        # 输出非法目标数量，帮助定位问题。
        invalid_count = int((~valid_target).sum())
        # 抛出明确异常。
        raise ValueError(f"survived中存在 {invalid_count} 个缺失或非法标签。")

    return cleaned

def clean_data(file_path):
    """清洗数据"""
    df = read_csv(file_path)
    normalized = normalize_column_names(df)
    validate_csv(normalized)
    cleaned = mark_and_modify_outliers(normalized)
    cleaned = normalize_numeric_columns(cleaned)
    return cleaned

def append_new_features(df):
    """添加新特征"""
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)
    df["is_single"] = (df["parch"] > 0).astype(int)
    df["cabin_deck"] = df["cabin"].str[0]
    return df


def main():
    """主函数：读取原始训练数据 → 清洗 → 加特征 → 保存到 outputs/"""
    df = clean_data(TRAIN_PATH)
    df = append_new_features(df)
    df.to_csv(CLEANED_TRAIN_PATH, index=False)
    print(f"清洗后的训练数据已保存到：{CLEANED_TRAIN_PATH}")
    print(f"共 {len(df)} 条记录，{len(df.columns)} 列")


if __name__ == "__main__":
    main()

