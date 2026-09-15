"""候选模型定义 + Pipeline 构建"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from .config import RANDOM_STATE, NUMERIC_FEATURE, CATEGORICAL_FEATURE

# 候选模型字典：(模型实例, 是否需要标准化数值列)
# 树模型不需要标准化；线性/SVM/KNN 需要
MODELS = {
    "LogisticRegression": (LogisticRegression(max_iter=1000), True),
    "DecisionTree": (DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE), False),
    "RandomForest": (RandomForestClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE), False),
    "GradientBoosting": (GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE), False),
    "SVM": (SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE), True),
    "KNN": (KNeighborsClassifier(n_neighbors=5), True),
}


def build_pipeline(model, scale_numeric=True):
    """构建预处理 + 模型的完整 Pipeline。

    所有统计学习步骤在 K 折内 fit，避免数据泄漏。
    缺失值已在数据清洗阶段填补，此处不再做填补。

    Args:
        model: 已实例化的 sklearn 分类器
        scale_numeric: 是否对数值列做标准化（树模型不需要）
    """
    # 数值列：（可选）标准化
    numeric_steps = []
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps) if numeric_steps else "passthrough"

    # 类别列：独热编码（handle_unknown=ignore 处理测试集未见类别）
    categorical_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    # ColumnTransformer 按列分流
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURE),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURE),
    ])

    # 完整管线：预处理 → 模型
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return full_pipeline
