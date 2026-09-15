"""K折交叉验证评估"""
from sklearn.model_selection import StratifiedKFold, cross_validate

from .config import RANDOM_STATE, NUMERIC_FEATURE, CATEGORICAL_FEATURE, TARGET
from .modeling import build_pipeline, MODELS


def k_fold_evaluate(df, model, scale_numeric=True, k=5, verbose=True):
    """分层 K 折交叉验证，输出 accuracy / precision / recall / f1 的均值±标准差。

    Args:
        df: 清洗后的 DataFrame
        model: 已实例化的 sklearn 分类器
        scale_numeric: 是否对数值列做标准化
        k: K 折数
        verbose: 是否打印结果
    """
    X = df[NUMERIC_FEATURE + CATEGORICAL_FEATURE].copy()
    y = df[TARGET].astype(int)

    pipeline = build_pipeline(model, scale_numeric=scale_numeric)
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)

    scores = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=["accuracy", "precision", "recall", "f1"],
        n_jobs=1,
    )

    if verbose:
        for metric in ["accuracy", "precision", "recall", "f1"]:
            values = scores[f"test_{metric}"]
            print(f"  {metric:10s}: {values.mean():.4f} ± {values.std():.4f}")

    return scores


def evaluate_all_models(df, k=5):
    """对 MODELS 字典中的所有模型做 K 折交叉验证，输出对比表格。"""
    results = {}
    print("=" * 78)
    print(f"{'模型':<20} {'Accuracy':<16} {'Precision':<16} {'Recall':<16} {'F1':<16}")
    print("-" * 78)

    for name, (model, scale) in MODELS.items():
        scores = k_fold_evaluate(df, model, scale_numeric=scale, k=k, verbose=False)
        results[name] = scores
        acc = scores["test_accuracy"]
        prec = scores["test_precision"]
        rec = scores["test_recall"]
        f1 = scores["test_f1"]
        print(
            f"{name:<20} "
            f"{acc.mean():.4f}±{acc.std():.4f}   "
            f"{prec.mean():.4f}±{prec.std():.4f}   "
            f"{rec.mean():.4f}±{rec.std():.4f}   "
            f"{f1.mean():.4f}±{f1.std():.4f}"
        )

    print("=" * 78)
    return results
