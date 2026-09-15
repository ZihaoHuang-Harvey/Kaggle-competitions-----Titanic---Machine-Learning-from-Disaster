"""训练评估入口：跑全量模型对比"""
import pandas as pd

from .config import CLEANED_TRAIN_PATH
from .evaluation import evaluate_all_models


def main():
    """主函数：读取清洗数据 → 跑全部模型对比"""
    df = pd.read_csv(CLEANED_TRAIN_PATH)
    evaluate_all_models(df, k=5)


if __name__ == "__main__":
    main()
