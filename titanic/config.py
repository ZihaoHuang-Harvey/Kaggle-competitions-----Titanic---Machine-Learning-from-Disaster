"""项目配置：路径常量、随机种子、特征列表"""
from pathlib import Path

# 项目根目录（config.py 在 titanic/ 下，所以 parent 是项目根）
ROOT_DIR = Path(__file__).resolve().parent.parent

# 数据目录与文件路径
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

TRAIN_PATH = DATA_DIR / "train.csv"           # 原始训练数据
TEST_PATH = DATA_DIR / "test.csv"            # 原始测试数据
CLEANED_TRAIN_PATH = OUTPUT_DIR / "cleaned_train.csv"  # 清洗后的训练数据
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"       # Kaggle 提交文件

# 确保输出目录存在
OUTPUT_DIR.mkdir(exist_ok=True)

# 随机种子
RANDOM_STATE = 42

# 目标列
TARGET = "survived"

# 必需列（校验用）
REQUIRED_COLUMNS = {
    "passengerid",
    "survived",
    "pclass",
    "name",
    "sex",
    "age",
    "sibsp",
    "parch",
    "ticket",
    "fare",
    "cabin",
    "embarked",
}

# 训练特征（不含 survived）
TRAIN_FEATURE = REQUIRED_COLUMNS - {"survived"}

# 数值特征：连续值 + 0/1二值特征
NUMERIC_FEATURE = ["age", "fare", "sibsp", "parch", "family_size", "is_alone", "is_single"]

# 类别特征：低基数，独热编码
CATEGORICAL_FEATURE = ["pclass", "sex", "embarked", "cabin_deck"]

# 不参与建模的列（标识符 / 高基数文本）
DROP_COLUMNS = ["passengerid", "name", "ticket", "cabin"]
