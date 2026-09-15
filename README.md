# Titanic 机器学习项目

预测泰坦尼克号乘客生还的二分类问题（Kaggle 入门竞赛）。

## 项目结构

```
titanic_project/
├── data/                       # 原始数据（只读）
│   ├── train.csv               # 891 条带标签训练数据
│   └── test.csv                # 418 条无标签测试数据
├── outputs/                    # 中间产物与提交文件（不入 git）
│   ├── cleaned_train.csv       # 清洗后的训练数据
│   └── submission.csv          # Kaggle 提交文件
├── titanic/                    # 源码包
│   ├── __init__.py             # 导出主要 API
│   ├── config.py               # 路径常量、特征列表、随机种子
│   ├── data.py                 # 数据读取、校验、清洗、特征工程
│   ├── modeling.py             # 候选模型字典 + Pipeline 构建
│   ├── evaluation.py           # K 折交叉验证评估
│   ├── training.py             # 入口：跑全量模型对比
│   └── submission.py           # 入口：预测并导出提交文件
├── requirements.txt
└── README.md
```

## 数据准备

原始数据 `train.csv` 和 `test.csv` 未随仓库一起提供（Kaggle 比赛规则禁止二次分发），请自行下载：

1. 访问 [Kaggle Titanic Data 页面](https://www.kaggle.com/competitions/titanic/data)
2. 加入比赛（点击 *Join Competition*，免费）
3. 下载 `train.csv` 和 `test.csv`，放到项目根目录的 `data/` 下：

```
titanic_project/
└── data/
    ├── train.csv
    └── test.csv
```

## 数据准备

原始数据 `train.csv` 和 `test.csv` 未随仓库一起提供（Kaggle 比赛规则禁止二次分发），请自行下载：

1. 访问 [Kaggle Titanic Data 页面](https://www.kaggle.com/competitions/titanic/data)
2. 加入比赛（点击 *Join Competition*，免费）
3. 下载 `train.csv` 和 `test.csv`，放到项目根目录的 `data/` 下：

```
titanic_project/
└── data/
    ├── train.csv
    └── test.csv
```

## 环境准备

```bash
pip install -r requirements.txt
```

## 使用方式

### 1. 生成清洗后的训练数据

```bash
python -m titanic.data
```

会读取 `data/train.csv`，输出到 `outputs/cleaned_train.csv`。

### 2. 跑 K 折交叉验证对比所有模型

```bash
python -m titanic.training
```

会用 5 折交叉验证评估 6 个候选模型，输出对比表。

### 3. 用最佳模型预测并生成 Kaggle 提交文件

```bash
python -m titanic.submission
```

会用 GradientBoosting 在全量训练数据上训练，预测 `data/test.csv`，
输出到 `outputs/submission.csv`（Kaggle 提交格式）。

## 提交结果

去 [Kaggle Titanic 比赛页面](https://www.kaggle.com/competitions/titanic)
上传 `outputs/submission.csv` 即可获得排行榜分数。

## 最佳模型

| 模型 | Accuracy | F1 |
|------|----------|----|
| GradientBoosting | 0.8361 | 0.7693 |
| SVM | 0.8249 | 0.7597 |
| RandomForest | 0.8171 | 0.7347 |

最终采用 **GradientBoosting** 作为提交模型。
最终得分：0.7751


