# End-to-End Data Science Pipeline - 项目要求与 Pipeline

## 1. 项目定位

这个项目用于申请 Master of Data Science、Master of Artificial Intelligence、Master of Information Technology 等专业。目标不是做一个普通 notebook，而是展示完整的数据科学工作流：数据获取、清洗、探索性分析、特征工程、模型训练、评估、解释、API/可视化部署和项目文档。

推荐项目标题：

```text
End-to-End Data Science Pipeline: Customer Churn Prediction and Model Deployment
```

推荐默认任务：客户流失预测。  
理由：这是通用商业数据科学问题，不绑定医疗、金融或某个狭窄行业；同时能覆盖分类建模、特征工程、模型解释和业务决策。

## 2. 申请加分点

这个项目应证明你具备以下能力：

- 能从原始数据开始完成完整 data science lifecycle。
- 能处理缺失值、异常值、类别变量、数值变量和数据泄漏风险。
- 能比较多个机器学习模型，而不是只跑一个模型。
- 能用严谨指标解释模型效果，如 Accuracy、Precision、Recall、F1、ROC-AUC。
- 能用 SHAP、Permutation Importance 或 feature importance 解释模型。
- 能把模型做成可运行 demo/API，体现工程能力。
- 能用清晰文档解释方法、结果和局限性。

这比单个 Kaggle notebook 更有申请价值，因为它展示的是“从问题到系统”的能力。

## 3. 推荐技术栈

核心技术：

```text
Python
pandas
numpy
scikit-learn
xgboost 或 lightgbm
matplotlib
seaborn
plotly
shap
joblib
FastAPI 或 Streamlit
pytest
```

可选增强：

```text
MLflow：实验追踪
DVC：数据版本管理
Docker：环境封装
GitHub Actions：自动测试
```

MVP 不强制使用 MLflow、DVC、Docker，但 README 里应说明后续可扩展方向。

## 4. 推荐数据集

首选数据集：

```text
Telco Customer Churn Dataset
```

适合原因：

- 数据规模适中，适合 GitHub 项目。
- 包含数值变量、类别变量和二分类标签。
- 有缺失值、类别编码、特征选择等真实数据处理步骤。
- 业务解释清晰：预测客户是否流失。

备选数据集：

```text
Bank Marketing Dataset
Credit Card Customer Churn Dataset
Ames Housing Dataset
Airline Delay Dataset
```

如果选择回归任务，可以用 Ames Housing；如果选择分类任务，优先用 Telco Churn。

## 5. 仓库结构要求

建议仓库结构：

```text
end-to-end-data-science-pipeline/
├── README.md
├── pyproject.toml 或 requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   └── 03_model_experiments.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   └── predict.py
├── app/
│   ├── streamlit_app.py
│   └── api.py
├── models/
│   └── .gitkeep
├── reports/
│   ├── figures/
│   ├── model_comparison.md
│   └── final_report.md
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_predict.py
└── docs/
    ├── project_design.md
    └── model_card.md
```

## 6. Pipeline 设计

### Step 1: Problem Definition

要求：

- 定义业务问题：预测客户是否可能流失。
- 定义目标变量。
- 明确这是 classification task。
- 写清楚为什么这个问题适合 data science。

输出：

```text
docs/project_design.md
README.md 中的 Problem Statement
```

### Step 2: Data Loading

要求：

- 从 `data/raw/` 读取原始数据。
- 检查行数、列数、字段类型。
- 统一列名格式，如 snake_case。
- 保留原始数据，不直接覆盖。

输出：

```text
notebooks/01_data_understanding.ipynb
src/data_loader.py
```

### Step 3: Data Quality Assessment

要求：

- 检查缺失值比例。
- 检查重复行。
- 检查异常值。
- 检查类别字段取值是否一致。
- 检查目标变量是否类别不平衡。
- 输出数据质量报告。

必须包含的分析：

```text
missing value table
duplicate count
target distribution
numeric feature summary
categorical feature cardinality
```

输出：

```text
reports/data_quality_report.md
```

### Step 4: Exploratory Data Analysis

要求：

- 目标变量分布图。
- 数值变量分布图。
- 类别变量与目标变量的关系图。
- 相关性热力图。
- 至少 5 个有解释价值的 EDA 结论。

输出：

```text
notebooks/02_eda.ipynb
reports/figures/
reports/final_report.md
```

### Step 5: Preprocessing

要求：

- 缺失值处理。
- 类别变量编码。
- 数值变量标准化或归一化。
- train/validation/test split。
- 使用 scikit-learn Pipeline 或 ColumnTransformer。
- 避免 data leakage：所有 preprocessing 必须只在训练集 fit。

输出：

```text
src/preprocessing.py
src/features.py
```

### Step 6: Model Training

至少训练以下模型：

```text
Logistic Regression
Random Forest
XGBoost 或 LightGBM
Support Vector Machine 或 Gradient Boosting
```

要求：

- 每个模型使用统一 train/test split。
- 每个模型输出相同评估指标。
- 至少对最佳模型进行简单超参数调优。
- 保存最佳模型到 `models/`。

输出：

```text
src/train.py
models/best_model.joblib
reports/model_comparison.md
```

### Step 7: Model Evaluation

必须包含：

```text
Accuracy
Precision
Recall
F1-score
ROC-AUC
Confusion Matrix
ROC Curve
Precision-Recall Curve
```

要求：

- 说明为什么不能只看 Accuracy。
- 如果类别不平衡，要解释 Recall / Precision 的取舍。
- 写出最佳模型选择原因。

输出：

```text
src/evaluate.py
reports/model_comparison.md
reports/figures/confusion_matrix.png
reports/figures/roc_curve.png
```

### Step 8: Explainability

要求：

- 对最佳模型进行解释。
- 至少输出 feature importance。
- 推荐使用 SHAP。
- 写清楚最重要的 5 个特征及其业务含义。

输出：

```text
src/explain.py
reports/figures/shap_summary.png
docs/model_card.md
```

### Step 9: Deployment Demo

至少实现一种 demo：

推荐：

```text
Streamlit dashboard
```

功能要求：

- 用户输入客户特征。
- 系统输出流失概率。
- 显示预测类别。
- 显示 top contributing features。
- 显示模型说明和免责声明。

可选 API：

```text
FastAPI /predict endpoint
```

输出：

```text
app/streamlit_app.py
app/api.py
```

### Step 10: Testing

至少包含：

```text
test data loading
test preprocessing output shape
test no missing values after preprocessing
test prediction output type
test API response structure
```

输出：

```text
tests/
```

## 7. README 要求

README 必须包含：

```text
Project Overview
Why this project matters
Dataset
Pipeline
Model comparison table
Best model result
Screenshots
How to run
Repository structure
Limitations
Future work
```

README 中必须有一张模型结果表：

```text
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | ... | ... | ... | ... | ... |
| Random Forest | ... | ... | ... | ... | ... |
| XGBoost | ... | ... | ... | ... | ... |
```

## 8. 验收标准

项目完成标准：

- `README.md` 能让招生官在 2 分钟内理解项目价值。
- `notebooks/` 有完整分析过程，但核心逻辑不只存在 notebook 中。
- `src/` 中代码可复用、模块化。
- 至少 3 个模型完成对比。
- 最佳模型保存并可被 demo 调用。
- Streamlit 或 FastAPI 可以运行。
- 有模型解释图。
- 有测试文件。
- 有最终报告。
- 仓库没有上传大型无关文件、缓存文件或隐私数据。

## 9. 时间安排

建议 10-14 天完成：

```text
Day 1: 数据集确定、仓库结构、README 初版
Day 2-3: 数据加载、质量检查、EDA
Day 4-5: preprocessing pipeline
Day 6-7: 模型训练和评估
Day 8: 模型解释和 model card
Day 9-10: Streamlit / FastAPI demo
Day 11: tests
Day 12: final report
Day 13-14: README、截图、代码清理
```

## 10. 可写进 CV 的项目描述

中文版本：

```text
构建端到端客户流失预测数据科学项目，完成数据质量评估、EDA、特征工程、模型训练、超参数调优、模型解释和 Streamlit 部署；比较 Logistic Regression、Random Forest 与 XGBoost 等模型，并使用 ROC-AUC、F1、Confusion Matrix 和 SHAP 分析模型表现与关键特征。
```

英文版本：

```text
Built an end-to-end customer churn prediction pipeline covering data quality assessment, EDA, feature engineering, model benchmarking, explainability and Streamlit deployment. Compared Logistic Regression, Random Forest and XGBoost using ROC-AUC, F1-score and confusion matrices, and interpreted key drivers with SHAP.
```

## 11. 加分增强项

如果时间充足，可以增加：

- MLflow 实验追踪。
- Docker 一键运行。
- GitHub Actions 自动测试。
- Model Card。
- Data Card。
- API endpoint。
- Deployed demo link。

优先级：

```text
高：README、结果表、Streamlit demo、SHAP
中：tests、model card、FastAPI
低：Docker、MLflow、DVC
```

