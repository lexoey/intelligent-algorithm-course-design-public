# A2 乳腺癌二分类实验

本目录完成《智能算法综合课程设计》A 类题的 A2 乳腺癌二分类任务。实验在统一的训练集、测试集划分下比较 Logistic Regression、SVM、Random Forest、Gradient Boosting 和 MLPClassifier 五种模型，其中包含两种集成模型与一种人工神经网络模型。

## 环境

- Windows 11
- Conda 环境：`machine_learning`
- Python 3.11
- 依赖版本见 `requirements.txt`

## 数据

原始数据不复制、不修改，程序读取课程根目录的 `A类题/A2_classification_breast_cancer.csv`。数据包含 569 条样本、30 个数值特征和二分类目标字段 `target`；`sample_id` 仅用于样本识别，不参与训练。

## 运行方式

在仓库根目录运行：

```powershell
conda run -n machine_learning python A类题_A2_乳腺癌二分类/src/run_experiment.py
conda run -n machine_learning python A类题_A2_乳腺癌二分类/src/plot_results.py
```

运行完成后，结果表保存于 `results/tables/`，证据图片保存于 `results/figures/`。固定随机种子为 2026，训练集与测试集按 80%/20% 分层划分；模型选择和交叉验证仅使用训练集。

## 目录说明

- `src/`：数据读取、训练评价和结果绘图代码。
- `tests/`：自动化测试。
- `data/`：原始数据路径及字段说明。
- `results/tables/`：本地运行后生成的模型指标、交叉验证、调参和预测明细。
- `results/figures/`：本地运行后生成的图表。

## 复现约定

所有图表和 CSV 均由同一次本地运行结果生成。训练集交叉验证用于模型比较；已调参模型的交叉验证分数属于参数选择后的训练集参考，不应视为无偏泛化估计。最终比较以独立测试集为准；若用于正式研究，应采用嵌套交叉验证。运行前请不要移动课程根目录中的 A2 CSV 文件。
