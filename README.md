# 智能算法综合课程设计（公开材料版）

本仓库整理自课程设计材料，用于保存题目、原始数据与可复现实验代码。它不含任何个人实验报告、已生成的实验结果、截图证据、模型文件或大体积的 C 类原始数据。

## 内容范围

- `智能算法综合课程设计实验指导书.docx`：课程任务与提交要求。
- `智能算法综合课程设计-报告模版.docx`：课程提供的报告模板。
- `数据集清单.csv`、`README_数据集使用说明.md`：数据集目录与使用说明。
- `A类题/`、`B类题/`：课程提供的原始 CSV 数据。
- `C类题/`：C 类数据的说明文件；原始档案因体积较大而未纳入公开仓库。
- `A类题_A2_乳腺癌二分类/`：匿名化的 A2 任务复现代码和自动化测试，不含运行产物。

## A2 复现

在已配置依赖的 Python 环境中，从仓库根目录运行：

```powershell
conda run -n machine_learning python A类题_A2_乳腺癌二分类/src/run_experiment.py
conda run -n machine_learning python A类题_A2_乳腺癌二分类/src/plot_results.py
conda run -n machine_learning python -m unittest discover -s A类题_A2_乳腺癌二分类/tests -v
```

程序会在本地生成 `results/`。该目录被 `.gitignore` 排除，避免将个人运行产物上传到仓库。

## 说明

本仓库仅用于课程学习与代码复现。A2 的指标仅反映固定随机种子和固定测试集划分下的一次实验结果，不能用于医学诊断或临床决策。
