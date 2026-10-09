# 《智能算法综合课程设计》配套数据集使用说明

本数据包与A、B、C三类题实验指导书配套使用。A、B类题的表格数据已整理为CSV文件；C类题数据体量较大，请运行 `C类题/download_c_datasets.py` 按选题方向下载，四个方向只需选择一个。

## 一、目录与选题对应关系

| 方向 | 文件或数据集 | 目标字段 | 主要要求 |
| --- | --- | --- | --- |
| A1 回归 | `A类题/A1_regression_diabetes.csv` | `target` | 至少4种回归模型，含集成模型和人工神经网络 |
| A2 分类 | `A类题/A2_classification_breast_cancer.csv` | `target` | 至少4种分类模型，含集成模型和人工神经网络 |
| A3 聚类 | `A类题/A3_clustering_wine.csv` | 无监督任务 | `reference_class`仅用于完成聚类后的对照，禁止作为输入特征 |
| B1 分类 | `B类题/B1_classification_digits.csv` | `target` | 1种基准模型和至少2种集成模型 |
| B2 回归 | `B类题/B2_regression_friedman1.csv` | `target` | 1种基准模型和至少2种集成模型 |
| C1 图像分类 | CIFAR-10 | 图像类别 | 运行下载程序选择 `image_classification` |
| C2 文本生成 | Tiny Shakespeare | 下一字符或词元 | 运行下载程序选择 `text_generation` |
| C3 语义分割 | Oxford-IIIT Pet | 像素级trimap掩码 | 运行下载程序选择 `semantic_segmentation` |
| C4 实例分割 | Penn-Fudan Pedestrian | 实例掩码及边界框 | 运行下载程序选择 `instance_segmentation` |

## 二、A、B类CSV读取示例

```python
import pandas as pd

df = pd.read_csv("A类题/A2_classification_breast_cancer.csv")
X = df.drop(columns=["sample_id", "target"])
y = df["target"]
```

聚类方向应使用：

```python
df = pd.read_csv("A类题/A3_clustering_wine.csv")
X = df.drop(columns=["sample_id", "reference_class"])
reference_class = df["reference_class"]  # 仅在聚类完成后用于对照
```

## 三、C类数据下载

先安装依赖，再进入 `C类题` 文件夹运行下载程序：

```bash
pip install torch torchvision
python download_c_datasets.py --task image_classification
```

可选任务名称：

- `image_classification`：CIFAR-10；
- `text_generation`：Tiny Shakespeare；
- `semantic_segmentation`：Oxford-IIIT Pet；
- `instance_segmentation`：Penn-Fudan Pedestrian；
- `all`：下载全部数据，不建议学生在未确定选题时使用。

如需自定义保存位置，可增加 `--root` 参数，例如：

```bash
python download_c_datasets.py --task semantic_segmentation --root D:/course_data
```

## 四、统一注意事项

1. `sample_id`只用于识别样本，不作为模型输入特征。
2. 数据划分前先固定随机种子；不同模型必须使用同一训练集、验证集和测试集。
3. 标准化、编码、特征选择等操作只能在训练数据上拟合，避免数据泄漏。
4. 报告中应注明数据集名称、样本数、特征数、目标变量、预处理方式和实际使用的数据范围。
5. 课程配套数据仅用于教学实验。外部数据集的版权和许可要求以其原发布页面为准。

## 五、数据来源

- Diabetes、Breast Cancer Wisconsin、Wine和Digits：scikit-learn内置教学数据集。
- Friedman #1：使用 `sklearn.datasets.make_friedman1` 按固定随机种子2026生成的非线性回归模拟数据。
- CIFAR-10与Oxford-IIIT Pet：通过 `torchvision.datasets` 下载。
- Tiny Shakespeare：TensorFlow公开教学语料下载地址。
- Penn-Fudan Pedestrian：宾夕法尼亚大学公开数据，下载方式参照PyTorch实例分割教程。

