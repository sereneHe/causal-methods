# 快速开始指南

## 安装依赖

```bash
pip install numpy pandas networkx scipy matplotlib seaborn
```

## 1. 最简单的例子

```python
from datasets import load_problem_dict
from evaluation import save_comprehensive_results
import numpy as np

# 加载数据
problem = {'name': 'er', 'number_of_variables': 5, 
           'number_of_samples': 100, 'edge_ratio': 2.0}
W_true, _, B_true, _, _, X, _, _, node_names, _ = load_problem_dict(problem)

# 运行你的算法 (这里用随机预测示例)
B_pred = (np.random.rand(5, 5) > 0.7).astype(int)
np.fill_diagonal(B_pred, 0)

# 保存所有结果 (edges.txt, graph.png, matrix.png, metrics等)
metrics = save_comprehensive_results(
    B_pred=B_pred,
    output_dir='results/test',
    algorithm_name='MyAlgorithm',
    B_true=B_true,
    node_names=node_names
)

print(f"F1: {metrics['F1']:.4f}, SHD: {metrics['shd']}")
```

## 2. 使用gCastle算法

```python
# 使用PC算法
from gcastle.algorithms import PC

pc = PC()
pc.learn(X)
B_pred = pc.causal_matrix

# 保存结果
save_comprehensive_results(B_pred, 'results/pc', 'PC', B_true=B_true)
```

## 3. 比较多个算法

```python
from evaluation import plot_f1_shd_runtime

# 运行多个算法，收集结果
results = {
    'PC': [],
    'GES': [],
    'NOTEARS': []
}

# 在不同设置下运行
for n_samples in [100, 250, 500, 1000]:
    problem = {'name': 'er', 'number_of_variables': 10, 
               'number_of_samples': n_samples, 'edge_ratio': 2.0}
    
    W_true, _, B_true, _, _, X, _, _, _, _ = load_problem_dict(problem)
    
    # 运行每个算法并记录结果
    for alg_name in ['PC', 'GES', 'NOTEARS']:
        # ... 运行算法得到 B_pred 和 runtime ...
        metrics = compute_all_metrics(B_true, B_pred, runtime)
        results[alg_name].append(metrics)

# 绘制对比图
plot_f1_shd_runtime(results, 'results/comparison', 
                   x_variable='Sample Size', 
                   x_values=[100, 250, 500, 1000])
```

## 4. 测试所有数据集

```python
test_datasets = [
    {'name': 'er', 'number_of_variables': 5, 'number_of_samples': 100},
    {'name': 'sf', 'number_of_variables': 5, 'number_of_samples': 100},
    {'name': 'admissions', 'number_of_samples': 50},
]

for problem in test_datasets:
    data = load_problem_dict(problem)
    print(f"{problem['name']}: Data shape {data[5].shape}")
```

## 输出文件

运行后会生成：

```
results/
├── test/
│   ├── MyAlgorithm_edges.txt              # 预测的边列表
│   ├── MyAlgorithm_graph.png              # 网络图
│   ├── MyAlgorithm_adjacency_matrix.png   # 邻接矩阵热图
│   ├── MyAlgorithm_adjacency_matrix.csv   # 邻接矩阵CSV
│   ├── MyAlgorithm_metrics.txt            # 性能指标
│   ├── MyAlgorithm_metrics.csv            # 指标CSV
│   ├── MyAlgorithm_comparison.png         # Ground truth对比
│   └── MyAlgorithm_summary.txt            # 总结
└── comparison/
    ├── f1_shd_runtime_comparison.png      # F1/SHD/Runtime对比图
    ├── metrics_comparison.png             # 所有指标对比
    └── results_summary.csv                # 结果汇总表
```

## 完整示例

参见 `examples_evaluation.py` 文件获取完整代码示例！
