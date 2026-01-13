"""
SID (Structural Intervention Distance) 使用示例和说明
"""

import numpy as np
from evaluation.metrics import compute_sid, compute_all_metrics

print("=" * 60)
print("SID (Structural Intervention Distance) 示例")
print("=" * 60)

# 示例1: 完全正确的预测
print("\n示例1: 完全正确的预测")
B_true = np.array([
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
])
B_pred = B_true.copy()

sid = compute_sid(B_true, B_pred)
print(f"Ground Truth = Predicted")
print(f"SID = {sid} (应该是0，表示完美预测)")

# 示例2: 边方向错误
print("\n示例2: 一条边方向相反")
B_pred_reversed = np.array([
    [0, 1, 1, 0],
    [0, 0, 0, 0],  # 原本 0->2 的边被移除
    [0, 1, 0, 1],  # 新增了 2->1 的边
    [0, 0, 0, 0]
])

sid = compute_sid(B_true, B_pred_reversed)
print(f"SID = {sid}")
print(f"说明: 边方向错误会影响因果效应，导致SID增加")

# 示例3: 缺少边
print("\n示例3: 缺少一条边")
B_pred_missing = np.array([
    [0, 1, 0, 0],  # 缺少 0->2
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
])

sid = compute_sid(B_true, B_pred_missing)
print(f"SID = {sid}")
print(f"说明: 缺少边会影响可达性，增加SID")

# 示例4: 多余的边
print("\n示例4: 多余的边")
B_pred_extra = np.array([
    [0, 1, 1, 1],  # 多了 0->3
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
])

sid = compute_sid(B_true, B_pred_extra)
print(f"SID = {sid}")
print(f"说明: 多余的边不一定增加SID，因为0已经能通过0->2->3到达3")

# 示例5: 完全错误的预测
print("\n示例5: 完全随机的图")
np.random.seed(42)
B_pred_random = (np.random.rand(4, 4) > 0.7).astype(int)
np.fill_diagonal(B_pred_random, 0)

sid = compute_sid(B_true, B_pred_random)
print(f"SID = {sid}")
print(f"说明: 随机图的SID通常很大")

# 使用完整的metrics函数
print("\n" + "=" * 60)
print("使用 compute_all_metrics 获取包含SID的完整指标")
print("=" * 60)

metrics = compute_all_metrics(B_true, B_pred_missing, runtime=0.5)
print(f"\nF1 Score: {metrics['F1']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"SHD: {metrics['shd']}")
print(f"SID: {metrics['sid']}")
print(f"Runtime: {metrics['runtime']:.4f}s")

print("\n" + "=" * 60)
print("SID vs SHD 的区别:")
print("=" * 60)
print("- SHD (Structural Hamming Distance): 计算边的差异数量")
print("  - 只关注边的存在与否")
print("  - 不考虑边的方向对因果效应的影响")
print("")
print("- SID (Structural Intervention Distance): 计算因果效应的差异")
print("  - 关注节点的可达性(后代集合)")
print("  - 考虑了因果传播的完整影响")
print("  - 更能反映图结构对因果推断的影响")
print("")
print("示例:")
print("  如果 A->B->C 被错误预测为 A->C (缺少B)")
print("  - SHD = 2 (少了A->B和B->C)")
print("  - SID = 0 (A仍然能到达C，因果效应相同)")
print("=" * 60)
