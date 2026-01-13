"""
Comprehensive evaluation metrics for causal discovery
Integrates gCastle metrics and additional evaluation methods
"""

import numpy as np
import networkx as nx
from typing import Dict, Tuple, Optional


def count_accuracy(B_true: np.ndarray, B_pred: np.ndarray, 
                   use_cpu_only: bool = True) -> Dict[str, float]:
    """
    Compute various accuracy metrics between true and predicted graphs
    
    Args:
        B_true: True binary adjacency matrix
        B_pred: Predicted binary adjacency matrix
        use_cpu_only: Whether to use CPU only (for compatibility)
    
    Returns:
        Dictionary containing fdr, tpr, fpr, shd, nnz, precision, recall, F1, gscore
    """
    if not ((B_true == 0) | (B_true == 1)).all():
        B_true = (B_true != 0).astype(int)
    if not ((B_pred == 0) | (B_pred == 1)).all():
        B_pred = (B_pred != 0).astype(int)
    
    d = B_true.shape[0]
    
    # Linear indices of predicted edges
    pred = np.flatnonzero(B_pred)
    # Linear indices of true edges  
    cond = np.flatnonzero(B_true)
    # Linear indices of reversed edges
    cond_reversed = np.flatnonzero(B_true.T)
    # Linear indices of true negative edges
    cond_skeleton = np.concatenate([cond, cond_reversed])
    
    # True positives
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # False positives (predicted but not true)
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    # Extra is predicted in reverse direction
    extra = np.intersect1d(pred, cond_reversed, assume_unique=True)
    # Reverse is true edge that was predicted in reverse
    reverse = np.intersect1d(cond, cond_reversed, assume_unique=True)
    
    # Predicted edges count
    pred_size = len(pred)
    # True edges count
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    
    # False discovery rate
    fdr = float(len(false_pos) + len(extra)) / max(pred_size, 1)
    # True positive rate (recall)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    # False positive rate
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # Structural Hamming Distance
    shd = len(false_pos) + len(reverse) + len(cond) - len(true_pos)
    # Number of predicted edges
    nnz = pred_size
    
    # Precision
    precision = float(len(true_pos)) / max(pred_size, 1)
    # Recall (same as tpr)
    recall = tpr
    # F1 score
    F1 = 2 * precision * recall / max(precision + recall, 1e-8)
    # G-score (geometric mean of precision and recall)
    gscore = np.sqrt(precision * recall)
    
    return {
        'fdr': fdr,
        'tpr': tpr, 
        'fpr': fpr,
        'shd': shd,
        'nnz': nnz,
        'precision': precision,
        'recall': recall,
        'F1': F1,
        'gscore': gscore
    }


def compute_sid(B_true: np.ndarray, B_pred: np.ndarray) -> int:
    """
    Compute Structural Intervention Distance (SID)
    
    SID measures the difference between the causal effects in two graphs by comparing
    the reachability (descendant) sets of each node.
    
    Args:
        B_true: True binary adjacency matrix
        B_pred: Predicted binary adjacency matrix
    
    Returns:
        SID score (lower is better, 0 means identical causal effects)
    """
    if not ((B_true == 0) | (B_true == 1)).all():
        B_true = (B_true != 0).astype(int)
    if not ((B_pred == 0) | (B_pred == 1)).all():
        B_pred = (B_pred != 0).astype(int)
    
    try:
        d = B_true.shape[0]
        
        # Ensure both graphs have same number of nodes
        if B_true.shape != B_pred.shape:
            raise ValueError("Graph dimensions must match")
        
        # Convert to graphs
        G_true = nx.DiGraph(B_true)
        G_pred = nx.DiGraph(B_pred)
        
        # Ensure all nodes are present (even isolated ones)
        G_true.add_nodes_from(range(d))
        G_pred.add_nodes_from(range(d))
        
        # Get descendants for each node
        def get_descendants_dict(G):
            desc = {}
            for node in range(d):
                try:
                    desc[node] = set(nx.descendants(G, node))
                except:
                    desc[node] = set()
            return desc
        
        desc_true = get_descendants_dict(G_true)
        desc_pred = get_descendants_dict(G_pred)
        
        # Compute SID as sum of symmetric differences
        sid = 0
        for node in range(d):
            true_desc = desc_true[node]
            pred_desc = desc_pred[node]
            # Symmetric difference: elements in either set but not both
            sid += len(true_desc.symmetric_difference(pred_desc))
        
        return int(sid)
    
    except Exception as e:
        print(f"Warning: SID computation failed with error: {e}")
        return -1


def compute_cpdag_metrics(B_true: np.ndarray, B_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics for CPDAG (completed partially directed acyclic graph)
    
    Args:
        B_true: True binary adjacency matrix
        B_pred: Predicted binary adjacency matrix
    
    Returns:
        Dictionary with CPDAG-specific metrics
    """
    # Convert to CPDAG representation (undirected edges for equivalence class)
    def to_cpdag(B):
        cpdag = B.copy()
        for i in range(B.shape[0]):
            for j in range(i+1, B.shape[0]):
                # If bidirectional edge, make it undirected in CPDAG
                if B[i, j] and B[j, i]:
                    cpdag[i, j] = 1
                    cpdag[j, i] = 1
        return cpdag
    
    cpdag_true = to_cpdag(B_true)
    cpdag_pred = to_cpdag(B_pred)
    
    # Compute skeleton metrics
    skel_true = (B_true + B_true.T) > 0
    skel_pred = (B_pred + B_pred.T) > 0
    
    # True positives in skeleton
    tp_skel = np.sum(skel_true & skel_pred) / 2  # Divide by 2 for undirected edges
    # False positives in skeleton
    fp_skel = np.sum(~skel_true & skel_pred) / 2
    # False negatives in skeleton
    fn_skel = np.sum(skel_true & ~skel_pred) / 2
    
    # Precision and recall for skeleton
    precision_skel = tp_skel / max(tp_skel + fp_skel, 1)
    recall_skel = tp_skel / max(tp_skel + fn_skel, 1)
    f1_skel = 2 * precision_skel * recall_skel / max(precision_skel + recall_skel, 1e-8)
    
    return {
        'skeleton_precision': precision_skel,
        'skeleton_recall': recall_skel,
        'skeleton_F1': f1_skel,
        'skeleton_shd': int(fp_skel + fn_skel)
    }


def compute_all_metrics(B_true: np.ndarray, B_pred: np.ndarray, 
                       runtime: Optional[float] = None) -> Dict[str, float]:
    """
    Compute all available metrics
    
    Args:
        B_true: True binary adjacency matrix
        B_pred: Predicted binary adjacency matrix
        runtime: Algorithm runtime in seconds (optional)
    
    Returns:
        Dictionary with all metrics
    """
    metrics = count_accuracy(B_true, B_pred)
    
    # Add SID
    sid = compute_sid(B_true, B_pred)
    metrics['sid'] = sid
    
    # Add CPDAG metrics
    cpdag_metrics = compute_cpdag_metrics(B_true, B_pred)
    metrics.update(cpdag_metrics)
    
    # Add runtime if provided
    if runtime is not None:
        metrics['runtime'] = runtime
    
    return metrics


def format_metrics(metrics, decimal_places=4):
    """
    Format metrics dictionary for display.
    
    Args:
        metrics: Dictionary of metrics
        decimal_places: Number of decimal places for floating point values
        
    Returns:
        Formatted string representation
    """
    lines = []
    lines.append("\nMetrics Summary:")
    lines.append("=" * 50)
    
    # Core metrics
    core_metrics = ['fdr', 'tpr', 'fpr', 'shd', 'nnz', 'precision', 'recall', 'F1']
    detail_metrics = ['tp', 'fp', 'fn', 'tn', 'extra', 'missing', 'reverse']
    skeleton_metrics = ['skeleton_tp', 'skeleton_fp', 'skeleton_fn', 'skeleton_precision', 
                       'skeleton_recall', 'skeleton_F1']
    
    # Display core metrics
    lines.append("\nCore Metrics:")
    for key in core_metrics:
        if key in metrics:
            value = metrics[key]
            if key == 'sid':
                if value == -1:
                    lines.append(f"  {key:20s}: N/A (computation failed)")
                else:
                    lines.append(f"  {key:20s}: {value}")
            elif isinstance(value, int):
                lines.append(f"  {key:20s}: {value}")
            else:
                lines.append(f"  {key:20s}: {value:.{decimal_places}f}")
    
    # Detail metrics
    if any(k in metrics for k in detail_metrics):
        lines.append("\nDetailed Metrics:")
        for key in detail_metrics:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, int):
                    lines.append(f"  {key:20s}: {value}")
                else:
                    lines.append(f"  {key:20s}: {value:.{decimal_places}f}")
    
    # Skeleton metrics
    if skeleton_metrics:
        lines.append("\nSkeleton Metrics:")
        for key in skeleton_metrics:
            value = metrics[key]
            if isinstance(value, int):
                lines.append(f"  {key:20s}: {value}")
            else:
                lines.append(f"  {key:20s}: {value:.{decimal_places}f}")
    
    # Runtime
    if 'runtime' in metrics:
        lines.append(f"\nRuntime: {metrics['runtime']:.4f} seconds")
    
    # Add explanation for SID if present
    if 'sid' in metrics and metrics['sid'] != -1:
        lines.append("\nNote: SID (Structural Intervention Distance) measures")
        lines.append("      differences in causal effects. Lower is better, 0 is perfect.")
    
    lines.append("=" * 50)
    return "\n".join(lines)

