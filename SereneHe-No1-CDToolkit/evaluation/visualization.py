"""
Comprehensive output and visualization for causal discovery results
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .metrics import compute_all_metrics, format_metrics


def save_predicted_edges(B_pred: np.ndarray, 
                        output_path: str,
                        node_names: Optional[List[str]] = None,
                        weighted: bool = False,
                        W_pred: Optional[np.ndarray] = None) -> None:
    """
    Save predicted edges to text file
    
    Args:
        B_pred: Predicted binary adjacency matrix
        output_path: Path to save edges file
        node_names: List of node names (optional)
        weighted: Whether to include edge weights
        W_pred: Weighted adjacency matrix (required if weighted=True)
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    d = B_pred.shape[0]
    if node_names is None:
        node_names = [f'X{i}' for i in range(d)]
    
    edges = []
    for i in range(d):
        for j in range(d):
            if B_pred[i, j]:
                if weighted and W_pred is not None:
                    edges.append(f"{node_names[i]} -> {node_names[j]} (weight: {W_pred[i, j]:.4f})")
                else:
                    edges.append(f"{node_names[i]} -> {node_names[j]}")
    
    with open(output_path, 'w') as f:
        f.write(f"Total edges: {len(edges)}\n")
        f.write("="*50 + "\n")
        for edge in edges:
            f.write(edge + "\n")
    
    print(f"✓ Saved {len(edges)} edges to {output_path}")


def plot_network_graph(B_pred: np.ndarray,
                      output_path: str,
                      node_names: Optional[List[str]] = None,
                      title: str = "Predicted Causal Graph",
                      node_size: int = 2000,
                      font_size: int = 12,
                      figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Plot and save network graph visualization
    
    Args:
        B_pred: Predicted binary adjacency matrix
        output_path: Path to save figure
        node_names: List of node names
        title: Plot title
        node_size: Size of nodes
        font_size: Font size for labels
        figsize: Figure size
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    d = B_pred.shape[0]
    if node_names is None:
        node_names = [f'X{i}' for i in range(d)]
    
    # Create directed graph
    G = nx.DiGraph()
    for i in range(d):
        G.add_node(node_names[i])
    
    for i in range(d):
        for j in range(d):
            if B_pred[i, j]:
                G.add_edge(node_names[i], node_names[j])
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Use different layout based on graph size
    if d <= 10:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif d <= 20:
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, k=1, iterations=30, seed=42)
    
    # Draw graph
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=node_size, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=font_size, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          arrows=True, arrowsize=20, 
                          arrowstyle='->', width=2,
                          connectionstyle='arc3,rad=0.1')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved network graph to {output_path}")


def plot_adjacency_matrix(B_pred: np.ndarray,
                         output_path: str,
                          node_names: Optional[List[str]] = None,
                         title: str = "Adjacency Matrix",
                         figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot and save adjacency matrix heatmap
    
    Args:
        B_pred: Predicted binary adjacency matrix
        output_path: Path to save figure
        node_names: List of node names
        title: Plot title
        figsize: Figure size
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    d = B_pred.shape[0]
    if node_names is None:
        node_names = [f'X{i}' for i in range(d)]
    
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(B_pred, annot=True, fmt='g', cmap='YlOrRd',
                xticklabels=node_names, yticklabels=node_names,
                cbar_kws={'label': 'Edge Presence'},
                linewidths=0.5, linecolor='gray')
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('To Node', fontsize=12)
    plt.ylabel('From Node', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved adjacency matrix heatmap to {output_path}")


def save_comprehensive_results(B_pred: np.ndarray,
                               output_dir: str,
                               algorithm_name: str = "algorithm",
                               B_true: Optional[np.ndarray] = None,
                               W_pred: Optional[np.ndarray] = None,
                               node_names: Optional[List[str]] = None,
                               runtime: Optional[float] = None,
                               additional_info: Optional[Dict] = None) -> Dict[str, float]:
    """
    Save all results comprehensively
    
    Args:
        B_pred: Predicted binary adjacency matrix
        output_dir: Output directory for all files
        algorithm_name: Name of the algorithm
        B_true: True binary adjacency matrix (optional, for metrics)
        W_pred: Weighted adjacency matrix (optional)
        node_names: List of node names
        runtime: Algorithm runtime in seconds
        additional_info: Additional information to save
    
    Returns:
        Dictionary of metrics (if B_true provided)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Save predicted edges
    edges_file = os.path.join(output_dir, f"{algorithm_name}_edges.txt")
    save_predicted_edges(B_pred, edges_file, node_names, 
                        weighted=(W_pred is not None), W_pred=W_pred)
    
    # 2. Plot network graph
    graph_file = os.path.join(output_dir, f"{algorithm_name}_graph.png")
    plot_network_graph(B_pred, graph_file, node_names, 
                      title=f"{algorithm_name} - Predicted Causal Graph")
    
    # 3. Plot adjacency matrix
    matrix_file = os.path.join(output_dir, f"{algorithm_name}_adjacency_matrix.png")
    plot_adjacency_matrix(B_pred, matrix_file, node_names,
                         title=f"{algorithm_name} - Adjacency Matrix")
    
    # 4. Save adjacency matrix as CSV
    csv_file = os.path.join(output_dir, f"{algorithm_name}_adjacency_matrix.csv")
    d = B_pred.shape[0]
    if node_names is None:
        node_names = [f'X{i}' for i in range(d)]
    df = pd.DataFrame(B_pred, columns=node_names, index=node_names)
    df.to_csv(csv_file)
    print(f"✓ Saved adjacency matrix CSV to {csv_file}")
    
    metrics = None
    # 5. If ground truth available, compute and save metrics
    if B_true is not None:
        metrics = compute_all_metrics(B_true, B_pred, runtime)
        
        # Save metrics
        metrics_file = os.path.join(output_dir, f"{algorithm_name}_metrics.txt")
        with open(metrics_file, 'w') as f:
            f.write(format_metrics(metrics))
        print(f"✓ Saved metrics to {metrics_file}")
        
        # Save metrics as CSV
        metrics_csv = os.path.join(output_dir, f"{algorithm_name}_metrics.csv")
        pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)
        print(f"✓ Saved metrics CSV to {metrics_csv}")
        
        # Plot comparison if ground truth available
        comparison_file = os.path.join(output_dir, f"{algorithm_name}_comparison.png")
        plot_comparison(B_true, B_pred, comparison_file, node_names, algorithm_name)
    
    # 6. Save additional information
    if additional_info:
        info_file = os.path.join(output_dir, f"{algorithm_name}_info.txt")
        with open(info_file, 'w') as f:
            for key, value in additional_info.items():
                f.write(f"{key}: {value}\n")
        print(f"✓ Saved additional info to {info_file}")
    
    # 7. Save summary
    summary_file = os.path.join(output_dir, f"{algorithm_name}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Algorithm: {algorithm_name}\n")
        f.write(f"Number of nodes: {B_pred.shape[0]}\n")
        f.write(f"Number of predicted edges: {np.sum(B_pred)}\n")
        if runtime is not None:
            f.write(f"Runtime: {runtime:.4f} seconds\n")
        if metrics:
            f.write(f"\nKey Metrics:\n")
            f.write(f"  F1 Score: {metrics['F1']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  SHD: {metrics['shd']}\n")
    print(f"✓ Saved summary to {summary_file}")
    
    print(f"\n✓✓✓ All results saved to {output_dir}")
    return metrics


def plot_comparison(B_true: np.ndarray,
                   B_pred: np.ndarray,
                   output_path: str,
                   node_names: Optional[List[str]] = None,
                   algorithm_name: str = "Algorithm") -> None:
    """
    Plot comparison between true and predicted graphs
    
    Args:
        B_true: True binary adjacency matrix
        B_pred: Predicted binary adjacency matrix
        output_path: Path to save figure
        node_names: List of node names
        algorithm_name: Name of the algorithm
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    d = B_true.shape[0]
    if node_names is None:
        node_names = [f'X{i}' for i in range(d)]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # True graph
    sns.heatmap(B_true, annot=True, fmt='g', cmap='Blues',
                xticklabels=node_names, yticklabels=node_names,
                cbar_kws={'label': 'Edge'}, ax=axes[0],
                linewidths=0.5, linecolor='gray')
    axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('To Node')
    axes[0].set_ylabel('From Node')
    
    # Predicted graph
    sns.heatmap(B_pred, annot=True, fmt='g', cmap='Oranges',
                xticklabels=node_names, yticklabels=node_names,
                cbar_kws={'label': 'Edge'}, ax=axes[1],
                linewidths=0.5, linecolor='gray')
    axes[1].set_title(f'Predicted ({algorithm_name})', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('To Node')
    axes[1].set_ylabel('From Node')
    
    # Difference (errors)
    diff = np.abs(B_true - B_pred)
    sns.heatmap(diff, annot=True, fmt='g', cmap='Reds',
                xticklabels=node_names, yticklabels=node_names,
                cbar_kws={'label': 'Error'}, ax=axes[2],
                linewidths=0.5, linecolor='gray')
    axes[2].set_title('Errors (FP + FN)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('To Node')
    axes[2].set_ylabel('From Node')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved comparison plot to {output_path}")

def plot_adjacency_heatmap(adj_matrix: np.ndarray,
                          output_path: str,
                          node_names: Optional[List[str]] = None,
                          title: str = "Adjacency Heatmap",
                          cmap: str = "seismic",
                          vmin: Optional[float] = None,
                          vmax: Optional[float] = None,
                          center: float = 0,
                          figsize: Tuple[int, int] = (10, 9),
                          annot: bool = False,
                          fmt: str = '.2f') -> None:
    """
    Create a high-quality heatmap visualization of adjacency matrix
    
    Args:
        adj_matrix: Adjacency matrix (can be binary or weighted)
        output_path: Path to save the heatmap
        node_names: List of node names for axis labels
        title: Plot title
        cmap: Colormap name (seismic, RdBu_r, coolwarm, etc.)
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        center: Center value for diverging colormap
        figsize: Figure size (width, height)
        annot: Whether to annotate cells with values
        fmt: String formatting for annotations
    
    Example:
        >>> # For weighted matrix
        >>> plot_adjacency_heatmap(
        ...     adj_matrix=W,
        ...     output_path='weighted_adj.png',
        ...     node_names=['X1', 'X2', 'X3'],
        ...     vmin=-5, vmax=5
        ... )
        
        >>> # For binary matrix
        >>> plot_adjacency_heatmap(
        ...     adj_matrix=B,
        ...     output_path='binary_adj.png',
        ...     cmap='Blues',
        ...     vmin=0, vmax=1
        ... )
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Convert to DataFrame
    if isinstance(adj_matrix, pd.DataFrame):
        df = adj_matrix
    else:
        # Generate default node names if not provided
        if node_names is None:
            n_nodes = adj_matrix.shape[0]
            node_names = [f'X{i}' for i in range(n_nodes)]
        df = pd.DataFrame(adj_matrix, index=node_names, columns=node_names)
    
    # Auto-detect value range if not specified
    if vmin is None:
        vmin = df.values.min()
    if vmax is None:
        vmax = df.values.max()
    
    # Configure seaborn style
    sns.set(style="white")
    plt.figure(figsize=figsize)
    
    # Create heatmap
    ax = sns.heatmap(
        df,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        linewidths=0.5,
        linecolor='gray',
        square=True,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"shrink": 0.8},
        annot=annot,
        fmt=fmt
    )
    
    # Set axis labels
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    # Set title
    plt.title(title, fontsize=14, pad=15)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved adjacency heatmap to {output_path}")


def plot_weighted_comparison_heatmaps(W_true: np.ndarray,
                                     W_pred: np.ndarray,
                                     output_path: str,
                                     node_names: Optional[List[str]] = None,
                                     algorithm_name: str = "Algorithm",
                                     vmin: Optional[float] = None,
                                     vmax: Optional[float] = None,
                                     figsize: Tuple[int, int] = (18, 6)) -> None:
    """
    Create side-by-side heatmaps comparing true and predicted weighted adjacency matrices
    
    Args:
        W_true: True weighted adjacency matrix
        W_pred: Predicted weighted adjacency matrix
        output_path: Path to save the comparison plot
        node_names: List of node names
        algorithm_name: Name of the algorithm
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        figsize: Figure size
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Generate default node names if not provided
    if node_names is None:
        n_nodes = W_true.shape[0]
        node_names = [f'X{i}' for i in range(n_nodes)]
    
    # Auto-detect value range
    if vmin is None:
        vmin = min(W_true.min(), W_pred.min())
    if vmax is None:
        vmax = max(W_true.max(), W_pred.max())
    
    # Create DataFrames
    df_true = pd.DataFrame(W_true, index=node_names, columns=node_names)
    df_pred = pd.DataFrame(W_pred, index=node_names, columns=node_names)
    df_diff = pd.DataFrame(W_true - W_pred, index=node_names, columns=node_names)
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    sns.set(style="white")
    
    # True weights
    sns.heatmap(df_true, cmap='seismic', vmin=vmin, vmax=vmax, center=0,
                xticklabels=True, yticklabels=True,
                cbar_kws={'label': 'Weight'}, ax=axes[0],
                linewidths=0.5, linecolor='gray', square=True)
    axes[0].set_title('Ground Truth (Weighted)', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45, labelsize=9)
    axes[0].tick_params(axis='y', rotation=0, labelsize=9)
    
    # Predicted weights
    sns.heatmap(df_pred, cmap='seismic', vmin=vmin, vmax=vmax, center=0,
                xticklabels=True, yticklabels=True,
                cbar_kws={'label': 'Weight'}, ax=axes[1],
                linewidths=0.5, linecolor='gray', square=True)
    axes[1].set_title(f'Predicted ({algorithm_name})', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45, labelsize=9)
    axes[1].tick_params(axis='y', rotation=0, labelsize=9)
    
    # Difference
    diff_max = max(abs(df_diff.values.min()), abs(df_diff.values.max()))
    sns.heatmap(df_diff, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max, center=0,
                xticklabels=True, yticklabels=True,
                cbar_kws={'label': 'Difference'}, ax=axes[2],
                linewidths=0.5, linecolor='gray', square=True)
    axes[2].set_title('Weight Difference (True - Pred)', fontsize=14, fontweight='bold')
    axes[2].tick_params(axis='x', rotation=45, labelsize=9)
    axes[2].tick_params(axis='y', rotation=0, labelsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved weighted comparison heatmaps to {output_path}")