"""
Advanced visualization functions for causal discovery results
Includes polar charts and grouped bar charts for comprehensive comparison
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import Dict, List, Optional, Tuple


def plot_polar_metrics(results_df: pd.DataFrame,
                       output_path: str,
                       metric: str = 'SID',
                       method_col: str = 'Method',
                       category_col: str = 'Category',
                       title: Optional[str] = None,
                       category_colors: Optional[Dict[str, str]] = None,
                       figsize: Tuple[int, int] = (10, 10)) -> None:
    """
    Create polar (circular) bar chart for metrics comparison
    
    Args:
        results_df: DataFrame with columns [Method, Category, metric]
        output_path: Path to save the figure
        metric: Name of the metric column to plot
        method_col: Name of the method column
        category_col: Name of the category column
        title: Plot title (optional)
        category_colors: Dictionary mapping categories to colors
        figsize: Figure size
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Extract data
    methods = results_df[method_col].values
    scores = results_df[metric].values
    categories = results_df[category_col].values
    
    # Default category colors
    if category_colors is None:
        category_colors = {
            'Constraint-based': '#1f77b4',
            'Function-based': '#2ca02c',
            'Score-based': '#ff7f0e',
            'Gradient-based': '#d62728',
            'Exact': '#9467bd',
            'Hybrid': '#8c564b'
        }
    
    # Parameters
    N = len(scores)
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    radii = scores
    width = 2 * np.pi / N * 0.85
    bar_colors = [category_colors.get(cat, '#808080') for cat in categories]
    
    # Create polar plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
    ax.set_aspect('equal')
    
    # Draw bars
    bars = ax.bar(theta, radii, width=width, color=bar_colors, 
                  edgecolor='white', alpha=0.9)
    
    # Add method labels
    for i, (angle, radius, label) in enumerate(zip(theta, radii, methods)):
        angle_deg = np.degrees(angle)
        
        # Label position: slightly below bar top, towards center
        n_label = len(label)
        label_radius = max(radius - n_label * 0.5 - 50, 10)
        
        # Text alignment and rotation
        if np.pi/2 < angle < 3*np.pi/2:
            rotation = angle_deg + 180  # Flip to prevent upside-down text
            ha = 'right'
        else:
            rotation = angle_deg
            ha = 'left'
        
        ax.text(angle, label_radius, label,
                rotation=rotation,
                rotation_mode='anchor',
                ha=ha, va='center',
                fontsize=8)
    
    # Add legend
    unique_categories = sorted(set(categories))
    legend_elements = [Patch(facecolor=category_colors.get(cat, '#808080'), label=cat) 
                      for cat in unique_categories]
    ax.legend(handles=legend_elements, loc='upper right', 
             bbox_to_anchor=(1.2, 1.1), fontsize=12, title='Category')
    
    # Beautify
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines['polar'].set_visible(False)
    
    if title:
        ax.set_title(title, va='bottom', fontsize=14, weight='bold', pad=70)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved polar chart to {output_path}")


def plot_metrics_error_bars(results_df: pd.DataFrame,
                            output_path: str,
                            true_values: Dict[str, float],
                            metrics_to_plot: Optional[List[str]] = None,
                            method_col: str = 'Method',
                            title: Optional[str] = None,
                            metric_colors: Optional[Dict[str, str]] = None,
                            figsize: Tuple[int, int] = (18, 6),
                            normalize: bool = True) -> pd.DataFrame:
    """
    Create grouped bar chart showing percentage error for multiple metrics
    
    Args:
        results_df: DataFrame with method and metric columns
        output_path: Path to save the figure
        true_values: Dictionary of true/ideal values for each metric
        metrics_to_plot: List of metrics to include (default: all in true_values)
        method_col: Name of the method column
        title: Plot title
        metric_colors: Dictionary mapping metrics to colors
        figsize: Figure size
        normalize: Whether to compute percentage error
    
    Returns:
        DataFrame with computed errors
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Default metrics
    if metrics_to_plot is None:
        metrics_to_plot = list(true_values.keys())
    
    # Default colors
    if metric_colors is None:
        metric_colors = {
            'F1': '#003366',
            'F-score': '#003366',
            'Recall': '#ff4500',
            'SID': '#ffd700',
            'SHD': '#4caf50',
            'Precision': '#800000',
            'FDR': '#87cefa',
            'TPR': '#ff6347',
            'FPR': '#9370db'
        }
    
    # Extract data
    methods = results_df[method_col].values
    df_metrics = results_df[metrics_to_plot].copy()
    
    # Compute percentage error
    def percent_error(computed, true):
        if true == 0:
            return computed
        return abs((computed - true) / true) * 100
    
    error_df = df_metrics.copy()
    if normalize:
        for col in metrics_to_plot:
            true_val = true_values.get(col, 0)
            error_df[col] = df_metrics[col].apply(lambda x: percent_error(x, true_val))
    else:
        error_df = df_metrics
    
    # Save normalized results
    error_df_with_methods = error_df.copy()
    error_df_with_methods[method_col] = methods
    norm_output_path = output_path.replace('.png', '_normalized.csv')
    error_df_with_methods.to_csv(norm_output_path, index=False)
    print(f"✓ Saved normalized data to {norm_output_path}")
    
    # Plot
    bar_width = 0.8 / len(metrics_to_plot)
    x = np.arange(len(methods))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw bars for each metric
    for i, metric in enumerate(metrics_to_plot):
        values = error_df[metric].values
        color = metric_colors.get(metric, f'C{i}')
        ax.bar(x + i * bar_width, values, width=bar_width, 
               label=metric, color=color)
    
    # Set X axis
    ax.set_xticks(x + bar_width * (len(metrics_to_plot) - 1) / 2)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel("Error (%)" if normalize else "Value", fontsize=13)
    
    # Legend & style
    ax.legend(title="Metric", bbox_to_anchor=(1.05, 1), 
             loc='upper left', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    if title:
        ax.set_title(title, va='bottom', fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved error bar chart to {output_path}")
    
    return error_df_with_methods


def create_comprehensive_comparison_plots(results_csv: str,
                                          output_dir: str,
                                          experiment_name: str = "experiment",
                                          method_col: str = 'Method',
                                          category_col: Optional[str] = 'Category',
                                          polar_metric: str = 'SID',
                                          bar_metrics: Optional[List[str]] = None,
                                          true_values: Optional[Dict[str, float]] = None) -> None:
    """
    Create both polar and bar chart visualizations from results CSV
    
    Args:
        results_csv: Path to CSV file with results
        output_dir: Directory to save output plots
        experiment_name: Name of the experiment (for file naming)
        method_col: Name of the method column
        category_col: Name of the category column (for polar chart)
        polar_metric: Metric to use for polar chart
        bar_metrics: Metrics to include in bar chart
        true_values: True/ideal values for metrics (for error calculation)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(results_csv)
    
    # Default bar metrics
    if bar_metrics is None:
        bar_metrics = ['SHD', 'SID', 'FDR', 'FPR', 'Recall', 'F1']
        # Filter to available columns
        bar_metrics = [m for m in bar_metrics if m in df.columns or 
                      m.replace('F1', 'F-score') in df.columns]
    
    # Default true values
    if true_values is None:
        true_values = {
            'SHD': 0,
            'SID': 0,
            'FDR': 0,
            'F1': 1.0,
            'F-score': 1.0,
            'Precision': 1.0,
            'Recall': 1.0,
            'TPR': 1.0,
            'FPR': 0
        }
    
    # Create polar chart
    if category_col and category_col in df.columns and polar_metric in df.columns:
        polar_path = os.path.join(output_dir, f'{experiment_name}_polar_{polar_metric}.png')
        plot_polar_metrics(
            results_df=df,
            output_path=polar_path,
            metric=polar_metric,
            method_col=method_col,
            category_col=category_col,
            title=f"{polar_metric} Scores by Algorithm Category"
        )
    
    # Create bar chart
    available_metrics = [m for m in bar_metrics if m in df.columns]
    if available_metrics:
        bar_path = os.path.join(output_dir, f'{experiment_name}_metrics_error_bars.png')
        error_df = plot_metrics_error_bars(
            results_df=df,
            output_path=bar_path,
            true_values=true_values,
            metrics_to_plot=available_metrics,
            method_col=method_col,
            title=f"Performance Metrics Comparison - {experiment_name}"
        )
    
    print(f"\n✓✓✓ All comparison plots saved to {output_dir}")


# Convenience function for quick comparison
def quick_comparison_plots(results: Dict[str, Dict],
                          output_dir: str,
                          experiment_name: str = "comparison") -> None:
    """
    Quick comparison plots from results dictionary
    
    Args:
        results: Dictionary mapping method names to metric dictionaries
                Example: {'PC': {'F1': 0.8, 'SHD': 5, 'SID': 10, 'Category': 'Constraint-based'}, ...}
        output_dir: Output directory
        experiment_name: Experiment name
    """
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(results, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Method'}, inplace=True)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, f'{experiment_name}_results.csv')
    df.to_csv(csv_path, index=False)
    
    # Create plots
    create_comprehensive_comparison_plots(
        results_csv=csv_path,
        output_dir=output_dir,
        experiment_name=experiment_name
    )
