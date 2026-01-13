"""
Plot performance metrics across multiple experiments with confidence intervals
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os


def plot_metrics_over_experiments(results: Dict[str, List[Dict]],
                                  output_dir: str,
                                  metrics_to_plot: List[str] = ['F1', 'precision', 'recall', 'shd', 'sid'],
                                  x_variable: str = 'experiment_id',
                                  x_values: Optional[List] = None,
                                  figsize: Tuple[int, int] = (15, 10),
                                  show_confidence: bool = True,
                                  confidence_alpha: float = 0.2) -> None:
    """
    Plot metrics across multiple experiments with confidence intervals
    
    Args:
        results: Dictionary mapping algorithm names to lists of metric dictionaries
                Example: {'PC': [{'F1': 0.8, 'shd': 5}, {'F1': 0.82, 'shd': 4}, ...],
                         'NOTEARS': [...]}
        output_dir: Directory to save plots
        metrics_to_plot: List of metric names to plot
        x_variable: Variable name for x-axis (default: 'experiment_id')
        x_values: Explicit x-axis values (e.g., sample sizes, edge densities)
        figsize: Figure size
        show_confidence: Whether to show confidence intervals (std deviation)
        confidence_alpha: Transparency of confidence interval shading
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine number of experiments
    n_experiments = max(len(runs) for runs in results.values())
    if x_values is None:
        x_values = list(range(1, n_experiments + 1))
    
    # Create subplots
    n_metrics = len(metrics_to_plot)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Color palette
    colors = sns.color_palette("husl", len(results))
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        for (alg_name, runs), color in zip(results.items(), colors):
            # Extract metric values
            values = []
            for run in runs:
                if metric in run:
                    values.append(run[metric])
                else:
                    values.append(np.nan)
            
            values = np.array(values)
            
            # If multiple runs per x-value, compute mean and std
            if len(values) != len(x_values):
                # Reshape to [n_x_values, n_repeats]
                n_repeats = len(values) // len(x_values)
                values_reshaped = values[:len(x_values) * n_repeats].reshape(len(x_values), n_repeats)
                
                mean_values = np.nanmean(values_reshaped, axis=1)
                std_values = np.nanstd(values_reshaped, axis=1)
            else:
                mean_values = values
                std_values = np.zeros_like(values)
            
            # Plot mean line
            ax.plot(x_values[:len(mean_values)], mean_values, 
                   marker='o', label=alg_name, color=color, linewidth=2)
            
            # Plot confidence interval
            if show_confidence and np.any(std_values > 0):
                ax.fill_between(x_values[:len(mean_values)],
                               mean_values - std_values,
                               mean_values + std_values,
                               alpha=confidence_alpha, color=color)
        
        ax.set_xlabel(x_variable, fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{metric} vs {x_variable}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved metrics comparison plot to {output_path}")


def plot_runtime_analysis(results: Dict[str, List[Dict]],
                         output_dir: str,
                         x_variable: str = 'n_variables',
                         x_values: Optional[List] = None,
                         log_scale: bool = False,
                         figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot runtime analysis across algorithms
    
    Args:
        results: Dictionary mapping algorithm names to lists of metric dictionaries
        output_dir: Directory to save plots
        x_variable: Variable name for x-axis
        x_values: Explicit x-axis values
        log_scale: Whether to use log scale for y-axis
        figsize: Figure size
    """
    os.makedirs(output_dir, exist_ok=True)
    
    n_experiments = max(len(runs) for runs in results.values())
    if x_values is None:
        x_values = list(range(1, n_experiments + 1))
    
    plt.figure(figsize=figsize)
    colors = sns.color_palette("husl", len(results))
    
    for (alg_name, runs), color in zip(results.items(), colors):
        # Extract runtime values
        runtimes = []
        for run in runs:
            if 'runtime' in run:
                runtimes.append(run['runtime'])
            else:
                runtimes.append(np.nan)
        
        runtimes = np.array(runtimes)
        
        # If multiple runs per x-value, compute mean and std
        if len(runtimes) != len(x_values):
            n_repeats = len(runtimes) // len(x_values)
            runtimes_reshaped = runtimes[:len(x_values) * n_repeats].reshape(len(x_values), n_repeats)
            mean_runtimes = np.nanmean(runtimes_reshaped, axis=1)
            std_runtimes = np.nanstd(runtimes_reshaped, axis=1)
        else:
            mean_runtimes = runtimes
            std_runtimes = np.zeros_like(runtimes)
        
        # Plot
        plt.plot(x_values[:len(mean_runtimes)], mean_runtimes,
                marker='s', label=alg_name, color=color, linewidth=2)
        
        # Confidence interval
        if np.any(std_runtimes > 0):
            plt.fill_between(x_values[:len(mean_runtimes)],
                           mean_runtimes - std_runtimes,
                           mean_runtimes + std_runtimes,
                           alpha=0.2, color=color)
    
    if log_scale:
        plt.yscale('log')
    
    plt.xlabel(x_variable, fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.title('Algorithm Runtime Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'runtime_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved runtime comparison plot to {output_path}")


def plot_f1_shd_runtime(results: Dict[str, List[Dict]],
                       output_dir: str,
                       x_variable: str = 'experiment_id',
                       x_values: Optional[List] = None,
                       figsize: Tuple[int, int] = (18, 5)) -> None:
    """
    Plot F1, SHD, and Runtime in a single figure with multiple subplots
    
    Args:
        results: Dictionary mapping algorithm names to lists of metric dictionaries
        output_dir: Directory to save plots
        x_variable: Variable name for x-axis
        x_values: Explicit x-axis values
        figsize: Figure size
    """
    os.makedirs(output_dir, exist_ok=True)
    
    n_experiments = max(len(runs) for runs in results.values())
    if x_values is None:
        x_values = list(range(1, n_experiments + 1))
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    colors = sns.color_palette("husl", len(results))
    
    metrics = ['F1', 'shd', 'runtime']
    titles = ['F1 Score', 'Structural Hamming Distance', 'Runtime (seconds)']
    
    for ax, metric, title in zip(axes, metrics, titles):
        for (alg_name, runs), color in zip(results.items(), colors):
            # Extract values
            values = []
            for run in runs:
                if metric in run:
                    values.append(run[metric])
                else:
                    values.append(np.nan)
            
            values = np.array(values)
            
            # Compute mean and std if needed
            if len(values) != len(x_values):
                n_repeats = len(values) // len(x_values)
                values_reshaped = values[:len(x_values) * n_repeats].reshape(len(x_values), n_repeats)
                mean_values = np.nanmean(values_reshaped, axis=1)
                std_values = np.nanstd(values_reshaped, axis=1)
            else:
                mean_values = values
                std_values = np.zeros_like(values)
            
            # Plot
            ax.plot(x_values[:len(mean_values)], mean_values,
                   marker='o', label=alg_name, color=color, linewidth=2, markersize=6)
            
            # Confidence interval
            if np.any(std_values > 0):
                ax.fill_between(x_values[:len(mean_values)],
                               mean_values - std_values,
                               mean_values + std_values,
                               alpha=0.2, color=color)
        
        ax.set_xlabel(x_variable, fontsize=11)
        ax.set_ylabel(metric.upper() if len(metric) <= 3 else metric.title(), fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'f1_shd_runtime_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved F1/SHD/Runtime comparison plot to {output_path}")


def create_results_summary_table(results: Dict[str, List[Dict]],
                                 output_dir: str,
                                 metrics: List[str] = ['F1', 'precision', 'recall', 'shd', 'sid', 'runtime']) -> None:
    """
    Create a summary table of results across all algorithms
    
    Args:
        results: Dictionary mapping algorithm names to lists of metric dictionaries
        output_dir: Directory to save table
        metrics: List of metrics to include in table
    """
    os.makedirs(output_dir, exist_ok=True)
    
    summary_data = []
    for alg_name, runs in results.items():
        row = {'Algorithm': alg_name}
        for metric in metrics:
            values = [run.get(metric, np.nan) for run in runs]
            values = np.array([v for v in values if not np.isnan(v)])
            if len(values) > 0:
                row[f'{metric}_mean'] = np.mean(values)
                row[f'{metric}_std'] = np.std(values)
            else:
                row[f'{metric}_mean'] = np.nan
                row[f'{metric}_std'] = np.nan
        summary_data.append(row)
    
    import pandas as pd
    df = pd.DataFrame(summary_data)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, 'results_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved results summary table to {csv_path}")
    
    # Save as formatted text
    txt_path = os.path.join(output_dir, 'results_summary.txt')
    with open(txt_path, 'w') as f:
        f.write(df.to_string(index=False))
    print(f"✓ Saved results summary text to {txt_path}")
