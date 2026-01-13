"""
Evaluation module for SereneHe-No1-CDToolkit
Provides comprehensive metrics, visualization, and plotting functions
"""

from .metrics import (
    count_accuracy,
    compute_sid,
    compute_cpdag_metrics,
    compute_all_metrics,
    format_metrics
)

from .visualization import (
    save_predicted_edges,
    plot_network_graph,
    plot_adjacency_matrix,
    save_comprehensive_results,
    plot_comparison,
    plot_adjacency_heatmap,
    plot_weighted_comparison_heatmaps
)

from .plotting import (
    plot_metrics_over_experiments,
    plot_runtime_analysis,
    plot_f1_shd_runtime,
    create_results_summary_table
)

from .advanced_plots import (
    plot_polar_metrics,
    plot_metrics_error_bars,
    create_comprehensive_comparison_plots,
    quick_comparison_plots
)

__all__ = [
    # Metrics
    'count_accuracy',
    'compute_sid',
    'compute_cpdag_metrics',
    'compute_all_metrics',
    'format_metrics',
    
    # Visualization
    'save_predicted_edges',
    'plot_network_graph',
    'plot_adjacency_matrix',
    'save_comprehensive_results',
    'plot_comparison',
    'plot_adjacency_heatmap',
    'plot_weighted_comparison_heatmaps',
    
    # Plotting
    'plot_metrics_over_experiments',
    'plot_runtime_analysis',
    'plot_f1_shd_runtime',
    'create_results_summary_table',
    
    # Advanced plots
    'plot_polar_metrics',
    'plot_metrics_error_bars',
    'create_comprehensive_comparison_plots',
    'quick_comparison_plots',
]
