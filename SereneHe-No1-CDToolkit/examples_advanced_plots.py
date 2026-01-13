"""
Advanced Visualization Examples for SereneHe-No1-CDToolkit

This script demonstrates how to use the new advanced plotting functions:
1. Polar (circular) bar charts for metric comparison
2. Grouped bar charts for percentage error analysis
3. Comprehensive comparison workflows
"""

import os
import numpy as np
import pandas as pd
from evaluation.advanced_plots import (
    plot_polar_metrics,
    plot_metrics_error_bars,
    create_comprehensive_comparison_plots,
    quick_comparison_plots
)


def example_1_polar_chart():
    """Example 1: Create a polar chart for SID comparison"""
    print("\n" + "="*60)
    print("Example 1: Polar Chart for SID Comparison")
    print("="*60)
    
    # Sample data: comparing different algorithms
    results = {
        'Method': ['PC', 'GES', 'DirectLiNGAM', 'ICALiNGAM', 'NOTEARS', 
                   'DAGMA', 'GOLEM', 'GraN-DAG', 'RL', 'GAE'],
        'SID': [45, 38, 52, 48, 35, 30, 42, 40, 55, 50],
        'Category': ['Constraint-based', 'Score-based', 'Function-based', 
                    'Function-based', 'Gradient-based', 'Gradient-based',
                    'Gradient-based', 'Gradient-based', 'Gradient-based', 
                    'Gradient-based']
    }
    df = pd.DataFrame(results)
    
    # Create output directory
    output_dir = 'examples/advanced_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate polar chart
    plot_polar_metrics(
        results_df=df,
        output_path=os.path.join(output_dir, 'polar_sid_example.png'),
        metric='SID',
        title='SID Scores by Algorithm Category'
    )
    
    print("\n✓ Polar chart created successfully!")
    print(f"  Location: {output_dir}/polar_sid_example.png")
    print("\nThis circular chart shows:")
    print("  - Each algorithm as a bar radiating from center")
    print("  - Bar height = SID score")
    print("  - Bar color = algorithm category")
    print("  - Perfect for comparing many algorithms at once")


def example_2_error_bars():
    """Example 2: Create grouped bar chart for percentage errors"""
    print("\n" + "="*60)
    print("Example 2: Percentage Error Comparison")
    print("="*60)
    
    # Sample data with multiple metrics
    results = {
        'Method': ['PC', 'GES', 'DirectLiNGAM', 'NOTEARS', 'DAGMA'],
        'SHD': [15, 12, 18, 10, 8],
        'SID': [45, 38, 52, 35, 30],
        'FDR': [0.25, 0.20, 0.30, 0.15, 0.12],
        'FPR': [0.18, 0.15, 0.22, 0.12, 0.10],
        'Recall': [0.75, 0.80, 0.70, 0.85, 0.88],
        'F1': [0.68, 0.72, 0.65, 0.76, 0.78]
    }
    df = pd.DataFrame(results)
    
    # Define ideal/true values for each metric
    true_values = {
        'SHD': 0,      # Lower is better
        'SID': 0,      # Lower is better
        'FDR': 0,      # Lower is better
        'FPR': 0,      # Lower is better
        'Recall': 1.0, # Higher is better
        'F1': 1.0      # Higher is better
    }
    
    output_dir = 'examples/advanced_plots'
    
    # Generate error bar chart
    error_df = plot_metrics_error_bars(
        results_df=df,
        output_path=os.path.join(output_dir, 'error_bars_example.png'),
        true_values=true_values,
        title='Performance Metrics - Percentage Error Analysis'
    )
    
    print("\n✓ Error bar chart created successfully!")
    print(f"  Location: {output_dir}/error_bars_example.png")
    print(f"  Normalized data: {output_dir}/error_bars_example_normalized.csv")
    print("\nThis chart shows:")
    print("  - Each method as a group of bars")
    print("  - Each bar = percentage error for one metric")
    print("  - Lower bars = better performance (closer to ideal)")
    print("\nTop 3 methods by average error:")
    avg_errors = error_df.drop('Method', axis=1).mean(axis=1)
    top3 = error_df.loc[avg_errors.nsmallest(3).index, 'Method'].values
    for i, method in enumerate(top3, 1):
        print(f"  {i}. {method}")


def example_3_comprehensive_workflow():
    """Example 3: Complete workflow from CSV to all visualizations"""
    print("\n" + "="*60)
    print("Example 3: Comprehensive Comparison Workflow")
    print("="*60)
    
    # Create sample results CSV
    results = {
        'Method': ['PC', 'GES', 'FCI', 'DirectLiNGAM', 'ICALiNGAM', 
                   'NOTEARS', 'DAGMA', 'GOLEM', 'GraN-DAG', 'RL'],
        'SHD': [15, 12, 20, 18, 16, 10, 8, 14, 12, 22],
        'SID': [45, 38, 55, 52, 48, 35, 30, 42, 40, 60],
        'FDR': [0.25, 0.20, 0.32, 0.30, 0.28, 0.15, 0.12, 0.22, 0.20, 0.35],
        'FPR': [0.18, 0.15, 0.25, 0.22, 0.20, 0.12, 0.10, 0.16, 0.15, 0.28],
        'Recall': [0.75, 0.80, 0.68, 0.70, 0.72, 0.85, 0.88, 0.78, 0.80, 0.65],
        'F1': [0.68, 0.72, 0.62, 0.65, 0.67, 0.76, 0.78, 0.70, 0.72, 0.60],
        'Category': ['Constraint-based', 'Score-based', 'Constraint-based',
                    'Function-based', 'Function-based', 'Gradient-based',
                    'Gradient-based', 'Gradient-based', 'Gradient-based',
                    'Gradient-based']
    }
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_dir = 'examples/advanced_plots'
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'comprehensive_results.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"\n✓ Created sample results CSV: {csv_path}")
    
    # Generate all visualizations
    create_comprehensive_comparison_plots(
        results_csv=csv_path,
        output_dir=output_dir,
        experiment_name='comprehensive_demo',
        polar_metric='SID',
        bar_metrics=['SHD', 'SID', 'FDR', 'FPR', 'Recall', 'F1']
    )
    
    print("\n✓ Generated comprehensive visualizations:")
    print(f"  - Polar chart: {output_dir}/comprehensive_demo_polar_SID.png")
    print(f"  - Error bars: {output_dir}/comprehensive_demo_metrics_error_bars.png")
    print(f"  - Normalized data: {output_dir}/comprehensive_demo_metrics_error_bars_normalized.csv")


def example_4_quick_comparison():
    """Example 4: Quick comparison from results dictionary"""
    print("\n" + "="*60)
    print("Example 4: Quick Comparison from Dictionary")
    print("="*60)
    
    # Results as nested dictionary
    results = {
        'PC': {
            'F1': 0.68, 'SHD': 15, 'SID': 45, 
            'FDR': 0.25, 'Recall': 0.75,
            'Category': 'Constraint-based'
        },
        'GES': {
            'F1': 0.72, 'SHD': 12, 'SID': 38,
            'FDR': 0.20, 'Recall': 0.80,
            'Category': 'Score-based'
        },
        'NOTEARS': {
            'F1': 0.76, 'SHD': 10, 'SID': 35,
            'FDR': 0.15, 'Recall': 0.85,
            'Category': 'Gradient-based'
        },
        'DAGMA': {
            'F1': 0.78, 'SHD': 8, 'SID': 30,
            'FDR': 0.12, 'Recall': 0.88,
            'Category': 'Gradient-based'
        }
    }
    
    output_dir = 'examples/advanced_plots'
    
    # Quick comparison (auto-converts dict to CSV and generates plots)
    quick_comparison_plots(
        results=results,
        output_dir=output_dir,
        experiment_name='quick_demo'
    )
    
    print("\n✓ Quick comparison complete!")
    print("  This function automatically:")
    print("  1. Converts dictionary to DataFrame")
    print("  2. Saves to CSV")
    print("  3. Generates polar and bar charts")
    print("\nPerfect for rapid prototyping and testing!")


def example_5_custom_colors():
    """Example 5: Customizing colors and styles"""
    print("\n" + "="*60)
    print("Example 5: Custom Colors and Styles")
    print("="*60)
    
    # Sample data
    results = {
        'Method': ['ExDAG-BOSS', 'ExDAG-GOBNILP', 'ExMAG', 'ExDBN-MILP'],
        'SID': [25, 20, 35, 30],
        'Category': ['Exact', 'Exact', 'Exact', 'Exact']
    }
    df = pd.DataFrame(results)
    
    output_dir = 'examples/advanced_plots'
    
    # Custom category colors
    custom_colors = {
        'Exact': '#9467bd',        # Purple for exact methods
        'Constraint-based': '#1f77b4',
        'Score-based': '#ff7f0e',
        'Gradient-based': '#2ca02c',
        'Function-based': '#d62728'
    }
    
    # Custom metric colors
    metric_colors = {
        'SID': '#ffd700',     # Gold
        'SHD': '#4caf50',     # Green
        'F1': '#003366',      # Dark blue
        'Recall': '#ff4500'   # Orange-red
    }
    
    # Generate with custom colors
    plot_polar_metrics(
        results_df=df,
        output_path=os.path.join(output_dir, 'polar_custom_colors.png'),
        metric='SID',
        category_colors=custom_colors,
        title='Exact Methods Performance (Custom Colors)'
    )
    
    print("\n✓ Custom styled plot created!")
    print(f"  Location: {output_dir}/polar_custom_colors.png")
    print("\nCustomization options:")
    print("  - category_colors: Dict mapping categories to hex colors")
    print("  - metric_colors: Dict mapping metrics to hex colors")
    print("  - figsize: Tuple (width, height) in inches")
    print("  - title: Custom plot title")


def example_6_real_world_krebs_cycle():
    """Example 6: Reproducing Krebs cycle visualization"""
    print("\n" + "="*60)
    print("Example 6: Krebs Cycle Analysis (Real-world Example)")
    print("="*60)
    
    # Simulated Krebs cycle results
    np.random.seed(42)
    methods = ['PC', 'GES', 'FCI', 'GFCI', 'DirectLiNGAM', 'ICALiNGAM', 
               'NOTEARS', 'DAGMA', 'GOLEM', 'GraN-DAG', 'RL', 'GAE',
               'ExDAG-BOSS', 'ExDAG-GOBNILP']
    
    categories = ['Constraint-based', 'Score-based', 'Constraint-based', 
                 'Constraint-based', 'Function-based', 'Function-based',
                 'Gradient-based', 'Gradient-based', 'Gradient-based',
                 'Gradient-based', 'Gradient-based', 'Gradient-based',
                 'Exact', 'Exact']
    
    results = pd.DataFrame({
        'Method': methods,
        'SHD': np.random.randint(50, 200, len(methods)),
        'SID': np.random.randint(80, 250, len(methods)),
        'FDR': np.random.uniform(0.3, 0.8, len(methods)),
        'FPR': np.random.uniform(0.2, 0.6, len(methods)),
        'Recall': np.random.uniform(0.4, 0.9, len(methods)),
        'F1': np.random.uniform(0.3, 0.7, len(methods)),
        'Category': categories
    })
    
    output_dir = 'examples/advanced_plots/krebs_cycle'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    csv_path = os.path.join(output_dir, 'krebs_cycle_results.csv')
    results.to_csv(csv_path, index=False)
    
    # Generate visualizations
    create_comprehensive_comparison_plots(
        results_csv=csv_path,
        output_dir=output_dir,
        experiment_name='Krebs_Cycle',
        polar_metric='SID',
        bar_metrics=['SHD', 'SID', 'FDR', 'FPR', 'Recall', 'F1'],
        true_values={
            'SHD': 200, 'SID': 150, 'FDR': 1.0, 
            'F1': 0.3, 'Recall': 1.0, 'FPR': 1.0
        }
    )
    
    print("\n✓ Krebs cycle analysis complete!")
    print(f"  Results saved to: {output_dir}")
    print("\nThis demonstrates:")
    print("  - Large-scale comparison (14 algorithms)")
    print("  - Multiple categories (Constraint/Score/Function/Gradient/Exact)")
    print("  - Real-world metric distributions")
    print("  - Custom true values for error calculation")


if __name__ == '__main__':
    print("\n" + "="*70)
    print(" Advanced Visualization Examples for SereneHe-No1-CDToolkit")
    print("="*70)
    
    # Run all examples
    example_1_polar_chart()
    example_2_error_bars()
    example_3_comprehensive_workflow()
    example_4_quick_comparison()
    example_5_custom_colors()
    example_6_real_world_krebs_cycle()
    
    print("\n" + "="*70)
    print(" All Examples Completed Successfully!")
    print("="*70)
    print("\nGenerated files:")
    print("  examples/advanced_plots/")
    print("    ├── polar_sid_example.png")
    print("    ├── error_bars_example.png")
    print("    ├── error_bars_example_normalized.csv")
    print("    ├── comprehensive_demo_polar_SID.png")
    print("    ├── comprehensive_demo_metrics_error_bars.png")
    print("    ├── quick_demo_results.csv")
    print("    ├── quick_demo_polar_SID.png")
    print("    ├── polar_custom_colors.png")
    print("    └── krebs_cycle/")
    print("          ├── krebs_cycle_results.csv")
    print("          ├── Krebs_Cycle_polar_SID.png")
    print("          └── Krebs_Cycle_metrics_error_bars.png")
    print("\nNext steps:")
    print("  1. View the generated plots to understand the visualizations")
    print("  2. Adapt examples to your own data")
    print("  3. Customize colors, metrics, and layouts as needed")
    print("  4. Integrate into your causal discovery pipeline")
    print("\n")
