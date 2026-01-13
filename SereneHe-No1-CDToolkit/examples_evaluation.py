"""
Example usage of the comprehensive evaluation and visualization functions
"""

import numpy as np
from evaluation import (
    save_comprehensive_results,
    plot_f1_shd_runtime,
    plot_metrics_over_experiments,
    create_results_summary_table
)
from datasets import load_problem_dict


def example_single_algorithm_evaluation():
    """
    Example: Evaluate a single algorithm and save all outputs
    """
    # 1. Load data
    problem = {
        'name': 'er',
        'number_of_variables': 10,
        'number_of_samples': 500,
        'edge_ratio': 2.0,
        'sem_type': 'gauss'
    }
    
    W_true, W_bi_true, B_true, B_bi_true, A_true, X, Y, tabu_edges, node_names, inter_nodes = \
        load_problem_dict(problem)
    
    # 2. Run your algorithm (example with random prediction)
    import time
    start_time = time.time()
    
    # TODO: Replace with actual algorithm
    B_pred = (np.random.rand(10, 10) > 0.7).astype(int)
    np.fill_diagonal(B_pred, 0)  # No self-loops
    
    runtime = time.time() - start_time
    
    # 3. Save comprehensive results
    metrics = save_comprehensive_results(
        B_pred=B_pred,
        output_dir='results/my_algorithm',
        algorithm_name='MyAlgorithm',
        B_true=B_true,
        W_pred=None,  # Can provide weighted matrix if available
        node_names=node_names,
        runtime=runtime,
        additional_info={
            'problem_type': 'er',
            'n_variables': 10,
            'n_samples': 500,
            'edge_ratio': 2.0
        }
    )
    
    print("\nMetrics:")
    print(f"F1 Score: {metrics['F1']:.4f}")
    print(f"SHD: {metrics['shd']}")
    print(f"Runtime: {runtime:.4f}s")


def example_multiple_algorithms_comparison():
    """
    Example: Compare multiple algorithms across different experiment settings
    """
    # Simulate results from multiple algorithms and experiments
    # In practice, you would run actual algorithms here
    
    results = {
        'PC': [],
        'NOTEARS': [],
        'GES': []
    }
    
    # Simulate 5 experiments with varying sample sizes
    sample_sizes = [100, 250, 500, 1000, 2000]
    
    for n_samples in sample_sizes:
        # For each algorithm, run 3 repeats
        for repeat in range(3):
            # PC results (example)
            results['PC'].append({
                'F1': 0.7 + np.random.normal(0, 0.05),
                'precision': 0.75 + np.random.normal(0, 0.05),
                'recall': 0.68 + np.random.normal(0, 0.05),
                'shd': int(10 + np.random.normal(0, 2)),
                'runtime': 0.5 + n_samples * 0.0001 + np.random.normal(0, 0.1)
            })
            
            # NOTEARS results (example)
            results['NOTEARS'].append({
                'F1': 0.75 + np.random.normal(0, 0.04),
                'precision': 0.78 + np.random.normal(0, 0.04),
                'recall': 0.72 + np.random.normal(0, 0.04),
                'shd': int(8 + np.random.normal(0, 2)),
                'runtime': 2.0 + n_samples * 0.001 + np.random.normal(0, 0.2)
            })
            
            # GES results (example)
            results['GES'].append({
                'F1': 0.72 + np.random.normal(0, 0.06),
                'precision': 0.74 + np.random.normal(0, 0.06),
                'recall': 0.70 + np.random.normal(0, 0.06),
                'shd': int(9 + np.random.normal(0, 2)),
                'runtime': 1.0 + n_samples * 0.0005 + np.random.normal(0, 0.15)
            })
    
    # Create output directory
    output_dir = 'results/comparison'
    
    # Plot F1, SHD, Runtime comparison
    plot_f1_shd_runtime(
        results=results,
        output_dir=output_dir,
        x_variable='Sample Size',
        x_values=sample_sizes
    )
    
    # Plot all metrics
    plot_metrics_over_experiments(
        results=results,
        output_dir=output_dir,
        metrics_to_plot=['F1', 'precision', 'recall', 'shd', 'runtime'],
        x_variable='Sample Size',
        x_values=sample_sizes,
        show_confidence=True
    )
    
    # Create summary table
    create_results_summary_table(
        results=results,
        output_dir=output_dir
    )
    
    print(f"\nComparison results saved to {output_dir}")


def example_dataset_testing():
    """
    Example: Test all datasets in load_problem
    """
    test_problems = [
        # Synthetic datasets
        {'name': 'er', 'number_of_variables': 5, 'number_of_samples': 100, 'edge_ratio': 2.0},
        {'name': 'sf', 'number_of_variables': 5, 'number_of_samples': 100, 'edge_ratio': 2.0},
        {'name': 'PATH', 'number_of_variables': 5, 'number_of_samples': 100},
        {'name': 'ermag', 'number_of_variables': 10, 'number_of_samples': 100, 
         'edge_ratio': 1.5, 'tabu_edges_ratio': 0.2, 'hidden_vertices_ratio': 0.3},
        {'name': 'dynamic', 'number_of_variables': 5, 'number_of_samples': 100, 
         'number_of_lags': 2, 'intra_edge_ratio': 2.0},
        
        # Real datasets
        {'name': 'admissions', 'number_of_samples': 50},
    ]
    
    print("Testing all datasets...\n")
    for problem in test_problems:
        try:
            W_true, W_bi_true, B_true, B_bi_true, A_true, X, Y, tabu_edges, node_names, inter_nodes = \
                load_problem_dict(problem)
            
            print(f"✓ {problem['name']:15s} - Shape: {X.shape}, Edges: {np.sum(B_true)}")
        except Exception as e:
            print(f"✗ {problem['name']:15s} - Error: {str(e)}")
    
    print("\n✓ Dataset testing complete!")


if __name__ == '__main__':
    print("=" * 60)
    print("SereneHe-No1-CDToolkit - Evaluation Examples")
    print("=" * 60)
    
    print("\n1. Single Algorithm Evaluation")
    print("-" * 60)
    example_single_algorithm_evaluation()
    
    print("\n2. Multiple Algorithms Comparison")
    print("-" * 60)
    example_multiple_algorithms_comparison()
    
    print("\n3. Dataset Testing")
    print("-" * 60)
    example_dataset_testing()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
