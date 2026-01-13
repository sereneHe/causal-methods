"""
Heatmap Visualization Examples for SereneHe-No1-CDToolkit

Demonstrates how to create various types of adjacency matrix heatmaps:
1. Single adjacency matrix heatmap (binary or weighted)
2. Side-by-side comparison heatmaps (true vs predicted)
3. Custom styling and colormaps
4. Real-world example with Krebs cycle data
"""

import os
import numpy as np
import pandas as pd
from evaluation import (
    plot_adjacency_heatmap,
    plot_weighted_comparison_heatmaps
)


def example_1_binary_heatmap():
    """Example 1: Basic binary adjacency matrix heatmap"""
    print("\n" + "="*60)
    print("Example 1: Binary Adjacency Matrix Heatmap")
    print("="*60)
    
    # Create sample binary DAG
    n = 5
    B = np.array([
        [0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0]
    ])
    
    node_names = ['X0', 'X1', 'X2', 'X3', 'X4']
    
    output_dir = 'examples/heatmaps'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create heatmap
    plot_adjacency_heatmap(
        adj_matrix=B,
        output_path=os.path.join(output_dir, 'binary_heatmap.png'),
        node_names=node_names,
        title='Binary Adjacency Matrix',
        cmap='Blues',
        vmin=0,
        vmax=1,
        annot=True,
        fmt='d'
    )
    
    print("\n✓ Binary heatmap created!")
    print(f"  Location: {output_dir}/binary_heatmap.png")
    print("\nFeatures:")
    print("  - Binary values (0 or 1)")
    print("  - Blue colormap for edges")
    print("  - Cell annotations showing exact values")
    print("  - Grid lines for clarity")


def example_2_weighted_heatmap():
    """Example 2: Weighted adjacency matrix with diverging colormap"""
    print("\n" + "="*60)
    print("Example 2: Weighted Adjacency Matrix Heatmap")
    print("="*60)
    
    # Create sample weighted DAG
    np.random.seed(42)
    n = 6
    W = np.array([
        [0.0,  2.3, 0.0, 0.0, -1.5, 0.0],
        [0.0,  0.0, 1.8, 0.0,  0.0, 0.0],
        [0.0,  0.0, 0.0, 3.2,  0.0, 1.1],
        [0.0,  0.0, 0.0, 0.0,  2.7, 0.0],
        [0.0,  0.0, 0.0, 0.0,  0.0, -2.1],
        [0.0,  0.0, 0.0, 0.0,  0.0, 0.0]
    ])
    
    node_names = ['A', 'B', 'C', 'D', 'E', 'F']
    
    output_dir = 'examples/heatmaps'
    
    # Create heatmap with diverging colormap
    plot_adjacency_heatmap(
        adj_matrix=W,
        output_path=os.path.join(output_dir, 'weighted_heatmap.png'),
        node_names=node_names,
        title='Weighted Adjacency Matrix (Causal Effects)',
        cmap='seismic',  # Red-blue diverging colormap
        vmin=-5,
        vmax=5,
        center=0,
        figsize=(10, 9),
        annot=True,
        fmt='.2f'
    )
    
    print("\n✓ Weighted heatmap created!")
    print(f"  Location: {output_dir}/weighted_heatmap.png")
    print("\nFeatures:")
    print("  - Weighted edges (positive and negative)")
    print("  - Seismic colormap (red=positive, blue=negative)")
    print("  - Centered at 0 for symmetric visualization")
    print("  - Annotations with 2 decimal places")


def example_3_comparison_heatmaps():
    """Example 3: Side-by-side comparison of true vs predicted"""
    print("\n" + "="*60)
    print("Example 3: True vs Predicted Comparison Heatmaps")
    print("="*60)
    
    # Create true and predicted weighted matrices
    np.random.seed(42)
    n = 5
    
    # True matrix
    W_true = np.array([
        [0.0,  2.0, 0.0, 0.0,  1.5],
        [0.0,  0.0, 1.8, 0.0,  0.0],
        [0.0,  0.0, 0.0, 3.0,  0.0],
        [0.0,  0.0, 0.0, 0.0,  2.5],
        [0.0,  0.0, 0.0, 0.0,  0.0]
    ])
    
    # Predicted matrix (with some errors)
    W_pred = np.array([
        [0.0,  1.8, 0.3, 0.0,  1.2],  # Small errors
        [0.0,  0.0, 2.1, 0.0,  0.0],
        [0.0,  0.0, 0.0, 2.8,  0.0],
        [0.0,  0.0, 0.0, 0.0,  2.3],
        [0.0,  0.0, 0.0, 0.0,  0.0]
    ])
    
    node_names = ['X1', 'X2', 'X3', 'X4', 'X5']
    
    output_dir = 'examples/heatmaps'
    
    # Create comparison heatmaps
    plot_weighted_comparison_heatmaps(
        W_true=W_true,
        W_pred=W_pred,
        output_path=os.path.join(output_dir, 'comparison_heatmaps.png'),
        node_names=node_names,
        algorithm_name='NOTEARS',
        vmin=-5,
        vmax=5,
        figsize=(18, 6)
    )
    
    print("\n✓ Comparison heatmaps created!")
    print(f"  Location: {output_dir}/comparison_heatmaps.png")
    print("\nFeatures:")
    print("  - Three panels: True | Predicted | Difference")
    print("  - Same color scale for true and predicted")
    print("  - Difference map highlights errors")
    print("  - Perfect for algorithm evaluation")


def example_4_krebs_cycle():
    """Example 4: Real-world Krebs cycle adjacency matrix"""
    print("\n" + "="*60)
    print("Example 4: Krebs Cycle Adjacency Matrix")
    print("="*60)
    
    # Krebs cycle node labels
    labels = [
        "FUMARATE", "GTP", "H2O", "CIS-ACONITATE", "MALATE", "OXALOACETATE", "FAD",
        "SUCCINYL-COA", "NAD", "A-K-GLUTARATE", "GDP", "NADH", "CITRATE",
        "SUCCINATE", "ISOCITRATE", "ACETYL-COA"
    ]
    
    # Create simulated Krebs cycle adjacency matrix
    # (In real use, you would load from CSV)
    np.random.seed(42)
    n = len(labels)
    
    # Simulate sparse metabolic network
    W = np.zeros((n, n))
    # Add some connections
    connections = [
        (15, 12, 3.5),  # ACETYL-COA -> CITRATE
        (12, 3, 2.8),   # CITRATE -> CIS-ACONITATE
        (3, 14, 2.5),   # CIS-ACONITATE -> ISOCITRATE
        (14, 9, 3.2),   # ISOCITRATE -> A-K-GLUTARATE
        (9, 7, 2.9),    # A-K-GLUTARATE -> SUCCINYL-COA
        (7, 13, 3.1),   # SUCCINYL-COA -> SUCCINATE
        (13, 0, 2.7),   # SUCCINATE -> FUMARATE
        (0, 4, 3.0),    # FUMARATE -> MALATE
        (4, 5, 3.3),    # MALATE -> OXALOACETATE
        (8, 11, 2.2),   # NAD -> NADH
        (10, 1, 1.8),   # GDP -> GTP
    ]
    
    for i, j, w in connections:
        W[i, j] = w
    
    output_dir = 'examples/heatmaps'
    
    # Create heatmap
    plot_adjacency_heatmap(
        adj_matrix=W,
        output_path=os.path.join(output_dir, 'krebs_cycle_heatmap.png'),
        node_names=labels,
        title='Krebs Cycle - Adjacency Heatmap (Ground Truth)',
        cmap='seismic',
        vmin=-5,
        vmax=5,
        center=0,
        figsize=(12, 10)
    )
    
    print("\n✓ Krebs cycle heatmap created!")
    print(f"  Location: {output_dir}/krebs_cycle_heatmap.png")
    print("\nFeatures:")
    print("  - 16 metabolites in Krebs cycle")
    print("  - Sparse metabolic network structure")
    print("  - Seismic colormap for biochemical reactions")
    print("  - Large figure size for readability")
    print("\nNote: Replace simulated data with your actual CSV:")
    print("  matrix = pd.read_csv('groundtruth_adj.csv', header=None)")


def example_5_custom_colormaps():
    """Example 5: Different colormap options"""
    print("\n" + "="*60)
    print("Example 5: Custom Colormaps and Styling")
    print("="*60)
    
    # Create sample weighted matrix
    np.random.seed(42)
    n = 4
    W = np.random.randn(n, n) * 2
    np.fill_diagonal(W, 0)
    W = np.tril(W)  # Make it DAG
    
    node_names = ['Gene1', 'Gene2', 'Gene3', 'Gene4']
    
    output_dir = 'examples/heatmaps'
    
    colormaps = [
        ('seismic', 'Red-Blue (Seismic)'),
        ('RdBu_r', 'Red-Blue Reversed'),
        ('coolwarm', 'Cool-Warm'),
        ('PRGn', 'Purple-Green'),
        ('RdYlBu_r', 'Red-Yellow-Blue')
    ]
    
    for cmap_name, cmap_label in colormaps:
        output_path = os.path.join(output_dir, f'colormap_{cmap_name}.png')
        plot_adjacency_heatmap(
            adj_matrix=W,
            output_path=output_path,
            node_names=node_names,
            title=f'Colormap: {cmap_label}',
            cmap=cmap_name,
            vmin=-5,
            vmax=5,
            center=0,
            figsize=(8, 7)
        )
    
    print("\n✓ Multiple colormaps created!")
    print(f"  Location: {output_dir}/colormap_*.png")
    print("\nAvailable colormaps:")
    for _, label in colormaps:
        print(f"  - {label}")
    print("\nChoose based on:")
    print("  - seismic: Standard diverging (red/blue)")
    print("  - RdBu_r: Reversed colors")
    print("  - coolwarm: Softer colors")
    print("  - PRGn: Color-blind friendly")
    print("  - RdYlBu_r: Three-way diverging")


def example_6_from_csv():
    """Example 6: Load from CSV and visualize"""
    print("\n" + "="*60)
    print("Example 6: Load Adjacency Matrix from CSV")
    print("="*60)
    
    # Create sample CSV file
    output_dir = 'examples/heatmaps'
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample data
    labels = ["FUMARATE", "GTP", "H2O", "CIS-ACONITATE", "MALATE", "OXALOACETATE"]
    n = len(labels)
    W = np.random.randn(n, n) * 2
    np.fill_diagonal(W, 0)
    W = np.tril(W)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'sample_groundtruth_adj.csv')
    pd.DataFrame(W).to_csv(csv_path, header=False, index=False)
    print(f"\n✓ Created sample CSV: {csv_path}")
    
    # Load and visualize
    matrix = pd.read_csv(csv_path, header=None)
    
    plot_adjacency_heatmap(
        adj_matrix=matrix.values,
        output_path=os.path.join(output_dir, 'from_csv_heatmap.png'),
        node_names=labels,
        title='Adjacency Matrix Loaded from CSV',
        cmap='seismic',
        vmin=-5,
        vmax=5,
        center=0,
        figsize=(10, 9)
    )
    
    print("\n✓ Heatmap created from CSV!")
    print(f"  CSV: {csv_path}")
    print(f"  Plot: {output_dir}/from_csv_heatmap.png")
    print("\nCode template:")
    print("""
    # Load your data
    matrix = pd.read_csv("path/to/your/adj_matrix.csv", header=None)
    
    # Create heatmap
    plot_adjacency_heatmap(
        adj_matrix=matrix.values,
        output_path='output.png',
        node_names=['X1', 'X2', ...],
        cmap='seismic',
        vmin=-5, vmax=5
    )
    """)


if __name__ == '__main__':
    print("\n" + "="*70)
    print(" Heatmap Visualization Examples for SereneHe-No1-CDToolkit")
    print("="*70)
    
    # Run all examples
    example_1_binary_heatmap()
    example_2_weighted_heatmap()
    example_3_comparison_heatmaps()
    example_4_krebs_cycle()
    example_5_custom_colormaps()
    example_6_from_csv()
    
    print("\n" + "="*70)
    print(" All Heatmap Examples Completed Successfully!")
    print("="*70)
    print("\nGenerated files:")
    print("  examples/heatmaps/")
    print("    ├── binary_heatmap.png")
    print("    ├── weighted_heatmap.png")
    print("    ├── comparison_heatmaps.png")
    print("    ├── krebs_cycle_heatmap.png")
    print("    ├── colormap_seismic.png")
    print("    ├── colormap_RdBu_r.png")
    print("    ├── colormap_coolwarm.png")
    print("    ├── colormap_PRGn.png")
    print("    ├── colormap_RdYlBu_r.png")
    print("    ├── from_csv_heatmap.png")
    print("    └── sample_groundtruth_adj.csv")
    print("\nNext steps:")
    print("  1. Replace sample data with your actual adjacency matrices")
    print("  2. Customize colormaps and styling to your needs")
    print("  3. Use plot_weighted_comparison_heatmaps() for algorithm evaluation")
    print("  4. Integrate into your causal discovery pipeline")
    print("\n")
