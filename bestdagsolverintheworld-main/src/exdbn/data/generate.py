import numpy as np

def generate_problem(cfg):
    """
    Generate a synthetic EXDBN dataset based on the provided configuration.

    Args:
        cfg (dict): Configuration dictionary containing dataset parameters.

    Returns:
        dict: Generated dataset containing variables and samples.
    """
    num_vars = cfg['problem']['number_of_variables']
    num_samples = cfg['problem']['number_of_samples']

    # Generate random data as a placeholder
    data = np.random.randn(num_samples, num_vars)

    return {
        'data': data,
        'config': cfg
    }

def save_problem(problem, out_path):
    """
    Save the generated dataset to a .npz file.

    Args:
        problem (dict): Generated dataset.
        out_path (str): Path to save the .npz file.
    """
    np.savez(out_path, **problem)