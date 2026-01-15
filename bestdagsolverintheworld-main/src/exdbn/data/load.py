import numpy as np

def load_problem_from_npz(file_path):
    """
    Load a problem dataset from a .npz file.

    Args:
        file_path (str): Path to the .npz file.

    Returns:
        dict: Loaded dataset.
    """
    data = np.load(file_path, allow_pickle=True)
    return {key: data[key] for key in data}