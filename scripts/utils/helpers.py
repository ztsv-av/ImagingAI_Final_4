import numpy as np
import random
import json

import torch

from scripts.utils.vars import SEED

def setGlobalSeed() -> None:
    """
    Sets the random seed for PyTorch, NumPy, and Python's random module to ensure reproducibility.
    """
    # set seed for Python's random module
    random.seed(SEED)
    # set seed for NumPy
    np.random.seed(SEED)
    # set seed for PyTorch
    torch.manual_seed(SEED)
    # set seed for PyTorch's CUDA operations, if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)  # if multiple GPUs are being used

def writeTXT(data, save_path) -> None:
    """
    Saves data to a TXT file.

    Parameters:
        - data: Data to save.
        - save_path (str): Where to save the data.
    """
    with open(save_path, "w") as file:
        file.write(f"{data}")

def writeJSON(data, save_path) -> None:
    """
    Saves data to a JSON file.

    Parameters:
        - data: Data to save.
        - save_path (str): Where to save the data.
    """
    with open(save_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

def readJSON(path) -> dict:
    """
    Reads JSON file at given path.

    Parameters:
        - path (str): Path to the JSON file.
    """
    with open(path, "r") as file:
        data = json.load(file)
    return data
