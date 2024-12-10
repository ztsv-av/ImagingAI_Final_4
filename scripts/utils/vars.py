import os
import torch

# absolute path to the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # This gets the directory containing vars.py
PROJECT_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "../../"))  # Adjust the path to the project root

# data paths
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "BraTS2021_Training_Data")
PATIENT_FOLDER_NAME = "BraTS2021_"

# splits paths
TRAIN_IDS_PATH = os.path.join(PROJECT_ROOT, "splits", "train.txt")
VAL_IDS_PATH = os.path.join(PROJECT_ROOT, "splits", "validation.txt")
TEST_IDS_PATH = os.path.join(PROJECT_ROOT, "splits", "test.txt")

# splits percentage
TRAIN_PERCENTAGE = 0.75
VAL_PERCENTAGE = 0.15

# path to file containing normalization values
STANDARDIZATION_STATS_PATH = os.path.join(PROJECT_ROOT, "scripts", "data", "standardization_stats.json")

# path to save/load the best model weights
BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, "scripts", "model", "best_model.pth")
KFOLD_RESULTS_PATH = os.path.join(PROJECT_ROOT, "scripts", "model", "kfold_results.json")

# seed
SEED = 1

# device for PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training variables
KFOLD_SPLITS = 5
EPOCHS = 2
BATCH_SIZE = 6
LR = 0.001
PATIENCE = 8
