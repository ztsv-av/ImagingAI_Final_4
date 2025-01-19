import os
import torch

# absolute path to the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # This gets the directory containing vars.py
PROJECT_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "../../"))  # Adjust the path to the project root

# data paths
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "BraTS2021_Training_Data_Cropped")
CROPPED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "BraTS2021_Training_Data_Cropped")
PATIENT_FOLDER_NAME = "BraTS2021_"
PATIENTS_CROPPING_IDXS_PATH = os.path.join(PROJECT_ROOT, "scripts/data", "patients_cropping_idxs.json")
DATA_CONFIG_PATH = os.path.join(PROJECT_ROOT, "scripts/data", "data_config.json")

# splits paths
TRAIN_IDS_PATH = os.path.join(PROJECT_ROOT, "splits", "train.txt")
VAL_IDS_PATH = os.path.join(PROJECT_ROOT, "splits", "validation.txt")
TEST_IDS_PATH = os.path.join(PROJECT_ROOT, "splits", "test.txt")

# splits percentage
TRAIN_PERCENTAGE = 0.75
VAL_PERCENTAGE = 0.15

# path to save/load the best model weights
BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, "scripts", "model", "best_model.pth")
CHECKPOINTS_PATH = os.path.join(PROJECT_ROOT, "scripts", "model", "checkpoints")

# predictions
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "predictions")

# seed
SEED = 1

# device for PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training variables
KFOLD_SPLITS = 3
EPOCHS = 300
BATCH_SIZE = 8
LR = 0.001
PATIENCE = 20
