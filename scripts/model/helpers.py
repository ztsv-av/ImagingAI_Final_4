import os
import re

import torch

from monai.networks.nets import UNet # default
from monai.networks.nets import UNETR # unet with transformer encoder
from monai.networks.nets import SwinUNETR # UNet with Swin Transformer as encoder
from monai.networks.nets import HighResNet # suitable for small object segmentation, retains high spatial resolution throughout the network

from scripts.utils.vars import DEVICE, BEST_MODEL_PATH, CHECKPOINTS_PATH, KFOLD_SPLITS

def getBestModelPath(modalities: list, model_name: str, kfold_idx=None):
    """
    Defines the path to the best trained model.
    Used to retrieve or upload the model on this path.

    Parameters:
        - modalities (list): List of modalities to include as channels (e.g., ["t1", "t2", "flair"]).
        - model_name (str): Name of the model to load.
        - kfold_idx (int): Index of the fold. None if no KFold is/was used for training.

    Returns:
        - model_path (str): Path to the best model
    """
    if kfold_idx == None:
        model_path = BEST_MODEL_PATH.replace(".pth", f"_{model_name}_" + "_".join(modalities) + ".pth")
    else:
        model_path = BEST_MODEL_PATH.replace(".pth", f"_{model_name}_" + "_".join(modalities) + f"_fold-{kfold_idx}.pth")
    return model_path


def defineMONAIModel(modalities: list, model_name: str):
    """
    Stores a dictionary of available for training models.
    Returns a model class based on the model_name.

    Parameters:
        - modalities (list): List of modalities to include as channels (e.g., ["t1", "t2", "flair"]).
        - model_name (str): Name of the model to load.

    Returns:
        - (monai.networks.nets): MONAI model class.
    """
    models = {
        "UNet": lambda: UNet(
            spatial_dims=3,
            in_channels=len(modalities),
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            dropout=0.2,
            num_res_units=2,
            norm="batch",
        ).to(DEVICE),
        "UNETR": lambda: UNETR(
            spatial_dims=3,
            in_channels=len(modalities),
            out_channels=3,
            img_size=(244, 244, 144),
            dropout_rate=0.2,
            feature_size=6,
            hidden_size=128,
            mlp_dim=512,
            num_heads=4,
        ).to(DEVICE),
        "SwinUNETR": lambda: SwinUNETR(
            img_size=(256, 256, 160),
            in_channels=len(modalities),
            out_channels=3,
            drop_rate=0.2,
            feature_size=12,
            depth=(2, 2, 2, 2),
            num_heads=(2, 4, 8, 16),
        ).to(DEVICE),
        "HighResNet": lambda: HighResNet(
            spatial_dims=3,
            in_channels=len(modalities),
            out_channels=3,
            dropout_prob=0.2,
        ).to(DEVICE),
    }
    # check if the model name is valid and return the model
    if model_name not in models:
        raise ValueError(f"Invalid model name: {model_name}. Choose from {list(models.keys())}")
    return models[model_name]()

def saveCheckpoint(
        model, optimizer, scheduler, 
        epoch, best_metric, best_metric_epoch, 
        epochs_no_improve, modalities):
    """
    Saves the training state to a checkpoint.

    Parameters:
        - model: The model being trained.
        - optimizer: The optimizer used for training, containing the current state.
        - scheduler: The learning rate scheduler used during training, containing the current state.
        - epoch (int): The current epoch number.
        - best_metric (float): The best validation metric achieved so far.
        - best_metric_epoch (int): The epoch at which the best validation metric was achieved.
        - epochs_no_improve (int): The number of consecutive epochs without improvement in the validation metric.
        - modalities (list[str]): A list of modalities used as input channels for the model (e.g., ["t1", "t2", "flair"]).
    """
    checkpoint_path = os.path.join(
        CHECKPOINTS_PATH, f"checkpoint_{model.name}_{'_'.join(modalities)}_epoch_{epoch + 1}.pth"
    )
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "best_metric_epoch": best_metric_epoch,
        "epochs_no_improve": epochs_no_improve,
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def loadCheckpoint(model, optimizer, scheduler, modalities):
    """
    Loads the training state from a checkpoint.

    Parameters:
        - model: The model to load state into.
        - optimizer: The optimizer to load state into.
        - scheduler: The learning rate scheduler to load state into.
        - modalities (list[str]): A list of modalities used as input channels for the model (e.g., ["t1", "t2", "flair"]).
    """
    os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
    # extract all checkpoint filenames
    all_checkpoints = [f for f in os.listdir(CHECKPOINTS_PATH) if f.endswith(".pth")]
    # ensure exact matching for modalities
    modality_pattern = f"^checkpoint_{model.name}_{'_'.join(modalities)}_epoch_(\\d+)\\.pth$"
    # sort checkpoints numerically based on the epoch number
    checkpoints = sorted(
        [f for f in all_checkpoints if re.match(modality_pattern, f)],
        key=lambda x: int(re.search(r"epoch_(\d+)", x).group(1))
    )
    # start from epoch 0 if checkpoint does not exist for the given modalities
    if not checkpoints:
        print("No checkpoints found, starting fresh training.")
        return None
    # resume training from the latest checkpoint
    checkpoint_path = os.path.join(CHECKPOINTS_PATH, checkpoints[-1])
    print(f"Resuming from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(DEVICE), weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint

def loadTrainedUNet(modalities: list, model_name: str):
    """
    Load the best model weights into the model.

    Parameters:
        - modalities (list): List of modalities to include as channels (e.g., ["t1", "t2", "flair"]).
        - model_name (str): Name of the model to load.
    """
    best_model_path_modalities = getBestModelPath(modalities=modalities, model_name=model_name)
    # define model architecture
    model = defineMONAIModel(modalities=modalities, model_name=model_name)
    # load the state_dict from the file
    state_dict = torch.load(best_model_path_modalities, map_location=torch.device(DEVICE), weights_only=True)
    # load the state_dict into the model
    model.load_state_dict(state_dict)
    print(f"Model weights loaded successfully from {best_model_path_modalities}")
    return model

def loadKFoldUNets(modalities: list, model_name: str):
    """
    Loads KFOLD_SPLITS UNet models trained on different folds.

    Parameters:
        - modalities (list): List of modalities to include as channels (e.g., ["t1", "t2", "flair"]).
        - model_name (str): Name of the model to load.
    
    Returns:
        - models (list): List of KFOLD_SPLITS UNet models trained on different folds.
    """
    models = []
    for fold_idx in range(KFOLD_SPLITS):
        model_path = getBestModelPath(modalities=modalities, model_name=model_name, kfold_idx=fold_idx)
        model = defineMONAIModel(modalities=modalities, model_name=model_name)
        state_dict = torch.load(model_path, map_location=torch.device(DEVICE), weights_only=True)
        model.load_state_dict(state_dict) # load weights into the model
        model.eval()  # set the model to evaluation mode
        models.append(model)
        print(f"Model weights loaded successfully from {model_path}")
    return models
