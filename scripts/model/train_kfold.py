from sklearn.model_selection import KFold

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete

from scripts.data.split import getSplits
from scripts.data.dataloader import createDataloader
from scripts.utils.utils import writeJSON
from scripts.utils.vars import DEVICE, KFOLD_SPLITS, KFOLD_RESULTS_PATH, EPOCHS, LR, PATIENCE, BEST_MODEL_PATH, SEED

def trainingPipelineKFold(modalities: list) -> None:
    """
    Training pipeline for the U-Net model using MONAI.

    Parameters:
        - modalities (list): List of modalities to include as channels (e.g., ["t1", "t2", "flair"]).
        - n_splits (int): Number of splits for the KFold.
    """
    # retrieve training and validation patient IDs
    print("Retrieving training and validation patient IDs...")
    train_ids, val_ids, _ = getSplits()
    all_ids = train_ids + val_ids
    # set up K-Fold
    kf = KFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=SEED)
    # TensorBoard writer
    writer = SummaryWriter()
    # kfold metrics
    best_fold_metric = 2.0
    best_fold_epoch = -1
    best_fold_idx = -1
    # kfold training loop
    print(f"Begin KFold training...")
    folds_results = {}
    for fold_idx, (train_index, val_index) in enumerate(kf.split(all_ids)):
        print(f"  Starting fold {fold_idx+1}/{KFOLD_SPLITS}...")
        # get the training and validation IDs for this fold
        fold_train_ids = [all_ids[i] for i in train_index]
        fold_val_ids = [all_ids[i] for i in val_index]
        # define training and validation dataloaders
        print("  Defining training DataLoader...")
        trainloader = createDataloader(patient_ids=fold_train_ids, modalities=modalities, train=True)
        print("  Defining validation DataLoader...")
        valloader = createDataloader(patient_ids=fold_val_ids, modalities=modalities, train=False)
        # initialize UNet from MONAI
        model = UNet(
            spatial_dims=3,
            in_channels=len(modalities),
            out_channels=3, # 3 segmentation classes, without background
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            dropout=0.2,
            num_res_units=2,
            norm="batch",
        ).to(DEVICE)
        # loss function
        loss_fn = DiceLoss(
            to_onehot_y=False,
            sigmoid=True, # raw outputs -> sigmoid
            include_background=True,
            reduction="mean", # reduction over the batch
            smooth_nr=1e-5, # numerator smoothing to avoid division by zero
            smooth_dr=1e-5, # denominator smoothing to avoid division by zero
            squared_pred=False # do not square predictions
        )
        # metric functions
        metric_fn = DiceMetric(
            include_background=True,
            reduction="mean", # reduction over classes and batch
            get_not_nans=False # ignore NaNs, recommended
        )
        metric_fn_batch = DiceMetric(
            include_background=True,
            reduction="mean_batch", # reduction over batch
            get_not_nans=False # ignore NaNs, recommended
        )
        metric_fn.name = "Dice Score"
        metric_fn_batch.name = "Dice Score Batch"
        # post-processing transforms for predictions and labels
        post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        # optimizer
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        # learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10, # number of epochs for the first cycle
            T_mult=1, # factor by which the cycle length increases after each restart
            eta_min=1e-6 # minimum learning rate
        )
        # early stopping parameters for this fold
        best_metric = 2.0
        best_metric_epoch = -1
        epochs_no_improve = 0
        # training for the current fold
        print("  Starting training...")
        best_model_path_modalities = BEST_MODEL_PATH.replace(".pth", "_".join(modalities) + ".pth")
        for epoch in range(EPOCHS):
            print(f"    Fold {fold_idx+1}, Epoch {epoch + 1}/{EPOCHS}")
            # train
            print("    Training...")
            train_loss  = train(
                model, trainloader, 
                loss_fn, optimizer, 
                epoch, writer, fold_idx
            )
            # validate
            print("    Validation...")
            val_loss, val_dice = validate(
                model, valloader, post_pred, 
                loss_fn, metric_fn, metric_fn_batch, 
                epoch, writer, fold_idx
            )
            # print results
            print(f"    Fold {fold_idx+1}, Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val {metric_fn.name}: {val_dice:.4f}")
            # learning rate scheduler step
            scheduler.step(val_loss)
            for param_group in optimizer.param_groups:
                lr = param_group["lr"]
                writer.add_scalar(f"Fold_{fold_idx+1}/Train/LearningRate", lr, epoch)
                break
            # early stopping
            if val_loss < best_metric:
                best_metric = val_loss
                best_metric_epoch = epoch + 1
                epochs_no_improve = 0
                # save the best model
                torch.save(model.state_dict(), best_model_path_modalities.replace(".pth", f"_fold-{fold_idx+1}.pth"))
                print(f"    Saved new best model for fold {fold_idx+1} at epoch {epoch + 1}")
            else:
                epochs_no_improve += 1
                print(f"    No improvement for {epochs_no_improve} epochs on fold {fold_idx+1}")
            if epochs_no_improve >= PATIENCE:
                print(f"    Early stopping triggered on fold {fold_idx+1}")
                break
        print(f"  Fold {fold_idx+1} completed. Best validation {metric_fn.name}: {best_metric:.4f} at epoch {best_metric_epoch}")
        folds_results[fold_idx+1] = {
            "best_metric": best_metric,
            "best_epoch": best_metric_epoch
        }
        # track if this fold is better than previous folds
        if best_metric < best_fold_metric:
            best_fold_metric = best_metric
            best_fold_epoch = best_metric_epoch
            best_fold_idx = fold_idx + 1
    kfold_results = {
        "best_fold_idx": best_fold_idx,
        "best_fold_metric": best_fold_metric,
        "folds_results": folds_results
    }
    writeJSON(kfold_results, KFOLD_RESULTS_PATH)
    print(f"All folds completed. Best fold: {best_fold_idx} with {metric_fn.name}: {best_fold_metric:.4f} at epoch {best_fold_epoch}")
    writer.close()

def train(model, dataloader, loss_fn, optimizer, epoch, writer, fold_idx):
    """
    Training loop for an epoch.

    Parameters:
        - model: UNet model.
        - dataloader: DataLoader for training data.
        - loss_fn: Loss function.
        - optimizer: Optimizer.
        - epoch: Current epoch.
        - writer: TensorBoard SummaryWriter.
        - fold_idx: Fold index.

    Returns:
        - Average training loss for the epoch.
    """
    # set model to train state (for proper gradients calculation)
    model.train()
    epoch_loss = 0
    step = 0
    # train
    for batch_data in dataloader:
        step += 1
        # retrieve data from dataloader for current batch
        inputs = batch_data["modalities"].to(DEVICE)
        labels = batch_data["mask"].to(DEVICE).long()
        # forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        # backward pass
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        # accumulate loss
        epoch_loss += loss.item()
        # write results
        if step % 5 == 0:
            print(f"      Fold_{fold_idx+1}, Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.4f}")
            writer.add_scalar(f"Fold_{fold_idx+1}/Train/BatchLoss", loss.item(), epoch * len(dataloader) + step)
    # compute average loss for the epoch
    epoch_loss /= step
    # write the results
    writer.add_scalar(f"Fold_{fold_idx+1}/Train/EpochLoss", epoch_loss, epoch)
    return epoch_loss

def validate(model, dataloader, post_pred, loss_fn, metric_fn, metric_fn_batch, epoch, writer, fold_idx):
    """
    Validation.

    Parameters:
        - model: UNet model.
        - dataloader: DataLoader for validation data.
        - loss_fn: Loss function.
        - metric_fn: Metric function over batch and classes.
        - metric_fn_batch: Metric function over batch.
        - epoch: Current epoch.
        - writer: TensorBoard SummaryWriter.
        - fold_idx: Fold index.

    Returns:
        - Average validation loss.
        - Average validation metric score.
    """
    model.eval() # set model to evaluation mode
    val_loss = 0
    step = 0
    # reset metrics at the beginning of validation
    metric_fn.reset()
    metric_fn_batch.reset()
    # validate
    with torch.no_grad():
        for batch_data in dataloader:
            step += 1
            # retrieve data from the dataloader
            inputs = batch_data["modalities"].to(DEVICE)
            labels = batch_data["mask"].to(DEVICE).long()
            # forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            # apply post-processing to outputs (sigmoid, discrete with threshold)
            outputs_post_pred = post_pred(outputs)
            # metric score
            metric_fn(y_pred=outputs_post_pred, y=labels)
            metric_fn_batch(y_pred=outputs_post_pred, y=labels)
    # total loss after processing all batches
    val_loss /= step
    # aggregate the metrics over all batches for the entire epoch
    epoch_metric = metric_fn.aggregate().item() # single scalar metric (mean over batch and classes)
    metric_batch = metric_fn_batch.aggregate() # vector of metrics per class
    metric_fn.reset()
    metric_fn_batch.reset()
    # extract per-class metrics from metric_batch
    metric_nc = metric_batch[0].item()
    metric_edema = metric_batch[1].item()
    metric_et = metric_batch[2].item()
    # log results
    writer.add_scalar(f"Fold_{fold_idx+1}/Validation/Loss/TotalLoss", val_loss, epoch)
    writer.add_scalar(f"Fold_{fold_idx+1}/Validation/Metric/TotalMetric", epoch_metric, epoch)
    writer.add_scalar(f"Fold_{fold_idx+1}/Validation/Metric/MetricNC", metric_nc, epoch)
    writer.add_scalar(f"Fold_{fold_idx+1}/Validation/Metric/MetricEDEMA", metric_edema, epoch)
    writer.add_scalar(f"Fold_{fold_idx+1}/Validation/Metric/MetricET", metric_et, epoch)
    return val_loss, epoch_metric
