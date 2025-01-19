import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete

from sklearn.model_selection import KFold

from scripts.data.split import getSplits
from scripts.data.dataloader import createDataloader
from scripts.model.helpers import getBestModelPath, defineMONAIModel, loadCheckpoint, saveCheckpoint
from scripts.utils.vars import DEVICE, LR, EPOCHS, PATIENCE, KFOLD_SPLITS, SEED

def trainingPipeline(modalities: list, model_name: str) -> None:
    """
    Training pipeline for the U-Net model using MONAI.

    Parameters:
        - modalities (list): List of modalities to include as channels (e.g., ["t1", "t2", "flair"]).
        - model_name (str): Name of the model to train.
    """
    # retrieve training and validation patient IDs
    print("Retrieving training and validation patient IDs...")
    train_ids, val_ids, _ = getSplits()
    # define training and validation dataloaders
    print("Defining training DataLoader...")
    trainloader = createDataloader(patient_ids=train_ids, modalities=modalities, train=True, model_name=model_name)
    print("Defining validation DataLoader...")
    valloader = createDataloader(patient_ids=val_ids, modalities=modalities, train=False, model_name=model_name)
    # initialize MONAI model
    model = defineMONAIModel(modalities=modalities, model_name=model_name)
    model.name = model_name
    # loss function
    loss_fn = DiceLoss(
        to_onehot_y=False,
        sigmoid=True, # raw outputs -> sigmoid
        include_background=True,
        smooth_nr=0,
        smooth_dr=1e-5, # avoid division by zero
        squared_pred=False # square predictions
    )
    # metric functions
    metric_fn = DiceMetric(
        include_background=True,
        reduction="mean", # reduction over classes and batch
        get_not_nans=False # ignore NaNs, recommended
    )
    # used extract per-class metrics from metric_batch
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
        T_0=20, # number of epochs for the first cycle
        T_mult=2, # factor by which the cycle length increases after each restart
        eta_min=1e-6 # minimum learning rate
    )
    # load checkpoint if exists
    checkpoint = loadCheckpoint(model, optimizer, scheduler, modalities)
    if checkpoint:
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint["best_metric"]
        best_metric_epoch = checkpoint["best_metric_epoch"]
        epochs_no_improve = checkpoint["epochs_no_improve"]
    else:
        start_epoch = 0
        best_metric = float("-inf")
        best_metric_epoch = -1
        epochs_no_improve = 0
    # TensorBoard writer will continue itself if checkpoint exists
    writer = SummaryWriter()
    # path to save the best model
    best_model_path_modalities = getBestModelPath(modalities=modalities, model_name=model_name)
    # start training
    print("Starting training...")
    for epoch in range(start_epoch, EPOCHS):
        print(f"  Epoch {epoch + 1}/{EPOCHS}")
        # train
        print("  Training...")
        train_loss  = train(
            model, trainloader, 
            loss_fn, optimizer, 
            epoch, writer, None
        )
        # step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar("Train/LearningRate", current_lr, epoch)
        # validate
        print("  Validation...")
        val_loss, val_dice = validate(
            model, valloader, post_pred, 
            loss_fn, metric_fn, metric_fn_batch, 
            epoch, writer, None
        )
        # print results
        print(f"  Epoch {epoch + 1} Train Loss: {train_loss:.4f} Learning Rate: {current_lr:.4f} Val Loss: {val_loss:.4f} Val Metric: {val_dice:.4f}")
        # check if model improved
        if val_dice > best_metric:
            best_metric = val_dice
            best_metric_epoch = epoch + 1
            epochs_no_improve = 0
            # save best model
            torch.save(model.state_dict(), best_model_path_modalities)
            print(f"  Saved new best model at epoch {epoch + 1}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epochs")
        # save checkpoint
        saveCheckpoint(
            model, optimizer, scheduler, 
            epoch, best_metric, best_metric_epoch, 
            epochs_no_improve, modalities
        )
        # early stopping
        if epochs_no_improve >= PATIENCE:
            print("  Early stopping triggered")
            break
    print(
        f"Training completed. Best validation {metric_fn.name}: {best_metric:.4f} at epoch {best_metric_epoch}"
    )
    writer.close()

def trainingPipelineKFold(modalities: list, model_name: str, fold_idx: int) -> None:
    """
    Training pipeline for the U-Net model using MONAI.

    Parameters:
        - modalities (list): List of modalities to include as channels (e.g., ["t1", "t2", "flair"]).
        - model_name (str): Name of the model to train.
        - fold_idx (int): Index of the fold to train the model on (0-indexed).
    """
    # retrieve training and validation patient IDs
    print("Retrieving training and validation patient IDs...")
    train_ids, val_ids, _ = getSplits()
    all_ids = train_ids + val_ids
    # set up K-Fold
    kf = KFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=SEED)
    # TensorBoard writer
    writer = SummaryWriter()
    # kfold training loop
    print(f"Begin KFold training...")
    for current_fold_idx, (train_index, val_index) in enumerate(kf.split(all_ids)):
        if current_fold_idx != fold_idx:
            continue  # skip all folds except the one specified by fold_idx
        print(f"  Starting fold {current_fold_idx+1}/{KFOLD_SPLITS}...")
        # get the training and validation IDs for this fold
        fold_train_ids = [all_ids[i] for i in train_index]
        fold_val_ids = [all_ids[i] for i in val_index]
        # define training and validation dataloaders
        print("  Defining training DataLoader...")
        trainloader = createDataloader(patient_ids=fold_train_ids, modalities=modalities, train=True, model_name=model_name)
        print("  Defining validation DataLoader...")
        valloader = createDataloader(patient_ids=fold_val_ids, modalities=modalities, train=False, model_name=model_name)
        # initialize MONAI model
        model = defineMONAIModel(modalities=modalities, model_name=model_name)
        model.name = model_name
        # loss function
        loss_fn = DiceLoss(
            to_onehot_y=False,
            sigmoid=True, # raw outputs -> sigmoid
            include_background=True,
            smooth_nr=0,
            smooth_dr=1e-5, # denominator smoothing to avoid division by zero
            squared_pred=True # square predictions
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
            T_0=20, # number of epochs for the first cycle
            T_mult=2, # factor by which the cycle length increases after each restart
            eta_min=1e-6 # minimum learning rate
        )
        # load checkpoint if exists
        checkpoint = loadCheckpoint(model, optimizer, scheduler, modalities + [f"fold_{current_fold_idx + 1}"])
        if checkpoint:
            start_epoch = checkpoint["epoch"] + 1
            best_metric = checkpoint["best_metric"]
            best_metric_epoch = checkpoint["best_metric_epoch"]
            epochs_no_improve = checkpoint["epochs_no_improve"]
        else:
            start_epoch = 0
            best_metric = float("-inf")
            best_metric_epoch = -1
            epochs_no_improve = 0
        # path to save the current best model
        best_model_path_modalities = getBestModelPath(modalities=modalities, model_name=model_name, kfold_idx=current_fold_idx)
        # training for the current fold
        print("  Starting training...")
        for epoch in range(start_epoch, EPOCHS):
            print(f"    Fold {current_fold_idx+1}, Epoch {epoch + 1}/{EPOCHS}")
            # train
            print("    Training...")
            train_loss  = train(
                model, trainloader, 
                loss_fn, optimizer, 
                epoch, writer, current_fold_idx
            )
            # step the scheduler
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar(f"Train/Fold_{current_fold_idx+1}/LearningRate", current_lr, epoch)
            # validate
            print("    Validation...")
            val_loss, val_dice = validate(
                model, valloader, post_pred, 
                loss_fn, metric_fn, metric_fn_batch, 
                epoch, writer, current_fold_idx
            )
            # print results
            print(f"    Fold {current_fold_idx+1}, Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Learning Rate:{current_lr:.4f}, Val Loss: {val_loss:.4f}, Val {metric_fn.name}: {val_dice:.4f}")
            # check if model improved
            if val_dice > best_metric:
                best_metric = val_dice
                best_metric_epoch = epoch + 1
                epochs_no_improve = 0
                # save the best model
                torch.save(model.state_dict(), best_model_path_modalities)
                print(f"    Saved new best model for fold {current_fold_idx+1} at epoch {epoch + 1}")
            else:
                epochs_no_improve += 1
                print(f"    No improvement for {epochs_no_improve} epochs on fold {current_fold_idx+1}")
            # save checkpoint
            saveCheckpoint(
                model, optimizer, scheduler, 
                epoch, best_metric, best_metric_epoch,
                epochs_no_improve, modalities + [f"fold_{current_fold_idx + 1}"]
            )
            # early stopping
            if epochs_no_improve >= PATIENCE:
                print(f"    Early stopping triggered on fold {current_fold_idx+1}")
                break
        print(f"  Fold {current_fold_idx+1} completed. Best validation {metric_fn.name}: {best_metric:.4f} at epoch {best_metric_epoch}")
    print(f"All folds completed.")
    writer.close()

def train(model, dataloader, loss_fn, optimizer, epoch, writer, fold_idx: int = None):
    """
    Training loop for an epoch.

    Parameters:
        - model: UNet model.
        - dataloader: DataLoader for training data.
        - loss_fn: Loss function.
        - optimizer: Optimizer.
        - epoch: Current epoch.
        - writer: TensorBoard SummaryWriter.
        - fold_idx (int): Fold index. Default is None, meaning not KFold training strategy.

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
            if fold_idx == None:
                print(f"    Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.4f}")
                writer.add_scalar("Train/BatchLoss", loss.item(), epoch * len(dataloader) + step)
            else:
                print(f"      Fold_{fold_idx+1}, Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.4f}")
                writer.add_scalar(f"Train/Fold_{fold_idx+1}/BatchLoss", loss.item(), epoch * len(dataloader) + step)
    # compute average loss for the epoch
    epoch_loss /= step
    # write the results
    if fold_idx == None:
        writer.add_scalar("Train/EpochLoss", epoch_loss, epoch)
    else:
        writer.add_scalar(f"Train/Fold_{fold_idx+1}/EpochLoss", epoch_loss, epoch)
    return epoch_loss

def validate(model, dataloader, post_pred, loss_fn, metric_fn, metric_fn_batch, epoch, writer, fold_idx: int = None):
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
        - fold_idx: Fold index. Default is None, meaning not KFold training strategy.

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
    if fold_idx == None:
        writer.add_scalar("Validation/Loss/TotalLoss", val_loss, epoch)
        writer.add_scalar("Validation/Metric/TotalMetric", epoch_metric, epoch)
        writer.add_scalar("Validation/Metric/MetricNC", metric_nc, epoch)
        writer.add_scalar("Validation/Metric/MetricEDEMA", metric_edema, epoch)
        writer.add_scalar("Validation/Metric/MetricET", metric_et, epoch)
    else:
        writer.add_scalar(f"Validation/Fold_{fold_idx+1}/Loss/TotalLoss", val_loss, epoch)
        writer.add_scalar(f"Validation/Fold_{fold_idx+1}/Metric/TotalMetric", epoch_metric, epoch)
        writer.add_scalar(f"Validation/Fold_{fold_idx+1}/Metric/MetricNC", metric_nc, epoch)
        writer.add_scalar(f"Validation/Fold_{fold_idx+1}/Metric/MetricEDEMA", metric_edema, epoch)
        writer.add_scalar(f"Validation/Fold_{fold_idx+1}/Metric/MetricET", metric_et, epoch)
    return val_loss, epoch_metric
