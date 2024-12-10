import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete

from scripts.data.split import getSplits
from scripts.data.dataloader import createDataloader
from scripts.utils.vars import DEVICE, LR, EPOCHS, PATIENCE, BEST_MODEL_PATH

def trainingPipeline(modalities: list) -> None:
    """
    Training pipeline for the U-Net model using MONAI.

    Parameters:
        - modalities (list): List of modalities to include as channels (e.g., ["t1", "t2", "flair"]).
    """
    # retrieve training and validation patient IDs
    print("Retrieving training and validation patient IDs...")
    train_ids, val_ids, _ = getSplits()
    # define training and validation dataloaders
    print("Defining training DataLoader...")
    trainloader = createDataloader(patient_ids=train_ids, modalities=modalities, train=True)
    print("Defining validation DataLoader...")
    valloader = createDataloader(patient_ids=val_ids, modalities=modalities, train=False)
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
    # early stopping parameters
    best_metric = 2 # dice loss is between 0 and 1
    best_metric_epoch = -1
    epochs_no_improve = 0
    # TensorBoard writer
    writer = SummaryWriter()
    # training
    best_model_path_modalities = BEST_MODEL_PATH.replace(".pth", "_".join(modalities) + ".pth")
    print("Starting training...")
    for epoch in range(EPOCHS):
        print(f"  Epoch {epoch + 1}/{EPOCHS}")
        # train
        print("  Training...")
        train_loss  = train(
            model, trainloader, 
            loss_fn, optimizer, 
            epoch, writer
        )
        # validate
        print("  Validation...")
        val_loss, val_dice = validate(
            model, valloader, post_pred, 
            loss_fn, metric_fn, metric_fn_batch, 
            epoch, writer
        )
        # print results
        print(f"  Epoch {epoch + 1} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Val Metric: {val_dice:.4f}")
        # learning rate scheduler step
        scheduler.step(val_loss)
        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
            writer.add_scalar("Train/LearningRate", lr, epoch)
            break
        # early stopping
        if val_loss < best_metric:
            best_metric = val_loss
            best_metric_epoch = epoch + 1
            epochs_no_improve = 0
            # save the best model
            torch.save(model.state_dict(), best_model_path_modalities)
            print(f"  Saved new best model at epoch {epoch + 1}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epochs")
        if epochs_no_improve >= PATIENCE:
            print("  Early stopping triggered")
            break
    print(
        f"Training completed. Best validation {metric_fn.name}: {best_metric:.4f} at epoch {best_metric_epoch}"
    )
    writer.close()

def train(model, dataloader, loss_fn, optimizer, epoch, writer):
    """
    Training loop for an epoch.

    Parameters:
        - model: UNet model.
        - dataloader: DataLoader for training data.
        - loss_fn: Loss function.
        - optimizer: Optimizer.
        - epoch: Current epoch.
        - writer: TensorBoard SummaryWriter.

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
            print(f"    Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.4f}")
            writer.add_scalar("Train/BatchLoss", loss.item(), epoch * len(dataloader) + step)
    # compute average loss for the epoch
    epoch_loss /= step
    # write the results
    writer.add_scalar("Train/EpochLoss", epoch_loss, epoch)
    return epoch_loss

def validate(model, dataloader, post_pred, loss_fn, metric_fn, metric_fn_batch, epoch, writer):
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
    writer.add_scalar("Validation/Loss/TotalLoss", val_loss, epoch)
    writer.add_scalar("Validation/Metric/TotalMetric", epoch_metric, epoch)
    writer.add_scalar("Validation/Metric/MetricNC", metric_nc, epoch)
    writer.add_scalar("Validation/Metric/MetricEDEMA", metric_edema, epoch)
    writer.add_scalar("Validation/Metric/MetricET", metric_et, epoch)
    return val_loss, epoch_metric
