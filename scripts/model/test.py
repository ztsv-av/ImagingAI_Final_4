import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, Activations

from scripts.data.split import getSplits
from scripts.data.dataloader import createDataloader
from scripts.utils.vars import DEVICE, PREDICTIONS_DIR
from scripts.model.helpers import loadTrainedUNet, loadKFoldUNets

def testingPipeline(modalities: list, model_name: str, ensemble: bool = False):
    """
    Testing pipeline for the trained U-Net model with Dice Score metric.

    Parameters:
        - modalities (list): List of modalities to include as channels (e.g., ["t1", "t2", "flair"]).
        - model_name (str): Name of the model to load.
        - kfold (bool): Whether to use ensemble of models or not.
    """
    # retrieve testing patient IDs
    print("Loading testing patient IDs...")
    _, _, test_ids = getSplits()
    # define testing dataloader
    print("Defining testing DataLoader...")
    testloader = createDataloader(patient_ids=test_ids, modalities=modalities, train=False, model_name=model_name)
    # load trained model(s)
    if not ensemble:
        model = loadTrainedUNet(modalities=modalities, model_name=model_name)
    else:
        models = loadKFoldUNets(modalities=modalities, model_name=model_name)
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
    # TensorBoard writer
    writer = SummaryWriter()
    # calculate metric on the test set
    print("Testing...")
    if not ensemble:
        metric = test(model=model, dataloader=testloader, metric_fn=metric_fn, metric_fn_batch=metric_fn_batch, writer=writer, test_ids=test_ids)
    else:
        metric = testEnsemble(models=models, dataloader=testloader, metric_fn=metric_fn, metric_fn_batch=metric_fn_batch, writer=writer)
    writer.close()
    print(f"Results for metric {metric_fn.name} on the test set: {metric}")

def test(model, dataloader, metric_fn, metric_fn_batch, writer, test_ids):
    """
    Test the trained model on the test set.

    Parameters:
        - model: Trained UNet model.
        - dataloader: DataLoader for test data.
        - metric_fn: Metric function over batch and classes.
        - metric_fn_batch: Metric function over batch.
        - writer: TensorBoard SummaryWriter.
        - test_ids: IDs of patients.

    Returns:
        - Test metric score.
    """
    model.eval() # set model to evaluation mode
    step = 0
    # reset metrics at the beginning of validation
    metric_fn.reset()
    metric_fn_batch.reset()
    # post-processing transforms for predictions
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    # start testing
    with torch.no_grad():
        for batch_data in dataloader:
            # retrieve data from the dataloader
            inputs = batch_data["modalities"].to(DEVICE)
            labels = batch_data["mask"].to(DEVICE).long()
            # forward pass
            outputs = model(inputs)
            # apply post-processing to outputs (sigmoid, discrete with threshold)
            outputs_post_pred = post_pred(outputs)
            # metric score
            metric_fn(y_pred=outputs_post_pred, y=labels)
            metric_fn_batch(y_pred=outputs_post_pred, y=labels)
            # visualize prediction
            for idx in range(inputs.shape[0]):
                step += 1
                # extract the corresponding slices for modality, ground truth, and prediction
                slice_idx = 72
                modality_img = inputs.cpu().numpy()[idx, :, :, :, slice_idx]
                seg_img = labels.cpu().numpy()[idx, :, :, :, slice_idx]
                pred_seg_img = outputs_post_pred.cpu().numpy()[idx, :, :, :, slice_idx]
                # call the plotting function
                plotPredictions(
                    modality_img=modality_img,
                    seg_img=seg_img,
                    pred_seg_img=pred_seg_img,
                    slice_idx=slice_idx,
                    step=step,
                    test_ids=test_ids,
                    show_plot=False
                )
                # log predictions
                if step % 5 == 0:
                    writer.add_images("Test/Imags/Inputs", modality_img, step)
                    writer.add_images("Test/Imags/Labels", seg_img, step)
                    writer.add_images("Test/Imags/Predictions", pred_seg_img, step)
    # aggregate the metrics over all batches for the entire epoch
    epoch_metric = metric_fn.aggregate().item() # single scalar metric (mean over batch and classes)
    metric_batch = metric_fn_batch.aggregate() # vector of metrics per class
    metric_fn.reset()
    metric_fn_batch.reset()
    # extract per-class metrics from metric_batch
    metric_nc = metric_batch[0].item()
    metric_edema = metric_batch[1].item()
    metric_et = metric_batch[2].item()
    # write metric results to TensorBoard and return it
    writer.add_scalar("Test/Metric/TotalMetric", epoch_metric, global_step=step)
    writer.add_scalar("Test/Metric/MetricNC", metric_nc, global_step=step)
    writer.add_scalar("Test/Metric/MetricEDEMA", metric_edema, global_step=step)
    writer.add_scalar("Test/Metric/MetricET", metric_et, global_step=step)
    return epoch_metric

def testEnsemble(models, dataloader, metric_fn, metric_fn_batch, writer):
    """
    Test the ensemble of trained models on the test set.

    Parameters:
        - models: List of trained UNet models.
        - dataloader: DataLoader for test data.
        - metric_fn: Metric function over batch and classes.
        - metric_fn_batch: Metric function over batch.
        - writer: TensorBoard SummaryWriter.

    Returns:
        - Test metric score.
    """
    step = 0
    # reset metrics at the beginning of validation
    metric_fn.reset()
    metric_fn_batch.reset()
    # post-processing transforms for predictions
    post_activation = Activations(sigmoid=True)
    post_pred = AsDiscrete(threshold=0.5)
    weights = [0.1, 0.45, 0.45]
    # start testing
    with torch.no_grad():
        for batch_data in dataloader:
            step += 1
            # retrieve data from the dataloader
            inputs = batch_data["modalities"].to(DEVICE)
            labels = batch_data["mask"].to(DEVICE).long()
            # ensemble forward pass (average prediction)
            summed_outputs = torch.zeros((inputs.shape[0], 3, *inputs.shape[2:]), device=DEVICE)
            for i, model in enumerate(models):
                outputs = model(inputs)
                probabilities = post_activation(outputs)  # convert raw outputs to probabilities
                summed_outputs += probabilities*weights[i]
            ensemble_outputs = summed_outputs
            # apply post-processing to outputs (sigmoid, discrete with threshold)
            outputs_post_pred = post_pred(ensemble_outputs)
            # metric score
            metric_fn(y_pred=outputs_post_pred, y=labels)
            metric_fn_batch(y_pred=outputs_post_pred, y=labels)
            # log predictions for visualization
            if step % 5 == 0:
                middle_depth = inputs.shape[-1] // 2
                writer.add_images("Test/KFold/Images/Inputs", inputs[0:1, 0:1, :, :, middle_depth], step)
                writer.add_images("Test/KFold/Images/Labels", labels[0:1, :, :, :, middle_depth], step)
                writer.add_images("Test/KFold/Images/Predictions", outputs_post_pred[0:1, :, :, :, middle_depth], step)
    # aggregate the metrics over all batches for the entire epoch
    epoch_metric = metric_fn.aggregate().item() # single scalar metric (mean over batch and classes)
    metric_batch = metric_fn_batch.aggregate() # vector of metrics per class
    metric_fn.reset()
    metric_fn_batch.reset()
    # extract per-class metrics from metric_batch
    metric_nc = metric_batch[0].item()
    metric_edema = metric_batch[1].item()
    metric_et = metric_batch[2].item()
    # write metric results to TensorBoard and return it
    writer.add_scalar("Test/KFold/Metric/TotalMetric", epoch_metric, global_step=step)
    writer.add_scalar("Test/KFold/Metric/MetricNC", metric_nc, global_step=step)
    writer.add_scalar("Test/KFold/Metric/MetricEDEMA", metric_edema, global_step=step)
    writer.add_scalar("Test/KFold/Metric/MetricET", metric_et, global_step=step)
    return epoch_metric

def plotPredictions(
    modality_img: np.array, 
    seg_img: np.array, 
    pred_seg_img: np.array, 
    slice_idx: int, 
    step: int, 
    test_ids: list[str],
    show_plot: bool = False
):
    """
    Plot a 3x2 figure:
        - 3 columns for the 3 classes (Necrotic Core, Peritumoral Edematous, Enhancing Tumor).
        - 2 rows: top row for Ground Truth, bottom row for Predictions.
    
    Each subplot shows the T1 modality with an overlay of the respective class mask.

    Parameters:
        - modality_img (np.array): Modality image of shape (channels, height, width).
                                   Channel 0 is T1.
        - seg_img (np.array): Ground truth segmentation mask of shape (3, height, width).
        - pred_seg_img (np.array): Predicted segmentation mask of shape (3, height, width).
        - slice_idx (int): Slice index used for the title.
        - step (int): Step number (or image ID in the test set).
        - test_ids (list[str]): Patient IDs.
        - show_plot (bool): Whether to display the plot (True) or save it (False).
    """

    # define colors (with RGBA) for the classes
    # channel -> color
    class_colors = {
        0: (0.0, 0.835, 1.0, 0.8), # necrotic core: #00d5ff
        1: (1.0, 0.9, 0.0, 0.8), # peritumoral edematous: #ffe600
        2: (0.5, 0.0, 0.0, 0.8), # enhancing tumor: #800000
    }
    # channel -> class label
    class_labels = {
        0: "Necrotic Core",
        1: "Peritumoral Edematous",
        2: "Enhancing Tumor"
    }
    def overlay_class(class_mask, color_rgba):
        """
        Create an RGBA overlay for one class.
        Parameters"
            - class_mask(np.array): 2D array (height, width) with 1/0 or True/False.
            - color_rgba(np.array): Tuple of (r, g, b, alpha).
        Returns:
            - overlay (np.array): RGBA image of shape (height, width, 4).
        """
        height, width = class_mask.shape
        overlay = np.zeros((height, width, 4), dtype=np.float32)
        # where the mask is > 0, set the given color
        overlay[class_mask > 0] = color_rgba
        return overlay
    # extract the T1 modality (channel 0)
    t1_modality = modality_img[0]  # shape: (height, width)
    # create a 3x2 figure: 2 rows (ground truth, prediction), 3 columns (each class)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(f"Patient ID: {test_ids[step]}, Slice Index: {slice_idx}", fontsize=20)
    # loop through each class (channel)
    for col, class_idx in enumerate([0, 1, 2]):
        # ground truth row
        axes[0, col].imshow(t1_modality, cmap="gray")
        gt_overlay = overlay_class(seg_img[class_idx], class_colors[class_idx])
        axes[0, col].imshow(gt_overlay, alpha=1.0)  # color_rgba already has alpha
        axes[0, col].set_title(f"GT: {class_labels[class_idx]}", fontsize=16)
        axes[0, col].axis("off")
        # prediction row
        axes[1, col].imshow(t1_modality, cmap="gray")
        pred_overlay = overlay_class(pred_seg_img[class_idx], class_colors[class_idx])
        axes[1, col].imshow(pred_overlay, alpha=1.0)  # color_rgba already has alpha
        axes[1, col].set_title(f"Pred: {class_labels[class_idx]}", fontsize=16)
        axes[1, col].axis("off")
    # create legend patches for the three classes
    patches = []
    for i in range(3):
        # the color in class_colors[i] is (r, g, b, alpha)
        # to display in a legend patch, we can use only the rgb part
        rgb = class_colors[i][:3]
        label = class_labels[i]
        patch = mpatches.Patch(color=rgb, label=label)
        patches.append(patch)
    # add the legend to the bottom of the figure
    fig.legend(
        handles=patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        fontsize=14
    )
    # adjust layout so legend fits
    fig.tight_layout(rect=[0, 0.05, 1, 1])  # leave space at bottom for legend
    # show or save the figure
    if show_plot:
        plt.show()
    else:
        plt.savefig(f"{PREDICTIONS_DIR}/{test_ids[step]}_t1_{slice_idx}.png", bbox_inches="tight")
        plt.close(fig)
