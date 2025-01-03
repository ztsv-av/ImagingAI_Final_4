import torch
from torch.utils.tensorboard import SummaryWriter

from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, Activations

from scripts.data.split import getSplits
from scripts.data.dataloader import createDataloader
from scripts.utils.vars import DEVICE
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
        metric = test(model=model, dataloader=testloader, metric_fn=metric_fn, metric_fn_batch=metric_fn_batch, writer=writer)
    else:
        metric = testEnsemble(models=models, dataloader=testloader, metric_fn=metric_fn, metric_fn_batch=metric_fn_batch, writer=writer)
    writer.close()
    print(f"Results for metric {metric_fn.name} on the test set: {metric}")

def test(model, dataloader, metric_fn, metric_fn_batch, writer):
    """
    Test the trained model on the test set.

    Parameters:
        - model: Trained UNet model.
        - dataloader: DataLoader for test data.
        - metric_fn: Metric function over batch and classes.
        - metric_fn_batch: Metric function over batch.
        - writer: TensorBoard SummaryWriter.

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
            step += 1
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
            # log predictions for visualization
            if step % 5 == 0:
                middle_depth = inputs.shape[-1] // 2
                writer.add_images("Test/Imags/Inputs", inputs[0:1, 0:1, :, :, middle_depth], step)
                writer.add_images("Test/Imags/Labels", labels[0:1, :, :, :, middle_depth], step)
                writer.add_images("Test/Imags/Predictions", outputs_post_pred[0:1, :, :, :, middle_depth], step)
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
    # start testing
    with torch.no_grad():
        for batch_data in dataloader:
            step += 1
            # retrieve data from the dataloader
            inputs = batch_data["modalities"].to(DEVICE)
            labels = batch_data["mask"].to(DEVICE).long()
            # ensemble forward pass (average prediction)
            summed_outputs = torch.zeros((inputs.shape[0], 3, *inputs.shape[2:]), device=DEVICE)
            for model in models:
                outputs = model(inputs)
                probabilities = post_activation(outputs)  # convert raw outputs to probabilities
                summed_outputs += probabilities
            ensemble_outputs = summed_outputs / len(models)
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
