from scripts.model.train import trainingPipeline, trainingPipelineKFold
from scripts.model.test import testingPipeline
from scripts.utils.helpers import setGlobalSeed

def run():
    # set python, random and pytorch seeds for reproducibility
    setGlobalSeed()
    # define modalities to load
    modalities = ["t1", "t2", "flair"] # this will load "t1", "t2", and "flair" modalities and segmentation mask later in createDataloader() function.
    available_models = ["UNet", "AttentionUNet", "UNETR", "SwinUNETR", "HighResNet"] # list of available models to train/test
    chosen_model = "UNet"
    kfold = False # not kfold training
    test = True # if to test or not
    if test:
        # start testing pipeline
        testingPipeline(modalities=modalities, model_name=chosen_model, ensemble=kfold)
    else:
        # start training pipeline
        if not kfold:
            trainingPipeline(modalities=modalities, model_name=chosen_model)
        else:
            trainingPipelineKFold(modalities=modalities, model_name=chosen_model)
            
if __name__ == "__main__":
    run()
