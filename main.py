from scripts.model.train import trainingPipeline
from scripts.model.test import testingPipeline
from scripts.utils.utils import setGlobalSeed

def run():
    # set python, random and pytorch seeds for reproducibility
    setGlobalSeed()
    # define modalities to load
    modalities = ["t1"] # this will load "t1", "t2", and "flair" modalities and segmentation mask later in createDataloader() function.
    # start training pipeline
    trainingPipeline(modalities=modalities)
    # start testing pipeline
    testingPipeline(modalities=modalities, kfold=False)

if __name__ == "__main__":
    run()
