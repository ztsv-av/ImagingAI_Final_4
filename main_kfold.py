from scripts.model.train_kfold import trainingPipelineKFold
from scripts.model.test import testingPipeline
from scripts.utils.utils import setGlobalSeed

def run():
    # set python, random and pytorch seeds for reproducibility
    setGlobalSeed()
    # define modalities to load
    modalities = ["t1"] # this will load "t1"modality and segmentation mask later in createDataloader() function.
    # start training pipeline
    trainingPipelineKFold(modalities=modalities)
    # start testing pipeline
    testingPipeline(modalities=modalities, kfold=True)

if __name__ == "__main__":
    run()
