import os
import random
from ast import literal_eval

from scripts.utils.vars import (
    DATA_PATH, TRAIN_IDS_PATH, VAL_IDS_PATH, TEST_IDS_PATH,
    TRAIN_PERCENTAGE, VAL_PERCENTAGE
)
from scripts.utils.helpers import writeTXT

def getPatientIDs():
    """
    Retrieves patient IDs for the data folder.
    """
    # retrieve patient IDs
    patient_ids = [
        patient_folder.split("_")[-1]
        for patient_folder in os.listdir(DATA_PATH)
        if os.path.isdir(os.path.join(DATA_PATH, patient_folder)) # ensure it is a directory
        and "_" in patient_folder # ensure it matches the expected format with underscore
    ]
    return patient_ids

def createSplits():
    """
    Randomly assigns patient IDs to train, validation and test sets.

    Returns:
        train_ids (list): Patient IDs used for training.
        val_ids (list): Patient IDs used for validation.
        test_ids (list): Patient IDs used for test.
    """
    # retrieve patient IDs
    patient_ids = getPatientIDs()
    # shuffle IDs
    random.shuffle(patient_ids)
    # get length for train, validation and test sets
    len_train = int(len(patient_ids)*TRAIN_PERCENTAGE)
    len_val = int(len(patient_ids)*VAL_PERCENTAGE)
    # retrieve patient IDs for train, validation and test sets
    train_ids = patient_ids[:len_train]
    val_ids = patient_ids[len_train:len_train + len_val]
    test_ids = patient_ids[len_train + len_val:]
    # store IDs to txt files and return them
    writeTXT(train_ids, TRAIN_IDS_PATH)
    writeTXT(val_ids, VAL_IDS_PATH)
    writeTXT(test_ids, TEST_IDS_PATH)
    return train_ids, val_ids, test_ids

def getSplits():
    """
    Load train, validation and test patient IDS from existing .txt files.
    If files do not exist, compute them using createSplits() function.

    Why is it important to have separate training, validation and test sets?
        - Having separate sets ensures the avoidance of data leakage and unbiased evaluation.
          Data leakage leads to overly optimistic performance metrics and invalid evaluation.
        - Training set is what the model is going to be trained on. 
          Ideally, it should cover the whole space for the task at hand.
          Othewise, the model will not be able to generalize good for the data instances
            not covered by the training data.
        - Validation set is used to evaluate the model training.
          It does not make sense to evaluate the model on the training set,
            because that is what the model is trained on. 
          Naturally, it will perform the best on the training set,
            so we use another set, validation, to evaluate the training process
            and guide the model towards most optimal solution using training callbacks.
          For example, we use early stopping which tracks validation metric
            and stops the training when the metric does not improve for some epochs.
          Using training metrics for early stopping does not work, because training metrics
            constantly improve as the training continues.
        - Test set is usually used to evaluate the model perfomance outside of the training loop.
          It provides an unbiased estimate of how well the model generalizes to unseen data.
          Ideally, this set should also represent the whole data space, so that our evaluation is valid.

    Returns:
        train_ids (list): Patient IDs used for training.
        val_ids (list): Patient IDs used for validation.
        test_ids (list): Patient IDs used for test.
    """
    try:
        with open(TRAIN_IDS_PATH, "r") as train_file:
            # convert string to list
            train_ids = literal_eval(train_file.read())
        with open(VAL_IDS_PATH, "r") as val_file:
            # convert string to list
            val_ids = literal_eval(val_file.read())
        with open(TEST_IDS_PATH, "r") as test_file:
            # convert string to list
            test_ids = literal_eval(test_file.read())
    except FileNotFoundError:
        train_ids, val_ids, test_ids = createSplits()
    return train_ids, val_ids, test_ids
