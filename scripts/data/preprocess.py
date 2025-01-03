import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

from scripts.utils.vars import DATA_PATH, CROPPED_DATA_PATH, PATIENTS_CROPPING_IDXS_PATH, DATA_CONFIG_PATH
from scripts.utils.helpers import writeJSON, readJSON

def calculateCropIndicies(modalities: list = ["t1", "t2", "flair"]):
    """
    Process all patients to calculate and save cropping indices for modality and segmentation images,
        where classes are not 0 (for segmentation masks) and pixel values are not 0 (for modality images).

    Parameters:
        - modalities (list): List of modalities names.
    
    Returns:
        - cropping_data (dict): Contains cropping indexes for width, height and depth for all the patients.
    """
    def calculateSegmentationIndices(seg_img: np.array) -> tuple:
        """
        Calculate min, max depth indices based on the segmentation mask,
            where class is not 0 (background)

        Parameters:
            - seg_img (numpy.ndarray): The segmentation mask.

        Returns:
            - (tuple): Minimum and maximum depth indices for a patient.
        """
        depth_indices = np.where(np.any(seg_img != 0, axis=(0, 1)))[0]
        return depth_indices[0], depth_indices[-1]

    def calculateModalityIndices(modality_img: np.array, depth_min: int, depth_max: int) -> dict:
        """
        Calculate min, max indices of modality images for width and height,
            where pixel values are not 0,
            using precomputed depth indices from the segmentation mask.

        Parameters:
            - modality_img (numpy.ndarray): The modality image.
            - depth_min (int): Minimum depth in segmentation mask where where class is not 0 (background).
            - depth_max (int): Maximum depth in segmentation mask where where class is not 0 (background).

        Returns:
            - cropping_data (dict): A dictionary containing the cropping indices for a patient.
        """
        # crop modality images to the relevant depth (for faster processing)
        modality_img_cropped = modality_img[:, :, depth_min:depth_max + 1]
        # calculate width and height indices based on modality images
        width_indices, height_indices = [], []
        for slice_idx in range(modality_img_cropped.shape[2]):
            slice_2d = modality_img_cropped[:, :, slice_idx]
            # find non-zero rows and columns
            rows = np.where(np.any(slice_2d != 0, axis=1))[0]
            cols = np.where(np.any(slice_2d != 0, axis=0))[0]
            if rows.size > 0:
                width_indices.append((rows[0], rows[-1]))
            if cols.size > 0:
                height_indices.append((cols[0], cols[-1]))
        # determine the global minimum and maximum for width and height
        width_min = min(idx[0] for idx in width_indices)
        width_max = max(idx[1] for idx in width_indices)
        height_min = min(idx[0] for idx in height_indices)
        height_max = max(idx[1] for idx in height_indices)
        return {
            "width_min": int(width_min),
            "width_max": int(width_max),
            "height_min": int(height_min),
            "height_max": int(height_max)}

    cropping_data = {}
    for pdir in tqdm(os.listdir(DATA_PATH), "Processing patients..."):
        patient_id = pdir.split("_")[-1]
        patient_data = {}
        # path to the segmentation image
        seg_file = os.path.join(DATA_PATH, f"{pdir}", f"{pdir}_seg.nii.gz")
        if not os.path.exists(seg_file):
            print(f"Segmentation file {seg_file} not found.")
            continue
        # load segmentation image once per patient
        seg_img = nib.load(seg_file).get_fdata()
        # calculate min and max depth crop dimensions
        depth_min, depth_max = calculateSegmentationIndices(seg_img)
        # calculate min and max width and height crop dimensions for each modality
        for modality in modalities:
            # path to the modality image
            modality_file = os.path.join(DATA_PATH, f"{pdir}", f"{pdir}_{modality}.nii.gz")
            if not os.path.exists(modality_file):
                print(f"Modality file {modality_file} not found.")
                continue
            # load modality image
            modality_img = nib.load(modality_file).get_fdata()
            try:
                # calculate cropping indices using the calculated crop depth indices
                cropping_indices = calculateModalityIndices(modality_img, depth_min, depth_max)
                # save indices for the current modality
                patient_data["depth_min"] = str(depth_min)
                patient_data["depth_max"] = str(depth_max)
                patient_data[modality] = cropping_indices
            except Exception as e:
                print(f"Error processing patient_id {patient_id} modality {modality}: {e}")
                break
        if patient_data:
            cropping_data[patient_id] = patient_data
    else:
        # save all cropping indices to a JSON file only if no error occurred
        writeJSON(data=cropping_data, save_path=PATIENTS_CROPPING_IDXS_PATH)
        print(f"Finished! Cropping data saved to {PATIENTS_CROPPING_IDXS_PATH}")
        return cropping_data

def getGlobalCropIndices(modalities: list = ["t1", "t2", "flair"], padding: int = 1, image_width_max: int = 240, image_height_max: int = 240, image_depth_max: int = 155):
    """
    Find the global minimum and maximum cropping indices for depth, width, and height
    across all patients and modalities and adjust them with the given padding.

    Parameters:
        - modalities (list): List of modalities names.
        - padding (int): Additional padding to include around the cropped region.
        - image_width_max (int): Original width of the .nii images.
        - image_height_max (int): Original height of the .nii images.
        - image_depth_max (int): Original depth of the .nii images.

    Returns:
        - global_cropping (dict): Dictionary containing global cropping indices.
    """
    # read the cropping data
    cropping_data = readJSON(PATIENTS_CROPPING_IDXS_PATH)
    skipped_ids = []
    global_cropping = {
        "depth_min": float("inf"),
        "depth_max": float("-inf"),
        "width_min": float("inf"),
        "width_max": float("-inf"),
        "height_min": float("inf"),
        "height_max": float("-inf")
    }
    # iterate over all patients and modalities
    for patient_id, patient_data in cropping_data.items():
        depth_min, depth_max = int(patient_data["depth_min"]), int(patient_data["depth_max"])
        # update global minimums and maximums for depth
        global_cropping["depth_min"] = min(global_cropping["depth_min"], depth_min)
        global_cropping["depth_max"] = max(global_cropping["depth_max"], depth_max)
        for modality in modalities:
            if modality in patient_data:
                modality_data = patient_data[modality]
                # update global minimums and maximums for width
                global_cropping["width_min"] = min(global_cropping["width_min"], modality_data["width_min"])
                global_cropping["width_max"] = max(global_cropping["width_max"], modality_data["width_max"])
                # update global minimums and maximums for height
                global_cropping["height_min"] = min(global_cropping["height_min"], modality_data["height_min"])
                global_cropping["height_max"] = max(global_cropping["height_max"], modality_data["height_max"])
    # adjust with padding
    if padding != 0:
        global_cropping["depth_min"] = max(0, global_cropping["depth_min"] - padding)
        global_cropping["depth_max"] = min(image_depth_max, global_cropping["depth_max"] + padding)
        global_cropping["width_min"] = max(0, global_cropping["width_min"] - padding)
        global_cropping["width_max"] = min(image_width_max, global_cropping["width_max"] + padding)
        global_cropping["height_min"] = max(0, global_cropping["height_min"] - padding)
        global_cropping["height_max"] = min(image_height_max, global_cropping["height_max"] + padding)
    # save global cropping indices
    global_cropping["skipped_ids"] = skipped_ids
    writeJSON(data=global_cropping, save_path=DATA_CONFIG_PATH)
    return global_cropping

def cropImages(modalities: list = ["t1", "t2", "flair"], depth_start=10, depth_end=-1):
    """
    Crops each patient images based on global cropping indices stored in DATA_CONFIG_PATH.
    Width and height dimensions are cropped on the stored indicies, while
        depth dimension is cropped using depth_start and depth_end parameters.
    
    Parameters:
        - modalities (list): List of modalities names.
        - depth_start (int): Starting dimension index of depth dimension.
        - depth_end (int): Ending dimension index of depth dimension (max_depth + depth_end).
    """
    # read global cropping indices from JSON
    data_config = readJSON(DATA_CONFIG_PATH)
    width_min = data_config["width_min"]
    width_max = data_config["width_max"]
    height_min = data_config["height_min"]
    height_max = data_config["height_max"]
    # process images for each patient
    for pdir in tqdm(os.listdir(DATA_PATH), "Processing patients..."):
        # load segmentation mask
        seg_file = os.path.join(DATA_PATH, f"{pdir}", f"{pdir}_seg.nii.gz")
        if not os.path.exists(seg_file):
            print(f"Segmentation file {seg_file} not found for patient {pdir}.")
            continue
        seg_img = nib.load(seg_file)
        seg_data = seg_img.get_fdata()
        # crop segmentation mask
        seg_cropped = seg_data[width_min:width_max, height_min:height_max, depth_start:depth_end].astype(seg_img.get_data_dtype())
        # save cropped segmentation mask
        cropped_seg_path = os.path.join(CROPPED_DATA_PATH, f"{pdir}", f"{pdir}_seg.nii.gz")
        directory = os.path.dirname(cropped_seg_path)
        os.makedirs(directory, exist_ok=True)
        nib.save(nib.Nifti1Image(seg_cropped, seg_img.affine), cropped_seg_path)
        # iterate through each modality
        for modality in modalities:
            modality_file = os.path.join(DATA_PATH, f"{pdir}", f"{pdir}_{modality}.nii.gz")
            if not os.path.exists(modality_file):
                print(f"Modality file {modality_file} not found for patient {pdir}.")
                continue
            modality_img = nib.load(modality_file)
            modality_data = modality_img.get_fdata()
            # crop modality image
            modality_cropped = modality_data[width_min:width_max, height_min:height_max, depth_start:depth_end].astype(modality_img.get_data_dtype())
            # save cropped modality image
            cropped_modality_path = os.path.join(CROPPED_DATA_PATH, f"{pdir}", f"{pdir}_{modality}.nii.gz")
            nib.save(nib.Nifti1Image(modality_cropped, modality_img.affine), cropped_modality_path)
