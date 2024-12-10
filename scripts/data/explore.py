import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import imageio
from IPython.display import clear_output

from scripts.utils.vars import DATA_PATH, PATIENT_FOLDER_NAME

def plotSlice(patient_id: str, modality: str, modality_slice: np.array, seg_slice: np.array, slice_idx: int, save_path: str = False):
    """
    Plot a single slice of the modality and segmentation.

    Parameters:
        - patient_id (str): Patient ID.
        - modality (str): Modality to visualize.
        - modality_slice (np.array): Slice of the modality image.
        - seg_slice (np.array): Slice of the segmentation mask.
        - slice_idx (int): Index of the slice.
        - save_path (str): Path where to save the plot (used for gif to save temporary plots).
    """
    clear_output(wait=True)
    # create figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    title = f"Patient ID: {patient_id}, Modality: {modality.upper()}, Slice: {slice_idx}"

    fig.suptitle(title, fontsize=16)
    # modality image
    ax[0].imshow(modality_slice, cmap="gray")
    ax[0].set_title(f"{modality.upper()} Image")
    ax[0].axis("off")
    # segmentation mask
    cmap = plt.get_cmap("jet")
    norm = mcolors.Normalize(vmin=0, vmax=3)  # Normalization for class values
    ax[1].imshow(seg_slice, cmap=cmap, norm=norm)
    ax[1].set_title("Segmentation Mask")
    ax[1].axis("off")
    # legend
    class_labels = {0: "Background", 1: "Necrotic Core", 2: "Peritumoral Edematous", 3: "Enhancing Tumor"}
    class_colors = {class_value: cmap(norm(class_value)) for class_value in class_labels.keys()}
    patches = [mpatches.Patch(color=color, label=label) for label, color in zip(class_labels.values(), class_colors.values())]
    ax[1].legend(handles=patches, loc="upper right", title="Classes")
    # save the plot if save_path is provided (used for gif creation)
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()

def visualizePatientData(patient_id: str = "00000", modality: str = "t1", slice_idx: int = None, create_gif: bool = False):
    """
    Visualize a patient's modality image and corresponding segmentation mask.
    Either plot only specific slice or create a gif going through all the slices.
    
    Parameters:
        - patient_id (str): ID of the patient.
        - modality (str): The MRI modality to visualize ('t1', 't1ce', 't2', 'flair').
        - slice_idx (int): Slice index to plot.
        - create_gif (bool): Whether to create a gif showing whole modality .nii file and segmentaton mask.
    """
    # define paths to patient .nii data
    patient_folder = PATIENT_FOLDER_NAME + patient_id
    patient_path = os.path.join(DATA_PATH, patient_folder)
    # define modality and segmentation file paths
    modality_file = os.path.join(patient_path, f"{patient_folder}_{modality}.nii.gz")
    seg_file = os.path.join(patient_path, f"{patient_folder}_seg.nii.gz")
    # check if modality and segmentation files exist
    if not os.path.exists(modality_file):
        print(f"Modality file {modality_file} not found.")
        return
    if not os.path.exists(seg_file):
        print(f"Segmentation file {seg_file} not found.")
        return
    
    # load modality and segmentation images
    modality_img = nib.load(modality_file).get_fdata()
    seg_img = nib.load(seg_file).get_fdata()
    
    # visualize
    if create_gif:
        # create gif for all slices
        slice_savepath = f"../../plots/{patient_id}/"
        gif_savepath = f"../../plots/{patient_id}/slices_visualization.gif"
        os.makedirs(slice_savepath, exist_ok=True) # make sure folder where to store gif file exists

        # get number of slices and iterate through them
        num_slices = modality_img.shape[2]
        frames = [] # to store slice images into a gif
        for idx in range(num_slices):
            # retrieve the slice
            modality_slice = modality_img[:, :, idx]
            seg_slice = seg_img[:, :, idx]
            # save each slice plot
            temp_slice_savepath = slice_savepath + f"{idx}.png"
            plotSlice(patient_id, modality, modality_slice, seg_slice, idx, temp_slice_savepath)
            frames.append(imageio.imread(temp_slice_savepath))

        # create the GIF
        imageio.mimsave(gif_savepath, frames, fps=5)
        # clean up temporary files
        for temp_path in [slice_savepath + f"{idx}.png" for idx in range(num_slices)]:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        print(f"GIF saved at {gif_savepath}")
    else: # plot single slice
        # plot single slice
        if slice_idx is None:
            slice_idx = modality_img.shape[2] // 2
        modality_slice = modality_img[:, :, slice_idx]
        seg_slice = seg_img[:, :, slice_idx]
        plot_savepath = f"../../plots/{patient_id}/{modality}_{slice_idx}.jpg"
        plotSlice(patient_id, modality, modality_slice, seg_slice, slice_idx, save_path=plot_savepath)

def countLabels():
    """
    Counts the number of labels for all segmentations masks in the dataset.

    Parameters:
        - data_path (str): Path to the folder containing data folders for each patient storing modality images and segmentation masks.
    """
    counts_dict = {}
    for pdir in tqdm(os.listdir(DATA_PATH), "Processing patients segmentation masks..."):
        # construct the path to the segmentation image
        seg_file = os.path.join(
            DATA_PATH,
            f"{pdir}",
            f"{pdir}_seg.nii.gz"
        )
        # load the segmentation image
        seg_img = nib.load(seg_file).get_fdata()
        # compute unique labels and their counts
        labels, counts = np.unique(seg_img, return_counts=True)
        # update the overall counts dictionary
        for label, count in zip(labels, counts):
            label = int(label)
            if label not in counts_dict:
                counts_dict[label] = count
            else:
                counts_dict[label] += count
    return counts_dict
