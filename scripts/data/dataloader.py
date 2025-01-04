from torch.utils.data import DataLoader

from monai.transforms import (
    MapTransform, Compose, LoadImaged, ToTensord, ConcatItemsd, 
    Orientationd, Spacingd, EnsureChannelFirstd,
    ConvertToMultiChannelBasedOnBratsClassesd,
    RandFlipd, SpatialPadd,
    ScaleIntensityd, NormalizeIntensityd, 
    RandGaussianNoised, RandBiasFieldd
)
from monai.data import Dataset as MonaiDataset

from scripts.utils.vars import DATA_PATH, PATIENT_FOLDER_NAME, BATCH_SIZE

class CustomSpatialCrop(MapTransform):
    """
    Cuts depth dimension from beginning and end.
    For example, having depth=155, start=5 and end=-6,
        the resulting depth dimension is 155-5-6=144.
    
    Parameters:
        - keys (list): Keys to standardize (e.g., ["modalities"]).
        - start (int): Dimension to start from.
        - end (int): Dimension to end at.
    """
    def __init__(self, keys, start=5, end=-6, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.start = start
        self.end = end

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                # crop the spatial dimensions
                d[key] = d[key][:, :, :, self.start:self.end]
        return d

def defineTransforms(modalities: list, train: bool = True, swin: bool = False):
    """
    Defines data transformations for MONAI dataset.
    Defines separate configurations for training (with augmentations) and validation/test (no augmentations).
    For possible transformations see https://docs.monai.io/en/latest/transforms.html.

    Parameters:
        - modalities (list): List of modalities to include as channels (e.g., ["t1", "t2", "flair"]).
        - train (bool): Whether to define transformations for training or validation/testing dataloaders.
        - swin (bool): Whether to pad images to (256, 256, 160) for Swin UNETR compatibility.

    Returns:
        - transforms (monai.transforms.Compose): Data transformations.
    """
    transforms = [
        LoadImaged(keys=[f"modality_{i}" for i in range(len(modalities))] + ["mask"]), # load images
        EnsureChannelFirstd(keys=[f"modality_{i}" for i in range(len(modalities))]), # ensure channels first
        ConvertToMultiChannelBasedOnBratsClassesd(keys="mask"), # to 3 channels for mask (3 classes): necrotic: 1, edema: 2, enhancing: 4
        ConcatItemsd(keys=[f"modality_{i}" for i in range(len(modalities))], name="modalities", dim=0),
        Orientationd(keys=["modalities", "mask"], axcodes="RAS"), # standardize orientation to RAS
        Spacingd(
            keys=["modalities", "mask"],
            pixdim=(1.0, 1.0, 1.0), # resample to 1mm isotropic spacing
            mode=("bilinear", "nearest") # use bilinear for images, nearest for labels
        ),
        ScaleIntensityd(keys="modalities"),  # normalize to [0, 1] range
    ]
    if train:
        transforms.extend([
            RandFlipd(keys=["modalities", "mask"], prob=0.5, spatial_axis=1), # random flip 
            RandFlipd(keys=["modalities", "mask"], prob=0.5, spatial_axis=2), # random flip 
            RandBiasFieldd(keys="modalities", degree=3, coeff_range=(0.0, 0.2), prob=0.75),  # bias field augmentation 
            RandGaussianNoised(keys="modalities", mean=0.0, std=0.03, prob=0.5), # gaussian noise
        ])
    transforms.extend([
        NormalizeIntensityd(keys="modalities", nonzero=True, channel_wise=True), # standardization
        SpatialPadd(keys=["modalities", "mask"], spatial_size=(192, 224, 144), mode="constant", value=0), # make dimensions divisible by 2^5 for UNet compatibility
        ToTensord(keys=["modalities", "mask"]) # convert to tensors
    ])
    if swin:
        transforms.append(
            SpatialPadd(keys=["modalities", "mask"], spatial_size=(256, 256, 160), mode="constant", value=0)
        )
    return Compose(transforms)

def createDataloader(patient_ids: list, modalities: list, batch_size: int = BATCH_SIZE, train: bool = True, model_name: str = "UNet"):
    """
    Creates a PyTorch DataLoader for the BraTS dataset.

    Parameters:
        - patient_ids (list): List of patient IDs to include in the dataloader.
        - modalities (list): List of modalities to include as channels (e.g., ["t1", "t2", "flair"]).
        - batch_size (int): Batch size for the dataloader.
        - train (bool): Train or validation/test set.
        - model_name (str): Name of the model.

    Returns:
        - dataloader (torch.utils.data.DataLoader): Configured dataloader for the given dataset.
    """
    # prepare data in MONAI format
    data_dicts = [
        {
            **{
                f"modality_{i}": f"{DATA_PATH}/{PATIENT_FOLDER_NAME}{pid}/{PATIENT_FOLDER_NAME}{pid}_{modality}.nii.gz"
                for i, modality in enumerate(modalities)
            },
            "mask": f"{DATA_PATH}/{PATIENT_FOLDER_NAME}{pid}/{PATIENT_FOLDER_NAME}{pid}_seg.nii.gz"
        }
        for pid in patient_ids
    ]
    # define data augmentation and transforms
    swin = True if model_name == "SwinUNETR" else False
    transforms = defineTransforms(modalities=modalities, train=train, swin=swin)
    # create MONAI dataset and DataLoader, return DataLoader
    monai_dataset = MonaiDataset(data=data_dicts, transform=transforms)
    dataloader = DataLoader(monai_dataset, batch_size=batch_size, shuffle=train)
    return dataloader
