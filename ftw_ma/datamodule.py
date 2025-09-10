# FTWMapAfricaDataModule for managing data loading and augmentation
import warnings
import kornia.augmentation as K
from torchgeo.transforms import SatSlideMix
from torchgeo.datamodules import NonGeoDataModule
from typing import Optional
from .dataset import FTWMapAfrica

class FTWMapAfricaDataModule(NonGeoDataModule):
    """
    Data module for FTW Mapping Africa, managing datasets and
    augmentations.

    If `aug_list` is None, all available augmentations are enabled by default.
    """
    def __init__(
        self,
        # temporal_options: str = "windowA",
        # num_samples: int = -1,
        batch_size: int = 32,
        num_workers: int = 0,
        aug_list: Optional[list] = None,
        **kwargs
    ):
        """
        Initialize the data module .
        Args:
            temporal_options (str): Temporal stacking options.
            num_samples (int): Number of samples to use (-1 for all).
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of DataLoader workers.
            aug_list (list, optional): List of augmentation names to apply.
            **kwargs: Additional arguments for FTWMapAfrica.
        """
        if "split" in kwargs:
            raise ValueError("Cannot specify split in FTWDataModule")
        
        super().__init__(FTWMapAfrica, batch_size, num_workers, **kwargs)

        # self.temporal_options = temporal_options
        # self.num_samples = num_samples
        # Only include RandomGamma if normalization_strategy is 'min_max'
        normalization_strategy = kwargs.get("normalization_strategy", None)
        available_augs = {
            # "rotation": K.RandomRotation(p=0.5, degrees=90),
            "rotation": K.RandomRotation(p=0.5, degrees=(90, 90)),
            "hflip": K.RandomHorizontalFlip(p=0.5),
            "vflip": K.RandomVerticalFlip(p=0.5),
            "rescale": K.RandomResizedCrop(
                size=(256, 256),
                scale=(0.75, 1.5),
                ratio=(1.0, 1.0),
                cropping_mode="slice",
                p=0.5
            ),
            "satslidemix": SatSlideMix(p=0.5),
            "sharpness": K.RandomSharpness(p=0.5),
            "gaussian_noise": K.RandomGaussianNoise(
                mean=0.0, std=0.05, p=0.25
            ),
            "brightness": K.RandomBrightness(brightness=(0.98, 1.02), p=0.25),
            "contrast": K.RandomContrast(contrast=(0.9, 1.2), p=0.25),
        }
        if normalization_strategy == "min_max":
            available_augs["gamma"] = K.RandomGamma(gamma=(0.2, 2.0), p=0.25)
        
        # Define geometric and photometric augmentation names
        geometric_augs = ["rotation", "hflip", "vflip", "rescale", 
                          "satslidemix"]
        photometric_augs = ["sharpness", "brightness", "contrast", 
                            "gaussian_noise"]

        if aug_list is None:
            aug_list = list(available_augs.keys())
        
        if "gamma" in aug_list and normalization_strategy != "min_max":
            print(f"Warning: 'gamma' augmentation requires 'min_max'" \
                  f"normalization. Skipping 'gamma'.")
            
        # Build selected_augs in the desired order
        selected_augs = []
        # Add geometric augmentations first
        selected_augs += [available_augs[name] for name in geometric_augs 
                          if name in aug_list and name in available_augs]

        # Add gamma if present
        if "gamma" in aug_list and "gamma" in available_augs and \
            normalization_strategy == "min_max":
            selected_augs.append(available_augs["gamma"])
        
        # Add photometric augmentations
        selected_augs += [available_augs[name] for name in photometric_augs 
                          if name in aug_list and name in available_augs]
        # print(selected_augs)

        self.train_aug = K.AugmentationSequential(
            *selected_augs, 
            data_keys=None,
            keepdim=True,
        )
        # self.aug = None
        # self.batch_size = batch_size
        # self.num_workers = num_workers
        self.kwargs = kwargs

    def setup(self, stage: str):
        """
        Set up datasets for the specified stage ('fit', 'validate',
        'test').
        """
        if stage == "fit":
            self.train_dataset = FTWMapAfrica(
                split="train",
                # temporal_options=self.temporal_options,
                # num_samples=self.num_samples,
                transforms=self.train_aug,
                **self.kwargs,
            )
        if stage in ["fit", "validate"]:
        # if stage == "validate":
            self.val_dataset = FTWMapAfrica(
                split="validate",
                # temporal_options=self.temporal_options,
                # num_samples=self.num_samples,
                transforms=None,
                **self.kwargs,
            )
        if stage == "test":
            self.test_dataset = FTWMapAfrica(
                split="test",
                # temporal_options=self.temporal_options,
                # num_samples=self.num_samples,
                transforms=None,
                **self.kwargs,
            )

