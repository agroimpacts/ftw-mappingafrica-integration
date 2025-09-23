# FTWMapAfricaDataModule for managing data loading and augmentation
from typing import Optional, Union, Dict, Any, Tuple, List
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
        batch_size: int = 32,
        num_workers: int = 0,
        aug_list: Optional[list] = None,
        global_stats: Optional[Union[Dict[str, Any], Tuple, List]] = None,
        normalization_strategy: str = "min_max",
        normalization_stat_procedure: str = "gpb",
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
            global_stats (dict/tuple/list, optional): Global stats for
                normalization, either mean/std or min/max.
            normalization_strategy (str): 'z_value' or 'min_max'.
            normalization_stat_procedure (str): Procedure for stats calc, either
                'lab', 'lpb', 'gab', or 'gpb'.
            **kwargs: Additional arguments for FTWMapAfrica.
        """
        if "split" in kwargs:
            raise ValueError("Cannot specify split in FTWDataModule")
        
         # normalize aug_list string "None" -> None
        # if isinstance(aug_list, str) and aug_list.lower() == "none":
        #     aug_list = None

        # ---- normalize inputs BEFORE calling super() ----
        # prefer explicit kwargs over possible duplicates inside kwargs
        if "aug_list" in kwargs and aug_list is None:
            aug_list = kwargs.pop("aug_list")
        if "global_stats" in kwargs and global_stats is None:
            global_stats = kwargs.pop("global_stats")
        if "normalization_strategy" in kwargs \
            and normalization_strategy == "min_max":
            normalization_strategy = kwargs.pop("normalization_strategy")
        if "normalization_stat_procedure" in kwargs \
            and normalization_stat_procedure == "gpb":
            normalization_stat_procedure = \
                kwargs.pop("normalization_stat_procedure")

        # normalize global_stats (accept dict/list/tuple)
        if isinstance(global_stats, dict):
            mean = global_stats.get("mean")
            std = global_stats.get("std")
            try:
                global_stats = {"mean": list(mean), "std": list(std)}
            except Exception:
                global_stats = None
        elif isinstance(global_stats, (list, tuple)) and len(global_stats) == 2:
            try:
                a, b = global_stats
                global_stats = {"mean": list(a), "std": list(b)} \
                    if normalization_strategy.startswith("z") \
                        else {"min": list(a), "max": list(b)}
            except Exception:
                global_stats = None

        # remove any keys we normalized so parent doesn't get unexpected objects
        kwargs.pop("aug_list", None)
        kwargs.pop("global_stats", None)
        kwargs.pop("normalization_strategy", None)
        kwargs.pop("normalization_stat_procedure", None)  
        # print(kwargs)      
        
        super().__init__(FTWMapAfrica, batch_size, num_workers, **kwargs)

        # # Only include RandomGamma if normalization_strategy is 'min_max'
        # normalization_strategy = kwargs.get("normalization_strategy", None)

        # # also accept string "None" from YAML by converting to actual None
        # if isinstance(aug_list, str) and aug_list.lower() == "none":
        #     self.train_aug = None
        
        # elif (aug_list == [] or aug_list is None):
        #     self.train_aug = None
        self.global_stats = global_stats
        self.normalization_strategy = normalization_strategy
        self.normalization_stat_procedure = normalization_stat_procedure
        # keep aug_list as provided (None means no augmentations)
        self.aug_list = aug_list

        if not self.aug_list:
            print("No augmentations will be applied.")
            self.train_aug = None
            self.kwargs = kwargs
            return

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
        if self.normalization_strategy == "min_max":
            available_augs["gamma"] = K.RandomGamma(gamma=(0.2, 2.0), p=0.25)
        
        # Define geometric and photometric augmentation names
        geometric_augs = ["rotation", "hflip", "vflip", "rescale", 
                          "satslidemix"]
        photometric_augs = ["sharpness", "brightness", "contrast", 
                            "gaussian_noise"]
        
        # if "gamma" in aug_list and normalization_strategy != "min_max":
        #     print(f"Warning: 'gamma' augmentation requires 'min_max'" \
        #         f"normalization. Skipping 'gamma'.")
        if "gamma" in (self.aug_list or []) \
            and self.normalization_strategy != "min_max":
             print(f"Warning: 'gamma' augmentation requires 'min_max'" \
                   f"normalization. Skipping 'gamma'.")            
        # Build selected_augs in the desired order
        # selected_augs = []
        # # Add geometric augmentations first
        # selected_augs += [available_augs[name] for name in geometric_augs 
        #                 if name in aug_list and name in available_augs]
        selected_augs = []
        # Add geometric augmentations first
        selected_augs += [available_augs[name] for name in geometric_augs
                          if (self.aug_list and name in self.aug_list) 
                          and name in available_augs]
        # Add gamma if present
        # if "gamma" in aug_list and "gamma" in available_augs and \
        #     normalization_strategy == "min_max":
        #     selected_augs.append(available_augs["gamma"])
        if (self.aug_list and "gamma" in self.aug_list) \
            and "gamma" in available_augs \
                and self.normalization_strategy == "min_max":
             selected_augs.append(available_augs["gamma"])
        
        # Add photometric augmentations
        # selected_augs += [available_augs[name] for name in photometric_augs 
        #                 if name in aug_list and name in available_augs]
        # print(selected_augs)
        # Add photometric augmentations
        selected_augs += [available_augs[name] for name in photometric_augs
                          if (self.aug_list and name in self.aug_list) 
                          and name in available_augs]

        self.train_aug = K.AugmentationSequential(
            *selected_augs, 
            data_keys=None,
            keepdim=True,
        )
        # print(self.train_aug)
        # print(self.normalization_strategy)
        # print(self.normalization_stat_procedure)
        # print(self.global_stats)
        self.kwargs = kwargs

    def setup(self, stage: str):
        """
        Set up datasets for the specified stage ('fit', 'validate',
        'test').
        """
        if stage == "fit":
            self.train_dataset = FTWMapAfrica(
                split="train",
                normalization_strategy=self.normalization_strategy, 
                normalization_stat_procedure=self.normalization_stat_procedure,
                global_stats=self.global_stats,
                transforms=self.train_aug,
                **self.kwargs,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = FTWMapAfrica(
                split="validate",
                normalization_strategy=self.normalization_strategy, 
                normalization_stat_procedure=self.normalization_stat_procedure,
                global_stats=self.global_stats,
                transforms=None,
                **self.kwargs,
            )
        if stage == "test":
            self.test_dataset = FTWMapAfrica(
                split="test",                
                normalization_strategy=self.normalization_strategy, 
                normalization_stat_procedure=self.normalization_stat_procedure,
                global_stats=self.global_stats,
                transforms=None,
                **self.kwargs,
            )

