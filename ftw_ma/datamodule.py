# FTWMapAfricaDataModule for managing data loading and augmentation
from typing import Optional, Union, Dict, Any, Tuple, List
import kornia
import kornia.augmentation as K
import kornia.constants
import torch
from lightning import LightningDataModule
from matplotlib.figure import Figure
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from .dataset import FTWMapAfrica


def randomChannelShuffle(x):
    if torch.rand(1) < 0.5:
        return x
    return torch.cat([x[:, 4:8], x[:, :4]], dim=1)


class FTWMapAfricaDataModule(LightningDataModule):
    """
    Data module for FTW Mapping Africa, managing datasets and
    augmentations.

    If `aug_list` is None, all available augmentations are enabled by default.
    """
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        global_stats: Optional[Union[Dict[str, Any], Tuple, List]] = None,
        normalization_strategy: str = "min_max",
        normalization_stat_procedure: str = "gab",
        random_shuffle: bool = False,
        crop_size: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs
    ):
        """
        Initialize the data module.
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
            random_shuffle (bool): Enable random channel shuffling.
            crop_size (int or tuple): Size for RandomResizedCrop. 
                Can be int (for square) or (height, width) tuple.
                If None, defaults to (256, 256).
            **kwargs: Additional arguments for FTWMapAfrica.
        """
        if "split" in kwargs:
            raise ValueError("Cannot specify split in FTWDataModule")
        
        # ---- normalize inputs BEFORE calling super() ----
        # prefer explicit kwargs over possible duplicates inside kwargs
        if "global_stats" in kwargs and global_stats is None:
            global_stats = kwargs.pop("global_stats")
        if "normalization_strategy" in kwargs:
            normalization_strategy = kwargs.pop("normalization_strategy")
        if "normalization_stat_procedure" in kwargs:
            normalization_stat_procedure = kwargs.pop("normalization_stat_procedure")
        if "crop_size" in kwargs:
            crop_size = kwargs.pop("crop_size")

        # normalize global_stats (accept dict/list/tuple)
        if isinstance(global_stats, dict):
            if normalization_strategy.startswith("z"):
                mean = global_stats.get("mean")
                std = global_stats.get("std")
                try:
                    global_stats = {"mean": list(mean), "std": list(std)}
                except Exception:
                    global_stats = None
            elif normalization_strategy == "min_max":
                min_ = global_stats.get("min")
                max_ = global_stats.get("max")
                try:
                    global_stats = {"min": list(min_), "max": list(max_)}
                except Exception:
                    global_stats = None
            else:
                global_stats = None
        elif isinstance(global_stats, (list, tuple)) and len(global_stats) == 2:
            try:
                a, b = global_stats
                global_stats = {"mean": list(a), "std": list(b)} \
                    if normalization_strategy.startswith("z") \
                        else {"min": list(a), "max": list(b)}
            except Exception:
                global_stats = None

        # Normalize crop_size
        if crop_size is None:
            crop_size = (256, 256)
        elif isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        elif isinstance(crop_size, (list, tuple)) and len(crop_size) == 2:
            crop_size = tuple(crop_size)
        else:
            raise ValueError(
                f"crop_size must be int or (height, width) tuple, "
                f"got {crop_size}"
            )

        print(f"Using normalization_strategy='{normalization_strategy}', ")
        print(f"normalization_stat_procedure='{normalization_stat_procedure}'")
        print(f"global_stats={global_stats}")
        print(f"crop_size={crop_size}")

        # remove any keys we normalized so parent doesn't get unexpected objects
        kwargs.pop("global_stats", None)
        kwargs.pop("normalization_strategy", None)
        kwargs.pop("normalization_stat_procedure", None)
        kwargs.pop("crop_size", None)
        # print(kwargs)      
        
        # super().__init__(FTWMapAfrica, batch_size, num_workers, **kwargs)
        super().__init__()
        if "split" in kwargs:
            raise ValueError("Cannot specify split in FTWDataModule")

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.global_stats = global_stats
        self.normalization_strategy = normalization_strategy
        self.normalization_stat_procedure = normalization_stat_procedure
        self.crop_size = crop_size

        augs = [
            K.RandomRotation(p=0.5, degrees=(90, 90)),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
            K.RandomBrightness(p=0.5, brightness=(0.5, 1.5)),
            K.RandomResizedCrop(
                size=self.crop_size,
                scale=(0.3, 0.9),
                ratio=(0.75, 1.33),
                p=0.5
            )
        ]
        
        if random_shuffle:
            augs.append(kornia.contrib.Lambda(randomChannelShuffle))

        print("Augmentations:")
        for aug in augs:
            print(aug)
     
        self.train_aug = K.AugmentationSequential(*augs, data_keys=None)

        self.kwargs = kwargs
        
    def setup(self, stage: str):
        """
        Set up datasets for the specified stage ('fit', 'validate',
        'test').
        """
        # remember current stage so helpers can behave accordingly
        self.stage = stage
        if stage == "fit":
            self.train_dataset = FTWMapAfrica(
                split="train",
                normalization_strategy=self.normalization_strategy, 
                normalization_stat_procedure=self.normalization_stat_procedure,
                global_stats=self.global_stats,
                **self.kwargs,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = FTWMapAfrica(
                split="validate",
                normalization_strategy=self.normalization_strategy, 
                normalization_stat_procedure=self.normalization_stat_procedure,
                global_stats=self.global_stats,
                **self.kwargs,
            )
        if stage == "test":
            self.test_dataset = FTWMapAfrica(
                split="test",                
                normalization_strategy=self.normalization_strategy, 
                normalization_stat_procedure=self.normalization_stat_procedure,
                global_stats=self.global_stats,
                **self.kwargs,
            )

    def train_dataloader(self) -> Any:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Any:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> Any:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def on_after_batch_transfer(self, batch: dict[str, Tensor], 
                                dataloader_idx: int):
        if self.trainer:
            if self.trainer.training:
                batch = self.train_aug(batch)
            # else:
            #     batch = self.aug(batch)
        return batch

    def plot(self, *args: Any, **kwargs: Any):
        fig: Figure | None = None
        dataset = self.val_dataset
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
        if dataset is not None:
            if hasattr(dataset, "plot"):
                fig = dataset.plot(*args, **kwargs)
        return fig
    
    def apply_train_aug(self, batch: dict[str, Tensor], 
                        seed: int | None = None):
        """Apply the configured train_aug to a single batched dict for 
        interactive use. batch must be batched (batch dimension present). 
        If seed provided, set deterministic seeds."""
        # Only apply train augmentations when datamodule is set for training.
        # If inspecting validate/test datasets, return original batch unchanged.
        if getattr(self, "stage", None) not in ("fit", "train"):
            # If stage is 'validate' or 'test' do not apply train augmentations.
            return batch
        if seed is not None:
            import random, numpy as _np, torch as _torch
            random.seed(seed); _np.random.seed(seed); _torch.manual_seed(seed)
        if self.train_aug is None:
            return batch
        return self.train_aug(batch)