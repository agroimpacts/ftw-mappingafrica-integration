import os
from typing import Optional

import torch
from torchvision.utils import make_grid, save_image
from lightning.pytorch.callbacks import Callback


def _safe_extract(batch):
    if isinstance(batch, dict):
        def pick(keys):
            for k in keys:
                if k in batch and batch[k] is not None:
                    return batch[k]
            return None
        img = pick(["image", "images", "img", "input", "image0"])
        tgt = pick(["mask", "target", "labels", "label"])
        return img, tgt
    if isinstance(batch, (list, tuple)):
        tensors = [x for x in batch if isinstance(x, torch.Tensor)]
        if len(tensors) >= 2:
            return tensors[0], tensors[1]
        if len(tensors) == 1:
            return tensors[0], None
        return (batch[0] if len(batch) > 0 else None, batch[1] 
                if len(batch) > 1 else None)
    if isinstance(batch, torch.Tensor):
        return batch, None
    return None, None


class AugmentationDebugCallback(Callback):
    def __init__(self, log_batch_idx: int = 0, max_images: int = 8):
        super().__init__()
        self.log_batch_idx = int(log_batch_idx)
        self.max_images = int(max_images)
        self._logged = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, 
                             dataloader_idx=0):
        if self._logged or batch_idx != self.log_batch_idx:
            return

        images, masks = _safe_extract(batch)
        if images is None:
            return

        images = images[: self.max_images].cpu()
        if masks is not None and isinstance(masks, torch.Tensor):
            if masks.dim() == 3:
                masks = masks[: self.max_images].unsqueeze(1).cpu()
            else:
                masks = masks[: self.max_images].cpu()

        try:
            img_grid = make_grid(images, nrow=min(4, len(images)), 
                                 normalize=True, scale_each=True)
        except Exception:
            img_grid = None

        try:
            mask_grid = None
            if masks is not None:
                mask_grid = make_grid(
                    masks.float(), 
                    nrow=min(4, masks.shape[0]), normalize=False
                )
        except Exception:
            mask_grid = None

        logger = trainer.logger
        logged = False

        try:
            if hasattr(logger, "experiment") and logger.experiment is not None:
                if img_grid is not None:
                    logger.experiment.add_image("debug/aug_images", img_grid, 
                                                global_step=trainer.global_step)
                if mask_grid is not None:
                    logger.experiment.add_image("debug/aug_masks", mask_grid, 
                                                global_step=trainer.global_step)
                logged = True
        except Exception:
            logged = False

        if not logged:
            outdir = trainer.default_root_dir or \
                getattr(trainer, "log_dir", ".")
            outdir = os.path.join(os.path.expanduser(outdir), "debug_augs")
            os.makedirs(outdir, exist_ok=True)
            try:
                if img_grid is not None:
                    save_image(
                        img_grid, 
                        os.path.join(
                            outdir, f"batch_{trainer.global_step}_images.png"
                        )
                    )
                if mask_grid is not None:
                    save_image(
                        mask_grid, 
                        os.path.join(
                            outdir, 
                            f"batch_{trainer.global_step}_masks.png"
                        )
                    )
            except Exception:
                pass

        self._logged