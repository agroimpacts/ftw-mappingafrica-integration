import os
import sys
import time
import yaml

# import tempfile
from tqdm import tqdm

import numpy as np
from lightning.pytorch.cli import LightningCLI

from torchgeo.datamodules import BaseDataModule
from torchgeo.trainers import BaseTask
from torchmetrics import JaccardIndex, MetricCollection, Precision, Recall

from ftw.metrics import get_object_level_metrics
# from .metrics import get_object_level_metrics
# from PIL import Image

from .trainers import *
from .datamodule import FTWMapAfricaDataModule

def fit(config, ckpt_path, cli_args):
    """Command to fit the model."""
    print("Running fit command")

    # Construct the arguments for PyTorch Lightning CLI
    cli_args = ["fit", f"--config={config}"] + list(cli_args)

    # If a checkpoint path is provided, append it to the CLI arguments
    if ckpt_path:
        cli_args += [f"--ckpt_path={ckpt_path}"]

    # Run LightningCLI by simulating sys.argv (avoid nested 
    # Click/Lit parsing issues)
    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0]] + list(cli_args)
        LightningCLI(
            model_class=BaseTask,
            datamodule_class=BaseDataModule,
            seed_everything_default=0,
            subclass_mode_model=True,
            subclass_mode_data=True,
            save_config_kwargs={"overwrite": True},
        )
    finally:
        sys.argv = saved_argv
    
    print("Finished")

def test(config, model_path, gpu, iou_threshold, out):
    """Command to test the model."""

    with open(config, "r") as f:
        config = yaml.safe_load(f)

    print("Running test command")
    if gpu is None:
        gpu = -1

    # Choose device consistently
    if gpu is not None and gpu >= 0:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print("Loading model")
    tic = time.time()
    trainer = CustomSemanticSegmentationTask.load_from_checkpoint(
        model_path, map_location="cpu"
    )
    model = trainer.model.eval().to(device)
    print(f"Model loaded in {time.time() - tic:.2f}s")

    print("Creating dataloader")
    # tic = time.time()

    dm = FTWMapAfricaDataModule(**config["data"].get("init_args"))
    dm.setup(stage="test")
    # print(f"Created dataloader with {len(ds)} samples "\
    #       f"in {time.time() - tic:.2f}s")

    metrics = MetricCollection(
        [
            JaccardIndex(
                task="multiclass", average="none", num_classes=3, ignore_index=3
            ),
            Precision(
                task="multiclass", average="none", num_classes=3, ignore_index=3
            ),
            Recall(
                task="multiclass", average="none", num_classes=3, ignore_index=3
            ),
        ]
    ).to(device)

    all_tps = 0
    all_fps = 0
    all_fns = 0
    for batch in tqdm(dm.test_dataloader(), desc="Testing"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        with torch.inference_mode():
            outputs = model(images)

        outputs = outputs.argmax(dim=1)

        new_outputs = torch.zeros(
            outputs.shape[0], outputs.shape[1], outputs.shape[2], device=device
        )
        new_outputs[outputs == 2] = 0  # Boundary pixels
        new_outputs[outputs == 0] = 0  # Background pixels
        new_outputs[outputs == 1] = 1  # Crop pixels
        outputs = new_outputs

        metrics.update(outputs, masks)
        outputs = outputs.cpu().numpy().astype(np.uint8)
        masks = masks.cpu().numpy().astype(np.uint8)

        # Save each output and mask as an image for inspection (for check)
        # output_dir = f"{os.path.dirname(out)}/test_outputs"
        # mask_dir = f"{os.path.dirname(out)}/test_masks"
        # os.makedirs(output_dir, exist_ok=True)
        # os.makedirs(mask_dir, exist_ok=True)

        # for i in range(outputs.shape[0]):
        #     output_path = os.path.join(output_dir, f"output_{i}.png")
        #     mask_path = os.path.join(mask_dir, f"mask_{i}.png")
        #     # Use PIL to save as PNG images
        #     Image.fromarray(outputs[i]).save(output_path)
        #     Image.fromarray(masks[i]).save(mask_path)        

        for i in range(len(outputs)):
            output = outputs[i]
            mask = masks[i]
            # if postprocess:
            #     post_processed_output = out.copy()
            #     output = post_processed_output
            tps, fps, fns = get_object_level_metrics(
                mask, output, iou_threshold=iou_threshold
            )
            all_tps += tps
            all_fps += fps
            all_fns += fns

    results = metrics.compute()
    pixel_level_iou = results["MulticlassJaccardIndex"][1].item()
    pixel_level_precision = results["MulticlassPrecision"][1].item()
    pixel_level_recall = results["MulticlassRecall"][1].item()

    if all_tps + all_fps > 0:
        object_precision = all_tps / (all_tps + all_fps)
    else:
        object_precision = float("nan")

    if all_tps + all_fns > 0:
        object_recall = all_tps / (all_tps + all_fns)
    else:
        object_recall = float("nan")

    print(f"Pixel level IoU: {pixel_level_iou:.4f}")
    print(f"Pixel level precision: {pixel_level_precision:.4f}")
    print(f"Pixel level recall: {pixel_level_recall:.4f}")
    print(f"Object level precision: {object_precision:.4f}")
    print(f"Object level recall: {object_recall:.4f}")

    if out is not None:
        if not os.path.exists(out):
            with open(out, "w") as f:
                f.write(
                    f"train_checkpoint,pixel_level_iou,"\
                    f"pixel_level_precision,pixel_level_recall,"\
                    f"object_level_precision,object_level_recall\n"
                )
        with open(out, "a") as f:
            f.write(
                f"{model_path},{pixel_level_iou},"\
                f"{pixel_level_precision},{pixel_level_recall},"\
                f"{object_precision},{object_recall}\n"
            )