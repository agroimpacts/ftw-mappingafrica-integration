import os
import time
import sys

import numpy as np
import torch
from lightning.pytorch.cli import LightningCLI
from torch.utils.data import DataLoader
from torchgeo.trainers import BaseTask
from torchmetrics import JaccardIndex, MetricCollection, Precision, Recall
from tqdm import tqdm

from .dataset import FTWMapAfrica
from .trainers import CustomSemanticSegmentationTask

import yaml

from ftw.metrics import get_object_level_metrics
# from .metrics import get_object_level_metrics
# from PIL import Image

from .trainers import *
from .dataset import FTWMapAfrica
from .datamodule import FTWMapAfricaDataModule

def fit(config, ckpt_path, cli_args):
    """Command to fit the model."""
    print("Running fit command")

    # Construct the arguments for PyTorch Lightning CLI
    cli_args = ["fit", f"--config={config}"] + list(cli_args)

    # If a checkpoint path is provided, append it to the CLI arguments
    if ckpt_path:
        cli_args += [f"--ckpt_path={ckpt_path}"]

    print(f"CLI arguments: {cli_args}")

    # Best practices for Rasterio environment variables
    rasterio_best_practices = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "AWS_NO_SIGN_REQUEST": "YES",
        "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
        "GDAL_SWATH_SIZE": "200000000",
        "VSI_CURL_CACHE_SIZE": "200000000",
    }
    os.environ.update(rasterio_best_practices)

    # Run LightningCLI by simulating sys.argv (avoid nested 
    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0]] + list(cli_args)
        LightningCLI(
            model_class=BaseTask,
            seed_everything_default=0,
            subclass_mode_model=True,
            subclass_mode_data=True,
            save_config_kwargs={"overwrite": True},
        )
    finally:
        sys.argv = saved_argv
    
    print("Finished")

def test(
    config,    
    gpu,
    model_path,
    data_dir,
    catalog,
    split,
    temporal_options,
    iou_threshold,
    out,
):

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
    tic = time.time()

    # Image normalization arugments from the config file
    data_args = config["data"].get("init_args")
    normalization_strategy = data_args.get("normalization_strategy", None)
    normalization_stat_procedure = data_args.get(
        "normalization_stat_procedure", None
    )
    global_stats = data_args.get("global_stats", None) 
    img_clip_val = data_args.get("img_clip_val", None)  
    nodata = data_args.get("nodata", None)

    print(f"normalization_strategy: {normalization_strategy}")
    print(f"normalization_stat_procedure: {normalization_stat_procedure}")
    print(f"global_stats: {global_stats}")  
    print(f"img_clip_val: {img_clip_val}")
    print(f"nodata: {nodata}")

    ds = FTWMapAfrica(
        catalog=catalog,
        data_dir=data_dir,
        split=split,
        temporal_options=temporal_options,
        normalization_strategy=normalization_strategy,
        normalization_stat_procedure=normalization_stat_procedure,
        global_stats=global_stats,
        img_clip_val=img_clip_val,
        nodata=nodata,        
    )
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=12)
    print(f"Created dataloader with {len(ds)} samples in" \
          "{time.time() - tic:.2f}s")

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
    for batch in tqdm(dl):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        with torch.inference_mode():
            outputs = model(images).argmax(dim=1)

        new_outputs = torch.zeros(
            outputs.shape[0], outputs.shape[1], outputs.shape[2], 
            device=device
        )
        new_outputs[outputs == 2] = 0  # Boundary pixels
        new_outputs[outputs == 0] = 0  # Background pixels
        new_outputs[outputs == 1] = 1  # Crop pixels
        outputs = new_outputs

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

        metrics.update(outputs, masks)
        outputs = outputs.cpu().numpy().astype(np.uint8)
        masks = masks.cpu().numpy().astype(np.uint8)

        for i in range(len(outputs)):
            output = outputs[i]
            mask = masks[i]
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