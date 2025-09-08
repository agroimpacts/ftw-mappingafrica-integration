import os
import sys
import time
import yaml
import tempfile

import numpy as np
from lightning.pytorch.cli import LightningCLI

from torchgeo.datamodules import BaseDataModule
from torchgeo.trainers import BaseTask

from .trainers import *

# def fit(config, ckpt_path, cli_args):
#     """Command to fit the model."""
#     print("Running fit command")

#     # Construct the arguments for PyTorch Lightning CLI
#     cli_args = ["fit", f"--config={config}"] + list(cli_args)

#     # If a checkpoint path is provided, append it to the CLI arguments
#     if ckpt_path:
#         cli_args += [f"--ckpt_path={ckpt_path}"]

#     # print(f"CLI arguments: {cli_args}")
#     # # Best practices for Rasterio environment variables
#     # rasterio_best_practices = {
#     #     "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
#     #     "AWS_NO_SIGN_REQUEST": "YES",
#     #     "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
#     #     "GDAL_SWATH_SIZE": "200000000",
#     #     "VSI_CURL_CACHE_SIZE": "200000000",
#     # }
#     # os.environ.update(rasterio_best_practices)

#     # # Run the LightningCLI with the constructed arguments
#     # # saved_argv = sys.argv
#     # # try:
#     #     # sys.argv = [saved_argv[0]] + list(cli_args)
#     # cli = LightningCLI(
#     #     model_class=BaseTask,
#     #     datamodule_class=BaseDataModule,
#     #     seed_everything_default=0,
#     #     subclass_mode_model=True,
#     #     subclass_mode_data=True,
#     #     save_config_kwargs={"overwrite": True},
#     #     args=cli_args,  # Pass the constructed cli_args
#     # )
#     # # finally:
#     # #     sys.argv = saved_argv
#     saved_argv = sys.argv
#     try:
#         sys.argv = [saved_argv[0]] + list(cli_args)
#         cli = LightningCLI(
#             model_class=BaseTask,
#             datamodule_class=BaseDataModule,
#             seed_everything_default=0,
#             subclass_mode_model=True,
#             subclass_mode_data=True,
#             save_config_kwargs={"overwrite": True},
#         )
#     finally:
#         sys.argv = saved_argv

#     # --- Diagnostics: inspect datamodule and one batch to see target types ---
#     try:
#         dm = getattr(cli, "datamodule", None)
#         print("Datamodule instance:", type(dm))
#         if dm is not None:
#             dl = dm.train_dataloader()
#             batch = next(iter(dl))
#             print("Sample batch keys:", getattr(batch, "keys", lambda: None)())
#             # attempt to display target info
#             targ = batch.get("target", batch.get("labels", batch))
#             print("target type:", type(targ))
#             if isinstance(targ, tuple) or isinstance(targ, list):
#                 print("target[0] type:", type(targ[0]))
#             else:
#                 try:
#                     import torch
#                     if isinstance(targ, torch.Tensor):
#                         print("target.shape, dtype, device:", targ.shape, targ.dtype, targ.device)
#                     else:
#                         print("target value:", targ)
#                 except Exception:
#                     print("Could not introspect target value")
#     except Exception as _exc:
#         print("Diagnostics failed:", _exc)
#     # --- end diagnostics ---
#     print("Finished")


def fit(config, ckpt_path, cli_args):
    """Command to fit the model."""
    print("Running fit command")

    # # Accept torchgeo-style configs that use top-level "data" as alias for "datamodule".
    # temp_cfg_path = None
    # if config and os.path.exists(config):
    #     with open(config, "r") as fh:
    #         cfg = yaml.safe_load(fh) or {}
    #     if "data" in cfg and "datamodule" not in cfg:
    #         cfg["datamodule"] = cfg.pop("data")
    #         tf = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    #         yaml.safe_dump(cfg, tf)
    #         tf.flush()
    #         tf.close()
    #         temp_cfg_path = tf.name
    #         config = temp_cfg_path

    # Construct the arguments for PyTorch Lightning CLI
    cli_args = ["fit", f"--config={config}"] + list(cli_args)

    # If a checkpoint path is provided, append it to the CLI arguments
    if ckpt_path:
        cli_args += [f"--ckpt_path={ckpt_path}"]

    # Run LightningCLI by simulating sys.argv (avoid nested Click/Lit parsing issues)
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
        # cleanup temporary config file if we created one
        # if temp_cfg_path is not None:
        #     try:
        #         os.remove(temp_cfg_path)
        #     except Exception:
        #         pass
    
    print("Finished")