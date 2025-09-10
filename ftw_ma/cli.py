# import datetime
# import enum
# import json
# import os
from pathlib import Path
from typing import Optional
import click

@click.group()
def ftw_ma():
    """Fields of The World (FTW) / Mapping Africa - Command Line Interface"""
    pass

@ftw_ma.group()
def model():
    """Training and testing FTW models."""
    pass

@model.command("fit", help="Fit the model")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the config file",
)
@click.option(
    "--ckpt_path",
    "-m",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    show_default=True,
    help="Path to a checkpoint file to resume training from",
)
@click.argument(
    "cli_args", nargs=-1, type=click.UNPROCESSED
)  # Capture all remaining arguments
def model_fit(config, ckpt_path, cli_args):
    try:
        from .compiler import fit
    except Exception as exc:
        raise click.ClickException(f"Failed to import training code: {exc}") \
            from exc

    fit(config, ckpt_path, cli_args)

@model.command("test", help="Test the model")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the config file",
)
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to model checkpoint",
)
@click.option(
    "--gpu",
    type=int,
    default=0,
    show_default=True,
    help=f"GPU to use, zero-based index. Set to -1 to use CPU."\
        "CPU is also always used if CUDA is not available.",
)
@click.option(
    "--iou_threshold",
    "-iou",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.5,
    show_default=True,
    help="IoU threshold for matching predictions to ground truths",
)
@click.option(
    "--out",
    "-o",
    type=click.Path(exists=False),
    default="metrics.json",
    show_default=True,
    help="Output file for metrics",
)
# @click.option(
#     "--temporal_options",
#     "-t",
#     type=click.Choice(TEMPORAL_OPTIONS),
#     default="stacked",
#     show_default=True,
#     help="Temporal option",
# )
def model_test(config, model, gpu, iou_threshold, out):
    from .compiler import test

    test(config, model, gpu, iou_threshold, out)

if __name__ == "__main__":
    ftw_ma()
