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

if __name__ == "__main__":
    ftw_ma()
