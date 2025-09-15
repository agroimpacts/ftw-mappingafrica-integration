"""
FTW Mapping Africa Integration

This package contains modules for loading, augmenting, and processing field 
boundary datasets for Africa.
"""
import warnings

# Ignore the specific torch grid_sample/affine_grid align_corners UserWarning
warnings.filterwarnings(
    "ignore",
    message=r".*grid_sample and affine_grid.*align_corners=False.*",
    category=UserWarning,
)

from .dataset import *
from .datamodule import *
from .utils import *
from .losses import *
from .trainers import *
from .cli import *
from .compiler import *
# from .metrics import *
