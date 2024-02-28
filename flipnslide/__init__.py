'''
flipnslide.

This is initialized with the following modules:
    - .ingest      : Import large data image in location and preprocess
    - .tiling      : Overlap tiling, with specific non-redundant transforms added in
    - .dataset     : Postprocess tiles into PyTorch or Tensorflow datasets
    - .viz         : Visualization tools for checking on data
    - .util        : Package utilities
'''

__version__ = '0.0.1'
__all__ = ['ingest', 'tiling', 'dataset', 'viz', 'util']
__author__ = 'elliesch <ellianna@berkeley.edu>'
__minimum_python_version__ = '3.8'

from .ingest import *
from .tiling import *
from .dataset import *
from .viz import *
from .util import *