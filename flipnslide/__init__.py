'''
flipnslide.

This is initialized with the following modules:
    - .pre         : Preprocess large image data for tiling
    - .tiling      : Overlap tiling, with specific non-redundant transforms added in
    - .post        : Postprocess tiles into PyTorch or Tensorflow datasets
    - .viz         : Visualization tools for checking on data
    - .util        : Package utilities
'''

__version__ = '0.0.1'
__all__ = ['pre', 'tiling', 'post', 'viz', 'util']
__author__ = 'elliesch <ellianna@berkeley.edu>'
__minimum_python_version__ = '3.7'

from .pre import *
from .tiling import *
from .post import *
from .viz import *
from .util import *