'''
flipnslide.

This is initialized with the following modules:
    - .preproc     : Preprocess large image data for tiling
    - .flipnslide  : Overlap tiling, with specific non-redundant transforms added in
    - .postproc    : Postprocess tiles into PyTorch or Tensorflow datasets
    - .viz         : Visualization tools for checking on data
    - .util        : Package utilities
'''

__version__ = '0.0.1'
__all__ = ['preproc', 'flipnslide', 'postproc', 'viz', 'util']
__author__ = 'elliesch <ellianna@berkeley.edu>'
__minimum_python_version__ = '3.7'

from .preproc import *
from .flipnslide import *
from .postproc import *
from .viz import *
from .util import *