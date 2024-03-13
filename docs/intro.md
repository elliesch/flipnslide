# `flipnslide`


Flip-n-Slide is a concise tiling and augmentation strategy to prepare large 
scientific images for use with GPU-enabled algorithms. The Flip-n-Slide
approach [@abrahams_2024] preserves multiple views of local semantic context
within a large image by creating multiple nodded tile overlaps for a given 
pixel position within the original large image. To eliminate the redundancies
within the dataset that are created by simply overlapping tiles, Flip-n-Slide
assigns a distinct, non-commutative matrix permutation to each overlapped 
tile, thereby providing a unique view of each contextual slice to any 
convolutional processing downstream. 


`flipnslide` is a Python package that outputs deep learning-ready preprocessed 
tiled datasets that follow the Flip-n-Slide strategy from a single large 
scientific image. The package is flexible, providing tiled outputs as `numpy` 
arrays [@harris_2020], `PyTorch` tensors [@paszke_2019], or `Tensorflow` tensors 
[@abadi 2015] depending on user preference,allowing it to be efficiently slotted into 
existing machine learning pipelines. `flipnslide` allows the user to select preferred 
tile size and save modes, and it can be implemented on existing data or used to 
download and tile any Earth Observation datasets from Planetary Computer 
[@microsoftopensource_2022].


`flipnslide` was developed for use with the large satellite images that are used 
in Earth Observation, but we welcome test cases from any scientific discipline where
spatiotemporal matrices are used as inputs to deep learning. Please let us know here 
you found it to be helpful in your machine learning pipeline, and we'll include
a link to your project on the [Projects page]()! 

This package is being actively developed in 
[a public repository on GitHub](https://github.com/elliesch/flipnslide), 
and we welcome new contributions. No contribution is too small, so if you encounter any
difficulties with this code, have questions about its use, find a typo, or have 
requests for new content (examples or features), please 
[open an issue on GitHub](https://github.com/elliesch/flipnslide/issues).


links to installation, easy_example, advanced_example, and contribute go here once those files exist in the jupyter book


## Contributors

Please see [the authors list](https://github.com/elliesch/flipnslide/blob/main/AUTHORS.md).


## Citation and Attribution

If you make use of this algorithm, please cite the ICLR 2024 paper:

    @inproceedings{flipnslide,
      author       = {Ellianna Abrahams and
                      Tasha Snow and
                      Matthew R. Siegfried and
                      Fernando Pérez},
      title        = {A Concise Tiling Strategy for Preserving Spatial Context in Earth Observation Imagery},
      booktitle    = {Machine Learning for Remote Sensing Workshop {ML4RS} at The Twelfth International Conference 
                      on Learning Representations, {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
      publisher    = {OpenReview.net},
      year         = {2024},
      url          = {upcoming},
    }