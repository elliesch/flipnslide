# `flipnslide`


Flip-n-Slide is a concise tiling and augmentation strategy to prepare large 
scientific images for use with GPU-enabled algorithms. The Flip-n-Slide
approach {cite}`abrahams_2024` preserves multiple views of local semantic context
within a large image by creating multiple nodded tile overlaps for a given 
pixel position within the original large image. To eliminate the redundancies
within the dataset that are created by simply overlapping tiles, Flip-n-Slide
assigns a distinct, non-commutative matrix permutation to each overlapped 
tile, thereby providing a unique view of each contextual slice to any 
convolutional processing downstream. 


`flipnslide` is a Python package that outputs deep learning-ready preprocessed 
tiled datasets that follow the Flip-n-Slide strategy from a single large 
scientific image. The package is flexible, providing tiled outputs as `numpy` 
arrays {cite}`harris_2020`, `PyTorch` tensors {cite}`paszke_2019`, or `Tensorflow` tensors 
{cite}`abadi_2015` depending on user preference, allowing it to be efficiently slotted into 
existing machine learning pipelines. `flipnslide` allows the user to select preferred 
tile size and save modes, and it can be implemented on existing data or used to 
download and tile any Earth science datasets from Planetary Computer 
{cite}`microsoftopensource_2022`.


`flipnslide` was developed for use with the large satellite images that are used 
in Earth observation, but we welcome test cases from any scientific discipline where
spatiotemporal matrices are used as inputs to deep learning. Please let us know here 
you found it to be helpful in your machine learning pipeline, and we will include
a link to your project on the [Projects page]()! 


This package is being actively developed in 
[a public repository on GitHub](https://github.com/elliesch/flipnslide), 
and we welcome new contributions. No contribution is too small, so if you encounter any
difficulties with this code, have questions about its use, find a typo, or have 
requests for new content (examples or features), please 
[open an issue on GitHub](https://github.com/elliesch/flipnslide/issues).


## Contributors

Please see [the authors list](https://github.com/elliesch/flipnslide/blob/main/AUTHORS.md).


## Citation and Attribution

If you make use of this code, please cite the companion conference paper from
*ML4RS @ ICLR 2024* that initially presented this strategy for Earth-observing 
remote sensing data:

    @inproceedings{flipnslide,
      author       = {Ellianna Abrahams and
                      Tasha Snow and
                      Matthew R. Siegfried and
                      Fernando Pérez},
      title        = {A Concise Tiling Strategy for Preserving Spatial Context in Earth Observation Imagery},
      booktitle    = {Machine Learning for Remote Sensing Workshop {ML4RS} at The Twelfth International Conference 
                      on Learning Representations, {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
      doi          = {10.48550/arXiv.2404.10927},
      year         = {2024},
      month        = may,
    }

The testing and development of the `flipnslide` codebase was done on the NASA supported CryoCloud cloud hub {cite}`snow_2023`, a NASA Earth Science Directorate funded project (grant numbers 80NSSC22K1877 and 80NSSC23K0002) and on the Jupyter Meets the Earth (JMTE) cloud hub, an NSF EarthCube funded project (grant numbers 1928406 and 1928374).

