# Flip-n-Slide

Flip-n-Slide is a concise tiling and augmentation strategy to prepare large scientific images for use with GPU-enabled algorithms. `flipnslide` is a Python package that outputs PyTorch-ready preprocessed datasets from a single large image.

## Documentation

The documentation for `flipnside` is available on [Read the Docs](https://flipnslide.readthedocs.io/).

## Installation and Dependencies

For now, `flipnslide` can be installed from PyPI using pip, by running:

```bash
pip install flipnslide
```
Check back later for instructions on installing from conda forge.

## Attribution

If you make use of this code, please cite the [companion conference paper](https://arxiv.org/abs/2404.10927) from *ML4RS @ ICLR 2024* that initially presents the algorithmic methods behind this implementation:

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

## License

Copyright 2024 Ellianna Abrahams, Tasha Snow, Matthew R. Siegfried, Fernando Pérez, and contributors.

``flipnslide`` is free software made available under the MIT License. For details see
the [LICENSE](https://github.com/elliesch/flipnslide/blob/main/LICENSE) file.

## Contributors

See the [AUTHORS](https://github.com/elliesch/flipnslide/blob/main/AUTHORS.md) file for a complete list of contributors to the project.
