---
title: 'flipnslide: A Python Package for for Preserving Spatial Context when Preprocessing Large Scientific Images'
tags:
  - Python
  - Tiling
  - Augmentation
  - Deep Learning
  - Earth Observation
authors:
  - name: Ellianna Abrahams
    orcid: 0000-0002-9879-1183
    equal-contrib: false
    affiliation: 1
    corresponding: true
  - name: Tasha Snow
    orcid: 0000-0001-5697-5470
    equal-contrib: false
    affiliation: 2
  - name: Matthew R. Siegfried
    orcid: 
    equal-contrib: false
    affiliation: 2
  - name: Fernando Pérez
    orcid: 
    equal-contrib: false
    affiliation: 1
affiliations:
 - name: Department of Statistics, University of California, Berkeley, Berkeley CA 94720, USA
   index: 1
 - name: Department of Geophysics, Colorado School of Mines, Golden CO 80401, USA
   index: 1
date: 15 March 2023
bibliography: paper.bib

---

#

# Summary

`flipnslide` is an open-source package that provides the python implementation of the Flip-n-Slide 
algorithm [@abrahams_2024] for simultaneously tiling and augmenting large, scientfic images in preparation 
for use with GPUs. Due to the onboard memory limitations of GPUs, large images need to be sliced into 
tiles in preparation for computer vision approaches that are built on deep learning architectures. When 
automated, these slices can accidentally remove important local spatial and temporal context. Traditionally 
this is avoided by sliding a tiling window across multiple overlaps at a single pixel location, but this 
creates redundancies within the data that alter the structure of the underlying training dataset. To 
eliminate these redundancies, Flip-n-Slide simultaneously tiles and augments each overlapped tile with a 
unique, physically-realistic permutation. This not only removes redundancies, but extends the training 
dataset without altering its true distribution, improving model accuracy in cases of class imbalance. 
`flipnslide` provides user-friendly access to this tiling approach, allowing the user to provide an input 
image or to download one, preprocess it, and create a machine learning ready dataset of augmented tiles at 
whatever size is needed all from a single line of code. While Flip-n-Slide is the first tiling strategy 
created to preserve context without creating redundancies, this package includes additional tiling strategies, 
allowing users to easily preprocess large scientific images and compare efficiency across different tiling 
approaches, enabling ease of use in ablation studies. `flipnslide` is designed for seamless integration into 
existing machine learning pipelines in Python, outputting data in arrays, tensors, or directly streamable 
dataloaders, depending on the needs of the user. 

#

# Statement of need

Given the growing influx of geospatial satellite imagery in recent decades, deep learning presents a promising 
opportunity for rapidly parsing meaningful scientific understanding from these images. Despite the remarkable 
accomplishments of deep neural networks in various vision classification tasks [@ronneberger_2015a; @zhao_2017; 
@chen_2018; @chen_2019; @tan_2020; @amara_2022], these methods can underperform on data that have noisy 
or underrepresented labels [@shin_2011; @guo_2019] or when one set of data representations is used for a wider 
set of downstream tasks [@yang_2018]. These are common challenges in Earth observation imagery. To overcome 
these issues, data augmentation is a widely adopted technique for generalizing a model fit to make better 
predictions by expanding the size and distribution of training data through a set of transformations 
[@vandyk_2001; @hestness_2017]. In recent years, much focus has been given to upstream augmentation methods that 
address overfitting through data mixing (such as [@zhang_2017; @yun_2019; @hong_2021; @baek_2021]) or proxy-free 
augmentations (such as [@cubuk_2019; @reed_2021; @li_2023]), both strategic approaches that expand the training 
data, but also create unrealistic transformations of the data. Furthermore, limited attention has been given to 
investigating the downstream impacts of upstream augmentation techniques on tiled imagery, an approach often 
employed to parse scientific imaging into smaller tiles to overcome the intractable size of the overall image 
for the GPU memory [@pinckaers_2018; @huang_2019]. 

Tiling is not only necessary for images that are larger than GPU memory [@ronneberger_2015a], but has also been 
shown to improve small object detection [@unel_2019]. In image classification, spatial context is needed to create 
greater distance at the latent level between classes with similar channel outputs and surface textures [@pereira_2021]. 
[@ronneberger_2015a] initially proposed a strategy to overlap tiles at test time, with a fixed size and overlap 
threshold, in order to avoid losing important spatial context for smaller features. This approach has largely remained 
the accepted convention since (with a typical overlap of 50%) and has been incorporated at training time as well 
[@unel_2019; @zeng_2019; @reina_2020; @akyon_2022]. However simply overlapping tiles has two drawbacks: overlapping tiles 
introduce redundancies at training time, as many pixel windows are repeated more than once, leading to overfitting and 
tiling can slice through objects which removes context from the slices at training and test time.

[@charette_2021] addresses issues with object slicing by combining objects into a super-object in post-processing when 
a first learning iteration indicates that adjacent tiles contain the same object. [@nguyen_2023] furthers this approach 
with a dynamic tiling routine that employs an iterative process to detect a dynamic size to assign to 
object-of-interest (OoI) patches after pre-learning on fixed, non-overlapping tiles, however this creates an initial 
training stage for every test image, creating large overhead. Although both of these approaches show promise in preserving 
local context when tiling an image, they require minimal degeneracy between classes so that separate objects in neighboring 
tiles are not mistakenly assigned as the same object. In large scale satellite imagery, especially in remote geophysical 
settings where classes of interest are often poorly characterized, small, similar objects can be adjacent within tolerance 
leading to mistaken combinations, creating issues of scale.

In a companion paper submitted to the Machine Learning for Remote Sensing (ML4RS) workshop at ICLR [@abrahams_2024], we 
present the Flip-n-Slide approach as a solution for creating image tiles that preserve local spatial contexts while 
eliminating overlap redundancies. Like earlier approaches, Flip-n-Slide uses overlapping tiles to retain spatial context 
around OoI, but addresses the issue of redundancy by applying a distinct transformation permutation to each overlapping tile. 
In this way, Flip-n-Slide avoids the overhead of recent tile combination approaches, which provide drawbacks at scale in 
large, real-world imagery where nearby degenerate classes could be too easily combined into a false super object in 
classification tasks. [Table 1](#tab1) highlights features of Flip-n-Slide as compared to previous methods.

| Method                             | DTS        | NPN        | IDS        | FCV        | NDR        |
|------------------------------------|------------|------------|------------|------------|------------|
| 50% Overlap <br> [@unel_2019, @zeng_2019, @reina_2020, @akyon_2022] |            | &#x2713;   | &#x2713;   | &#x2713;   |            |
| Tile Stitching <br> [@charette_2021] |            | &#x2713;   |            |            | &#x2713;   |
| Dynamic Tiling <br> [@nguyen_2023] | &#x2713;   |            |            |            | &#x2713;   |
| **Flip-n-Slide** <br> [@abrahams_2024] |            | &#x2713;   | &#x2713;   | &#x2713;   | &#x2713;   |

<a name="tab1"></a> *Table 1*: Comparison of other recent input tiling methods with the one presented in this paper. The 
column abbreviations are explained here: **DTS**, dynamic tile size. **NPN**, no pre-training necessary to determine tile 
size. **IDS**, increases data samples for training. **FCV**, full spatio-contextual view preserved. **NDR**, no data 
redundancies in overlapping tiles.

Flip-n-Slide is a concise tiling and augmentation strategy, built intentionally for use with large, scientific images where: 
One) tiling is necessary; Two) data transformations must be limited to rotations and reflections to be realistic; and Three) there is
no prior knowledge of the pixel locations for which spatial context will be necessary. Physically realistic transformations
of the data are implemented *alongside* the tiling overlap process, thereby minimizing redundancy when training convolutional
neural networks (CNNs), in which orientation matters for learning [@ghosh_2018; @szeliski_2022]. This strategy naturally
creates a larger set of samples without the superfluity of simply overlapping the tiles, leading to enhanced downstream model 
generalization. To achieve this goal, the algorithm first slides through multiple overlaps of the tiling window, exposing
the model to more contextual views of each location ([Figure 1](#fig1)). Each overlapping window is then distinctly
permuted to remove redundancies with other tiles that share pixels. In the companion paper, [@abrahams_2024], we demonstrated
the power of this approach to increase accuracy in vision classification tasks, particularly in cases of underrepresentation.
Here we present the open-source Python package, `flipnslide`, which seamlessly integrates into machine-learning pipelines
in Scikit-learn [@pedregosa_2011], PyTorch [@paszke_2019] and Tensorflow [@abadi_2015], making this method accessible and
easy to use in existing and new vision classification analyses. 

![<a name="fig1"></a>Figure 1. Flip-n-Slide's tile overlap strategy creates eight overlapping tiles for any image region more than a 75% tile threshold away 
from the overall image edge. Three tiling strategies, shown in false color to illustrate overlap, are visualized here. a) Tiles 
do not overlap. b) The conventional tile overlap strategy, shown at the recommended 50% overlap. c) Flip-n-Slide includes more 
tile overlaps, capturing more OoI tile position views for the training set.](figures/overlap_strategy.pdf)

#

# Implementing `flipnslide`

The Python package `flipnslide` can be installed from [PyPI](https://pypi.org/project/flipnslide/) using `pip` or from the 
source code, available on [GitHub](https://github.com/elliesch/flipnslide). The code is well documented on 
[Read the Docs](http://flipnslide.readthedocs.io), including examples for using `flipnslide` to create custom tiled datasets 
from large imagery. Users can choose to run `flipnslide` to download Earth Observation imagery from Planetary Computer (cite). 
Using this route, the code creates a time-averaged satellite image, removing NaNs by interpolation and standardizing the averaged 
image. The image is then simultaneously tiled and augmented, and the output tiles can be held in memory or saved to disk as either 
arrays (`numpy`) or tensors (`PyTorch` or `Tensorflow`). If a user already has an existing image, `flipnslide` can be implemented 
to directly tile and augment the pre-downloaded image as well, and just as before, the output tiles can be held in memory or saved 
to disk as either arrays or tensors. Our goal in writing this codebase was to enable a user to modularly drop this tiling method 
into any pre-existing machine learning pipeline in Python, and so either of these pathways can be implemented from a single line 
of code, using the same overall class. 

Ablation studies are common cross-validation technique in deep learning and computer vision analyses. To enable users to test 
the ablation of their preprocessing approach, we also provide two other tiling methods with low overhead in this codebase: one with 
no tile overlap and one that follows the previous convention, simply tiling the image with a 50% overlap window but no simultaneous augmentation [@ronneberger_2015a]. Each of these tiling approaches can also be implemented from a single line of code, further 
enabling ease-of-use in existing machine learning pipelines.

We include a verbose flag for all of these classes which is especially useful when using `flipnslide` to download, preprocess, 
then tile and augment an image. We have optimized this approach to be used with large imagery and large images can take some 
time to download, especially when employed on machines that are not server-side to the stored cloud images. The verbose flag 
not only prints out arrivals at each stage of the algorithmic pipeline, but it also provides visualizations showing the overall 
downloaded image, the image after standardization and NaN inpainting, and a selection of the final tiles. We show an example of 
this in [Figure 2](#fig2).

Since we anticipate that any output tiled datasets will be used with a GPU, we also provide useful one-line methods for creating 
a `PyTorch` dataset and streamable dataloader from `flipnslide` tiles within our codebase. The goal of this is to enable users 
who are new to working with GPU-enabled datasets to be able to drop their data directly into `flipnslide` and within a few lines 
of code arrive at a product that allows a user to begin testing and building ML architectures without needing to spend extra time 
on preprocessing stages.

![<a name="fig2"></a>Figure 2. The core function of the `flipnslide` codebase tiles a large scientific image either via download (with input coordinates and a time range) or from existing data (as an input image). It tiles in several steps: 1) Standardizing the image and removing NaNs. 2) Cropping the image to a size divisible by the tile size. 3) Tiling the image in one of three approaches. Here we feature tiles created using the Flip-n-Slide algorithm for preserving spatial context in image tiles [].](figures/output.jpg)

#

# Acknowledgements

The majority of this work was conducted on the unceded territories of the xučyun, the ancestral land of the Chochenyo speaking 
Muwekma Ohlone people. We have benefited, and continue to benefit, from the use of this land. In this acknowledgement, we recognize 
the importance of taking actions in support of American Indian and Indigenous peoples who are living, flourishing members of our 
communities today. 

The authors are grateful to ER and HK for helpful discussions on this method. EA acknowledges support from a Two Sigma PhD Fellowship. 
The testing and development of this codebase was done on the NASA supported CryoCloud cloud hub [@snow_2023] and on the Jupyter Meets 
the Earth (JMTE) cloud hub, an NSF EarthCube funded project (grant nos. 1928406 and 1928374). 

#

# References
