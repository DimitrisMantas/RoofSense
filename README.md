![](docs/logo.png)

[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/DimitrisMantas/RoofSense)
[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg)](https://huggingface.co/datasets/DimitrisMantas/RoofSense)

<h2>Table of Contents</h2>

<!-- TOC -->
  * [Installation](#installation)
    * [End Users](#end-users)
    * [Developers & Researchers](#developers--researchers)
  * [Documentation](#documentation)
    * [Dataset Generation](#dataset-generation)
      * [HuggingFace Hub](#huggingface-hub)
      * [Direct Import & Export](#direct-import--export)
    * [Training & Hyperparameter Optimisation (HPO)](#training--hyperparameter-optimisation-hpo)
      * [Training](#training)
      * [HPO](#hpo)
    * [Testing](#testing)
    * [Inference](#inference)
      * [Tiled Inference](#tiled-inference)
    * [Reproducibility](#reproducibility)
  * [Citation](#citation)
  * [Contributing](#contributing)
<!-- TOC -->

## Installation

### End Users

To install RoofSense for end use on your local machine, first clone this repository, then navigate to its root
directory, and finally execute the following command:

```txt
pip install .
```

on your terminal of choice.

Note that only Windows platforms are currently supported.

### Developers & Researchers

To install RoofSense for development use on your local machine, first clone this repository, then navigate to its root
directory, and finally execute the following command:

```txt
pip install -e .[dev]
```

This will install an editable version of RoofSense along with any additional development requirements.
It is always recommended that you install the program on a separate virtual environment.

## Documentation

The various functions of RoofSense are presented in relevant [implementation examples](implementation).

As providing in-depth documentation is a work in progress, it is possible that certain, and perhaps auxiliary, functions
do not correspond to an example, yet.
In this case, you are welcome to request or add examples as needed, according to
our [Contribution Guide](CONTRIBUTING.md).

### Dataset Generation

#### HuggingFace Hub

The recommended way to obtain RoofSense is by [cloning it](https://huggingface.co/datasets/DimitrisMantas/RoofSense)
from HuggingFace Hub.

The ground truth masks are provided in both raster and COCO JSON format.
Although our implementation requires the former, the latter can be used to import the existing annotations to the
annotation platform of your choice and edit or extend it, as required.
Note that the COCO annotations have been automatically generated
using [Roboflow Annotate](https://roboflow.com/annotate), which uses only three decimal places to define the
corresponding polygon vertices.
Hence, there may be slight differences between the two dataset versions.

#### Direct Import & Export

To generate RoofSense from scratch, execute

```txt
cd roofsense
python ./main.py
```

without modification on your terminal of choice.
Note that an active internet connection is required to initially download the required source datasets.
Furthermore, there may be minor differences in the RGB component of some images due to upstream changes made by Het Waterschapshuis between the initial release of RoofSense and this repository.
In any case, these differences are minor and not expected to significantly influence any experimental results obtained using the original model and dataset.

This script will generate the image component of RoofSense in the current user's home directory.
The dataset directory should initially have the following structure:

```text
C:\Users\<USERNAME>\.roofsense\<3DBAG_VERSION>/
├─ dataset/
├─ chips/
│  ├─ <TILE_ID>_<ROW>_<COLUMN>.png
│  ├─ ...
├─ images/
│  ├─ <TILE_ID>_<ROW>_<COLUMN>.tif
│  ├─ ...
```

Note that the working 3DBAG version is stored in the `BAG3DTileStore().version` attribute, which defaults to
`'2024.02.28'`.

The `images` subfolder contains the actual images meant for downstream tasks.
On the other hand, the `chips` subfolder contains the RGB component of each original image and is meant to be used for
annotation purposes.
Image names contain three components separated by underscores.
The first component represents the ID of the corresponding 3DBAG tile, and the latter two the row and column of the
parent image from which the image was extracted.

Once constructed, the corresponding ground truth masks must be exported in raster format and imported into RoofSense.
This process must be performed by subclassing and using an `AnnotationImporter, as required.
Currently, only [Roboflow Annotate](https://roboflow.com/annotate) is supported.
To learn how the import process works on a high level, follow the
relevant [import example](implementation/annotation/importers.py).

Once the masks have been imported, the dataset directory should have the following structure:

```text
C:\Users\<USERNAME>\.roofsense\<3DBAG_VERSION>/
├─ dataset/
│  ├─ colors.clr
│  ├─ colors.json
│  ├─ counts.json
│  ├─ names.json
│  ├─ scales.bin
│  ├─ splits.json
│  ├─ weights.bin
├─ chips/
│  ├─ <TILE_ID>_<ROW>_<COLUMN>.png
│  ├─ ...
├─ images/
│  ├─ <TILE_ID>_<ROW>_<COLUMN>.tif
│  ├─ ...
├─ masks/
│  ├─ <TILE_ID>_<ROW>_<COLUMN>.tif
│  ├─ ...
```

In addition to the masks, the import process generates the following auxiliary files:

- `colors.*`: Contains the class colors for use in QGIS (`colors.clr`) and by `TrainingDataset` (`colors.json`) for
  logging and visualisation purposes.
- `counts.json`: Contains the class pixel counts per mask in an $N \times C$ matrix format, where $N$ is the total
  number of masks (300) and $C$ the represented classes, excluding the background (8), for use
  by `AnnotationImporter` to split the dataset and compute class weights.
- `names.json`: Contains the class names for use by `TrainingDataset` for logging and visualisation purposes.
- `scales.bin`: A NumPy binary file containing the band ranges of the images in the training set for use by
  `TiledInferenceEngine` and `TrainingDataModule` to scale the input images in downstream tasks.
- `splits.json`: Contains the image names comprising the training, validation, and test splits for use by
  `TrainingDataset` to index and load the appropriate image-mask pairs when queried.
- `weights.bin`: Contains the class weights, including the background (0), for use by `TrainingTask` to set up the loss
  function.

Note that, although most of these files are provided in a human-readable format, with the sole exception of
`colors.clr`, they are meant for internal use, and should hence not be edited manually.
If these files become corrupted, which can lead to undefined behaviour, they can be regenerated by reimporting the
masks.

### Training & Hyperparameter Optimisation (HPO)

#### Training

To train the model presented in our [paper](#citation), follow the
relevant [training example](implementation/training/baseline.py).

For convenience, `TrainingTaskHyperparameterConfig` can be used to set up the model and certain training parameters in a
concise and reproducible fashion.
The default arguments of this data class correspond to an internal baseline.
The training protocol presented in our [paper](#citation) is given in the following table:

| Parameter                   | Value    |
|:----------------------------|:---------|
| Batch Size                  | 8        |
| Epochs                      | 400      |
| Encoder Dropout             | 3.024%   |
| Encoder Stochastic Depth    | 43.23%   |
| Optimiser                   | AdamW    |
| Base Learning Rate (LR)     | 1.245e-4 |
| Squared Gradient Decay Rate | 93.20%   |
| Weight Decay                | 9.114e-3 |
| LR Annealing Schedule       | Cosine   |
| LR Warmup Duration          | 3 Epochs |

and corresponds to the following configuration:

```python
TrainingTaskHyperparameterConfig(
    append_lab=True,
    encoder="tu-resnet-18d",
    global_pool="avgmax",
    aa_layer=True,
    drop_rate=0.030244232449387346,
    zero_init_last=True,
    attn_layer='eca',
    ecoder_atrous_rate1=20,
    decoder_atrous_rate2=15,
    decoder_atrous_rate3=6,
    label_smoothing=0.1,
    optimizer='AdamW',
    lr=0.00012454686917577712,
    beta2=0.9319903315454402,
    weight_decay=0.009113815580614238,
    lr_scheduler='CosineAnnealingLR',
    warmup_epochs=3
)
```

#### HPO

To perform HPO and derive the reported training protocol from the baseline, follow the
relevant [optimisation example](implementation/training/optimization/train.py).

Note that this script requires Optuna and Optuna Integration to execute.

### Testing

To test the model presented in our [paper](#citation), follow the
relevant [testing example](implementation/training/test.ipynb).

Note that, although extensive measures have been taken to minimise randomness, the relatively small size of RoofSense
means that test results using a retrained model, even with the same training protocol, may be slightly different from
those reported.

### Inference

If you are only interested in using RoofSense for inference on your own data, follow the
relevant [inference example](implementation/inference/simple.ipynb).
This example will automatically download the required
standalone [model weights](https://huggingface.co/DimitrisMantas/RoofSense) from HuggingFace Hub and shows how to run
inference on a single image and save the corresponding predictions to a file.

The provided input augmentations are in accordance with our [paper](#citation).
In order to construct them, `scales.bin`, a NumPy binary file containing the band ranges of the images in the training
set is required.
You can obtain this file by either [cloning](https://huggingface.co/datasets/DimitrisMantas/RoofSense)
or [generating](#direct-import--export) RoofSense.
Alternatively, these values are given in the following table:

| Band Index | Band Name   | Min. Value | Max. Value |
|:-----------|-------------|-----------:|-----------:|
| 1          | Red         |     0.0000 |   254.0000 |
| 2          | Green       |     0.0000 |   254.0000 |
| 3          | Blue        |     0.0000 |   254.0000 |
| 4          | Reflectance |     0.0000 |     1.0000 |
| 5          | Slope       |     0.0000 |    89.4805 |
| 6          | nDRM        |    -5.8354 |    51.1758 |
| 7          | Density     |     0.0000 |     2.1533 |

#### Tiled Inference

### Reproducibility

To reproduce the experimental results of our [paper](#citation), execute the desired examples without modification.

[//]: # (TODO: Note that results may differ.)

[//]: # (TODO: Note that we also provide the training checkpoint.)


[//]: # (- To run the ablation study)

## Citation

If you use this software in your work, please cite the
following [thesis](https://resolver.tudelft.nl/uuid:c463e920-61e6-40c5-89e9-25354fadf549).

```bibtex
@MastersThesis{Mantas2024,
    author      = {Mantas, Dimitris},
    school      = {Delft University of Technology},
    title       = {{CNN-based Roofing Material Segmentation using Aerial Imagery and LiDAR Data Fusion}},
    year        = {2024},
    address     = {Delft, The Netherlands},
    month       = oct,
    type        = {mathesis},
    date        = {2024-10-31},
    institution = {Delft University of Technology},
    location    = {Delft, The Netherlands},
    url         = {https://resolver.tudelft.nl/uuid:c463e920-61e6-40c5-89e9-25354fadf549},
}

```

## Contributing

RoofSense welcomes contributions and suggestions via GitHub pull requests.
If you would like to submit a pull request, please see our [Contribution Guide](CONTRIBUTING.md) for more information.

