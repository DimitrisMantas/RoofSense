<h1>RoofSense</h1>

![](docs/logo.png)

[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/DimitrisMantas/RoofSense)
[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg)](https://huggingface.co/datasets/DimitrisMantas/RoofSense)

<h2>Table of Contents</h2>

<!-- TOC -->
  * [Installation](#installation)
    * [End Users](#end-users)
    * [Developers](#developers)
  * [Documentation](#documentation)
    * [End Use](#end-use)
    * [Reproducibility](#reproducibility)
  * [Citation](#citation)
  * [Contributing](#contributing)
<!-- TOC -->

## Installation

### End Users

To install RoofSense for end use on your local machine, first clone this repository, then navigate to its root directory, and finally execute the following command:

```txt
pip install .
```

on your terminal of choice.

*Note that only Windows platforms are currently supported.*

### Developers

To install RoofSense for development use on your local machine, first clone this repository, then navigate to its root directory, and finally execute the following command:

```txt
pip install -e .[dev]
```

This will install an editable version of RoofSense along with any additional development requirements.
It is always recommended that you install the program on a separate virtual environment.

## Documentation

The various functions of RoofSense are presented in relevant [implementation examples](implementation).

### End Use

If you are only interested in using RoofSense for inference on your own data, simply download the default model [checkpoint](https://huggingface.co/DimitrisMantas/RoofSense) and then follow the relevant inference example.

### Reproducibility

To reproduce the experimental results presented in the work of Mantas [(2014)](#citation), simply execute the desired examples without modification.

[//]: # ([//]: # &#40;TODO: Fill this in.&#41;)
[//]: # (- To generate the dataset,)

[//]: # (- To train the model,)

[//]: # (- To perform hyperparameter optimisation)

[//]: # (- To test the model)

[//]: # (- To test the generalised perf)

[//]: # (- To run the ablation study)

[//]: # (- To use the model for inference)

## Citation
If you use this software in your work, please cite the following [thesis](https://resolver.tudelft.nl/uuid:c463e920-61e6-40c5-89e9-25354fadf549).

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

