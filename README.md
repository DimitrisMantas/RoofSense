![](docs/logo.png)

![](https://app.roboflow.com/images/download-dataset-badge.svg)

<h2>Table of Contents</h2>

<!-- TOC -->
  * [Installation](#installation)
    * [End Users](#end-users)
    * [Developers](#developers)
    * [Important (Training & Inference)](#important-training--inference)
  * [Documentation](#documentation)
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

### Important (Training & Inference)

RoofSense relies on [TorchSeg](https://github.com/isaaccorley/torchseg) to build its models because [TorchGeo](https://github.com/microsoft/torchgeo/blob/main/pyproject.toml), its other relevant dependency, uses [Segmentation Models Pytorch (SMP)](https://github.com/qubvel-org/segmentation_models.pytorch), which does not fully support [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models) encoders.
However, this functionality is required for specifying additional configuration parameters, such as stochastic depth and attention.

Note that TorchSeg is *not* installed automatically due to dependency clashes between itself and TorchGeo, and must hence be must be installed manually by executing the following command:

```txt
pip install torchseg
```

on your terminal of choice *after* RoofSense has been installed.

## Documentation

The various functions of RoofSense are presented in relevant [implementation examples](implementation).

### Reproducibility

To reproduce the experimental results presented in Mantas ([2024](#citation)), simply execute the desired examples without modification.

[//]: # ([//]: # &#40;TODO: Fill this in.&#41;)
[//]: # (- To generate the dataset,)

[//]: # (- To train the model,)

[//]: # (- To perform hyperparameter optimisation)

[//]: # (- To test the model)

[//]: # (- To test the generalised perf)

[//]: # (- To run the ablation study)

[//]: # (- To use the model for inference)

## Citation
If you use this software in your work, please cite the following work:

```bibtex
@article{Mantas2024,
   author = {Dimitris Mantas},
   city = {Delft, The Netherlands},
   institution = {Delft University of Technology},
   month = {10},
   title = {CNN-based Roofing Material Segmentation using Aerial Imagery and LiDAR Data Fusion},
   year = {2024},
}
```

## Contributing

RoofSense welcomes contributions and suggestions via GitHub pull requests.
If you would like to submit a pull request, please see our [Contribution Guide](CONTRIBUTING.md) for more information.

