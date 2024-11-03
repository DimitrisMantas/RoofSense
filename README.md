![](docs/logo.png)

![](https://app.roboflow.com/images/download-dataset-badge.svg)

## Installation

### End Users

To install RoofSense for end use on your local machine, first clone this repository, then navigate to its root directory, and finally execute the following command:

```txt
pip install .
```

on your terminal of choice.
*Please note that only Windows platforms are currently supported.*

### Contributors

To install RoofSense for end use on your local machine, first clone this repository, then navigate to its root directory, and finally execute the following command:

```txt
pip install -e .[dev]
```

This will install an editable version of RoofSense along with any additional development requirements.

RoofSense welcomes contributions and suggestions via GitHub pull requests.
If you would like to submit a pull request, please see our [Contribution Guide](CONTRIBUTING.md) for more information.

### Important (Training & Inference)

RoofSense relies on [TorchSeg](https://github.com/isaaccorley/torchseg) to build its models because [TorchGeo](https://github.com/microsoft/torchgeo/blob/main/pyproject.toml), its other relevant dependency, uses [Segmentation Models Pytorch (SMP)](https://github.com/qubvel-org/segmentation_models.pytorch), which does not fully support [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models) encoders.
However, this functionality is required for specifying additional configuration parameters, such as stochastic depth and attention.
*Please note that TorchSeg is *not* installed automatically due dependency clashes between itself and TorchGeo, and must hence be must be installed manually by executing the following command:

```txt
pip install torchseg
```

on your terminal of choice *after* RoofSense has been installed.

## Documentation

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
