![](colors.png)

<a href="https://universe.roboflow.com/my-workspace-lg4pq/roofsense">
    <img src="https://app.roboflow.com/images/download-dataset-badge.svg"></img>
</a>

# Installation

## End Users

To install RoofSense for end use on your local machine, clone this repository, navigate to its root directory, and
execute
the following command:

```txt
pip install -r requirements/common.txt
```

on your terminal of choice.
*Please note that only Windows platforms are currently supported.*

## Developers & Contributors

To install RoofSense for development use on your local machine, first follow
the [corresponding end-user procedure](#end-users) and
then install the additionally relevant dependencies:

```txt
pip install -r requirements/dev.txt
```

[//]: # (TODO -  Link to CONTRIBUTING.md.)
If you are planning to contribute to RoofSense, please see the pertinent guidelines.

## Important (Training & Inference)

RoofSense relies on `torchseg` to build its models because `torchgeo`, its other relevant
dependency, uses `segmentation_models_pytorch`, which does not fully support `timm` encoders.
However, this
functionality is required for
specifying additional
encoder configuration parameters, such as stochastic depth and attention.
*Please note that `torchseg` is *not* installed
automatically due dependency clashes between `torchseg` and `torchgeo`* and must hence be
must be installed manually by executing the following command:

```txt
pip install torchseg
```

on your terminal of choice *after* RoofSense has been installed.

# Sample Data

# Mess with the Stack Creation Options

# Import Annotations

# Train

# Test

# Perform HPO

# Infer

# Legal

Copyright © 2024 Dimitris Mantas

Except where otherwise noted, this work is licensed under the [MIT License](LICENSE).