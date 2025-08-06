<!---
SPDX-FileCopyrightText: Copyright 2022-2025, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->
# ML Inference Advisor - Introduction

The ML Inference Advisor (MLIA) helps AI developers design and optimize
neural network models for efficient inference on Arm® targets (see
[supported targets](#target-profiles)). MLIA provides
insights on how the ML model will perform on Arm early in the model
development cycle. By passing a model file and specifying an Arm hardware target,
users get an overview of possible areas of improvement and actionable advice.
The advice can cover operator compatibility, performance analysis and model
optimization (e.g. pruning and clustering). With the ML Inference Advisor,
we aim to make the Arm ML IP accessible to developers at all levels of abstraction,
with differing knowledge on hardware optimization and machine learning.

## Inclusive language commitment

This product conforms to Arm's inclusive language policy and, to the best of
our knowledge, does not contain any non-inclusive language.

If you find something that concerns you, email <terms@arm.com>.

## Releases

Release notes can be found in [MLIA releases](RELEASES.md).

## Getting support

In case you need support or want to report an issue, give us feedback or
simply ask a question about MLIA, please send an email to <mlia@arm.com>.

Alternatively, use the
[AI and ML forum](https://community.arm.com/support-forums/f/ai-and-ml-forum)
to get support by marking your post with the **MLIA** tag,
or tag the @mlia team directly for assistance.

## Reporting vulnerabilities

Information on reporting security issues can be found in
[Reporting vulnerabilities](SECURITY.md).

## License

ML Inference Advisor is licensed under [Apache License 2.0](LICENSES/Apache-2.0.txt)
unless otherwise indicated. This project contains software under a range of
permissive licenses, see [LICENSES](LICENSES/).

## Trademarks and copyrights

* Arm®, Arm® Ethos™-U, Arm® Cortex®-A, Arm® Cortex®-M, Arm® Corstone™ are
  registered trademarks or trademarks of Arm® Limited (or its subsidiaries) in
  the U.S. and/or elsewhere.
* TensorFlow™ is a trademark of Google® LLC.
* Keras™ is a trademark by François Chollet.
* Linux® is the registered trademark of Linus Torvalds in the U.S. and
  elsewhere.
* Python® is a registered trademark of the PSF.
* Ubuntu® is a registered trademark of Canonical.
* Microsoft and Windows are trademarks of the Microsoft group of companies.

# General usage

## Prerequisites and dependencies

It is recommended to use a virtual environment for MLIA installation, and a
typical setup requires:

* Ubuntu® 20.04.03 LTS (other OSs may work, the ML Inference Advisor has been
  tested on this one specifically)
* Python® >= 3.8.1
* Ethos™-U Vela dependencies (Linux® only)

  For more details, please refer to the
  [prerequisites of Vela](https://pypi.org/project/ethos-u-vela/).

## Installation

MLIA can be installed with `pip` using the following command:

```bash
pip install mlia
```

It is highly recommended to create a new virtual environment for the installation.

## First steps

After the installation, you can check that MLIA is installed correctly by
opening your terminal, activating the virtual environment and typing the
following command that should print the help text:

```bash
mlia --help
```

The ML Inference Advisor works with sub-commands, i.e. in general a command
would look like this:

```bash
mlia [sub-command] [arguments]
```

Where the following sub-commands are available:

* ["check"](#check): perform compatibility or performance checks on the model
* ["optimize"](#optimize): apply specified optimizations

Detailed help about the different sub-commands can be shown like this:

```bash
mlia [sub-command] --help
```

The following sections go into further detail regarding the usage of MLIA.

# Sub-commands

This section gives an overview of the available sub-commands for MLIA.

## **check**

### compatibility

Lists the model's operators with information about their compatibility with
the specified target.

*Examples:*

```bash
# List operator compatibility with Ethos-U55 with 256 MAC
mlia check ~/models/mobilenet_v1_1.0_224_quant.tflite --target-profile ethos-u55-256

# List operator compatibility with Cortex-A
mlia check ~/models/mobilenet_v1_1.0_224_quant.tflite --target-profile cortex-a

# Get help and further information
mlia check --help
```

### performance

Estimates the model's performance on the specified target and prints out
statistics.

*Examples:*

```bash
# Use default parameters
mlia check ~/models/mobilenet_v1_1.0_224_quant.tflite \
    --target-profile ethos-u55-256 \
    --performance

# Explicitly specify the target profile and backend(s) to use
# with --backend option
mlia check ~/models/ds_cnn_large_fully_quantized_int8.tflite \
    --target-profile ethos-u65-512 \
    --performance \
    --backend "vela" \
    --backend "corstone-300"

# Get help and further information
mlia check --help
```

## **optimize**

This sub-command applies optimizations to a Keras model (.h5 or SavedModel) or
a TensorFlow Lite model and shows the performance improvements compared to
the original unoptimized model.

There are currently three optimization techniques available to apply:

* **pruning**: Sets insignificant model weights to zero until the specified
    sparsity is reached.
* **clustering**: Groups the weights into the specified number of clusters and
    then replaces the weight values with the cluster centroids.

More information about these techniques can be found online in the TensorFlow
documentation, e.g. in the
[TensorFlow model optimization guides](https://www.tensorflow.org/model_optimization/guide).

*Examples:*

```bash
# Custom optimization parameters: pruning=0.6, clustering=16
mlia optimize ~/models/ds_cnn_l.h5 \
    --target-profile ethos-u55-256 \
    --pruning \
    --pruning-target 0.6 \
    --clustering \
    --clustering-target 16

# Get help and further information
mlia optimize --help
```

**Note:** A ***Keras model*** (.h5 or SavedModel) is required as input to
perform pruning and clustering.

## **rewrite**

Replaces certain subgraph/layer of the pre-trained model with candidates from the rewrite library, with or without training using a small portion of the training data, to achieve local performance gains.

The following rewrites are supported:

* fully-connected - replaces a subgraph with a fully connected layer
* fully-connected-sparsity - replaces a subgraph with a pruned M:N sparse fully connected layer
* fully-connected-unstructured-sparsity - replaces a subgraph with an unstructured pruned fully connected layer
* fully-connected-clustering - replaces a subgraph with a clustered fully connected layer
* conv2d - replaces a subgraph with a conv2d layer
* conv2d-sparsity - replaces a subgraph with a pruned M:N sparse conv2d layer
* conv2d-unstructured-sparsity - replaces a subgraph with an unstructured pruned conv2d layer
* conv2d-clustering  - replaces a subgraph with a clustered conv2d layer
* depthwise-separable-conv2d - replaces a subgraph with a depthwise seperable conv2d layer
* depthwise-separable-conv2d-sparsity - replaces a subgraph with a pruned M:N sparse depthwise seperable conv2d layer
* depthwise-separable-conv2d-unstructured-sparsity - replaces a subgraph with an unstructured pruned depthwise seperable conv2d layer
* depthwise-separable-conv2d-clustering - replaces a subgraph with a clustered depthwise seperable conv2d layer

**Note:** A ***TensorFlow Lite model*** is required as input
to perform a rewrite.

*Examples:*

```bash
# Rewrite Example 1
mlia optimize ~/models/ds_cnn_large_fp32.tflite \
    --target-profile ethos-u55-256 \
    --rewrite \
    --dataset input.tfrec \
    --rewrite-target fully-connected \
    --rewrite-start MobileNet/avg_pool/AvgPool \
    --rewrite-end MobileNet/fc1/BiasAdd

# Rewrite Example 2
mlia optimize ~/models/ds_cnn_large_fp32.tflite \
    --target-profile ethos-u55-256 \
    --rewrite \
    --dataset input.tfrec \
    --rewrite-target conv2d-clustering \
    --rewrite-start model/re_lu_9/Relu \
    --rewrite-end model/re_lu_10/Relu
```

### Random Dataset

The dataset flag is optional. If you do not provide a dataset, then the rewrite will occur using random data to give the user an idea of the performance benefits of the rewrite on the model.

### Conv2d Rewrites

For conv2d rewrites, the conv2d layer parameters are calculated as followed:

We first assume that valid (no) padding will be used, we calculate the conv2d parameters using the following formulae:

Kernel size: set by the user, defaults to 3x3

Output filters = $output\_shape[-1]$

$stride[0] = input\_shape[0] / output\_shape[0]$ (rounded to nearest integer)

$stride[1] = input\_shape[1] / output\_shape[1]$ (rounded to nearest integer)

The input and output shapes are then calculated using the following formulae:

$output\_shape[0] = \lfloor(input\_shape[0] - kernel\_size[0]) / stride[0] \rfloor + 1$

$output\_shape[1] = \lfloor(input\_shape[1] - kernel\_size[1]) / stride[1] \rfloor + 1$

If these resulting sizes do not match the desired output shape, we set the padding to 'same' such that they match it.

This introduces some constraints into the size of the kernel that can be used with the rewrite subgraph to produce the desired output shape. The user should be aware of these formulae when performing rewrites.

### Optimization Profiles

Training parameters for rewrites can be specified.

There are a number of predefined profiles for rewrites. Some examples of these are shown below:

|    Name      | Batch Size |  LR  | Show Progress | Steps | LR Schedule | Num Procs | Num Threads | Checkpoints |
| :----------: | :--------: | :--: | :-----------: | :---: | :---------: | :-------: | :---------: | :---------: |
| optimization |     32     | 1e-3 |      True     | 48000 |   "cosine"  |     1     |      0      |     None    |

|    Name                                 | Batch Size |  LR  | Show Progress | Steps | LR Schedule | Num Procs | Num Threads | Checkpoints | Num Clusters | Cluster Centroids Init             |
| :-------------------------------------: | :--------: | :--: | :-----------: | :---: | :---------: | :-------: | :---------: | :---------: | :----------: | :--------------------------------: |
| optimization-fully-connected-clustering |     32     | 1e-3 |      True     | 48000 |   "cosine"  |     1     |      0      |     None    |      16      |    "CentroidInitialization.LINEAR" |

|    Name                               | Batch Size |  LR  | Show Progress | Steps | LR Schedule | Num Procs | Num Threads | Checkpoints | Sparsity M | Sparsity N |
| :-----------------------------------: | :--------: | :--: | :-----------: | :---: | :---------: | :-------: | :---------: | :---------: | :--------: | :--------: |
| optimization-fully-connected-pruning  |     32     | 1e-3 |      True     | 48000 |   "cosine"  |     1     |      0      |     None    |     2      |      4     |

|    Name                                           | Batch Size |  LR  | Show Progress | Steps | LR Schedule | Num Procs | Num Threads | Checkpoints | Initial Sparsity | End Sparsity | End Step   |
| :-----------------------------------------------: | :--------: | :--: | :-----------: | :---: | :---------: | :-------: | :---------: | :---------: | :--------------: | :----------: | :--------: |
| optimization-fully-connected-unstructured-pruning |     32     | 1e-3 |      True     | 48000 |   "cosine"  |     1     |      0      |     None    |     0.25         |      0.5     | 48000      |

|    Name                        | Batch Size |  LR  | Show Progress | Steps | LR Schedule | Num Procs | Num Threads | Checkpoints | Num Clusters | Cluster Centroids Init             | Activation | Kernel Size |
| :----------------------------: | :--------: | :--: | :-----------: | :---: | :---------: | :-------: | :---------: | :---------: | :----------: | :--------------------------------: | :--------: | :---------: |
| optimization-conv2d-clustering |     32     | 1e-3 |      True     | 48000 |   "cosine"  |     1     |      0      |     None    |      16      |    "CentroidInitialization.LINEAR" | "relu"     | 3x3         |

|    Name                     | Batch Size |  LR  | Show Progress | Steps | LR Schedule | Num Procs | Num Threads | Checkpoints | Sparsity M | Sparsity N | Activation | Kernel Size |
| :-------------------------: | :--------: | :--: | :-----------: | :---: | :---------: | :-------: | :---------: | :---------: | :--------: | :--------: | :--------: | :---------: |
| optimization-conv2d-pruning |     32     | 1e-3 |      True     | 48000 |   "cosine"  |     1     |      0      |     None    |     2      |      4     | "relu"     | 3x3         |

|    Name                                  | Batch Size |  LR  | Show Progress | Steps | LR Schedule | Num Procs | Num Threads | Checkpoints | Initial Sparsity | End Sparsity | End Step   | Activation | Kernel Size |
| :--------------------------------------: | :--------: | :--: | :-----------: | :---: | :---------: | :-------: | :---------: | :---------: | :--------------: | :----------: | :--------: | :---------:| :---------: |
| optimization-conv2d-unstructured-pruning |     32     | 1e-3 |      True     | 48000 |   "cosine"  |     1     |      0      |     None    |     0.25         |      0.5     |     48000  |    "relu"  |         3x3 |

The complete list of built in optimization profiles is shown below. Each profile provides training parameters and parameters specific to the rewrite.

* optimization
* optimization-fully-connected-clustering
* optimization-fully-connected-pruning
* optimization-fully-connected-unstructured-pruning
* optimization-conv2d
* optimization-conv2d-clustering
* optimization-conv2d-pruning
* optimization-conv2d-unstructured-pruning
* optimization-depthwise-separable-conv2d
* optimization-depthwise-separable-conv2d-clustering
* optimization-depthwise-separable-conv2d-pruning
* optimization-conv2d-depthwise-separable-unstructured-pruning

**Note:** For convolutional rewrites (e.g. optimization-conv2d-pruning). The activation function for the rewrite can be selected in the optimization profile from the following list:

* "relu" - Standard ReLU activation function
* "relu6" - ReLU6 activation function i.e. ReLU activation function capped at 6
* "none" - No activation function

The user can also specify custom augmentations as part of the training parameters. An example of this can be found in the following optimization profile:

|               Name               | Batch Size |  LR  | Show Progress | Steps | LR Schedule | Num Procs | Num Threads | Checkpoints | Augmentations - gaussian_strength | Augmentations - mixup_strength |
| :------------------------------: | :--------: | :--: | :-----------: | :---: | :---------: | :-------: | :---------: | :---------: | :-------------------------------: | :----------------------------: |
| optimization-custom-augmentation |     32     | 1e-3 |      True     | 48000 |   "cosine"  |     1     |      0      |     None    |               0.1                 |                0.1             |

The augmentations consist of 2 parameters: mixup strength and gaussian strength.

Augmentations can be selected from a number of pre-defined profiles (see the table below) or each individual parameter can be chosen (see optimization_custom_augmentation above for an example):

|         Name         | MixUp Strength | Gaussian Strength |
| :------------------: | :------------: | :---------------: |
|         "none"       |       None     |        None       |
|         "gaussian"   |       None     |        1.0        |
|         "mixup"      |       1.0      |        None       |
|         "mixout"     |       1.6      |        None       |
| "mix_gaussian_large" |       2.0      |        1.0        |
| "mix_gaussian_small" |       1.6      |        0.3        |

An example of using an optimization profile can be seen below:

```bash
##### An example for using optimization Profiles
mlia optimize ~/models/ds_cnn_large_fp32.tflite \
    --target-profile ethos-u55-256 \
    --optimization-profile optimization \
    --rewrite \
    --dataset input.tfrec \
    --rewrite-target fully-connected \
    --rewrite-start MobileNet/avg_pool/AvgPool \
    --rewrite-end MobileNet/fc1/BiasAdd
```

#### Custom optimization Profiles

For the _custom optimization profiles_, the configuration file for a custom
optimization profile is passed as path and needs to conform to the TOML file format.
Each optimization in MLIA has a pre-defined set of parameters which can be present
in the config file. When using the built-in optimization profiles, the appropriate
toml file is copied to `mlia-output` and can be used to understand what parameters
apply for each optimization.

*Example:*

``` bash
# for custom profiles
mlia optimize --optimization-profile ~/my_custom_optimization_profile.toml
```

When providing rewrite-specific parameters e.g. for clustering, the rewrite name should be specified in the toml:

For example, the following provides rewrite-specific parameters for the conv2d-clustering rewrite

``` bash
[rewrite.conv2d-clustering]
num_clusters = 16
cluster_centroids_init = "CentroidInitialization.LINEAR"
```

# Target profiles

The targets currently supported are described in the sections below.
All sub-commands require a target profile as input parameter.
That target profile can be either a name of a built-in target profile
or a custom file. MLIA saves the target profile that was used for a run
in the output directory.

The support of the above sub-commands for different targets is provided via
backends that need to be installed separately, see
[Backend installation](#backend-installation) section.

## Ethos-U

There are a number of predefined profiles for Ethos-U with the following
attributes:

```table
+--------------------------------------------------------------------+
| Profile name  | MAC | System config               | Memory mode    |
+=====================================================================
| ethos-u55-256 | 256 | Ethos_U55_High_End_Embedded | Shared_Sram    |
+---------------------------------------------------------------------
| ethos-u55-128 | 128 | Ethos_U55_High_End_Embedded | Shared_Sram    |
+---------------------------------------------------------------------
| ethos-u65-512 | 512 | Ethos_U65_High_End          | Dedicated_Sram |
+---------------------------------------------------------------------
| ethos-u65-256 | 256 | Ethos_U65_High_End          | Dedicated_Sram |
+--------------------------------------------------------------------+
```

Example:

```bash
mlia check ~/model.tflite --target-profile ethos-u65-512 --performance
```

Ethos-U is supported by these backends:

* [Corstone-300](#corstone-300)
* [Corstone-310](#corstone-310)
* [Vela](#vela)

As described in section [Custom target profiles](#custom-target-profiles), you can customize
the target using the following parameters in the .toml files:

* mac: number of MACs [256, 512]
* memory_mode: [SRAM Only, Shared SRAM, Dedicated SRAM]
* system_config: name of the system configuration. For Vela backend, it's defined in `vela.ini`.
* config: for the Vela backend - the path to Vela configuration file,
          passed in the `--config` argument.
          If not given, uses the builtin path: `mlia/resources/vela/vela.ini`

## Cortex-A

The profile *cortex-a* can be used to get the information about supported
operators for Cortex-A CPUs when using the Arm NN TensorFlow Lite Delegate.
Please, find more details in the section for the
[corresponding backend](#arm-nn-tensorflow-lite-delegate).

## TOSA

The target profile *tosa* can be used for TOSA compatibility checks of your
model. It requires the [TOSA Checker](#tosa-checker) backend. Please note that
TOSA is currently only available for x86 architecture.

For more information, see TOSA Checker's:

* [repository](https://review.mlplatform.org/plugins/gitiles/tosa/tosa_checker/+/refs/heads/main)
* [pypi.org page](https://pypi.org/project/tosa-checker/)

## Custom target profiles

For the _custom target profiles_, the configuration file for a custom
target profile is passed as path and needs to conform to the TOML file format.
Each target in MLIA has a pre-defined set of parameters which need to be present
in the config file. When using the built-in target profiles, the appropriate
toml file is copied to `mlia-output` and can be used to understand what parameters
apply for each target.

*Example:*

``` bash
# for custom profiles
mlia ops --target-profile ~/my_custom_profile.toml sample_model.tflite
```

# Backend installation

The ML Inference Advisor is designed to use backends to provide different
metrics for different target hardware. Some backends come pre-installed,
but others can be added and managed using the command `mlia-backend`, that
provides the following functionality:

* **install**
* **uninstall**
* **list**

 *Examples:*

```bash
# List backends installed and available for installation
mlia-backend list

# Install Corstone-300 backend for Ethos-U
mlia-backend install Corstone-300 --path ~/FVP_Corstone_SSE-300/

# Uninstall the Corstone-300 backend
mlia-backend uninstall Corstone-300

# Get help and further information
mlia-backend --help
```

**Note:** Some, but not all, backends can be automatically downloaded, if no
path is provided.

## Available backends

This section lists available backends. As not all backends work on any platform
the following table shows some compatibility information:

```table
+----------------------------------------------------------------------------+
| Backend       | Linux                  | Windows        | Python           |
+=============================================================================
| Arm NN        |                        |                |                  |
| TensorFlow    | x86_64 and AArch64     | Windows 10     | Python>=3.8      |
| Lite Delegate |                        |                |                  |
+-----------------------------------------------------------------------------
| Corstone-300  | x86_64 and  AArch64    | Not compatible | Python>=3.8      |
+-----------------------------------------------------------------------------
| Corstone-310  | x86_64 and  AArch64    | Not compatible | Python>=3.8      |
+-----------------------------------------------------------------------------
| TOSA checker  | x86_64 (manylinux2014) | Not compatible | 3.7<=Python<=3.9 |
+-----------------------------------------------------------------------------
| Vela          | x86_64 and  AArch64    | Windows 10     | Python~=3.7      |
+----------------------------------------------------------------------------+
```

### Arm NN TensorFlow Lite Delegate

This backend provides general information about the compatibility of operators
with the Arm NN TensorFlow Lite Delegate for Cortex-A. It comes pre-installed.

For version 23.05 the classic delegate is used.

For more information see:

* [Arm NN TensorFlow Lite Delegate documentation](https://arm-software.github.io/armnn/latest/md_delegate__delegate_quick_start_guide.html)

### Corstone-300

Corstone-300 is a backend that provides performance metrics for systems based
on Cortex-M55 and Ethos-U. It is only available on the Linux platform.

*Examples:*

```bash
# Download and install Corstone-300 automatically
mlia-backend install Corstone-300
# Point to a local version of Corstone-300 installed using its installation script
mlia-backend install Corstone-300 --path YOUR_LOCAL_PATH_TO_CORSTONE_300
```

For further information about Corstone-300 please refer to:
<https://developer.arm.com/Processors/Corstone-300>

### Corstone-310

Corstone-310 is a backend that provides performance metrics for systems based
on Cortex-M85 and Ethos-U.

* For access to AVH for Corstone-310 please refer to:
  <https://developer.arm.com/Processors/Corstone-310>
* Please use the examples of MLIA using Corstone-310 here to get started:
  <https://github.com/ARM-software/open-iot-sdk>

### TOSA Checker

The TOSA Checker backend provides operator compatibility checks against the
TOSA specification. Please note that TOSA is currently only available for x86 architecture.

Please, install it into the same environment as MLIA using this command:

```bash
mlia-backend install tosa-checker
```

Additional resources:

* Source code: <https://gitlab.arm.com/tosa/tosa-checker>
* PyPi package <https://pypi.org/project/tosa-checker/>

### Vela

The Vela backend provides performance metrics for Ethos-U based systems. It
comes pre-installed.

Additional resources:

* <https://pypi.org/project/ethos-u-vela/>
