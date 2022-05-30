<!---
SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->
# ML Inference Advisor

## Introduction

This tool is used to help AI developers design and optimize neural network
models for efficient inference on Arm® targets by enabling performance analysis
and providing actionable advice early in the model development cycle. The final
advice can cover the operator list, performance analysis and suggestions for
model inference run on certain hardware before/after applying model optimization
(e.g. pruning, clustering, etc.).

## Prerequisites and dependencies

It is recommended to use virtual environments for MLIA installation, and a
typical setup for MLIA requires:

* Ubuntu® 20.04.03 LTS (other OSs may work, the ML Inference Advisor has been
  tested on this one specifically)
* Python® >= 3.8
* Ethos™-U Vela dependencies (Linux® only)
  * For more details, please refer to the
    [prerequisites of Vela](https://pypi.org/project/ethos-u-vela/)

## Backend installation

### Generic case using Corstone™-300 as an example

The ML Inference Advisor is designed to support multiple performance
estimators (backends) that could generate performance analysis for individual
types of hardware. In this guide, we use the backend for
Ethos™-U (Corstone™-300) as an example.

The install command can automatically download the necessary components and
dependencies, install them and configure them properly.

The usage is:

```bash
mlia backend install --help
```

and the result looks like:

positional arguments:

* name: Name of the backend to install

optional arguments:

* -h/--help: Show this help message and exit
* --path PATH: Path to the installed backend
* --download: Download and install a backend
* --noninteractive: Non interactive mode with automatic confirmation of every action

Some examples of the installation process are:

```bash
# reply 'y' or 'n' when prompted to download and install a Corstone-300
mlia backend install --download
# for downloading and installing a specific backend
mlia backend install Corstone-300 --download
# for installing backend from the path of your downloaded backend
mlia backend install --path your_local_path_for_the_installed_backend
```

Please note: Corstone™-300 used in the example above is available only
on the Linux® platform.

After a successful installation of the backend(s), start using mlia in your
virtual environment. Please note: backends cannot be removed once installed.
Consider creating new environment and reinstall backends when needed.

### Using Corstone™-310

For instructions on installing Corstone™-310, please refer to
<https://github.com/ARM-software/open-iot-sdk>

## Usage

After the initial setup, you can start the program by opening your terminal and
typing the following command:

```bash
mlia [command] [arguments]
```

where [command] is to be substituted by one of the supported options, discussed in
the next section.

To get a list of all available options, use:

```bash
mlia --help
```

To get help on a specific command, use:

```bash
mlia [command] --help
```

Choices of commands: you can use ["operators"](#operators-ops) command for the
model's operator list, run the specified optimizations using
["model optimization"](#model-optimization-opt) command, or measure the
performance of inference on hardware using ["performance"](#performance-perf)
command. In the end, you can use ["all tests"](#all-tests-all) command to
have a full report.

Most commands accept the name of the target profile name as input parameter.
There are a number of predefined profiles with following attributes:

```
+--------------------------------------------------------------------+
| Profile name  | MAC | System config               | Memory mode    |
+=====================================================================
| ethos-u55-256 | 256 | Ethos_U55_High_End_Embedded | Shared_Sram    |
+---------------------------------------------------------------------
| ethos-u55-128 | 128 | Ethos_U55_High_End_Embedded | Shared_Sram    |
+---------------------------------------------------------------------
| ethos-u65-512 | 512 | Ethos_U65_High_End          | Dedicated_Sram |
+--------------------------------------------------------------------+
```

### **Operators** (ops)

#### *Description*

Prints the model's operator list.

#### *Arguments*

##### Optional arguments

* -h/--help: Show the general help document and exit.
* --supported-ops-report: Generate the SUPPORTED_OPS.md file in the current working
  directory and exit.

##### Target profile options

* --target-profile: Target profile that will set the target options such as
  target, mac value, memory mode, etc ...
  * default: ethos-u55-256
  * options:
    * ethos-u55-256
    * ethos-u55-128
    * ethos-u65-512

##### TFLite model options

* model: Input model in TFLite format [required].

##### Output options

* --output: Name of the file where the report will be saved.
  The report is also displayed the standard output, as plain text.
  Valid file extensions (formats) are {.txt,.json,.csv},
  anything else will be formatted as plain text.

### **Performance** (perf)

#### *Description*

Prints the model's performance statistics.

#### *Arguments*

##### optional arguments

* -h/--help: Show the general help document and exit.
* --supported-ops-report: Generate the SUPPORTED_OPS.md file in the current
  working directory and exit.

##### Target profile options

* --target-profile: Target profile that will set the target options such as
  target, mac value, memory mode, etc ...
  * default: ethos-u55-256
  * options:
    * ethos-u55-256
    * ethos-u55-128
    * ethos-u65-512

##### TFLite model options

* model: Input model in TFLite format [required].

##### Output options

* --output: Name of the file where the report will be saved.
  The report is also displayed the standard output, as plain text.
  Valid file extensions (formats) are {.txt,.json,.csv},
  anything else will be formatted as plain text.

##### Debug options

* --verbose: Produce verbose output (for debugging purposes).

### **Model optimization** (opt)

#### *Description*

Shows the performance improvements after applying optimizations to the model.

#### *Arguments*

##### optional arguments

* -h/--help: Show the general help document and exit.
* --supported-ops-report: Generate the SUPPORTED_OPS.md file in the current
  working directory and exit.

##### Target profile options

* --target-profile: Target profile that will set the target options such as
  target, mac value, memory mode, etc ...
  * default: ethos-u55-256
  * options:
    * ethos-u55-256
    * ethos-u55-128
    * ethos-u65-512

##### Keras™ model options

* model: Input model in Keras™ (.h5 or SavedModel) format [required].

##### optimization options

* --optimization-type: Type of optimization to apply to the model [required].
  * options:
    * pruning
    * clustering
* --optimization-target: Target for optimization (for pruning this is sparsity
  between (0,1), for clustering this is the number of clusters
  (positive integer)) [required].
* --layers-to-optimize: Name of the layers to optimize (separated by space).
  Example: conv1 conv2 conv3
  * default: every layer

##### Debug options

* --verbose: Produce verbose output (for debugging purposes).

### **All tests** (all)

#### *Description*

Generates a full report on the input model's operator list,
runs the specified optimizations and lists the performance improvements.

#### *Arguments*

##### Optional arguments

* -h/--help: show this help message and exit

##### Target profile options

* --target-profile: Target profile that will set the target options such as
  target, mac value, memory mode, etc ...
  * default: ethos-u55-256
  * options:
    * ethos-u55-256
    * ethos-u55-128
    * ethos-u65-512

##### Keras™ model options

* model: Input model in Keras™ (.h5 or SavedModel) format [required].

##### Optimization options

* --optimization-type: List of the optimization types separated by comma
  * default: pruning, clustering
* --optimization-target: List of the optimization targets separated by comma,
  (for pruning this is sparsity between (0,1), for clustering this is the
  number of clusters (positive integer))
  * default: 0.5, 32

##### Output options

* --output: Name of the file where the report will be saved.
  The report is also displayed the standard output, as plain text.
  Valid file extensions (formats) are {.txt,.json,.csv},
  anything else will be formatted as plain text.

##### Debug options

* --verbose: Produce verbose output (for debugging purposes).

## Resources

Additional useful information:

* [Corstone™-300](https://developer.arm.com/Processors/Corstone-300)

## License

ML Inference Advisor is licensed under [Apache License 2.0](LICENSE.txt).

## Trademarks and Copyrights

Arm®, Ethos™-U, Cortex®-M, Corstone™ are registered trademarks or trademarks
of Arm® Limited (or its subsidiaries) in the U.S. and/or elsewhere.
TensorFlow™ is a trademark of Google® LLC.
Keras™ is a trademark by François Chollet.
Linux® is the registered trademark of Linus Torvalds in the U.S. and elsewhere.
Python® is a registered trademark of the PSF.
Ubuntu® is a registered trademark of Canonical.
