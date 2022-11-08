<!---
SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->
# ML Inference Advisor

## Inclusive language commitment

This product conforms to Arm's inclusive language policy and, to the best of
our knowledge, does not contain any non-inclusive language.

If you find something that concerns you, email terms@arm.com.

## Introduction

This tool is used to help AI developers design and optimize neural network
models for efficient inference on Arm® targets (e.g. Cortex®-M55 and
Ethos™-U55/Ethos™-U65, Cortex®-M85 and Ethos™-U55) by enabling performance analysis
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

The ML Inference Advisor is designed to support multiple performance
estimators (backends) that could generate performance analysis for individual
types of hardware.

The `backend` command is used to manage the installation of new backends.
The `install` sub-command can be used to either

* install a backend installed locally already (option `--path`) or
* (if available) automatically download the necessary components and
  dependencies, install them and configure them properly (option `--download`).

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

Example:

```bash
# Use this command to see what backends can be downloaded.
mlia backend install --download
```

After a successful installation of the backend(s), start using mlia in your
virtual environment.

*Please note*: Backends cannot be removed once installed.
Consider creating a new environment and reinstall backends when needed.

### Backend compatibility table

Not all backends work on any platform. Please refer to the compatibility table
below:

```
+---------------------------------------------------------------------------+
| Backend      | Linux                  | Windows        | Python           |
+============================================================================
| Corstone-300 | x86_64                 | Not compatible | Python>=3.8      |
+----------------------------------------------------------------------------
| Corstone-310 | x86_64                 | Not compatible | Python>=3.8      |
+----------------------------------------------------------------------------
| TOSA checker | x86_64 (manylinux2014) | Not compatible | 3.7<=Python<=3.9 |
+----------------------------------------------------------------------------
| Vela         | x86_64                 | Windows 10     | Python~=3.7      |
+---------------------------------------------------------------------------+
```

### Using Corstone™-300

To install Corstone™-300 as a backend for Ethos™-U both options (`--download`
and `--path`) can be used:

```bash
# To download and install Corstone-300 automatically
mlia backend install --download Corstone-300
# To point MLIA to an already locally installed version of Corstone-300
mlia backend install --path YOUR_LOCAL_PATH_TO_CORSTONE_300
```

Please note: Corstone™-300 used in the example above is available only
on the Linux® platform.

### Using Corstone™-310

Corstone™-310 is available as Arm® Virtual Hardware (AVH).

* For access to AVH for Corstone™-310 please refer to:
  <https://developer.arm.com/Processors/Corstone-310>
* Please use the examples of MLIA using Corstone™-310 here to get started:
  <https://github.com/ARM-software/open-iot-sdk>

### Using TOSA checker

TOSA compatibility checker is available in MLIA as an external dependency.
Please, install it into the same environment as MLIA using next command:

```bash
pip install mlia[tosa]
```

TOSA checker resources:

* Source code: <https://review.mlplatform.org/admin/repos/tosa/tosa_checker>
* PyPi package <https://pypi.org/project/tosa-checker/>

## Usage

After the initial setup, you can start the program by opening your terminal and
typing the following command:

```bash
mlia [command] [arguments]
```

Choices of commands:

* ["operators"](#operators-ops): show the model's operator list
* ["optimization"](#model-optimization-opt): run the specified optimizations
* ["performance"](#performance-perf): measure the performance of inference on hardware
* ["all_tests"](#all-tests-all): have a full report

To get a list of all available options, use:

```bash
mlia --help
```

To get help on a specific command, use:

```bash
mlia [command] --help
```

Most commands accept the name of the target profile as input parameter.
There are a number of predefined profiles for Ethos™-U with following attributes:

```
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

Target profile "tosa" could be used for TOSA compatibility checks.

### **Operators** (ops)

#### *Description*

Prints the model's operator list.

#### *Example*

```bash
# List operator compatibility for a specific target profile
mlia operators --target-profile ethos-u55-256 ~/models/mobilenet_v1_1.0_224_quant.tflite
```

#### *Arguments*

##### TensorFlow Lite model options

* model: Input model in TensorFlow Lite format [required].

##### Target profile options

* --target-profile: Target profile that will set the target options such as
  target, MAC value, memory mode, etc ...
  * default: ethos-u55-256
  * options:
    * ethos-u55-256
    * ethos-u55-128
    * ethos-u65-512
    * ethos-u65-256
    * tosa

##### Output options

* --output: Name of the file where the report will be saved. The report format
  is automatically detected based on the file extension. Supported formats are:
  csv, json.

##### Optional arguments

* -h/--help: Show the general help document and exit.
* --supported-ops-report: Generate the SUPPORTED_OPS.md file in the current working
  directory and exit (Ethos-U target profiles only).

### **Performance** (perf)

#### *Description*

Prints the model's performance statistics.

#### *Example*

```bash
# Explicitly specify backend(s) to use with --evaluate-on
mlia performance ~/models/mobilenet_v1_1.0_224_quant.tflite \
--evaluate-on "Vela" "Corstone-310"
```

#### *Arguments*

##### TensorFlow Lite model options

* model: Input model in TensorFlow Lite format [required].

##### Target profile options

* --target-profile: Target profile that will set the target options such as
  target, MAC value, memory mode, etc ...
  * default: ethos-u55-256
  * options:
    * ethos-u55-256
    * ethos-u55-128
    * ethos-u65-512
    * ethos-u65-256

##### Output options

* --output: Name of the file where the report will be saved. The report format
  is automatically detected based on the file extension. Supported formats are:
  csv, json.

##### Debug options

* --verbose: Produce verbose output (for debugging purposes).

##### Evaluation options

* --evaluate-on: Backends to use for evaluation.
  * default: Vela
  * options:
    * Vela
    * Corstone-300
    * Corstone-310

##### optional arguments

* -h/--help: Show the general help document and exit.
* --supported-ops-report: Generate the SUPPORTED_OPS.md file in the current
  working directory and exit (Ethos-U target profiles only).

### **Model optimization** (opt)

#### *Description*

Shows the performance improvements after applying optimizations to the model.

#### *Example*

```bash
# Custom optimization parameters: pruning=0.6, clustering=16
mlia optimization \
--optimization-type pruning,clustering \
--optimization-target 0.6,16 \
~/models/ds_cnn_l.h5
```

#### *Arguments*

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

##### Target profile options

* --target-profile: Target profile that will set the target options such as
  target, MAC value, memory mode, etc ...
  * default: ethos-u55-256
  * options:
    * ethos-u55-256
    * ethos-u55-128
    * ethos-u65-512
    * ethos-u65-256

##### Evaluation options

* --evaluate-on: Backends to use for evaluation.
  * default: Vela
  * options:
    * Vela
    * Corstone-300
    * Corstone-310

##### Debug options

* --verbose: Produce verbose output (for debugging purposes).

##### optional arguments

* -h/--help: Show the general help document and exit.
* --supported-ops-report: Generate the SUPPORTED_OPS.md file in the current
  working directory and exit (Ethos-U target profiles only).

### **All tests** (all)

#### *Description*

Generates a full report on the input model's operator list,
runs the specified optimizations and lists the performance improvements.

#### *Example*

```bash
# Create full report and save it as JSON file
mlia all_tests --output ./report.json ~/models/ds_cnn_l.h5
```

#### *Arguments*

##### Keras™ model options

* model: Input model in Keras™ (.h5 or SavedModel) format [required].

##### Optimization options

* --optimization-type: List of the optimization types separated by comma
  * default: pruning, clustering
* --optimization-target: List of the optimization targets separated by comma,
  (for pruning this is sparsity between (0,1), for clustering this is the
  number of clusters (positive integer))
  * default: 0.5, 32

##### Target profile options

* --target-profile: Target profile that will set the target options such as
  target, MAC value, memory mode, etc ...
  * default: ethos-u55-256
  * options:
    * ethos-u55-256
    * ethos-u55-128
    * ethos-u65-512
    * ethos-u65-256
    * tosa

##### Output options

* --output: Name of the file where the report will be saved. The report format
  is automatically detected based on the file extension. Supported formats are:
  csv, json.

##### Debug options

* --verbose: Produce verbose output (for debugging purposes).

##### Optional arguments

* -h/--help: show this help message and exit

## Releases

Release notes can be found in [MLIA releases](RELEASES.md)

# Reporting vulnerabilities

Information on reporting security issues can be found in [Reporting vulnerabilities](SECURITY.md)

## Resources

Additional useful information:

* [Corstone™-300](https://developer.arm.com/Processors/Corstone-300)
* [Corstone™-310](https://developer.arm.com/Processors/Corstone-310)
* [TOSA Checker](https://review.mlplatform.org/admin/repos/tosa/tosa_checker)

## License

ML Inference Advisor is licensed under [Apache License 2.0](LICENSES/Apache-2.0.txt).

## Trademarks and copyrights

* Arm®, Ethos™-U, Cortex®-A, Cortex®-M, Corstone™ are registered trademarks or
  trademarks of Arm® Limited (or its subsidiaries) in the U.S. and/or
  elsewhere.
* TensorFlow™ is a trademark of Google® LLC.
* Keras™ is a trademark by François Chollet.
* Linux® is the registered trademark of Linus Torvalds in the U.S. and
  elsewhere.
* Python® is a registered trademark of the PSF.
* Ubuntu® is a registered trademark of Canonical.
* Microsoft and Windows are trademarks of the Microsoft group of companies.
