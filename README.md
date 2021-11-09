# Inference Advisor README

## Introduction

A tool to help AI developers to design and optimize neural network models for
efficient inference on Arm targets by enabling performance analysis and providing
actionable advice early in the model development cycle.

## Prerequisites

* Ubuntu 18.04.05 LTS (other OSs may work, this has been tested on that one specifically)
* Python >= 3.8
* CMake >= 3.20.5
* armclang 6.15 (later versions don't work)
* IPSS-ML (see below)
* Docker
* Virtualenv
* Git

## Setup

The mlia package assumes a [virtualenv]
(<https://virtualenv.pypa.io/en/stable/>)
managed development environment.

Install Virtualenv:

```shell
apt install virtualenv
```

Change current working directory and create the virtual environment with Python
3.8 inside:

```shell
cd mlia
virtualenv -p python3.8 venv
```

Activate the virtual environment:

```shell
source venv/bin/activate
```

### CMake

Go to <https://cmake.org/download/> and get the latest version of CMake
(anything above 3.20 should be fine)

Install CMake so that this new version will be available to your current user
(e.g. put a link in /usr/local/bin)

### Armclang

Go to <https://developer.arm.com/tools-and-software/embedded/arm-compiler/downloads/version-6>
and get verion 6.15 of armclang (at the moment AIET does not work with later versions)

Untar the archive in a temp directory, then run the install script
(install_x86_64.sh). Follow the instructions on the screen.

Like for CMake, put a link to the executable in /usr/local/bin, so that this
version will be available to your current user

### License file

In order to be able to run the software, you first need to set the license file:

```shell
export ARMLMD_LICENSE_FILE=[link to license file]
```

## Installation

Please refer to the installation steps for your use case:

* For end users, please refer to USER_GUIDE.md
* For developers, please refer to DEV_GUIDE.md

## Usage

After the initial setup, you can start the program by opening your terminal and
typing the following command:

```shell
mlia [command] [arguments]
```

where [command] is to be substituted by one of the following options:

### **All tests**

#### *Description*

Generates full report.
Shows the original model's operator list, runs the specified optimizations and
lists the performance improvements.

#### *Arguments*

##### Optional arguments

* -h/--help: show this help message and exit

##### Device options

* --device: Device type
  * options:
    * ethos-u55
    * ethos-u65
  * default: ethos-u55
* --mac: MAC value
  * options:
    * 32
    * 64
    * 128
    * 256
    * 512
  * default: 256
* --config: Vela configuration file(s) in Python ConfigParser .ini file format
* --system-config: System configuration
  * default: internal-default
* --memory-mode: Memory mode
  * default: internal-default
* --max-block-dependency: Max block dependency
  * default: 3
* --arena-cache-size: Arena cache size
* --tensor-allocator: Tensor allocator algorithm
  * options:
    * LinearAlloc
    * Greedy
    * HillClimb
* --cpu-tensor-alignment: CPU tensor alignment
  * default: 16
* --optimization-strategy: Optimization strategy
  * options:
    * Performance
    * Size
  * default: Performance

##### Keras model options

* model: Input model in keras (.h5 or SavedModel) format [required].

##### Optimization options

* --optimization-type: List of the optimization types separated by comma.
  * default: pruning, clustering
* --optimization-target: List of the optimization targets separated by comma,
                        (for pruning this is sparsity between (0,1), for
                        clustering this is the number of clusters (positive
                        integer))
  * default: 0.5, 32

##### Output options

* --output: Name of the file where the report will be saved.
  The report is also displayed the standard output, as plain text.
  Valid file extensions (formats) are {.txt,.json,.csv},
  anything else will be formatted as plain text.

##### Debug options

* --verbose: Produce verbose output (for debugging purposes).

### **Operators** (ops)

#### *Description*

Prints the model's operator list.

#### *Arguments*

##### Optional arguments

* -h/--help: Show the general help document and exit.
* --supported-ops-report: Generate the SUPPORTED_OPS.md file in the current working
  directory and exit.

##### Device options

* --device: Type of device to optimise for.
  * options:
    * ethos-u55
    * ethos-u65
  * default: ethos-u55
* --mac: MAC value of the target device.
  * options:
    * 32
    * 64
    * 128
    * 256
    * 512
  * default: 256
* --config: Vela configuration file(s) in Python ConfigParser .ini file format
* --system-config: System configuration
  * default: internal-default
* --memory-mode: Memory mode
  * default: internal-default
* --max-block-dependency: Maximum block dependency
  * default: 3
* --arena-cache-size: Arena cache size
* --tensor-allocator: Tensor allocator algorithm
* --cpu-tensor-alignment: CPU tensor alignment
  * default: 16
* --optimization-strategy:
  * options:
    * Performance
    * Size
  * default: Performance

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

##### device options

* --device: Type of device to optimise for.
  * options:
    * ethos-u55
    * ethos-u65
  * default: ethos-u55
* --mac: MAC value of the target device.
  * options:
    * 32
    * 64
    * 128
    * 256
    * 512
  * default: 256
* --config: Vela configuration file(s) in Python ConfigParser .ini file format
* --system-config: System configuration
  * default: internal-default
* --memory-mode: Memory mode
  * default: internal-default
* --max-block-dependency: Maximum block dependency
  * default: 3
* --arena-cache-size: Arena cache size
* --tensor-allocator: Tensor allocator algorithm
* --cpu-tensor-alignment: CPU tensor alignment
  * default: 16
* --optimization-strategy:
  * options:
    * Performance
    * Size
  * default: Performance

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

##### device options

* --device: Type of device to optimise for.
  * options:
    * ethos-u55
    * ethos-u65
  * default: ethos-u55
* --mac: MAC value of the target device.
  * options:
    * 32
    * 64
    * 128
    * 256
    * 512
  * default: 256
* --config: Vela configuration file(s) in Python ConfigParser .ini file format
* --system-config: System configuration
  * default: internal-default
* --memory-mode: Memory mode
  * default: internal-default
* --max-block-dependency: Maximum block dependency
  * default: 3
* --arena-cache-size: Arena cache size
* --tensor-allocator: Tensor allocator algorithm
* --cpu-tensor-alignment: CPU tensor alignment
  * default: 16
* --optimization-strategy:
  * options:
    * Performance
    * Size
  * default: Performance

##### Keras model options

* model: Input model in keras (.h5 or SavedModel) format [required].

##### optimization options

* --optimization-type: Type of optimization to apply to the model [required].
  * options:
    * pruning
    * clustering
* --optimization-target: Target for optimization (for pruning this is sparsity
  between (0,1), for clustering this is the number of clusters (positive integer))
  [required].
* --layers-to-optimize: Name of the layers to optimize (separated by space).
  Example: conv1 conv2 conv3
  * default: every layer

##### Debug options

* --verbose: Produce verbose output (for debugging purposes).
