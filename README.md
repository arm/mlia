# Inference Advisor README

## Introduction

A tool to help AI developers to design and optimize neural network models for
efficient inference on Arm targets by enabling performance analysis and providing
actionable advice early in the model development cycle.

## Usage

After the initial setup, you can start the program by opening your terminal and
typing the following command:

```shell
mlia [command] [arguments]
```

where [command] is to be substituted by one of the supported options, discussed in
the next section.

To get a list of all available options, use:

```shell
mlia --help
```

To get help on a specific command, use:

```shell
mlia [command] --help
```

### **All tests** (all)

#### *Description*

Generates a full report on the input model performance and operator support.
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
