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

## User Guide

### Commands

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

* --output-format: How to output the resulting table.
  * options:
    * plain_text: prints it in a human readable way
    * json: saves it into a json file
    * csv: saves it into a csv file
  * default: plain_text
* --output: Name of the file where report will be saved. If no file name is specified,
  the report will be displayed on the standard output.

##### Debug options

* --verbose: Produce verbose output (for debugging purposes).

#### *Examples*

```shell
mlia all_tests mlia/tests/test_resources/models/simple_model.h5

ML Inference Advisor 0.1

Help the design and optimization of neural network models for efficient inference on a target CPU, GPU and NPU

Supported targets:

 - Ethos-U55 <op compatibility, perf estimation, model opt>
 - Ethos-U65 <op compatibility, perf estimation, model opt>

ARM 2021 Copyright Reserved

Device information:
╒════════════╤═══════╤══════════════════════╤══════════════════╤══════════════════╕
│ IP class   │ MAC   │ Accelerator config   │ System config    │ Memory mode      │
╞════════════╪═══════╪══════════════════════╪══════════════════╪══════════════════╡
│ ethos-u55  │ 256   │ ethos-u55-256        │ internal-default │ internal-default │
╘════════════╧═══════╧══════════════════════╧══════════════════╧══════════════════╛

=== Model Analysis =========================================================

fully_quantize: 0, inference_type: 6, input_inference_type: 9, output_inference_type: 9
Checking operator compatibility ...
Done
Evaluating performance ...

Original model:

fully_quantize: 0, inference_type: 6, input_inference_type: 9, output_inference_type: 9
Getting the memory usage metrics ...
Done
Compiling the model ...
Done
Getting the performance metrics ...
WARNING: This task may require several minutes (press ctrl-c to interrupt)
Done

Optimized model:

Applying optimizations (pruning: 0.5 - clustering: 32) ...
Done
fully_quantize: 0, inference_type: 6, input_inference_type: 9, output_inference_type: 9
Getting the memory usage metrics ...
Done
Compiling the model ...
Done
Getting the performance metrics ...
WARNING: This task may require several minutes (press ctrl-c to interrupt)
Done

Operators:
╒═════╤════════════════════════════════╤═════════════════╤════════════════════╤═══════════════════════════════════════════════════╕
│ #   │ Operator name                  │ Operator type   │ Supported on NPU   │ Reasons                                           │
╞═════╪════════════════════════════════╪═════════════════╪════════════════════╪═══════════════════════════════════════════════════╡
│ 1   │ sequential/reshape/Shape       │ SHAPE           │ No                 │ * CPU only operator                               │
├─────┼────────────────────────────────┼─────────────────┼────────────────────┼───────────────────────────────────────────────────┤
│ 2   │ sequential/reshape/strided_sli │ STRIDED_SLICE   │ No                 │ * Input(s) and Output tensors must not be dynamic │
│     │ ce                             │                 │                    │ * Op has dynamic tensor(s):                       │
│     │                                │                 │                    │ sequential/reshape/strided_slice                  │
├─────┼────────────────────────────────┼─────────────────┼────────────────────┼───────────────────────────────────────────────────┤
│ 3   │ sequential/reshape/Reshape/sha │ PACK            │ No                 │ * Input(s) and Output tensors must not be dynamic │
│     │ pe                             │                 │                    │ * Op has dynamic tensor(s):                       │
│     │                                │                 │                    │ sequential/reshape/strided_slice                  │
├─────┼────────────────────────────────┼─────────────────┼────────────────────┼───────────────────────────────────────────────────┤
│ 4   │ sequential/reshape/Reshape     │ RESHAPE         │ Yes                │                                                   │
├─────┼────────────────────────────────┼─────────────────┼────────────────────┼───────────────────────────────────────────────────┤
│ 5   │ sequential/conv1/Relu;sequenti │ CONV_2D         │ Yes                │                                                   │
│     │ al/conv1/BiasAdd;sequential/co │                 │                    │                                                   │
│     │ nv2/Conv2D;sequential/conv1/Co │                 │                    │                                                   │
│     │ nv2D                           │                 │                    │                                                   │
├─────┼────────────────────────────────┼─────────────────┼────────────────────┼───────────────────────────────────────────────────┤
│ 6   │ sequential/conv2/Relu;sequenti │ CONV_2D         │ Yes                │                                                   │
│     │ al/conv2/BiasAdd;sequential/co │                 │                    │                                                   │
│     │ nv2/Conv2D                     │                 │                    │                                                   │
├─────┼────────────────────────────────┼─────────────────┼────────────────────┼───────────────────────────────────────────────────┤
│ 7   │ sequential/max_pooling2d/MaxPo │ MAX_POOL_2D     │ Yes                │                                                   │
│     │ ol                             │                 │                    │                                                   │
├─────┼────────────────────────────────┼─────────────────┼────────────────────┼───────────────────────────────────────────────────┤
│ 8   │ sequential/flatten/Reshape     │ RESHAPE         │ Yes                │                                                   │
├─────┼────────────────────────────────┼─────────────────┼────────────────────┼───────────────────────────────────────────────────┤
│ 9   │ Identity                       │ FULLY_CONNECTED │ Yes                │                                                   │
╘═════╧════════════════════════════════╧═════════════════╧════════════════════╧═══════════════════════════════════════════════════╛
Operators statistics:
  Number of operators                                        9
  Number of NPU supported operators                          6
  Unsupported ops ratio                                    33%

Performance metrics:
╒════════════════════════════════╤════════════╤═════════════╤════════╤═══════════════════╕
│ Metric                         │ Original   │ Optimized   │ Unit   │ Improvement (%)   │
╞════════════════════════════════╪════════════╪═════════════╪════════╪═══════════════════╡
│ SRAM used                      │ 21.33      │ 20.52       │ KiB    │ 3.81              │
├────────────────────────────────┼────────────┼─────────────┼────────┼───────────────────┤
│ Off-chip flash used            │ 23.58      │ 10.92       │ KiB    │ 53.68             │
├────────────────────────────────┼────────────┼─────────────┼────────┼───────────────────┤
│ NPU active cycles              │ 33,997     │ 22,902      │ cycles │ 32.64             │
├────────────────────────────────┼────────────┼─────────────┼────────┼───────────────────┤
│ NPU idle cycles                │ 175        │ 270         │ cycles │ -54.29            │
├────────────────────────────────┼────────────┼─────────────┼────────┼───────────────────┤
│ NPU total cycles               │ 34,172     │ 23,172      │ cycles │ 32.19             │
├────────────────────────────────┼────────────┼─────────────┼────────┼───────────────────┤
│ NPU AXI0 RD data beat received │ 4,144      │ 3,688       │ beats  │ 11.00             │
├────────────────────────────────┼────────────┼─────────────┼────────┼───────────────────┤
│ NPU AXI0 WR data beat written  │ 2,990      │ 2,876       │ beats  │ 3.81              │
├────────────────────────────────┼────────────┼─────────────┼────────┼───────────────────┤
│ NPU AXI1 RD data beat received │ 2,967      │ 1,347       │ beats  │ 54.60             │
╘════════════════════════════════╧════════════╧═════════════╧════════╧═══════════════════╛
IMPORTANT: The performance figures above refer to NPU only

=== Advice Generation ======================================================

1  You have 33% of operators that cannot be placed on the NPU.
   For better performance, please review the reasons reported in the table, and adjust the model accordingly where possible.

2  You have at least 1 operator that is CPU only: SHAPE.
   Using operators that are supported by the NPU will improve performance.
   For guidance on supported operators, run: mlia operators --supported-ops-report

3  With the selected optimization (pruning: 0.5 + clustering: 32)
   - You have achieved 3.81% performance improvement in SRAM used (KiB)
   - You have achieved 53.68% performance improvement in Off chip flash used (KiB)
   - You have achieved 32.19% performance improvement in NPU total cycles
   You can try to push the optimization target higher (e.g. pruning 0.6 and/or clustering 16) to check if those results can be further improved.
   For more info, see: mlia optimization --help
   Pruning command: mlia optimization --optimization-type pruning --optimization-target 0.6 tests/test_resources/models/simple_model.h5
   Clustering command: mlia optimization --optimization-type clustering --optimization-target 16 tests/test_resources/models/simple_model.h5

4  The applied tooling techniques have an impact on accuracy. Additional hyperparameter tuning may be required after any optimization.
```

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

* --output-format: How to output the resulting table.
  * options:
    * plain_text: prints it in a human readable way
    * json: saves it into a json file
    * csv: saves it into a csv file
  * default: plain_text
* --output: Name of the file where report will be saved. If no file name is specified,
  the report will be displayed on the standard output.

#### *Examples*

```shell

mlia operators mlia/tests/test_resources/models/simple_3_layers_model.tflite

ML Inference Advisor 0.1

Help the design and optimization of neural network models for efficient inference on a target CPU, GPU and NPU

Supported targets:

 - Ethos-U55 <op compatibility, perf estimation, model opt>
 - Ethos-U65 <op compatibility, perf estimation, model opt>

ARM 2021 Copyright Reserved

Device information:
╒════════════╤═══════╤══════════════════════╤══════════════════╤══════════════════╕
│ IP class   │ MAC   │ Accelerator config   │ System config    │ Memory mode      │
╞════════════╪═══════╪══════════════════════╪══════════════════╪══════════════════╡
│ ethos-u55  │ 256   │ ethos-u55-256        │ internal-default │ internal-default │
╘════════════╧═══════╧══════════════════════╧══════════════════╧══════════════════╛

=== Model Analysis =========================================================

Checking operator compatibility ...
Done

Operators:
╒═════╤════════════════════════════════╤═════════════════╤════════════════════╤═══════════╕
│ #   │ Operator name                  │ Operator type   │ Supported on NPU   │ Reasons   │
╞═════╪════════════════════════════════╪═════════════════╪════════════════════╪═══════════╡
│ 1   │ sequential/dense/MatMul;sequen │ FULLY_CONNECTED │ Yes                │           │
│     │ tial/dense/BiasAdd             │                 │                    │           │
├─────┼────────────────────────────────┼─────────────────┼────────────────────┼───────────┤
│ 2   │ sequential/dense_1/MatMul;sequ │ FULLY_CONNECTED │ Yes                │           │
│     │ ential/dense_1/BiasAdd;sequent │                 │                    │           │
│     │ ial/dense_1/Relu               │                 │                    │           │
├─────┼────────────────────────────────┼─────────────────┼────────────────────┼───────────┤
│ 3   │ Identity                       │ FULLY_CONNECTED │ Yes                │           │
╘═════╧════════════════════════════════╧═════════════════╧════════════════════╧═══════════╛
Operators statistics:
  Number of operators                                        3
  Number of NPU supported operators                          3
  Unsupported ops ratio                                     0%

=== Advice Generation ======================================================

1  You don't have any unsupported operators, your model will run completely on NPU.
   Check the estimated performance by running the following command:
   mlia performance mlia/tests/test_resources/models/simple_3_layers_model.tflite

```

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

* --output-format: How to output the resulting table.
  * options:
    * plain_text: prints it in a human readable way
    * json: saves it into a json file
    * csv: saves it into a csv file
  * default: plain_text
* --output: Name of the file where report will be saved. If no file name is
  specified, the report will be displayed on the standard output.

##### Debug options

* --verbose: Produce verbose output (for debugging purposes).

#### *Examples*

```shell

mlia performance mlia/tests/test_resources/models/simple_3_layers_model.tflite

ML Inference Advisor 0.1

Help the design and optimization of neural network models for efficient inference on a target CPU, GPU and NPU

Supported targets:

 - Ethos-U55 <op compatibility, perf estimation, model opt>
 - Ethos-U65 <op compatibility, perf estimation, model opt>

ARM 2021 Copyright Reserved

Device information:
╒════════════╤═══════╤══════════════════════╤══════════════════╤══════════════════╕
│ IP class   │ MAC   │ Accelerator config   │ System config    │ Memory mode      │
╞════════════╪═══════╪══════════════════════╪══════════════════╪══════════════════╡
│ ethos-u55  │ 256   │ ethos-u55-256        │ internal-default │ internal-default │
╘════════════╧═══════╧══════════════════════╧══════════════════╧══════════════════╛

=== Model Analysis =========================================================

Getting the memory usage metrics ...
Done
Compiling the model ...
Done
Getting the performance metrics ...
WARNING: This task may require several minutes (press ctrl-c to interrupt)
Done

Performance metrics:
╒════════════════════════════════╤═════════╤════════╕
│ Metric                         │ Value   │ Unit   │
╞════════════════════════════════╪═════════╪════════╡
│ SRAM used                      │ 0.03    │ KiB    │
├────────────────────────────────┼─────────┼────────┤
│ Off-chip flash used            │ 0.98    │ KiB    │
├────────────────────────────────┼─────────┼────────┤
│ NPU active cycles              │ 1,100   │ cycles │
├────────────────────────────────┼─────────┼────────┤
│ NPU idle cycles                │ 1,072   │ cycles │
├────────────────────────────────┼─────────┼────────┤
│ NPU total cycles               │ 2,172   │ cycles │
├────────────────────────────────┼─────────┼────────┤
│ NPU AXI0 RD data beat received │ 5       │ beats  │
├────────────────────────────────┼─────────┼────────┤
│ NPU AXI0 WR data beat written  │ 5       │ beats  │
├────────────────────────────────┼─────────┼────────┤
│ NPU AXI1 RD data beat received │ 97      │ beats  │
╘════════════════════════════════╧═════════╧════════╛
IMPORTANT: The performance figures above refer to NPU only

=== Advice Generation ======================================================

1  You can improve the inference time by using only operators that are supported by the NPU.
   Try running the following command to verify that:
   mlia operators mlia/tests/test_resources/models/simple_3_layers_model.tflite

2  Check if you can improve the performance by applying tooling techniques to your model.
   Note: you will need a Keras/TF.saved_model input for that.
   For example: mlia optimization --optimization-type pruning --optimization-target 0.5 /path/to/keras_model
   For more info: mlia optimization --help

```

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

#### *Examples*

```shell

mlia optimization --optimization-type pruning --optimization-target 0.5 mlia/tests/test_resources/models/simple_model.h5

ML Inference Advisor 0.1

Help the design and optimization of neural network models for efficient inference on a target CPU, GPU and NPU

Supported targets:

 - Ethos-U55 <op compatibility, perf estimation, model opt>
 - Ethos-U65 <op compatibility, perf estimation, model opt>

ARM 2021 Copyright Reserved

Device information:
╒════════════╤═══════╤══════════════════════╤══════════════════╤══════════════════╕
│ IP class   │ MAC   │ Accelerator config   │ System config    │ Memory mode      │
╞════════════╪═══════╪══════════════════════╪══════════════════╪══════════════════╡
│ ethos-u55  │ 256   │ ethos-u55-256        │ internal-default │ internal-default │
╘════════════╧═══════╧══════════════════════╧══════════════════╧══════════════════╛

=== Model Analysis =========================================================

Original model:

fully_quantize: 0, inference_type: 6, input_inference_type: 9, output_inference_type: 9
Getting the memory usage metrics ...
Done
Compiling the model ...
Done
Getting the performance metrics ...
WARNING: This task may require several minutes (press ctrl-c to interrupt)
Done

Optimized model:

Applying optimizations (pruning: 0.5) ...
Done
fully_quantize: 0, inference_type: 6, input_inference_type: 9, output_inference_type: 9
Getting the memory usage metrics ...
Done
Compiling the model ...
Done
Getting the performance metrics ...
WARNING: This task may require several minutes (press ctrl-c to interrupt)
Done

Performance metrics:
╒════════════════════════════════╤════════════╤═════════════╤════════╤═══════════════════╕
│ Metric                         │ Original   │ Optimized   │ Unit   │ Improvement (%)   │
╞════════════════════════════════╪════════════╪═════════════╪════════╪═══════════════════╡
│ SRAM used                      │ 21.33      │ 20.61       │ KiB    │ 3.37              │
├────────────────────────────────┼────────────┼─────────────┼────────┼───────────────────┤
│ Off-chip flash used            │ 23.58      │ 14.14       │ KiB    │ 40.03             │
├────────────────────────────────┼────────────┼─────────────┼────────┼───────────────────┤
│ NPU active cycles              │ 33,997     │ 25,768      │ cycles │ 24.21             │
├────────────────────────────────┼────────────┼─────────────┼────────┼───────────────────┤
│ NPU idle cycles                │ 175        │ 404         │ cycles │ -130.86           │
├────────────────────────────────┼────────────┼─────────────┼────────┼───────────────────┤
│ NPU total cycles               │ 34,172     │ 26,172      │ cycles │ 23.41             │
├────────────────────────────────┼────────────┼─────────────┼────────┼───────────────────┤
│ NPU AXI0 RD data beat received │ 4,144      │ 3,736       │ beats  │ 9.85              │
├────────────────────────────────┼────────────┼─────────────┼────────┼───────────────────┤
│ NPU AXI0 WR data beat written  │ 2,990      │ 2,888       │ beats  │ 3.41              │
├────────────────────────────────┼────────────┼─────────────┼────────┼───────────────────┤
│ NPU AXI1 RD data beat received │ 2,967      │ 1,759       │ beats  │ 40.71             │
╘════════════════════════════════╧════════════╧═════════════╧════════╧═══════════════════╛
IMPORTANT: The performance figures above refer to NPU only

=== Advice Generation ======================================================

1  With the selected optimization (pruning: 0.5)
   - You have achieved 3.37% performance improvement in SRAM used (KiB)
   - You have achieved 40.03% performance improvement in Off chip flash used (KiB)
   - You have achieved 23.41% performance improvement in NPU total cycles
   You can try to push the optimization target higher (e.g. pruning 0.6) to check if those results can be further improved.

2  For better performance, make sure that all the operators of your final TFLite model are supported by the NPU.
   For more details, run: mlia operators --help

```

## Developer Guide

### Quick Start

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

Install pre-commit framework:

```shell
pip install pre-commit
```

Setup pre-commit and pre-push hooks to run linting and unit testing:

```shell
pre-commit install -t pre-commit
pre-commit install -t pre-push
```

Manually run pre-commit linting and unit testing hooks:

```shell
pre-commit run --all-files
```

Alternatively use the script provided. This will build and spawn a docker
container with all dependencies needed to run pre-commit hooks.

```shell
./check-me.sh
```

Temporarily disabling pre-commit hooks on git commit:

```shell
git commit --no-verify
```

Likewise, to disable pre-commit hooks on git push:

```shell
git push --no-verify
```

Manually running unit tests (with coverage)

```shell
pytest --cov --cov-fail-under=100
```

### Python Environment and Package Dependencies

#### CMake

Go to <https://cmake.org/download/> and get the latest version of CMake
(anything above 3.20 should be fine)

Install CMake so that this new version will be available to your current user
(e.g. put a link in /usr/local/bin)

#### Armclang

Go to <https://developer.arm.com/tools-and-software/embedded/arm-compiler/downloads/version-6>
and get verion 6.15 of armclang (at the moment AIET does not work with later versions)

Untar the archive in a temp directory, then run the install script
(install_x86_64.sh). Follow the instructions on the screen.

Like for CMake, put a link to the executable in /usr/local/bin, so that this
version will be available to your current user

#### IPSS-ML dependencies

We will need three things: the wheel file, a set of softwares and a set of systems.

Below are the list of components and their location:

Name | Version | Filename | URL | Size
--- | --- | --- | --- | ---
IPSS-ML (AIET middleware) | 21.09.0 | aiet-21.9.0-py3-none-any.whl | https://artifactory.eu02.arm.com/artifactory/ml-tooling.pypi-local/aiet/21.9.0/aiet-21.9.0-py3-none-any.whl | 44K
CS-300 (Corstone 300) | 21.08.0 | fvp_corstone_sse-300_ethos-u55-21.08.0.tar.gz | https://artifactory.eu02.arm.com/artifactory/mleng.ecosystem-maven-releases/com/arm/ml/eco/platform/fvp_corstone_sse-300_ethos-u55/21.08.0/fvp_corstone_sse-300_ethos-u55-21.08.0.tar.gz | 20M
SGM-775 (IPSS-ML system) | 21.03.0 | sgm775_ethosu_platform-21.03.0.tar.gz | https://artifactory.eu02.arm.com/artifactory/mleng.ecosystem-maven-releases/com/arm/ml/eco/platform/sgm775_ethosu_platform/21.03.0/sgm775_ethosu_platform-21.03.0.tar.gz | 24M
SGM-775 OSS (IPSS-ML system) | 21.08.0 |sgm775_ethosu_platform-21.08.0-oss.tar.gz | https://artifactory.eu02.arm.com/artifactory/mleng.ecosystem-maven-releases/com/arm/ml/eco/platform/sgm775_ethosu_platform/21.08.0/sgm775_ethosu_platform-21.08.0-oss.tar.gz | 547M
Ethos-U55 Eval Platform (IPSS-ML software) | 21.08.0 |ethosu_eval_platform_release_aiet-21.08.0.tar.gz | https://artifactory.eu02.arm.com/artifactory/mleng.ecosystem-maven-releases/com/arm/ml/eco/ethosu_eval_platform_release_aiet/21.08.0/ethosu_eval_platform_release_aiet-21.08.0.tar.gz | 188M
Ethos-U65 Eval App (IPSS-ML software) | 21.08.0 | ethosU65_eval_app-21.08.0.tar.gz | https://artifactory.eu02.arm.com/artifactory/mleng.ecosystem-maven-releases/com/arm/ml/eco/ethosU65_eval_app/21.08.0/ethosU65_eval_app-21.08.0.tar.gz | 21M

Put all the packages above in a directory, let's call it PACKAGE_DIR

Still within the virtual environment, run the following commands:

```shell
pip install PACKAGE_DIR/aiet-21.9.0-py3-none-any.whl
aiet --version
aiet, version 21.9.0

$ aiet system install -s PACKAGE_DIR/fvp_corstone_sse-300_ethos-u55-21.08.0.tar.gz
$ aiet software install -s PACKAGE_DIR/ethosu_eval_platform_release_aiet-21.08.0.tar.gz

$ tar xzf PACKAGE_DIR/sgm775_ethosu_platform-21.03.0.tar.gz -C PACKAGE_DIR
$ tar xzf PACKAGE_DIR/sgm775_ethosu_platform-21.08.0-oss.tar.gz -C PACKAGE_DIR

$ aiet system install -s PACKAGE_DIR/sgm775_ethosu_platform
$ aiet software install -s PACKAGE_DIR/ethosU65_eval_app-21.08.0.tar.gz

$ aiet system list
Available systems:

CS-300: Cortex-M55
CS-300: Cortex-M55+Ethos-U55
SGM-775

$ aiet software list
Available softwares:

automatic_speech_recognition
generic_inference
image_classification
keyword_spotting
noise_reduction
person_detection
```

Note: the install script that comes with the source code (mlia/scripts/install.sh)
is a good reference for installing the AIET.
Keep in mind that the script is also designed to install a pre-packaged build of
the MLIA.
If you want to skip that, in order not to conflict with your current master,
use mlia/scripts/install_dev.sh.

In case you want to use the install script for setting everything up, at the end
redo:

```shell
pip install -e .
```

to install the latest MLIA for development

#### License file

In order to be able to run the software, you first need to set the license file:

```shell
export ARMLMD_LICENSE_FILE=[link to license file]
```

#### Installation

The package itself is defined in the setup.py as a local install
dependency of the form:

```shell
pip install -e .
```

Further install dependencies can be added using the command:

```shell
pip install PACKAGE
```

### Package Layout

The package directory hierarchy is laid out to follow python best practice:

```shell
./setup.py
./src/<package>/*.py
./tests/test_*.py
./scripts/*.{py,sh}
./docker/<docker files>
./mlia_output/*.log
```

### Linting and Unit Testing

The mlia package is setup with a [pre-commit]
(<https://pre-commit.com/>)
script to run a variety of common, python centric linters, fixers and
tests.

Note: pre-commit will install automatically packages needed.

#### Usage

  There are three primary use cases for the [pre-commit]
(<https://pre-commit.com/>) script:

1. git hook for commit and push

    Setup [pre-commit] (<https://pre-commit.com/>) to automatically
execute the [pre-commit] (<https://pre-commit.com/>) checks on *git
commit* and *git push*.

    ```shell
    pre-commit install -t pre-commit
    pre-commit install -t pre-push
    ```

    There are situations where it is convenient to disable pre-commit
when running a git command, this can be achieved by specify the
--no-verify option on the git command.  e.g.

    ```shell
    git commit --no-verify
    git push --no-verify
    ```

1. Manual Execution

    Manually execute the tests and checks at the command line:

    ```shell
    pre-commit run --all-files
    ```

1. CI Test Script

    Setup the CI system to execute the [pre-commit]
(<https://pre-commit.com/>) checks. The CI infrastructure invokes
the [pre-commit] (<https://pre-commit.com/>) script as per Manual
Execution.

Many of the programs invoked by the [pre-commit]
(<https://pre-commit.com/>) script can be configured to either check for
an issue and return an exit code indicating success or failure, or
conversely to fix the issue in the source code and return an exit code
indicating if a fix was applied or not.

Where this choice exists, the [pre-commit] (<https://pre-commit.com/>)
script is setup to fix the source.

This choice means that on manual execution, the [pre-commit]
(<https://pre-commit.com/>) hooks will fix the source code.

On git hook execution the hooks will fix the source code, but if a
modification is made the git operation will fail, in general
re-executing the git command will subsequently succeed.

In a CI test, any modification of the source will be silently ignored
in CI, where modification occurs [pre-commit]
(<https://pre-commit.com/>) will return a non zero exit code causing the
CI check to fail.

In order to facilitate the development and the maintainability of the CI
infrastructure, a script has been provided to run in a docker container all the
pre-commit hooks.

```shell
./check-me.sh
```

In this way the checks run in a controlled environment independently from the
host are running on.

#### Linter and Unit Test Summary

The python project pre-commit hooks provide the following checks:

* Pre-populated basic setuptools setup.py file.
* Package [pre-commit] (<https://pre-commit.com/>) hooks including:
  * basic project sanity linting:
    * detect attempts to commit private keys
    * detect attempts to commit oversize files
    * executable scripts without shebangs
  * basic whitespace linting:
    * trailing whitespace
    * use of TABs
    * mixed line endings
    * missing final line ending
  * yaml linting
  * python pep? import re-ordering via [reorder_python_imports](
    https://github.com/asottile/reorder_python_imports>)
  * python pep8 linting via [black](https://github.com/psf/black)
  * python pep257 linting via [pydocstlye](https://github.com/PyCQA/pydocstyle/)
  * markdown linting via [markdownlint](https://github.com/markdownlint/markdownlint)
  * python unittest via [pytest](https://docs.pytest.org/en/latest/)
  * python unittest coverage via pytest-cov

### Markdown Rendering

While authoring markdown content, the allmark program provides
convenient browser rendering, with live reload.

The allmark renderer can be started using the provided convenience script:

```shell
./start-allmark.sh
```

Now point your browser to localhost:33001

### End to end testing

Please refer to the document E2E_TESTS.md
