# Inference Advisor USER GUIDE

## Introduction

This guide covers the specific use case of the end user, for details on prerequisites
and general setup, please refer to README.md.

## Installation

To install the Inference Advisor and its dependencies, the recommended way is to
use the install script at scripts/install.sh

```shell
Usage: ./scripts/install.sh [-v] -d package_dir -e venv_dir

Options:
  -h print this help message and exit
  -v enable verbose output
  -d path to the directory containing the install packages
  -e virtual environment directory name
```

The script needs all required packages to be available in one location, that
has to be passed to the install script itself as the -d argument.
A name for the virtual environment to use has also to be provided as the -e argument.

The required packages:

* ML Inference Advisor
* AI Evaluation Toolkit
* FVP Corstone-300 Ecosystem
* Ethos-U55 Eval Platform
* SGM-775
* Ethos-U65 Eval Platform

Example:

```shell
./scripts/install.sh -d /path/to/packages -e venv
```

Run the install script and follow the instructions, the script will install all
packages and dependecies in the correct order, it will then set everything up and
create a virtual environment for running the Inference Advisor.

## Usage

For details on usage and commands, please refer to README.md
