# Inference Advisor DEVELOPER GUIDE

## Introduction

This guide covers the specific use case of the developer, providing instructions
on how to set up and run both the unit tests and the end-to-end tests.
It is primarly aimed at developers who want to contribute to the code,
or modify it for their needs.

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

## Installation

With the virtual environment activated, check out the code from the [mlia repo]
(<https://eu-gerrit-2.euhpc.arm.com/admin/repos/mlg/tooling/mlia>)

You have to get the IPSS-ML dependencies separately:

* IPSS-ML (AIET middleware)
* HW emulation systems
* Generic inference runner software

### IPSS-ML dependencies

A few things are required: the wheel file, a set of software and a set of systems.

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

### Install script

To install all the dependencies for development, the recommended way is to use
the install script for developers at scripts/install_dev.sh directly from
within the repo.

```shell
Usage: ./scripts/install_dev.sh [-v] -d package_dir -e venv_dir

Options:
  -h print this help message and exit
  -v enable verbose output
  -d path to the directory containing the install packages
  -e virtual environment directory name
```

The script needs all required packages to be available in one location, that
has to be passed to the install script itself as the -d argument.
In the example in the previous section, this would be PACKAGE_DIR.
A name for the virtual environment to use has also to be provided as the -e argument.

The required packages:

* AI Evaluation Toolkit
* FVP Corstone-300 Ecosystem
* Ethos-U55 Eval Platform
* SGM-775
* Ethos-U65 Eval Platform

At the end, redo:

```shell
pip install -e .
```

to install your latest MLIA for development

### Manual installation

You can also install the dependencies manually.

The IPSS-ML middleware (aiet) requires both systems (the backends) and
software (the applications to run) in order to work.
They have to be installed individually.

#### Generic installation process

The generic process requires to first install the middleware itself:

```shell
$ pip install aiet-<version>.whl
$ aiet --version
aiet, version <version>
```

Then all the required systems have to be installed.
A "system" is a collection of data, executables, configuration files etc.,
specifically bundled together in an archive file to be consumed by the aiet middleware:

```shell
aiet system install -s /path/to/your/system_package.tar.gz
```

The list of the installed systems can be displayed by the following command:

```shell
$ aiet system list
Available systems:

<system name 1>
<system name 2>
...
```

Similarly, a "software" is an application that will eventually run on one of
the systems installed previously.
A "software" is bundled in an archive file in a specific way for being compiled
and run by the aiet middleware.

```shell
aiet software install -s /path/to/your/software_package.tar.gz
```

The list of the installed systems can be displayed by the following command:

```shell
$ aiet software list
Available softwares:

<software name 1>
<software name 2>
...
```

#### Specific installation process for MLIA

Still within the virtual environment, run the following commands:

```shell
$ pip install PACKAGE_DIR/aiet-21.9.0-py3-none-any.whl
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

Note: the install script that comes with the source code (mlia/scripts/install_dev.sh)
is a good reference for installing the AIET.

### Installation update and extras

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
./docs/<docs build files>
./examples/*.ipynb
```

## Linting and Unit Testing

The mlia package is setup with a [pre-commit]
(<https://pre-commit.com/>)
script to run a variety of common, python centric linters, fixers and
tests.

Note: pre-commit will install automatically packages needed.

### Usage

For details on usage and commands, please refer to README.md

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

1. Fix the git hook for pre-commit using tricks in Dockerfile.

    ```shell
    make fake-repo
    cp .pre-commit-config.yaml fake-repo/foo.yaml
    cd fake-repo && git init
    pre-commit install-hooks -c foo.yaml
    rm -rf fake-repo
    ```

1. There are situations where it is convenient to disable pre-commit
   when running a git command, this can be achieved by specify the --no-verify
   option on the git command.

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

### Linter and Unit Test Summary

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

## End to end testing

### Running E2E tests

E2E tests could be launched locally using script `check-e2e.sh`
in the root of the project.

Next environment variables should be set before run:

* ARMLMD_LICENSE_FILE - license info

* MLIA_E2E_CONFIG - path to the E2E tests configuration directory

* AIET_ARTIFACT_PATH - path to the AIET artifact which will be used for
  AIET installation

Directory which MLIA_E2E_CONFIG points to should have two subfolders:

* systems - artifacts for the AIET systems should be placed here
* software - artifacts for the AIET software should be placed here

In order to test MLIA commands configuration file in JSON format should
be created and placed into configuration directory with name
e2e_tests_config.json

This configuration file describes what commands to test and with what
parameters. It allows to provide combinations of the parameters in order
to test several scenarios at once.

The file should have following structure:

* executions - list of the command execution definitions. Each definition
  contains command name and groups of parameters. Test will run all
  possible parameters combinations from these groups.

For example with next configuration command "all" will be launched twice,
one per each device.

``` json
{
    "executions": [
        {
            "command": "all",
            "parameters": {
                "optimizations": [
                    [
                        "--optimization-type",
                        "pruning,clustering",
                        "--optimization-target",
                        "0.5,32"
                    ]
                ],
                "device": [
                    [
                        "--device",
                        "ethos-u55",
                        "--mac",
                        "256",
                        "--system-config",
                        "Ethos_U55_High_End_Embedded"
                    ],
                    [
                        "--device",
                        "ethos-u65",
                        "--mac",
                        "512",
                        "--system-config",
                        "Ethos_U65_High_End"
                    ]
                ],
                "default": [
                    [
                        "--config",
                        "tests/test_resources/vela/sample_vela.ini",
                        "--memory-mode",
                        "Shared_Sram"
                    ]
                ],
                "models": [
                    [
                        "e2e_config/sample_model.h5"
                    ]
                ]
            }
        }
    ]
}
```

## Example layout for the E2E configuration directory

```shell
e2e_config/
    systems/
        fvp_corstone_sse-300_ethos-u55-21.08.0-SNAPSHOT.tar.gz
        sgm775_ethosu_platform-21.03.0.tar.gz
        sgm775_ethosu_platform-21.08.0-SNAPSHOT-oss.tar.gz
    software/
        ethosu_eval_platform_release_aiet-21.08.0-SNAPSHOT.tar.gz
        ethosU65_eval_app-21.08.0-SNAPSHOT.tar.gz
    aiet-21.9.0rc2-py3-none-any.whl
    e2e_tests_config.json
```

## Example E2E tests launch

```shell
export ARMLMD_LICENSE_FILE=<actual value for the license env variable>
export MLIA_E2E_CONFIG=e2e_config
export AIET_ARTIFACT_PATH=e2e_config/aiet-21.9.0rc2-py3-none-any.whl
./check-e2e.sh
```
