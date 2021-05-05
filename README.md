# Introduction to mlia

A brief introduction to mlia.

<!-- markdownlint-disable MD026 -->
## Quick Start TL;DR;
<!-- markdownlint-enable -->

Initial setup after template instantiation:

```shell
cd mlia
git init
```

The mlia package assumes a
[virtualenv](<https://virtualenv.pypa.io/en/stable/>) installation.
Install development dependencies:

```shell
virtualenv venv
source venv/bin/activate
```

Install pre-commit framework:

```
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

## Package Versioning

The package is setup to version using
[setuptools_scm](<https://pypi.org/project/setuptools-scm/>).  The
package version is automatically derived from the most recent git tag
of the form vX.Y.Z where X, Y and Z are numbers.  The derived version
is constructed to be unique even in the presence of commits since the
version tag, local commits and local changes.  Refer to the
setuptools_scm for a detailed explanation of the Default Verioning
Scheme.

The mlia package version is updated by
creating a version tag in git:

For example:

```shell
git tag v2.0.1
```

## Python Environment and Package Dependencies

The mlia package assumes a [virtualenv]
(<https://virtualenv.pypa.io/en/stable/>)
managed development environment.

The package itself is defined in the setup.py as a local install
dependency of the form:

```shell
pip install -e .
```

Further install dependencies can be added using the command:

```shell
pip install PACKAGE
```

## Package Layout

The package directory hierarchy is laid out to follow python best practice:

```
./setup.py
./src/<package>/*.py
./tests/test_*.py

```

## Linting and Unit Testing

The mlia package is setup with a [pre-commit]
(<https://pre-commit.com/>)
script to run a variety of common, python centric linters, fixers and
tests.

Note: pre-commit will install automatically packages needed.

### Usage

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

### Linter and Unit Test Summary

The python project pre-commit hooks provide the following checks:

- Pre-populated basic setuptools setup.py file.
- Package versioning via [setuptools_scm](<https://pypi.org/project/setuptools-scm/>)
- Package [pre-commit] (<https://pre-commit.com/>) hooks including:
  - basic project sanity linting:
    - detect attempts to commit private keys
    - detect attempts to commit oversize files
    - executable scripts without shebangs
  - basic whitespace linting:
    - trailing whitespace
    - use of TABs
    - mixed line endings
    - missing final line ending
  - yaml linting
  - python pep? import re-ordering via [reorder_python_imports](https://github.com/asottile/reorder_python_imports>)
  - python pep8 linting via [black](https://github.com/psf/black)
  - python pep257 linting via [pydocstlye](https://github.com/PyCQA/pydocstyle/)
  - markdown linting via [markdownlint](https://github.com/markdownlint/markdownlint)
  - python unittest via [pytest](https://docs.pytest.org/en/latest/)
  - python unittest coverage via pytest-cov

## Markdown Rendering

While authoring markdown content, the allmark program provides
convenient browser rendering, with live reload.

The allmark renderer can be started using the provided convenience script:

```shell
./start-allmark.sh
```

Now point your browser to localhost:33001
