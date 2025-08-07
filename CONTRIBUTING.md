<!---
SPDX-FileCopyrightText: Copyright 2023,2025, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->
# Contribution Guidelines

The ML Inference Advisor (MLIA) project is open for external contributors and
welcomes contributions. MLIA is licensed under the [Apache-2.0 license](https://spdx.org/licenses/Apache-2.0.html)
and all accepted contributions must have the same license.

This document contains the rules for contributing code to MLIA. All contributed
code must follow these rules before it can be accepted to the main branch of
MLIA.

## Setting up MLIA repo

First clone the MLIA repository.

```bash
    # Using SSH
    git clone "ssh://git@github.com:arm/mlia.git"
    # Or HTTPS
    git clone "https://github.com/arm/mlia.git"
    cd mlia
    git checkout main
    # git pull is not required upon initial clone but good practice before
    # creating a patch
    git pull
    # Set your username, this must be your real name no pseudonyms or anonymous
    # contributions are accepted.
    git config user.name "FIRST_NAME SECOND_NAME"
    # use the same e-mail you set up your github account with
    git config user.email your@email.address
```

### Git Commit Msg Hook

Install the git commit msg hook. This automatically adds a Gerrit "Change-Id"
line to every commit.

```bash
    # Download the commit hook
    curl -o .git/hooks/commit-msg https://gerrit-review.googlesource.com/tools/hooks/commit-msg
    # Make it executable
    chmod +x .git/hooks/commit-msg
```

### Pre-Commit Tests

To help with the contribution process you can run some tests using the pre-commit
hooks. These tests ensure that the contributed code is compliant with the MLIA
coding style. The pre-commit process may reformat your code to make it compliant.
Code that does not pass these tests cannot be accepted to the main branch of MLIA.

You can install the pre-commit hooks in your MLIA folder like this:

```bash
    pre-commit install
```

The pre-commit tests are run on each commit. You can also run them manually
like this:

```bash
    pre-commit run --all-files --hook-stage commit
    pre-commit run --all-files --hook-stage push
```

### Commit Messages

For the commit messages, the codebase follows [Conventional Commits](https://www.conventionalcommits.org),
with some customizations. Header description is be capitalized, and the following
commit types are allowed: build, ci, docs, feat, fix, perf, refactor, style, test.

### Sign off

Commit your code using [sign-off](#developer-certificate-of-origin-dco) this
adds a "Signed-off-by" line, required for MLIA contributions.

```bash
    git commit -s -m "fix: your commit message"
```

### Code reviews

This project follows the conventional GitHub pull request flow. See [here](https://docs.github.com/en/pull-requests)
for details of how to create a pull request.

Contributions must go through code review on GitHub. Only reviewed contributions
can go to the main branch of MLIA.

## Developer Certificate of Origin (DCO)

Before the MLIA project accepts your contribution, you need to certify its
origin and give us your permission. To manage this process we use
[Developer Certificate of Origin (DCO) V1.1](https://developercertificate.org/).

To indicate that you agree to the the terms of the DCO, you "sign off" your contribution
by adding a line with your name and e-mail address to every git commit message:

```bash
Signed-off-by: FIRST_NAME SECOND_NAME <your@email.address>
```

You must use your real name, no pseudonyms or anonymous contributions are accepted.

## In File Copyright Notice

In each source file, include the following copyright notice:

```bash
# SPDX-FileCopyrightText: Copyright <years changes were made> <copyright holder>.
# SPDX-License-Identifier: Apache-2.0
```

Note: if an existing file does not conform, please update the license header
as part of your contribution.

## Releases

Official MLIA releases are published through [PyPI](https://pypi.org/project/mlia).

## Development Repository

The MLIA development repository is hosted on [github.com](https://github.com/arm/mlia.git/).

## Continuous Integration

Contributions to MLIA go through testing at the Arm CI system. All unit,
integration and regression tests must pass before a contribution gets merged
to the MLIA main branch.
