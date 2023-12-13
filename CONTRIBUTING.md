<!---
SPDX-FileCopyrightText: Copyright 2023, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->
# Contribution Guidelines

The ML Inference Advisor (MLIA) project is open for external contributors and
welcomes contributions. MLIA is licensed under the [Apache-2.0 license](https://spdx.org/licenses/Apache-2.0.html)
and all accepted contributions must have the same license.
Below is an overview on contributing code to MLIA.

## Contributing code to MLIA

- All code reviews are performed on [ML Platform Gerrit](https://review.mlplatform.org)
- GitHub account credentials are required for creating an account on ML Platform
- Configure your ML Platform account with your [e-mail](https://review.mlplatform.org/settings/#EmailAddresses)
- Configure your ML Platform account for [SSH access](https://review.mlplatform.org/settings/#SSHKeys)
- Set up MLIA git repo

```bash
    git clone "ssh://<your_github_username>@review.mlplatform.org:29418/ml/mlia"
    # set up commit-msg hook to automatically add Gerrit Change-ID to each commit
    scp -p -P 29418 \
    <your_github_username>@review.mlplatform.org:hooks/commit-msg "mlia/.git/hooks/"
    cd mlia
    git checkout main
    # git pull is not required upon initial clone but good practice before
    # creating a patch
    git pull
    git config user.name "FIRST_NAME SECOND_NAME"
    # use the same e-mail you set up your ML Platform account with
    git config user.email your@email.address
```

- Commit using [sign-off](#developer-certificate-of-origin-dco)

```bash
    git commit -s
```

- For the commit messages, the codebase follows [Conventional Commits](https://www.conventionalcommits.org),
  with some customizations. Header description is be capitalized, and the following
  commit types are allowed: build, ci, docs, feat, fix, perf, refactor, style, test.

- Push patch for code review

```bash
    git push origin HEAD:refs/for/main
```

- Patch will appear on ML Platform Gerrit [here](https://review.mlplatform.org/q/is:open+project:ml/mlia+branch:main)
- See below for details on copyright notice and developer certificate of origin

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

## Development repository

The MLIA development repository is hosted on the [mlplatform.org](https://git.mlplatform.org/ml/mlia.git/).

## Code reviews

Contributions must go through code review. Code reviews are performed through
the [mlplatform.org Gerrit server](https://review.mlplatform.org).
Contributors need to signup to this Gerrit server with their GitHub account
credentials.

Only reviewed contributions can go to the main branch of MLIA.

## Continuous integration

Contributions to MLIA go through testing at the Arm CI system. All unit,
integration and regression tests must pass before a contribution gets merged
to the MLIA main branch.
