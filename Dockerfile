# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-FileCopyrightText: Copyright (c) 2013-2019 Yamashita, Yuu
# SPDX-FileCopyrightText: Copyright (c) 2013 Sam Stephenson
# SPDX-License-Identifier: Apache-2.0 AND MIT

# Execution environment for self-check and tests of this repository.
# The build context should be the MLIA repository.
#
# Example - Build the image with Python 3.10 installed:
#
# docker build \
#   --build-arg UID="$(id -u)" \
#   --build-arg GID="$(id -g)" \
#   --build-arg PYTHON_VERSIONS="3.10" \
#   --build-arg BASE_IMAGE="ubuntu:20.04" \
#   -t "mlia-test" \
#   .
#
# Example - Run the linters in the container:
#
# docker run --rm --user "$(id -u):$(id -g)" --pid=host \
#   -v "$PWD:/workspace" \
#   -w "/workspace" \
#   "mlia-test" \
#   tox --workdir /home/foo/tox/ -e lint

ARG BASE_IMAGE=ubuntu:20.04

FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    locales \
    ruby-dev \
    shellcheck \
    cmake \
    # Dependencies required by pyenv to build Python
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

ARG UID
ARG GID

RUN groupadd -g ${GID} -o foo
RUN useradd -m -l -u ${UID} -g foo foo

USER foo
ENV HOME=/home/foo
ENV USER=foo
ENV PATH="/home/foo/.local/bin:${PATH}"

# Install pyenv
ENV PYENV_GIT_TAG v2.3.3
RUN curl https://pyenv.run | bash

ENV PYENV_ROOT /home/foo/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# Python versions separated by semicolons. E.g. "3.8;3.9"
ARG PYTHON_VERSIONS
# Install Python versions and set them to be available globally
COPY docker/install_python_versions.sh /home/foo
RUN /home/foo/install_python_versions.sh "${PYTHON_VERSIONS}" --set-all-global

# Create a temporary mock MLIA repository to setup the tox and pre-commit
# environments. Copy only relevant files to facilitate caching.
ENV TMP_REPO /tmp/mlia/
RUN mkdir $TMP_REPO
WORKDIR $TMP_REPO
RUN git init \
    && mkdir src \
    && touch README.md
COPY pyproject.toml setup.cfg setup.py tox.ini ./

ENV TOX_WORK_DIR $HOME/tox
COPY .pre-commit-config.yaml .

# Install tox, create tox environment for linting (as this takes most time to set up)
# in a specific cache directory so that it is cached in the docker image.
# Set up pre-commit hooks to cache the hook environments

# pip.conf may contain sensitive information such as login information for the
# chosen pip index. By loading it as a secret, the secret information will not
# be leaked into the final build or cache.
RUN --mount=type=secret,id=pip_conf,mode=755,target=/home/foo/.pip/pip.conf \
    pip3 install -U tox~=3.27.1 &&\
    tox --workdir $TOX_WORK_DIR --notest --recreate -e lint &&\
    tox --workdir $TOX_WORK_DIR -e lint_setup

WORKDIR $HOME
RUN rm -r $TMP_REPO
