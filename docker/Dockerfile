# Copyright 2021, Arm Ltd.
# SPDX-License-Identifier: Apache-2.0

# Execution environment for self check and tests of this repository.
# The self check and test entry point is check-me.sh in the top level
# directory.

FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
    git \
    locales \
    python3 \
    python3-dev \
    python3-pip \
    ruby-dev \
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
ENV PATH="/home/foo/.local/bin:${PATH}"

RUN pip3 install virtualenv
RUN pip3 install pre-commit

RUN mkdir /home/foo/.pip
COPY pip.conf /home/foo/.pip/pip.conf
COPY pydistutils.cfg /home/foo/.pydistutils.cfg

# Pre-install the pre-commit hooks.  This requires some hoop jumping
# because pre-commit insists on running in a git repo even if the
# actual command running does not necessarily require the .git repo.
# We create a fake, install the pre-commit hooks, then blow the fake.

RUN mkdir /home/foo/fake-repo
COPY pre-commit-config.yaml /home/foo/fake-repo/foo.yaml
RUN cd /home/foo/fake-repo && git init
RUN cd /home/foo/fake-repo && pre-commit install-hooks -c foo.yaml
RUN rm -rf /home/foo/fake-repo

# Create the virtualenv that CI will run within, we do this here in
# the Dockerfile to avoid the time of creating and populating this
# virtualenv on every test run.  The downside of this decision is that
# we fix package versions at image creation, the upside of this
# decision is.. we fix package versions at image creation ;-)
RUN virtualenv --quiet --clear --python python3.6 "/home/foo/v"
RUN /home/foo/v/bin/pip install --upgrade pip setuptools
RUN /home/foo/v/bin/pip install pytest pytest-cov pytest-mock pyfakefs

COPY self-check-helper.sh .
