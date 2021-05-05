#!/bin/bash

# Copyright 2021, Arm Ltd.
# SPDX-License-Identifier: Apache-2.0

# Top level driver script to sanity check the tooling and scripts in
# this repository.
#
# This interface is intended as the entry point for both CI bots and
# developers.
#
# The script runs a bunch of self checks and tests.  These are run in
# a docker container described by the Dockerfile in this directory.
# The actual logic executed inside the container is captured over in
# scripts/self-check-helper.sh

set -e
set -u
set -o pipefail

execdir=$(dirname "$0")
execdir=$(cd "$execdir" && pwd)

tmpdir=$(mktemp -d)

cleanup()
{
  rm -rf "$tmpdir"
}

trap cleanup 0

# The name of the docker image used for linting this project both
# locally and via CI, this image name must be globally unique.
tag="lint--mlia"

# Take care to suppress docker build output, unless there is an error...
docker build \
       --quiet \
       --build-arg UID="$(id -u)" \
       --build-arg GID="$(id -g)" \
       -t "$tag" "$execdir/docker"

docker run --rm \
       --user "$(id -u):$(id -g)" \
       --pid=host \
       -v "$execdir:/workspace" \
       "$tag" \
       ./self-check-helper.sh \
       /workspace
