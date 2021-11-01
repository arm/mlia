#!/bin/bash

# Copyright 2021, Arm Ltd.

set -e
set -u
set -o pipefail

execdir=$(dirname "$0")
execdir=$(cd "$execdir" && pwd)

tag="mlia-docs"

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
       ./run_gen_docs.sh \
       /workspace
