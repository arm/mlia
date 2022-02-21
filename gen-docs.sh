#!/bin/bash

# Copyright (C) 2021-2022, Arm Ltd.

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
       python3 runner.py gen_docs /workspace
