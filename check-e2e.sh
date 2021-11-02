#!/bin/bash

# Copyright 2021, Arm Ltd.

set -e
set -u
set -o pipefail

if [ -z "${ARMLMD_LICENSE_FILE+x}" ]; then
  echo "ARMLMD_LICENSE_FILE is not set"
  exit 1
fi

if [ -z "${MLIA_E2E_CONFIG+x}" ]; then
  echo "MLIA_E2E_CONFIG is not set"
  exit 1
fi

execdir=$(dirname "$0")
execdir=$(cd "$execdir" && pwd)

tag="test-mlia-e2e"

docker build \
       --quiet \
       --build-arg UID="$(id -u)" \
       --build-arg GID="$(id -g)" \
       -t "$tag" -f "$execdir/docker/Dockerfile.e2e" "$execdir/docker"

docker run --rm \
       --user "$(id -u):$(id -g)" \
       --pid=host \
       -v "$execdir:/workspace" \
       -e ARMLMD_LICENSE_FILE="$ARMLMD_LICENSE_FILE" \
       -e MLIA_E2E_CONFIG="$MLIA_E2E_CONFIG" \
       -e AIET_ARTIFACT_PATH="${AIET_ARTIFACT_PATH:-}" \
       "$tag" \
       ./run_e2e_tests.sh \
       /workspace
