#!/bin/bash

# Copyright (C) 2021-2022, Arm Ltd.

# Usage:
#
#    check-e2e.sh [<tests_to_run>] [<tests_to_skip>]
#
#
# Pass to this script the comma separated list of the markers of end-to-end tests
# that you want to run or skip:
#
#  * all: runs all end-to-end tests (default, can be omitted)
#  * install: runs the installation tests only
#  * command: runs the command test only
#  * model_gen: run the model generation tests only
#
# Examples:
#
#  * MLIA_E2E_CONFIG=e2e_config AIET_ARTIFACT_PATH=e2e_config/aiet-21.12.1-py3-none-any.whl ./check-e2e.sh
#  * MLIA_E2E_CONFIG=e2e_config AIET_ARTIFACT_PATH=e2e_config/aiet-21.12.1-py3-none-any.whl ./check-e2e.sh all (same as the above)
#  * MLIA_E2E_CONFIG=e2e_config ./check-e2e.sh install (does not require AIET_ARTIFACT_PATH, can be omitted)
#  * MLIA_E2E_CONFIG=e2e_config AIET_ARTIFACT_PATH=e2e_config/aiet-21.12.1-py3-none-any.whl ./check-e2e.sh command
#  * MLIA_E2E_CONFIG=e2e_config AIET_ARTIFACT_PATH=e2e_config/aiet-21.12.1-py3-none-any.whl ./check-e2e.sh model_gen
#  * MLIA_E2E_CONFIG=e2e_config AIET_ARTIFACT_PATH=e2e_config/aiet-21.12.1-py3-none-any.whl ./check-e2e.sh all model_gen (run all tests but skip model_gen)

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

tests_to_run="${1:-all}"
tests_to_skip="${2:-}"

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
       python3 runner.py run_e2e_tests --tests-to-run="$tests_to_run" \
                                       --tests-to-skip="$tests_to_skip" \
                                       /workspace
