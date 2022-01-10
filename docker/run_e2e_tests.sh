#!/bin/bash

# Copyright 2021, Arm Ltd.

set -e
set -u
set -o pipefail

# shellcheck disable=SC1091
source utils.sh

WORKSPACE="$1"
utils::check_workspace_path "$WORKSPACE"

# shellcheck disable=SC1091
cd "$WORKSPACE" || exit 1

utils::activate_virtual_env

if [[ -n "$AIET_ARTIFACT_PATH" ]]; then
  echo "Installing AIET from $AIET_ARTIFACT_PATH ..."
  pip install "$AIET_ARTIFACT_PATH"
fi

echo "Installing the ML Inference Advisor from the local directory ..."
# in order to test installation script we must produce MLIA wheel
# to make things easier wheel should have fixed version
# (the same version as mentioned in install.sh)
FIXED_WHEEL_VERSION=0.1.1
SETUPTOOLS_SCM_PRETEND_VERSION="$FIXED_WHEEL_VERSION" python setup.py -q bdist_wheel
WHEEL_PATH="dist/mlia-$FIXED_WHEEL_VERSION-py3-none-any.whl"
pip install "$WHEEL_PATH"

echo "Extracting the AIET artifacts ..."
cat "$MLIA_E2E_CONFIG"/systems/*.tar.gz | tar -xzf - -i -C "$MLIA_E2E_CONFIG"/systems
cat "$MLIA_E2E_CONFIG"/applications/*.tar.gz | tar -xzf - -i -C "$MLIA_E2E_CONFIG"/applications

echo "Running E2E tests ..."
export PYTHONUNBUFFERED=1
pytest --collect-only -m e2e
pytest -v --capture=tee-sys --durations=0 --durations-min=5 --tb=long --junit-xml=report/report.xml -m e2e
