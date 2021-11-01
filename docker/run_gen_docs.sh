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

pip install .

utils::generate_docs "$WORKSPACE"
