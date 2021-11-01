#!/bin/bash

# Copyright 2021, Arm Ltd.

set -e
set -u
set -o pipefail

utils::check_workspace_path() {
    if [ -z "$1" ]; then
        echo "Please, provide a workspace path"
        exit 1
    fi

    WORKSPACE=$1
    if [ ! -d "$WORKSPACE" ]; then
        echo "$WORKSPACE is not a valid directory path"
        exit 1
    fi
}

utils::activate_virtual_env() {
    # shellcheck disable=SC1091
    source "/home/foo/v/bin/activate"
}

utils::generate_docs() {
    WORKSPACE=$1

    cd "$WORKSPACE/docs"
    sphinx-apidoc -f -o source "$WORKSPACE/src/mlia"
    make html
}
