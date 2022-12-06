#!/bin/bash

# SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -e
set -u
set -o pipefail

# Set one or multiple python versions through pyenv to be available globally
function set_python_global_versions() {
    py_versions=$1
    parsed_versions=""

    # Find correct parsed subversion for each provided version
    for i in $(echo "$py_versions" | tr ";" "\n")
    do
        parsed_versions+=$(pyenv versions | grep "$i")
        parsed_versions+=" "
    done
    # Set global versions
    echo "$parsed_versions" | xargs pyenv global
}

# Install all python versions provided as parameter using pyenv
# if the --set-all-global flag is set, make all installed versions
# available globally.
py_versions=$1
parameters=$2
for i in $(echo "$py_versions" | tr "," "\n")
do
    pyenv install "$i":latest
done

if [ "$parameters" == "--set-all-global" ]; then
    set_python_global_versions "$1"
fi
