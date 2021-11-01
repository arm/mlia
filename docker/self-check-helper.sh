#!/bin/bash

# Copyright 2021, Arm Ltd.

# Helper script to run sanity checks and tests on this project.  The
# top level entry point script is check-me.sh in the top level
# directory.  This script captures the logic and executes inside a
# docker instance created by the top level script.

set -e
set -u
set -o pipefail

usage()
{
  cat <<EOF
usage: self-check-helper.sh [OPTION] WORKSPACE

  -h, --help
    Print brief usage information and exit.

  -x
    Enable shell tracing in this script.

EOF
}

base=$(basename "$0")
args=$(getopt -ohx -l help -n "$base" -- "$@")
eval set -- "$args"
while [ $# -gt 0 ]; do
  if [ -n "${opt_prev:-}" ]; then
    eval "$opt_prev=\$1"
    opt_prev=
    shift 1
    continue
  elif [ -n "${opt_append:-}" ]; then
    eval "$opt_append=\"\${$opt_append:-} \$1\""
    opt_append=
    shift 1
    continue
  fi
  case $1 in
  -h | --help)
    usage
    exit 0
    ;;
  -x)
    set -x
    ;;
  --)
    shift
    break 2
    ;;
  esac
  shift 1
done

# shellcheck disable=SC1091
source utils.sh

WORKSPACE="$1"
utils::check_workspace_path "$WORKSPACE"

# shellcheck disable=SC1091
cd "$WORKSPACE" || exit 1

utils::activate_virtual_env

pip install .

pre-commit run --all-files --hook-stage=push

python3 setup.py -q check
python3 setup.py -q sdist bdist_wheel

utils::generate_docs "$WORKSPACE"
