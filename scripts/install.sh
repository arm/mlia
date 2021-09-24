#!/bin/bash
# Copyright 2021, Arm Ltd.
set -e

init_packages() {
    AIET_PACKAGE="$PACKAGE_DIR/aiet-21.9.0-py3-none-any.whl"
    MLIA_PACKAGE="$PACKAGE_DIR/mlia-0.1-py3-none-any.whl"

    CORSTONE_PACKAGE="$PACKAGE_DIR/fvp_corstone_sse-300_ethos-u55-21.08.0.tar.gz"
    CORSTONE_PACKAGE_APPS="$PACKAGE_DIR/ethosu_eval_platform_release_aiet-21.08.0.tar.gz"

    SGM_PACKAGE="$PACKAGE_DIR/sgm775_ethosu_platform-21.03.0.tar.gz"
    SGM_PACKAGE_OSS="$PACKAGE_DIR/sgm775_ethosu_platform-21.08.0-oss.tar.gz"
    SGM_PACKAGE_DIR="$PACKAGE_DIR/sgm775_ethosu_platform"
    SGM_PACKAGE_APPS="$PACKAGE_DIR/ethosU65_eval_app-21.08.0.tar.gz"
}

check_package() {
    if [ -z "$1" ]; then
        echo -e "Configuration error\n"
        usage
    fi

    if [ ! -f "$1" ]; then
        echo "Error: package $1 does not exist"
        exit 1
    fi
}

check_packages() {
    check_package "$AIET_PACKAGE"

    check_package "$CORSTONE_PACKAGE"
    check_package "$CORSTONE_PACKAGE_APPS"

    check_package "$SGM_PACKAGE"
    check_package "$SGM_PACKAGE_OSS"
    check_package "$SGM_PACKAGE_APPS"

    check_package "$MLIA_PACKAGE"
}

create_and_init_virtual_env() {
    # shellcheck disable=SC2086
    virtualenv $VIRT_ENV_OPTIONS "$1"
    # shellcheck disable=SC1090
    source "$1/bin/activate"
}

install_aiet() {
    # shellcheck disable=SC2086
    pip $PIP_OPTIONS install "$AIET_PACKAGE"

    aiet system install -s "$CORSTONE_PACKAGE"
    aiet software install -s "$CORSTONE_PACKAGE_APPS"

    tar xzf "$SGM_PACKAGE" -C "$PACKAGE_DIR"
    tar xzf "$SGM_PACKAGE_OSS" -C "$PACKAGE_DIR"
    aiet system install -s "$SGM_PACKAGE_DIR"
    aiet software install -s "$SGM_PACKAGE_APPS"
}

install_mlia() {
    # shellcheck disable=SC2086
    pip $PIP_OPTIONS install "$MLIA_PACKAGE"
}

usage() {
    USAGE_NOTE="ML Inference Advisor installation script

This script creates virtual environment and installs the required packages:
  - ML Inference Advisor
  - AI Evaluation Toolkit
  - FVP Corstone-300 Ecosystem
  - Ethos-U55 Eval Platform
  - SGM-775
  - Ethos-U65 Eval Platform

Usage: $0 [-v] -d package_dir -e venv_dir

Options:
  -h print this help message and exit
  -v enable verbose output
  -d path to the directory containing the install packages
  -e virtual environment directory name"

    echo "$USAGE_NOTE"
    exit 1
}

# disable virtualenv output by default
VIRT_ENV_OPTIONS=-q
# disable pip output by default (including warnings)
PIP_OPTIONS=-qq
# path to the directory with MLIA and AIET packages
PACKAGE_DIR=
# name of the virt env directory
VENV_PATH=

while getopts "hvd:e:" o; do
    case "${o}" in
        d)
            PACKAGE_DIR=${OPTARG}
            ;;
        e)
            VENV_PATH=${OPTARG}
            ;;
        v)
            VIRT_ENV_OPTIONS=
            PIP_OPTIONS=
            ;;
        h)
            usage
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "$PACKAGE_DIR" ]; then
    echo -e "Error: No package directory provided\n"
    usage
fi

if [ ! -d "$PACKAGE_DIR" ]; then
    echo -e "Error: $PACKAGE_DIR is not a directory\n"
    usage
fi

if [ -z "$VENV_PATH" ]; then
    echo -e "Error: No directory name for the virtual environment provided\n"
    usage
fi

if [ -d "$VENV_PATH" ]; then
    echo -e "Error: Directory $VENV_PATH already exists\n"
    usage
fi

echo "Checking packages ..."
init_packages "$PACKAGE_DIR"
check_packages

echo "Creating virtual environment [$VENV_PATH] ..."
create_and_init_virtual_env "$VENV_PATH"

echo "Installing AI Evaluation Toolkit ..."
install_aiet

echo "Installing ML Inference Advisor ..."
install_mlia

echo "Installation complete."
echo "Please activate the virtual environment $VENV_PATH to start working with the ML Inference Advisor."
