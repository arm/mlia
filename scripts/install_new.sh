#!/bin/bash
# Copyright 2021, Arm Ltd.
set -e

# === GLOBAL VARIABLES ===

# Required Python version
PYTHON_REQUIRED_MAJOR_VERSION=3
PYTHON_REQUIRED_MINOR_VERSION=8
PYTHON_REQUIRED_VERSION="$PYTHON_REQUIRED_MAJOR_VERSION.$PYTHON_REQUIRED_MINOR_VERSION"

# Current Python version
PYTHON_MAJOR_VERSION=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR_VERSION=$(python3 -c 'import sys; print(sys.version_info.minor)')
PYTHON_VERSION="$PYTHON_MAJOR_VERSION.$PYTHON_MINOR_VERSION"

# Default temp path where all the dependencies will be downloaded
PACKAGE_DIR=$(mktemp -d -t mlia-XXXXXX)

# ML Inference Advisor params
MLIA_NAME="ML Inference Advisor"
MLIA_VERSION="0.1.1"
MLIA_WHEEL_FILE="mlia-$MLIA_VERSION-py3-none-any.whl"
MLIA_URL="https://artifactory.eu02.arm.com:443/artifactory/ml-tooling.pypi-local/mlia/$MLIA_VERSION/$MLIA_WHEEL_FILE"

# AI Evaluation Toolkit params
AIET_NAME="AI Evaluation Toolkit"
AIET_VERSION="21.12.1"
AIET_WHEEL_FILE="aiet-$AIET_VERSION-py3-none-any.whl"
AIET_URL="https://artifactory.eu02.arm.com:443/artifactory/ml-tooling.pypi-local/aiet/$AIET_VERSION/$AIET_WHEEL_FILE"

# Generic Inference Runner AIET application
AIET_GENERIC_INFERENCE_RUNNER_NAME="Generic Inference Runner"
AIET_GENERIC_INFERENCE_RUNNER_VERSION="0.1.1"
AIET_GENERIC_INFERENCE_RUNNER_ARCHIVE="ethosu_generic_inference_runner_aiet-$AIET_GENERIC_INFERENCE_RUNNER_VERSION.tar.gz"
AIET_GENERIC_INFERENCE_RUNNER_URL="https://artifactory.eu02.arm.com:443/artifactory/ml-tooling.pypi-local/mlia/generic_inference_runner/$AIET_GENERIC_INFERENCE_RUNNER_VERSION/$AIET_GENERIC_INFERENCE_RUNNER_ARCHIVE"

# FVP Corstone-300 Ecosystem params
CS_300_FVP_NAME="FVP Corstone-300 Ecosystem"
CS_300_FVP_DIRECTORY="FVP_Corstone_SSE-300"
CS_300_FVP_DEFAULT_PATHS=("/opt/$CS_300_FVP_DIRECTORY" \
                          "$HOME/$CS_300_FVP_DIRECTORY" \
                          "$PWD/$CS_300_FVP_DIRECTORY")

CS_300_FVP_MODELS_PATH="models/Linux64_GCC-6.4"
CS_300_FVP_AIET_CONFIG_NAME="aiet-config.json"
CS_300_FVP_AIET_CONFIG_RELATIVE_PATH="resources/aiet/system/cs-300"
CS_300_FVP_VALID_PATH=""
CS_300_FVP_VERSION="11.16_26"
CS_300_FVP_TAR_FILE="FVP_Corstone_SSE-300_$CS_300_FVP_VERSION.tgz"
CS_300_FVP_WEB_LINK="https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/$CS_300_FVP_TAR_FILE"
CS_300_FVP_DOWNLOADED="false"
# Name of the virtual environment directory
VENV_PATH=

# Disable pip output by default (including warnings)
PIP_OPTIONS=-qq

# Verbose output disabled by default
VERBOSE=0

# === FUNCTIONS ===

log() {
    echo -e "$@"
}

verbose () {
    if [[ $VERBOSE -eq 1 ]]; then
        log "$@"
    fi
}

error() {
    # shellcheck disable=SC2145
    log "ERROR: $@"
}

check_fvp_instance() {
    local FVP_PATH=$1

    verbose "Checking if path \"$FVP_PATH\" contains a valid $CS_300_FVP_NAME instance ..."

    # Remove the trailing slash from the FVP path if present
    FVP_PATH="${FVP_PATH%/}"

    # Build the paths to the Ethos-U55 and Ethos-U65 model executables
    ETHOS_U55_MODEL="$FVP_PATH/FVP_Corstone_SSE-300_Ethos-U55"
    ETHOS_U65_MODEL="$FVP_PATH/FVP_Corstone_SSE-300_Ethos-U65"

    # Check if both the model executables exist
    if [ -f "$ETHOS_U55_MODEL" ] && [ -f "$ETHOS_U65_MODEL" ]; then
        verbose "Path \"$FVP_PATH\" contains a valid $CS_300_FVP_NAME instance"
        CS_300_FVP_VALID_PATH="$FVP_PATH"
    else
        verbose "Path \"$FVP_PATH\" does NOT contain a valid $CS_300_FVP_NAME instance"
        CS_300_FVP_VALID_PATH=""
    fi
}

check_fvp_path() {
    local FVP_PATH=$1

    # Check if a path has been passed as an argument
    if [ -n "$FVP_PATH" ]; then
        # Remove the trailing slash from the FVP path if present
        FVP_PATH="${FVP_PATH%/}"

        # Check the current FVP path
        check_fvp_instance "$FVP_PATH"
        # Check if the current FVP path is valid, otherwise check if the FVP path refers to the "models" directory of the FVP
        if [ -z "$CS_300_FVP_VALID_PATH" ] && [[ "$FVP_PATH" != *"$CS_300_FVP_MODELS_PATH" ]]; then
            # Check also the "models" sub directory
            check_fvp_instance "$FVP_PATH/$CS_300_FVP_MODELS_PATH"
        fi

        # Do not check the default paths
        return
    fi

    verbose "No path specified, checking the default paths ..."

    # No path specified, check the default paths
    for FVP_PATH in "${CS_300_FVP_DEFAULT_PATHS[@]}"; do
        # Remove the trailing slash from the FVP path if present
        FVP_PATH="${FVP_PATH%/}"

        # Check the current FVP path
        check_fvp_instance "$FVP_PATH"
        # Check if the current FVP path is valid
        if [ -n "$CS_300_FVP_VALID_PATH" ]; then
            # The current FVP path is valid
            return

        # Check if the FVP path refers to the "models" directory of the FVP
        elif [[ "$FVP_PATH" != *"$CS_300_FVP_MODELS_PATH" ]]; then
            # Check also the "models" sub directory
            check_fvp_instance "$FVP_PATH/$CS_300_FVP_MODELS_PATH"
            # Check if the current FVP path is valid
            if [ -n "$CS_300_FVP_VALID_PATH" ]; then
                # The current FVP path is valid
                return
            fi
        fi
    done
}

init_packages() {
    MLIA_PACKAGE="$PACKAGE_DIR/$MLIA_WHEEL_FILE"
    AIET_PACKAGE="$PACKAGE_DIR/$AIET_WHEEL_FILE"
    CS_300_APP_PACKAGE="$PACKAGE_DIR/$AIET_GENERIC_INFERENCE_RUNNER_ARCHIVE"
}

check_path() {
    if [ -z "$1" ]; then
        error "No path specified to check"
        usage
    fi

    if [ ! -d "$1" ]; then
        error "Path $1 does not exist or it's not a directory. Exiting ..."
        exit 1
    fi
}

check_file() {
    if [ -z "$1" ]; then
        error "No file specified to check"
        usage
    fi

    if [ ! -f "$1" ]; then
        error "File $1 does not exist. Exiting ..."
        exit 1
    fi
}

check_package() {
    if [ -z "$1" ]; then
        error "No package specified to check"
        usage
    fi

    check_file "$1"
}

check_packages() {
    check_package "$AIET_PACKAGE"
    check_package "$MLIA_PACKAGE"
    check_package "$CS_300_APP_PACKAGE"
}

create_and_init_virtual_env() {
    # Create the virtual environment
    python3 -m venv "$1"

    # Activate the virtual environment
    # shellcheck disable=SC1091
    source "$1/bin/activate"

    # Update setuptools in the virtual environment
    pip install -U pip setuptools
}

install_aiet() {
    # Install the AI Evaluation Toolkit, but not the systems or the applications
    pip $PIP_OPTIONS install "$AIET_PACKAGE"
}

configure_aiet() {
    # Install the AI Evaluation Toolkit systems and applications
    aiet system install -s "$CS_300_FVP_VALID_PATH"
    aiet application install -s "$CS_300_APP_PACKAGE"
}

install_mlia() {
    # Install the Inference Advisor
    pip $PIP_OPTIONS install "$MLIA_PACKAGE"
}

download_fvp() {
    CS_300_FVP_PATH="$PACKAGE_DIR"
    wget -nv $CS_300_FVP_WEB_LINK -O "$CS_300_FVP_PATH/$CS_300_FVP_TAR_FILE"
    CS_300_FVP_DOWNLOADED="true"
}

install_fvp() {
    CORSTONE_PACKAGE="$CS_300_FVP_PATH/$CS_300_FVP_TAR_FILE"
    tar xzf "$CORSTONE_PACKAGE" -C "$CS_300_FVP_PATH"
    "$CS_300_FVP_PATH/FVP_Corstone_SSE-300.sh" -q --i-agree-to-the-contained-eula -d "$CS_300_FVP_PATH" --nointeractive
    CS_300_FVP_VALID_PATH="$CS_300_FVP_PATH/$CS_300_FVP_MODELS_PATH"
}

print_manual_fvp_installation_instructions() {
    echo "For downloading the FVP: wget -nv $CS_300_FVP_WEB_LINK -O $PACKAGE_DIR/$CS_300_FVP_TAR_FILE"
    echo "For installing the FVP, please run the install_new.sh with the command line for example like this: install_new.sh -f your_fvp_path -e name_of_your_env"
}

download_maybe() {
    local MS="Please confirm downloading FVP from developer.arm.com? y/[n]: "
    local TMOUT=10

    while true; do
            if ! read -p "$MS" -r; then
                    echo "Timed out so nothing will be downloaded. Exiting ..."
                    exit 0
            fi
            case $REPLY in
                    [yY]*)
                            download_fvp
                            break;;
                    [nN]*)
                            print_manual_fvp_installation_instructions
                            echo "Nothing downloaded. Exiting ..."
                            exit 0;;
                    *)
                            echo "Sorry, try again" >&2
            esac
    done
}

usage() {
    USAGE_NOTE="$MLIA_NAME installation script

This script creates a virtual environment and installs the required packages:
  - $MLIA_NAME
  - $AIET_NAME
  - $CS_300_FVP_NAME
  - Ethos-U55/65 Generic Inference Runner

Usage: $0 [-v] [-f fvp_path] [-d package_dir] -e venv_dir

Options:
  -h Print this help message and exit
  -v [optional] Enable verbose output
  -f [optional] Path to a local instance of the $CS_300_FVP_NAME
     If not specified, the script will check in the following locations in that order:
      1. /opt/FVP_Corstone_SSE-300
      2. $HOME/FVP_Corstone_SSE-300
      3. $PWD/FVP_Corstone_SSE-300
  -d [optional] Path to the directory where to download the install packages
  -e The name of the virtual environment directory
  "

    log "$USAGE_NOTE"
    exit 1
}

# === ENTRY POINT ===

# Check the Python version
if [[ $PYTHON_MAJOR_VERSION -lt $PYTHON_REQUIRED_MAJOR_VERSION ]]; then
    error "The minimum Python version required is $PYTHON_REQUIRED_VERSION, you have $PYTHON_VERSION"
    exit 1
fi
if [[ $PYTHON_MINOR_VERSION -lt $PYTHON_REQUIRED_MINOR_VERSION ]]; then
    error "The minimum Python version required is $PYTHON_REQUIRED_VERSION, you have $PYTHON_VERSION"
    exit 1
fi

# Argument parsing
while getopts "hvf:e:d:" o; do
    case "${o}" in
        f)
            CS_300_FVP_PATH=${OPTARG}
            ;;
        d)
            PACKAGE_DIR=${OPTARG}
            ;;
        e)
            VENV_PATH=${OPTARG}
            ;;
        v)
            VERBOSE=1
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

# Argument checking
if [ -n "$CS_300_FVP_PATH" ] && [ ! -d "$CS_300_FVP_PATH" ]; then
    error "\"$CS_300_FVP_PATH\" is not a directory"
    usage
fi

if [ -z "$PACKAGE_DIR" ]; then
    error "No package directory provided"
    usage
fi

if [ ! -d "$PACKAGE_DIR" ]; then
    error "\"$PACKAGE_DIR\" is not a directory"
    usage
fi

if [ -z "$VENV_PATH" ]; then
    error "No directory provided for the virtual environment"
    usage
fi

if [ -d "$VENV_PATH" ]; then
    error "Directory \"$VENV_PATH\" already exists"
    usage
fi

# Installation process
log "\nInstalling the $MLIA_NAME and its dependencies ..."

log "\nCreating virtual environment \"$VENV_PATH\" ..."
create_and_init_virtual_env "$VENV_PATH"

log "\nChecking local $CS_300_FVP_NAME instance ..."
check_fvp_path "$CS_300_FVP_PATH"

# If no FVP file exists, we need to download and install them
if [ -z "$CS_300_FVP_VALID_PATH" ]; then
    download_maybe
    log "\nDownloaded the $CS_300_FVP_NAME version $CS_300_FVP_VERSION to \"$CS_300_FVP_PATH\" ..."
    if [ "$CS_300_FVP_DOWNLOADED" == "true" ]; then
        install_fvp
    fi
fi

log "\nUsing the local instance of the $CS_300_FVP_NAME version $CS_300_FVP_VERSION at \"$CS_300_FVP_VALID_PATH\" ..."

# Downloading components
log "\nDownloading the $MLIA_NAME version $MLIA_VERSION to \"$PACKAGE_DIR\" ..."
wget "$MLIA_URL" -O "$PACKAGE_DIR/$MLIA_WHEEL_FILE"

log "\nDownloading the $AIET_NAME version $AIET_VERSION to \"$PACKAGE_DIR\" ..."
wget "$AIET_URL" -O "$PACKAGE_DIR/$AIET_WHEEL_FILE"

log "\nDownloading the $AIET_GENERIC_INFERENCE_RUNNER_NAME version $AIET_GENERIC_INFERENCE_RUNNER_VERSION to \"$PACKAGE_DIR\" ..."
wget "$AIET_GENERIC_INFERENCE_RUNNER_URL" -O "$PACKAGE_DIR/$AIET_GENERIC_INFERENCE_RUNNER_ARCHIVE"

verbose "Checking packages ..."
init_packages "$PACKAGE_DIR"
check_packages

# Installing components
log "\nInstalling the $AIET_NAME ..."
# The AI Evaluation Toolkit has to be installed first (before the Inference Advisor)
# due to a less strict dependency from Vela than that the Inference Advisor, but without
# installing the systems and the applications just yet. Installing the systems requires
# the configuration files included in the Inference Advisor
install_aiet

log "\nInstalling the $MLIA_NAME ..."
# The Inference Advisor has to be installed after the AI Evaluation Toolkit, since it has
# a stronger dependency on Vela
install_mlia

# Get the Inference Advisor install path (i.e. its location in site-packages)
MLIA_PACKAGE_PATH=$(python3 -c "import os, mlia; print(os.path.dirname(mlia.__file__))")
check_path "$MLIA_PACKAGE_PATH"
verbose "MLIA package found at \"$MLIA_PACKAGE_PATH\""

verbose "Checking the $CS_300_FVP_NAME configuration file ... "
# Check that the configuration file for the AI Evaluation Toolkit system is included in
# the Inference Advisor
CS_300_FVP_AIET_CONFIG_PATH="$MLIA_PACKAGE_PATH/$CS_300_FVP_AIET_CONFIG_RELATIVE_PATH"
CS_300_FVP_AIET_CONFIG="$CS_300_FVP_AIET_CONFIG_PATH/$CS_300_FVP_AIET_CONFIG_NAME"
check_file "$CS_300_FVP_AIET_CONFIG"

# Prepare the FVP package for AIET
log "\nInitializing the $CS_300_FVP_NAME instance at \"$CS_300_FVP_VALID_PATH\" ..."
# Copy the AIET configuration file to the FVP directory
cp -f "$CS_300_FVP_AIET_CONFIG" "$CS_300_FVP_VALID_PATH"

log "\nConfiguring the $AIET_NAME ..."
# Installing the AI Evaluation Toolkit systems and applications using the configuration file
# included in the Inference Advisor
configure_aiet

log "\nInstallation complete"
log "Please activate the virtual environment \"$VENV_PATH\" to start working with the $MLIA_NAME [source $VENV_PATH/bin/activate]"

# All done
exit 0
