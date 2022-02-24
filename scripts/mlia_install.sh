#!/bin/bash
# Copyright (C) 2021-2022, Arm Ltd.
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
MLIA_VERSION="0.2.0"
MLIA_WHEEL_FILE="mlia-$MLIA_VERSION-py3-none-any.whl"
MLIA_URL="https://artifactory.eu02.arm.com:443/artifactory/ml-tooling.pypi-local/mlia/$MLIA_VERSION/$MLIA_WHEEL_FILE"

# AI Evaluation Toolkit params
AIET_NAME="AI Evaluation Toolkit"
AIET_WHEEL_FILE_PATTERN="aiet-*-py3-none-any.whl"
AIET_WHEEL_FILE_RELATIVE_PATH="resources/aiet/package/"

# AI Evaluation Toolkit application params
AIET_APPLICATIONS_RELATIVE_PATH="resources/aiet/applications"

# FVP Corstone-300 Ecosystem params
CS_300_FVP_NAME="FVP Corstone-300 Ecosystem"
CS_300_FVP_DIRECTORY="FVP_Corstone_SSE-300"
CS_300_FVP_DEFAULT_PATHS=("/opt/$CS_300_FVP_DIRECTORY" \
                          "$HOME/$CS_300_FVP_DIRECTORY" \
                          "$PWD/$CS_300_FVP_DIRECTORY")
CS_300_FVP_MODELS_PATH="models/Linux64_GCC-6.4"
CS_300_FVP_AIET_CONFIG_NAME="aiet-config.json"
CS_300_FVP_AIET_CONFIG_RELATIVE_PATH="resources/aiet/systems/cs-300"
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

create_and_init_virtual_env() {
    # Create the virtual environment
    python3 -m venv "$1"

    # Activate the virtual environment
    # shellcheck disable=SC1091
    source "$1/bin/activate"

    # Update setuptools in the virtual environment
    pip $PIP_OPTIONS install -U pip setuptools
}

install_aiet() {
    # Install the AI Evaluation Toolkit, but not the systems or the applications
    pip $PIP_OPTIONS install "$AIET_PACKAGE"
}

configure_aiet() {
    # Copy the AIET configuration file to the FVP directory
    verbose "Initializing the $CS_300_FVP_NAME instance at \"$CS_300_FVP_VALID_PATH\" ..."
    cp -f "$CS_300_FVP_AIET_CONFIG" "$CS_300_FVP_VALID_PATH"

    # Install the AI Evaluation Toolkit systems and applications
    verbose "Installing the $AIET_NAME systems ..."
    aiet system install -s "$CS_300_FVP_VALID_PATH"
    verbose "Installing the $AIET_NAME applications ..."
    AIET_APPLICATIONS_TEMP_DIR="$MLIA_TEMP_PACKAGE_DIR/$AIET_APPLICATIONS_RELATIVE_PATH"
    find "$AIET_APPLICATIONS_TEMP_DIR" -mindepth 1 -maxdepth 1 -type f -name "*.tar.gz" \
      -exec sh -c 'aiet application install -s ${0}' {} \;
}

install_mlia() {
    # Install the Inference Advisor
    pip $PIP_OPTIONS install "$MLIA_PACKAGE"
}

download_fvp() {
    log "\nDownloading the $CS_300_FVP_NAME version $CS_300_FVP_VERSION to \"$PACKAGE_DIR\" ..."

    CS_300_FVP_PATH="$PACKAGE_DIR"
    wget -nv $CS_300_FVP_WEB_LINK -O "$CS_300_FVP_PATH/$CS_300_FVP_TAR_FILE"

    CS_300_FVP_DOWNLOADED="true"
}

install_fvp() {
    local FVP_INSTALL_PATH=$1
    CORSTONE_PACKAGE="$CS_300_FVP_PATH/$CS_300_FVP_TAR_FILE"
    tar xzf "$CORSTONE_PACKAGE" -C "$FVP_INSTALL_PATH"
    "$FVP_INSTALL_PATH/FVP_Corstone_SSE-300.sh" -q --i-agree-to-the-contained-eula -d "$FVP_INSTALL_PATH" --nointeractive
    CS_300_FVP_VALID_PATH="$FVP_INSTALL_PATH/$CS_300_FVP_MODELS_PATH"
}

print_manual_fvp_installation_instructions() {
    log "\nFor downloading the FVP, use for example: wget $CS_300_FVP_WEB_LINK -O $PACKAGE_DIR/$CS_300_FVP_TAR_FILE"
    log "\nFor installing the FVP, please untar the FVP archive, run the install script FVP_Corstone_SSE-300.sh and follow the instructions."
    log "Then, run mlia_install.sh again specifying the FVP path with the '-f' option. For example: mlia_install.sh -f your_fvp_path -e name_of_your_venv"
}

download_maybe() {
    local MS="Please confirm downloading FVP from developer.arm.com? y/[n]: "
    local TMOUT=10

    while true; do
            log ""
            if ! read -p "$MS" -r; then
                    log "\nTimed out so nothing will be downloaded. Exiting ..."
                    exit 0
            fi
            case $REPLY in
                    [yY]*)
                            download_fvp
                            break;;
                    [nN]* | "")
                            print_manual_fvp_installation_instructions
                            log "\nNothing downloaded. Exiting ..."
                            exit 0;;
                    *)
                            log "Sorry, try again" >&2
            esac
    done
}

usage() {
    USAGE_NOTE="$MLIA_NAME installation script

This script creates a virtual environment and installs the required packages:
  - $MLIA_NAME
  - $AIET_NAME
  - $CS_300_FVP_NAME
  - Ethos-U55/65 Inference Runner applications

Usage: $0 [-v] [-f fvp_path] [-d package_dir] -e venv_dir

Options:
  -h Print this help message and exit
  -v [optional] Enable verbose output
  -f [optional] Path to a local instance of the $CS_300_FVP_NAME
     If not specified, the script will check in the following locations in this order:
      1. /opt/FVP_Corstone_SSE-300
      2. $HOME/FVP_Corstone_SSE-300
      3. $PWD/FVP_Corstone_SSE-300
  -d [optional] Path to the directory where to download the packages to install
  -e The name of the virtual environment directory
  "

    log "$USAGE_NOTE"
    exit 1
}

# === ENTRY POINT ===

# Check the Python version
PYTHON_VERSION_ERROR_MESSAGE="The minimum Python version required is $PYTHON_REQUIRED_VERSION, you have $PYTHON_VERSION"
if [[ $PYTHON_MAJOR_VERSION -lt $PYTHON_REQUIRED_MAJOR_VERSION ]]; then
    error "$PYTHON_VERSION_ERROR_MESSAGE"
    exit 1
fi
if [[ $PYTHON_MINOR_VERSION -lt $PYTHON_REQUIRED_MINOR_VERSION ]]; then
    error "$PYTHON_VERSION_ERROR_MESSAGE"
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

verbose "Using directory \"$PACKAGE_DIR\" to download the packages ..."

# Installation process
log "\nInstalling the $MLIA_NAME and its dependencies ..."

log "\nCreating virtual environment \"$VENV_PATH\" ..."
create_and_init_virtual_env "$VENV_PATH"

log "\nChecking local $CS_300_FVP_NAME instance ..."
check_fvp_path "$CS_300_FVP_PATH"

# If no FVP file exists, we need to download and install them
if [ -z "$CS_300_FVP_VALID_PATH" ]; then
    download_maybe
    log "\nSuccessfully downloaded from developer.arm.com: the $CS_300_FVP_NAME version $CS_300_FVP_VERSION to \"$CS_300_FVP_PATH\" ..."
    if [ "$CS_300_FVP_DOWNLOADED" == "true" ]; then
        mkdir -p "$PWD/$CS_300_FVP_DIRECTORY"
        install_fvp "$PWD/$CS_300_FVP_DIRECTORY"
    fi
fi

log "\nUsing the local instance of the $CS_300_FVP_NAME version $CS_300_FVP_VERSION at \"$CS_300_FVP_VALID_PATH\" ..."

log "\nDownloading the $MLIA_NAME version $MLIA_VERSION to \"$PACKAGE_DIR\" ..."
MLIA_PACKAGE="$PACKAGE_DIR/$MLIA_WHEEL_FILE"
wget -nv "$MLIA_URL" -O "$MLIA_PACKAGE"
check_package "$MLIA_PACKAGE"

log "\nExtracting the $AIET_NAME from the $MLIA_NAME package ..."
TEMP_DIR=$(mktemp -d -t mlia-XXXXXX)
wheel unpack "$PACKAGE_DIR/$MLIA_WHEEL_FILE" -d "$TEMP_DIR"
MLIA_TEMP_PACKAGE_DIR="$TEMP_DIR/mlia-$MLIA_VERSION/mlia"
AIET_WHEEL_FILE_PATH="$MLIA_TEMP_PACKAGE_DIR/$AIET_WHEEL_FILE_RELATIVE_PATH"
# shellcheck disable=SC2206
AIET_WHEEL_FILES=( $AIET_WHEEL_FILE_PATH/$AIET_WHEEL_FILE_PATTERN )
AIET_PACKAGE="${AIET_WHEEL_FILES[0]}"
check_package "$AIET_PACKAGE"

log "\nExtracting the $CS_300_FVP_NAME configuration file ... "
# Check that the configuration file for the AI Evaluation Toolkit system is included in
# the Inference Advisor
CS_300_FVP_AIET_CONFIG_PATH="$MLIA_TEMP_PACKAGE_DIR/$CS_300_FVP_AIET_CONFIG_RELATIVE_PATH"
CS_300_FVP_AIET_CONFIG="$CS_300_FVP_AIET_CONFIG_PATH/$CS_300_FVP_AIET_CONFIG_NAME"
check_file "$CS_300_FVP_AIET_CONFIG"

log "\nInstalling the $AIET_NAME ..."
# The AI Evaluation Toolkit has to be installed first (before the Inference Advisor)
# due to a less strict dependency from Vela than that the Inference Advisor, but without
# installing the systems and the applications just yet. Installing the systems requires
# the configuration files included in the Inference Advisor
install_aiet

log "\nConfiguring the $AIET_NAME ..."
# Installing the AI Evaluation Toolkit systems and applications using the configuration file
# included in the Inference Advisor
configure_aiet

log "\nInstalling the $MLIA_NAME ..."
# The Inference Advisor has to be installed after the AI Evaluation Toolkit, since it has
# a stronger dependency on Vela
install_mlia

log "\nInstallation complete"
log "Please activate the virtual environment \"$VENV_PATH\" to start working with the $MLIA_NAME [source $VENV_PATH/bin/activate]"

# All done
exit 0
