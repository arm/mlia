#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Please, provide workspace path"
  exit 1
fi

WORKSPACE=$1
if [ ! -d "$WORKSPACE" ]; then
  echo "$WORKSPACE is not valid directory path"
  exit 1
fi

# shellcheck disable=SC1091
cd "$WORKSPACE" || exit 1

# shellcheck disable=SC1091
source "/home/foo/v/bin/activate"

if [[ -n "$AIET_ARTIFACT_PATH" ]]; then
  echo "Install AIET from $AIET_ARTIFACT_PATH"
  pip install "$AIET_ARTIFACT_PATH"
fi

echo "Install application from the local directory"
pip install .

export PYTHONUNBUFFERED=1
pytest --collect-only -m e2e
pytest -v --capture=tee-sys --durations=0 --durations-min=5 --tb=long --junit-xml=report/report.xml -m e2e
