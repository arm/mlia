#!/bin/bash

# Copyright 2021, Arm Ltd.
# SPDX-License-Identifier: Apache-2.0

set -e
set -u

if ! type -p allmark > /dev/null
then
  printf "error: can't find allmark\n" >&2
  printf "\n" >&2
  printf "  allmark project:\n" >&2
  printf "    https://allmark.io/\n" >&2
  printf "\n" >&2
  printf "  Linux install hint:\n" >&2
  printf "\n" >&2
  printf "    sudo su\n" >&2
  printf "    curl -o /usr/local/bin/allmark -s --insecure https://allmark.io/bin/files/allmark_linux_amd64 \n" >&2
  printf "    chmod +x /usr/local/bin/allmark\n" >&2
  printf "\n" >&2
  exit 2
fi
allmark serve --livereload
