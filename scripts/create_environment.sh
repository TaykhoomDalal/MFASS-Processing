#!/usr/bin/env bash
set -euo pipefail

name="${1:-mfass-processing}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if mamba env list | awk '{print $1}' | grep -qx "$name"; then
  mamba env update --name "$name" --file "$root/environment.yml" --prune --yes
else
  mamba env create --name "$name" --file "$root/environment.yml" --yes
fi
