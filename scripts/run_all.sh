#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$root/scripts/download_data.py" "$@"
python "$root/MFASS/process.py" --full
python "$root/scripts/verify_outputs.py" --full
