#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
name="mfass-processing"
locked=false

usage() {
  cat <<'EOF'
Usage: scripts/create_environment.sh [--locked] [--name NAME]

Without --locked, solve environment.yml. With --locked, recreate the named
Linux environment from the explicit conda and hash-locked pip specifications.
EOF
}

while (($#)); do
  case "$1" in
    --locked)
      locked=true
      shift
      ;;
    --name)
      if (($# < 2)); then
        echo "missing value for --name" >&2
        usage >&2
        exit 2
      fi
      name="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      name="$1"
      shift
      ;;
  esac
done

if $locked; then
  if mamba env list | awk '{print $1}' | grep -qx "$name"; then
    mamba env remove --name "$name" --yes
  fi
  mamba create \
    --name "$name" \
    --file "$root/environment-lock-linux-64.txt" \
    --yes
  mamba run --name "$name" python -m pip install \
    --no-deps \
    --require-hashes \
    --requirement "$root/requirements-lock.txt"
  exit
fi

if mamba env list | awk '{print $1}' | grep -qx "$name"; then
  mamba env update \
    --name "$name" \
    --file "$root/environment.yml" \
    --prune \
    --yes
else
  mamba env create \
    --name "$name" \
    --file "$root/environment.yml" \
    --yes
fi
