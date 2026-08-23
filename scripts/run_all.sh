#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment="mfass-processing"
data_root="$root/data"
full=false
reuse_source=""
reuse_benchmark_root=""
reuse_fasta=""

usage() {
  cat <<'EOF'
Usage: scripts/run_all.sh [--full] [--data-root PATH]
                          [--reuse-source PATH]
                          [--reuse-benchmark-root PATH]
                          [--reuse-fasta PATH]

The compact output is the default. --full additionally builds and verifies
mfass-full.parquet. Every Python step runs in the mfass-processing Mamba
environment.
EOF
}

while (($#)); do
  case "$1" in
    --full)
      full=true
      shift
      ;;
    --data-root|--reuse-source|--reuse-benchmark-root|--reuse-fasta)
      if (($# < 2)); then
        echo "missing value for $1" >&2
        usage >&2
        exit 2
      fi
      option="$1"
      value="$2"
      case "$option" in
        --data-root) data_root="$value" ;;
        --reuse-source) reuse_source="$value" ;;
        --reuse-benchmark-root) reuse_benchmark_root="$value" ;;
        --reuse-fasta) reuse_fasta="$value" ;;
      esac
      shift 2
      ;;
    --data-root=*)
      data_root="${1#*=}"
      shift
      ;;
    --reuse-source=*)
      reuse_source="${1#*=}"
      shift
      ;;
    --reuse-benchmark-root=*)
      reuse_benchmark_root="${1#*=}"
      shift
      ;;
    --reuse-fasta=*)
      reuse_fasta="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

download_args=(--data-root "$data_root")
[[ -z "$reuse_source" ]] || download_args+=(--reuse-source "$reuse_source")
[[ -z "$reuse_benchmark_root" ]] || \
  download_args+=(--reuse-benchmark-root "$reuse_benchmark_root")
[[ -z "$reuse_fasta" ]] || download_args+=(--reuse-fasta "$reuse_fasta")

process_args=(--data-root "$data_root" --output "$root/MFASS")
verify_args=(--data-root "$data_root" --output "$root/MFASS")
if $full; then
  process_args+=(--full)
  verify_args+=(--full)
fi

mamba run --name "$environment" \
  python "$root/scripts/download_data.py" "${download_args[@]}"
mamba run --name "$environment" \
  python "$root/MFASS/process.py" "${process_args[@]}"
mamba run --name "$environment" \
  python "$root/scripts/verify_outputs.py" "${verify_args[@]}"
