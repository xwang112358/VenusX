#!/usr/bin/env bash
# Download all VenusX InterPro CSV splits and AlphaFold2 PDB archives.
#
# Usage:
#   bash script/data/download_interpro_datasets.sh
#   bash script/data/download_interpro_datasets.sh --output_dir /path/to/output

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python "${REPO_ROOT}/evaluation/get_dataset_csv.py" --all "$@"
