#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPOS_DIR="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
HAPI_DIR="${REPOS_DIR}/hapi"

python $HAPI_DIR/predict_ppi.py \
    --settings-alphafolder $REPOS_DIR/settings_alphafolder/alphafolder_ppi.json \
    --settings-paths $REPOS_DIR/settings_paths/paths.json  \
    --pdb-dir $REPOS_DIR/pdbs/pdb_dir \
    --output-dir $REPOS_DIR/outputs/ppi_outputs \
    --cache-dir $REPOS_DIR/cache/ppi_cache\
    --save-metrics 
    