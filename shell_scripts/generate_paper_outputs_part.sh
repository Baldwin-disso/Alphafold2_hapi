#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPOS_DIR="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
HAPI_DIR="${REPOS_DIR}/hapi"

python $HAPI_DIR/predict_ppi.py \
    --settings-alphafolder $REPOS_DIR/settings_alphafolder/alphafolder_ppi.json \
    --settings-paths $REPOS_DIR/settings_paths/paths.json  \
    --npz-file $REPOS_DIR/npz_generation/ligand_receptor_combinations.npz \
    --ppi-database-path $REPOS_DIR/database \
    --output-dir $REPOS_DIR/outputs/output_database \
    --cache-dir $REPOS_DIR/cache/cache_database \
    --array-task-id 2 \
    --array-size 100 \
    --no-output-pdbs \
    --save-metrics