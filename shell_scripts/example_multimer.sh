#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPOS_DIR="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
HAPI_DIR="${REPOS_DIR}/hapi"

python $HAPI_DIR/hapi_main.py \
    --fasta-files $REPOS_DIR/fastas/dummy_multimer.fasta \
    --cases-names example_multimer \
    --settings-alphafolder $REPOS_DIR/settings_alphafolder/alphafolder_multimer.json \
    --settings-paths $REPOS_DIR/settings_paths/paths_teddy.json



