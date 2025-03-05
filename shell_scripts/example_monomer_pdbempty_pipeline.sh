#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPOS_DIR="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
HAPI_DIR="${REPOS_DIR}/hapi"

python $HAPI_DIR/hapi_main.py \
    --pdb-files $REPOS_DIR/pdbs/1ema.pdb \
    --cases-names example_monomer_pdbempty_pipeline \
    --settings-alphafolder $REPOS_DIR/settings_alphafolder/alphafolder_monomer_pdbempty_pipeline.json\
    --settings-paths $REPOS_DIR/settings_paths/paths_teddy.json
    
