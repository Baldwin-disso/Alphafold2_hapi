#!/bin/bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"



${SCRIPT_DIR}/example_monomer_empty_pipeline.sh 
${SCRIPT_DIR}/example_monomer_pdb_pipeline.sh 
${SCRIPT_DIR}/example_monomer_pdbempty_pipeline.sh 
${SCRIPT_DIR}/example_monomer.sh 
${SCRIPT_DIR}/example_multimer.sh 
${SCRIPT_DIR}/example_ppi_prediction.sh 
${SCRIPT_DIR}/generate_paper_outputs_part.sh 
