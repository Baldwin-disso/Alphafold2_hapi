# Copyright 2025 - Lena Conesson, Baldwin Dumortier, Gabriel Krouk 
#
# This code includes modified code coming from AlphaFold 2, 
# which is licensed under the Apache License 2.0
# Copyright 2021 DeepMind Technologies Limited.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.    

import argparse
import json
from pathlib import Path


PATH_TEMPLATE_DB = {
    "data_dir": "",
    "uniref90_database_path": "uniref90/uniref90.fasta",
    "mgnify_database_path": "mgnify/mgy_clusters_2022_05.fa",
    "template_mmcif_dir": "pdb_mmcif/mmcif_files",
    "obsolete_pdbs_path": "pdb_mmcif/obsolete.dat",
    "bfd_database_path": "bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt",
    "uniref30_database_path": "uniref30/UniRef30_2021_03",
    "uniprot_database_path": "uniprot/uniprot.fasta",
    "pdb70_database_path": "pdb70/pdb70",
    "pdb_seqres_database_path": "pdb_seqres/pdb_seqres.txt",
}

PATH_TEMPLATE_BIN = {"jackhmmer_binary_path": "jackhmmer", 
    "hhblits_binary_path": "hhblits",
    "hhsearch_binary_path": "hhsearch",
    "hmmsearch_binary_path": "hmmsearch",
    "hmmbuild_binary_path": "hmmbuild",
    "kalign_binary_path": "kalign"
}

PATH_TEMPLATE_OTHER = {
    "small_bfd_database_path": "",
}


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="generation of json path file")

    parser.add_argument('--bin-root', default=None, required=True, type=str, help="provide root directory of binary tools")
    parser.add_argument('--db-root', default=None, required=True, type=str, help="provide root directory of databases")
    parser.add_argument('--output-file', default='settings_paths/paths.json', type=str, help="provide root directory of binary tools")
    args = parser.parse_args()

    Path(args.output_file).parent.mkdir(exist_ok=True, parents=True)
    bin_dict = { k : str(Path(args.bin_root,v)) for k,v  in PATH_TEMPLATE_BIN.items()}
    db_dict = { k : str(Path(args.db_root,v)) for k,v  in PATH_TEMPLATE_DB.items()}
    global_dict = {**bin_dict, **db_dict, **PATH_TEMPLATE_OTHER}

    with open(args.output_file, "w") as json_file:
      json.dump(global_dict, json_file, indent=4)
    
    print(f"paths were recorded in {args.output_file}")