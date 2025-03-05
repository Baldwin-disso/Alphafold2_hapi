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
from hapi import hapi_ppi
import os
from pathlib import Path

import numpy as np
from Bio import PDB


def number_interaction(file_path):
    data = np.load(file_path, allow_pickle=True)
    keys = list(data.keys())
    if keys:
        first_key = keys[0]
        first_value = data[first_key]
        nb_interaction = len(first_value)
        data.close()
    else:
        print("The .npz file is empty or does not contain any keys.")
    return nb_interaction

def calculate_sequence_length(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    sequence_length = 0
    for model in structure:
        for chain in model:
            sequence_length += len([residue for residue in chain if PDB.is_aa(residue, standard=True)])
    return sequence_length




def pdb_paths_from_database(path_database_pdb,file_path_npz,nb_data_per_task, array_task_id):

    data = np.load(file_path_npz, allow_pickle=True)

    keys = list(data.keys())
    selected_paths = []
    selected_names = []
    if keys:
        first_key = keys[0]
        first_value = data[first_key]
        start_index = array_task_id * nb_data_per_task
        end_index = (array_task_id + 1) * nb_data_per_task

        print(f"Array_task_id {array_task_id}") 
        print(f"\n  we run couples from  {start_index} to {end_index} (excluded)")

        for i in range(start_index, min(end_index, len(first_value))):
            ligand = first_value[i]['ligand']
            receptor = first_value[i]['receptor']
            selected_paths.append(f"{path_database_pdb}/receptor/{receptor}.pdb")
            selected_paths.append(f"{path_database_pdb}/peptide/{ligand}.pdb")
            selected_paths.append("SEP")

            selected_names.append(f"{ligand}_{receptor}") 
        data.close()
    else:
        print("The .npz file is empty or does not contain any keys.")
    # drop last
    if selected_paths and selected_paths[-1] == "SEP":
        selected_paths.pop()
    return selected_paths, selected_names



def pdb_paths_from_directory(directory):
    sequence_lengths = []
    selected_paths = []
    selected_names = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdb'):
            pdb_path = os.path.join(directory, filename)
            seq_length = calculate_sequence_length(pdb_path)
            sequence_lengths.append((seq_length, pdb_path))
    sorted_paths = sorted(sequence_lengths, key=lambda x: x[0])
    for length, path in sorted_paths:
        selected_paths.append(path)
        selected_paths.append('SEP')
        selected_names.append(Path(path).stem)
    
    if selected_paths and selected_paths[-1] == "SEP":
        selected_paths.pop()
    return selected_paths, selected_names


    
if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Alphafold HAPI usage')
    # settings args
    parser.add_argument('--settings-alphafolder', type=str, required=True, help='required settings for alphafolder object')
    parser.add_argument('--settings-paths', type=str, required=True, help='required settings for paths')
    # to set input
    parser.add_argument('--npz-file', type=str, default=None, help='PDB files to process.') 
    parser.add_argument('--ppi-database-path', default = None, type=str, help="path of the database folder pdbs 'peptide' and 'protein'") 
    parser.add_argument('--pdb-dir', default = None, help="Path to the directory containing the 2 chains PDBs files")   

    # for jeanzay 
    parser.add_argument('--array-size', default = 10, type=int, help="number of task on jeanzay")  # not in json

    # required both by this script and alphafolder
    parser.add_argument('--array-task-id', default = 1, type=int, help='identifier of the task') 
    parser.add_argument('--output-dir', default = None, type=str, help="path of the output folder (metrics, msas..)", required=True) # to alphafolder 
    parser.add_argument('--cache-dir', default = './cache', type=str, help="path of the output folder (metrics, msas..)", required=True) # to alphafolder 

    # alphafolder attributes override
    parser.add_argument('--save-metrics', action="store_true", default = False, help="When active, the model will save complete matrix of score (pae, plddt, distogram..) in a pickle file")
    parser.add_argument('--no-output-pdbs', action="store_true", default = False, help="When active, the model will not save the pdb file of the prediction in the output folder")

   
    # as input args of methods
    parser.add_argument('--receptor-offset', type=int, default=1, help='offset number of receptor from original pdb file (number of first amino acid passed to pipeline)') 
    parser.add_argument('--hotspots', default = None, nargs='+' ,type=int, help='residue hotspot numbered as they are out of piple line (binder then ligand without renumbering from 1 when changing chain ) ')
    
    args = parser.parse_args()
    kwargs = vars(args)

    # map kwargs
    array_task_id = kwargs["array_task_id"]

    # get override kwargs 
    required_keys_per_type = hapi_ppi.AlphaFolderPPI.get_required_keys_per_type()
    ovrd_keys = required_keys_per_type["settings_alphafolder"]
    ovrd_kwargs = { k:kwargs[k] for k in ovrd_keys if k in kwargs  }


    # get pdb files and cases_names
    if kwargs["npz_file"] is not None:
        interaction = number_interaction(kwargs["npz_file"])
        nb_data_per_task = interaction // kwargs["array_size"]
        print(f"Total number of interactions: {interaction}"
              + '\n' + f"Number of data per task: {nb_data_per_task}")
        list_pdb_files, cases_names = pdb_paths_from_database(kwargs["ppi_database_path"], kwargs["npz_file"], nb_data_per_task, array_task_id)
        # force saving metrics
        if "save_metrics" in ovrd_kwargs:
            ovrd_kwargs["save_metrics"] = True
    if  kwargs["pdb_dir"] is not None:
        list_pdb_files, cases_names = pdb_paths_from_directory(kwargs["pdb_dir"]) 
    


    hotspots = kwargs.pop("hotspots") if "hotspots" in kwargs  else None
    receptor_offset = kwargs.pop("receptor_offset"); 

    # instanciating alphafold and running
    alphafolder = hapi_ppi.AlphaFolderPPI.from_jsons(
        af_json_path = kwargs["settings_alphafolder"],
        paths_json_path = kwargs["settings_paths"],
        **ovrd_kwargs
        )

    alphafolder.predict_mutiple(
        pdbs_paths=list_pdb_files,
        cases_names=cases_names,
        receptor_offset=receptor_offset,
        hotspots=hotspots, 
    )
    print("End of main function.")
