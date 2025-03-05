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

from typing import Dict, Union
from pathlib import Path
import string
import random
import pickle
import os
import tempfile

import numpy as np
from Bio import PDB
from Bio import SeqUtils

from alphafold.common import protein
from alphafold.common import residue_constants

import re




class AltLocSelect(PDB.Select):
    def __init__(self, altloc='A'):
        self.altloc = altloc

    def accept_residue(self, residue):
        return residue.id[2].strip() in ('', self.altloc)

def filter_residues(structure, altloc=[' ','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']):
    for model in structure:
        for chain in model:
            residues = list(chain)
            residues_dict = {}
            # Populate the residues_dict with each residue's alternate locations
            for res in residues:
                res_id = res.id[1]
                alt_loc = res.id[2].strip() or ' '
                if res_id not in residues_dict:
                    residues_dict[res_id] = {}
                residues_dict[res_id][alt_loc] = res

            # Loop through each residue id and choose the appropriate conformation
            for res_id in residues_dict.keys():
                selected_res = None
                # Priority for conformations: main chain (' '), 'A', then 'B'
                for alt_loc in altloc:
                    if alt_loc in residues_dict[res_id]:
                        selected_res = residues_dict[res_id][alt_loc]
                        break
                # Detach all alternate conformations and add back the selected one
                for alt_loc, res in residues_dict[res_id].items():
                    chain.detach_child(res.id)
                if selected_res:
                    chain.add(selected_res)





def generate_random_string(length):
      # choose from all lowercase letter
      letters = string.ascii_lowercase
      result_str = ''.join(random.choice(letters) for i in range(length))
      return result_str






def convert_pdb_to_seqs(pdb_file):
    assert Path(pdb_file).suffix in ['.pdb', '.cif']
    parser = PDB.PDBParser(QUIET=True) if Path(pdb_file).suffix == '.pdb' else PDB.MMCIFParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)

    filter_residues(structure)
    
    result = {}
    for model in structure:
        for chain in model:
            name = f"{chain.id}" # only model 1
            seq = ''.join([
                SeqUtils.seq1(r.get_resname()) 
                for r in chain.get_residues() 
                if SeqUtils.seq1(r.get_resname()) != 'X'
            ])
            result.update({name:seq})  
    return result



def convert_pdbs_to_fasta(pdbs, fasta_output_dir):
    seqs = []
    names = []
    if isinstance(pdbs, str): #Case unique_pdb in multimer mode
        #TODO adjust chain names
        seqs = convert_pdb_to_seqs(pdbs)
        names = ['Binder', 'Target']
        fasta = Path(fasta_output_dir) / f"{Path(pdbs).stem}.fasta"
    else : 
        for pdb in pdbs:
            seq = convert_pdb_to_seqs(pdb)[0]
            seqs.append(seq)
            names.append(Path(pdb).stem)
        fasta = Path(fasta_output_dir) / f"{names[0]}_{names[1]}.fasta"
    with  open(fasta,'w') as f:
        for i,seq in enumerate(seqs):
                f.write(">" + names[i] \
                + "\n" + seq + "\n")
    return f.name



def separate_pdb_multimer_file(multimer_pdb_path, use_tempfile = False, output_dir=None, rename_chains=False):
    multimer_pdb_path = Path(multimer_pdb_path)
    parser = PDB.PDBParser()
    structure = parser.get_structure("multimer", str(multimer_pdb_path))
    io = PDB.PDBIO()
    assert len(structure) == 1
    assert use_tempfile or output_dir
    monomer_pdbs_list = []
    
    chain_ids = iter(protein.PDB_CHAIN_IDS) 

    for model in structure:
        for chain in model:
            io.set_structure(chain)
            io.save(structure.get_id() + "_" + chain.get_id() + ".pdb")
            original_chain_id = chain.id
            if rename_chains:
                try:
                    chain.id = next(chain_ids) 
                except StopIteration:
                    raise ValueError("Exceeded available PDB_CHAIN_IDS for renaming.")
            
            io.set_structure(chain)
            
            if use_tempfile:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
                output_file = Path(temp_file.name)
            else :
                output_file = Path(
                output_dir,
                 multimer_pdb_path.stem + '_' + original_chain_id + '.pdb'
            )
                if not os.path.exists(f'{output_dir}'):
                    os.makedirs(f'{output_dir}')
                
            io.save(str(output_file))

            monomer_pdbs_list.append(str(output_file))
    return monomer_pdbs_list



def separate_and_sort_chains_from_multiple_pdbs(pdb_files, output_dir):
    """
    Splits multiple PDB files into separate files for each chain, with renamed chain IDs.

    Args:
        pdb_files (list): List of paths to PDB files.
        output_dir (str): Directory to save the output PDB files.

    Returns:
        list: Paths to the generated monomer PDB files.
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Chain ID iterator (e.g., A, B, C, ...)
    chain_ids = iter(protein.PDB_CHAIN_IDS)

    parser = PDB.PDBParser()
    io = PDB.PDBIO()
    monomer_pdbs_list = []

    # Map to track chains and their sizes
    chains_to_process = []

    for pdb_file in pdb_files:
        pdb_file = Path(pdb_file)
        structure = parser.get_structure(pdb_file.stem, str(pdb_file))
        
        # Iterate over all models and chains
        for model in structure:
            for chain in model:
                original_chain_id = chain.id
                chain_size = len(chain)
                chains_to_process.append((pdb_file.stem, chain, original_chain_id, chain_size))

    # Sort chains by size (ascending) for renaming
    sorted_chains = sorted(chains_to_process, key=lambda x: x[3])  # Sort by chain size

    # Process each chain, assign new IDs, and save
    for file_name, chain, original_chain_id, _ in sorted_chains:
        try:
            new_chain_id = next(chain_ids)  # Assign next available chain ID
        except StopIteration:
            raise ValueError("Exceeded available PDB_CHAIN_IDS for renaming.")
        
        # Update chain ID
        chain.id = new_chain_id
        io.set_structure(chain)

        # Define output file path
        output_file = output_dir / f"{file_name}_{original_chain_id}.pdb"
        io.save(str(output_file))
        monomer_pdbs_list.append(str(output_file))

    return monomer_pdbs_list






def check_unique_chain_ids_and_single_chain_per_file(pdb_files):
    """
    Checks that each PDB file contains exactly one chain and that chain IDs are unique 
    across all PDB files in the provided list.
    
    :param pdb_files: List of paths to PDB files.
    :raises ValueError: If validation fails, raises an error with a detailed message.
    :return: None (if all validations pass).
    """
    parser = PDB.PDBParser(QUIET=True)
    seen_chains = set()  # Set to track all encountered chain IDs
    errors = []  # List to store detected errors

    for pdb_file in pdb_files:
        structure = parser.get_structure("structure", pdb_file)
        chain_ids = []  # List to store chain IDs in the current file

        # Extract chain IDs from the current PDB file
        for model in structure:
            for chain in model:
                chain_ids.append(chain.id)

        # Check if the file contains exactly one chain
        if len(chain_ids) != 1:
            errors.append(f"File '{pdb_file}' contains {len(chain_ids)} chains (must contain exactly 1).")

        # Check for duplicate chain IDs across all files
        for chain_id in chain_ids:
            if chain_id in seen_chains:
                errors.append(f"Duplicate chain ID '{chain_id}' found in file '{pdb_file}'.")
            else:
                seen_chains.add(chain_id)

    # Raise a ValueError if any errors are found
    if errors:
        error_message = "\n".join(errors)
        raise ValueError(f"Validation failed with the following errors:\n{error_message}")

    # If no errors, validation passed
    print("All files meet the requirements.")



def af2_get_atom_positions(pdb_path):
    parser = PDB.PDBParser(QUIET=True) if Path(pdb_path).suffix == '.pdb' else PDB.MMCIFParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_path)

    filter_residues(structure)

    all_positions = []
    all_positions_mask = []
    residue_indices = []

    for model in structure:
        for chain in model:
            for res in chain:
                 # Skip non-standard amino acids and heteroatoms (HETATM) including water
                if not PDB.is_aa(res, standard=True):
                    continue  # Skip HETATM and water residues
                atom_positions = np.zeros((residue_constants.atom_type_num, 3))
                atom_masks = np.zeros(residue_constants.atom_type_num, dtype=np.int64)
                for atom in res:
                    atom_name = atom.name
                    if atom_name in residue_constants.atom_order:
                        atom_positions[residue_constants.atom_order[atom_name]] = atom.coord
                        atom_masks[residue_constants.atom_order[atom_name]] = 1
                if np.sum(atom_masks) > 0:  # Only add residues with at least one atom
                    all_positions.append(atom_positions)
                    all_positions_mask.append(atom_masks)
                    residue_indices.append(res.id[1])

    all_positions = np.array(all_positions)
    all_positions_mask = np.array(all_positions_mask)

    return all_positions, all_positions_mask, residue_indices


def generate_template_features(pdb_path, residue_mask):
    all_atom_positions, all_atom_masks, residue_indices = af2_get_atom_positions(pdb_path)
    seq = list(convert_pdb_to_seqs(pdb_path).values())[0]
    # Split the all atom positions and masks 
    all_atom_positions = np.split(all_atom_positions, all_atom_positions.shape[0])
    all_atom_masks = np.split(all_atom_masks, all_atom_masks.shape[0])

    output_templates_sequence = []
    output_confidence_scores = []
    templates_all_atom_positions = []
    templates_all_atom_masks = []

    # initialized with nul values
    for _ in seq:
        templates_all_atom_positions.append(
            np.zeros((residue_constants.atom_type_num, 3)))
        templates_all_atom_masks.append(np.zeros(residue_constants.atom_type_num))
        output_templates_sequence.append('-')
        output_confidence_scores.append(-1)

    confidence_scores = []
    for _ in seq: confidence_scores.append(9)

    for idx, i in enumerate(seq):
        if not residue_mask[idx]: continue

        # check if indices existes in filtered positions
        if idx >= len(all_atom_positions):
            continue

        templates_all_atom_positions[idx] = all_atom_positions[idx][0]  # assign target indices to model coordinates
        templates_all_atom_masks[idx] = all_atom_masks[idx][0]
        output_templates_sequence[idx] = seq[idx]
        output_confidence_scores[idx] = confidence_scores[idx]  # 0-9 where high means confident

    output_templates_sequence = ''.join(output_templates_sequence)

    templates_aatype = residue_constants.sequence_to_onehot(
        output_templates_sequence, residue_constants.HHBLITS_AA_TO_ID)

    template_feat_dict = {'template_all_atom_positions': np.array(templates_all_atom_positions)[None],
        'template_all_atom_masks': np.array(templates_all_atom_masks)[None],
        'template_sequence': [output_templates_sequence.encode()],
        'template_aatype': np.array(templates_aatype)[None],
        'template_confidence_scores': np.array(output_confidence_scores)[None],
        'template_domain_names': ['none'.encode()],
        'template_release_date': ["none".encode()]}

    return template_feat_dict




def concatenate_pickles(directory, outpath):
    '''Concatenate all the files .pkl in the directory given in one file outpath (.pkl)
    NEEDS : Have all the files .pkl to concatanate in one directory '''

    consolidated_data = {}
    files_to_delete = []

    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            filepath = os.path.join(directory, filename)

            with open(filepath, 'rb') as file:
                data = pickle.load(file)

                for key, value in data.items():
                    if key in consolidated_data:
                        if isinstance(value, list):
                            consolidated_data[key].extend(value)
                        else:
                            consolidated_data[key].append(value)
                    else:
                        consolidated_data[key] = [value] if not isinstance(value, list) else value
            files_to_delete.append(filepath)
            
    output_path = os.path.join(directory, outpath)
    with open(output_path, 'wb') as file:
        pickle.dump(consolidated_data, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    for file_path in files_to_delete:
        os.remove(file_path)
    
    print(f"All pickle files have been concatenated and saved to {output_path}.")



def adjust_pdb_chain_names(pdb_paths, A_is_ligand):
    ''''Verify the chain names in the PDB files and rename them to A and B if necessary'''
    parser = PDB.PDBParser()
    chain_lengths = {}
    structures = {}

    # Parse each PDB and calculate chain lengths
    for pdb_path in pdb_paths:
        structure = parser.get_structure(Path(pdb_path).stem, pdb_path)
        structures[pdb_path] = structure
        for model in structure:
            for chain in model:
                sequence = [res.get_resname() for res in chain.get_unpacked_list() if PDB.is_aa(res)]
                chain_lengths[(pdb_path, chain.id)] = len(sequence)

    # Sort chains by sequence length to determine ligand and receptor
    sorted_chains = sorted(chain_lengths.items(), key=lambda x: x[1])
    smallest_chain = sorted_chains[0][0]
    largest_chain = sorted_chains[-1][0]

    # Determine correct chain IDs based on A_is_ligand flag
    if A_is_ligand:
        correct_order = {'A': smallest_chain, 'B': largest_chain}
    else:
        correct_order = {'A': largest_chain, 'B': smallest_chain}

    # Rename chains directly in the structure
    renamed_chains = {}
    for (pdb_path, chain_id), length in chain_lengths.items():
        structure = structures[pdb_path]
        for model in structure:
            for chain in model:
                original_id = chain.id
                if (pdb_path, chain.id) == correct_order['A']:
                    chain.id = 'A'
                elif (pdb_path, chain.id) == correct_order['B']:
                    chain.id = 'B'
                # Update the dictionary after renaming
                renamed_chains[chain.id] = str(Path(pdb_path))

    # Save the modified structures back to their original files
    io = PDB.PDBIO()
    for pdb_path, structure in structures.items():
        io.set_structure(structure)
        io.save(pdb_path)

    return renamed_chains


def extract_name_from_pdb(pdbs):
    names = []
    for pdb in pdbs:
         names.append(Path(pdb).stem)
    name = names[0] + '_' + names[1]
    return name







def extract_sequences(prot):
    """
    Converts an AlphaFold Protein object into separate amino acid sequences for each chain.

    Args:
        protein_obj: An instance of Protein containing `aatype` and `chain_index`.

    Returns:
        dict: A dictionary where keys are chain names ('A', 'B', ...) and values are sequences.
    """
    # List of amino acids (A, R, N, D, ..., V)
    restypes = residue_constants.restypes  # ['A', 'R', 'N', 'D', ..., 'V']

    # Initialize dictionary to store sequences by chain
    chain_sequences = {}

    # Iterate over residues
    for aa_idx, chain_idx in enumerate(prot.chain_index):
        chain_idx = int(chain_idx)  # Ensure integer indices
        aa_type = prot.aatype[aa_idx]  # Amino acid index
        aa_letter = restypes[aa_type]  # Corresponding one-letter code

        if chain_idx not in chain_sequences:
            chain_sequences[chain_idx] = []

        chain_sequences[chain_idx].append(aa_letter)

    # Convert lists to strings and use chain letters instead of numbers
    chain_sequences = {chr(65 + chain_id): ''.join(seq) for chain_id, seq in chain_sequences.items()}

    return chain_sequences

# Example usage:
# protein = ...  # Load a Protein object
# sequences = protein_to_sequences(protein)
# print(sequences)




def extract_canc_coordinates(prot: protein.Protein):
    """
    Extracts coordinates of C, A (Alpha Carbon), and N (Nitrogen) atoms for each residue in a protein structure.
    
    Args:
        atom_positions: Numpy array containing the positions of atoms.
        atom_mask: Numpy array indicating which atoms are present.
        aatype: Numpy array indicating the amino acid type for each residue.
        residue_index: Numpy array with the indices of residues.
        
    Returns:
        A dictionary with residue indices as keys and tuples of coordinates for C, A, and N atoms.
    """
    
    atom_mask = prot.atom_mask
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    atom_types = residue_constants.atom_types  # List of atom types like ['C', 'CA', 'N', ...]
    relevant_atoms = ['C', 'CA', 'N']
    
    coordinates_canc = {}

    # Iterate through all residues
    for i in range(len(residue_index)):
        res_coords = {}
        
        # Collect coordinates for C, A, and N if they exist
        for atom_idx, atom_name in enumerate(atom_types):
            if atom_name in relevant_atoms and atom_mask[i, atom_idx] > 0.5:  # Check if the atom is present
                res_coords[atom_name] = atom_positions[i, atom_idx]

        # Only add to the dictionary if all required atoms are found
        if all(key in res_coords for key in relevant_atoms):
            coordinates_canc[i+1] = (res_coords['N'], res_coords['CA'], res_coords['C'])
    
    return coordinates_canc

def generate_pae_score(binderlen, pae, hotspots=None, receptor_offset=None):
    pae_interaction1 = np.mean( pae[:binderlen,binderlen:] )
    pae_interaction2 = np.mean( pae[binderlen:,:binderlen] )
    pae_binder = np.mean( pae[:binderlen,:binderlen] )
    pae_target = np.mean( pae[binderlen:,binderlen:] )
    pae_interaction_total = ( pae_interaction1 + pae_interaction2 ) / 2
    if hotspots is not None:
        absolute_hotspots = get_absolute_hotspot_indices(binderlen, hotspots, receptor_offset)
        pae_hotspot1 = np.mean(pae[:binderlen, absolute_hotspots])
        pae_hotspot2 = np.mean(pae[absolute_hotspots, :binderlen])
        pae_hotspot = ( pae_hotspot1 + pae_hotspot2 ) / 2
    else:
        pae_hotspot = None

    
    return pae_binder, pae_target, pae_interaction_total, pae_hotspot

def generate_proxsite_score(binderlen, final_atom_positions, hotspots, receptor_offset):
    absolute_hotspots = get_absolute_hotspot_indices(binderlen, hotspots, receptor_offset)
    coords = final_atom_positions[:,residue_constants.atom_types.index('CA')] #size : (N,3)
    coords_binder = coords[:binderlen] # size (binder_len, 3)
    coords_hotspots = coords[np.array(absolute_hotspots)] # (hotspot_len, 3)

    coords_diff = coords_binder[:,None] - coords_hotspots[None] # size (binder_len, hotspots_len, 3 )
    coords_norm = np.linalg.norm(coords_diff, axis=-1) # size (binder_len, hotspots_len)
    proxsite = coords_norm.min()
    return proxsite


def get_absolute_hotspot_indices(binderlen, hotspots, receptor_offset):
        """
            ex : 
                - receptor offset = 5 (5th residue included, "one" indexed)
                - binder_len = 10
                - hotspot = [6]
                => absolute hotspot = 11 = 6 +  10  -  5  =  hotspot + 10 - receptor_offset 

        
        """
        offset = binderlen - receptor_offset 
        absolute_hotspots = [ h + offset for h in hotspots]   # indices of residues that are targeted in chain B
        return absolute_hotspots

def remove_cases_after_checkpoints(df, fastas_paths, pdbs_paths, cases_names, sequences):
    """
    Removes elements from fastas_paths, pdbs_paths, cases_names, and sequences 
    if the case name in the CSV file is found in cases_names.

    Args:
        df (pd.DataFrame): DataFrame loaded from the CSV file.
        fastas_paths (list): List of fasta file paths.
        pdbs_paths (list): List of pdb file paths.
        cases_names (list): List of case names.
        sequences (list): List of associated sequences.

    Returns:
        tuple: Updated lists (fastas_paths, pdbs_paths, cases_names, sequences).
    """
    if not cases_names:
        print("cannot use checkpoint if cases_names are empty, restarting processing from start ")
    # Check if the DataFrame contains data
    elif df.empty:
        print("checkpoint is empty, processing starts from the beginning")
    else:
    
        # Iterate over each row in the DataFrame
        for _, row in df.iterrows():
            case_name = row[0]  # The name is in the first column of the CSV
            
            if case_name in cases_names:
                # Find the position of case_name in cases_names
                index = cases_names.index(case_name)
                
                # Remove the corresponding elements from each list
                del cases_names[index]
                if fastas_paths:
                    del fastas_paths[index]
                if pdbs_paths:
                    del pdbs_paths[index]
                if sequences:
                    del sequences[index]
        
    return fastas_paths, pdbs_paths, cases_names, sequences


def extract_L_R_format_from_str(s):
    pattern = r"^L_(\d+)_R_(\d+)$"
    match = re.match(pattern, s)
    if match:
        L_value = f"L_{match.group(1)}"
        R_value = f"R_{match.group(2)}"
        return L_value, R_value
    return None  



if __name__ == '__main__':
    seq = convert_pdb_to_seqs('/media/honeypot/lena/data/structures/peptide/L_214.pdb')
    print(seq)
    