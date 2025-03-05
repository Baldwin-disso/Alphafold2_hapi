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

import os
from typing import Dict, Union, List
import copy
import numpy as np
from Bio import PDB
from Bio import SeqUtils
from Bio.PDB.StructureBuilder import StructureBuilder


from alphafold.common import residue_constants
from alphafold.data import parsers
from alphafold.data import pipeline
from alphafold.data import feature_processing
from alphafold.data.templates import TemplateSearchResult
from alphafold.data.tools import hhblits
from alphafold.data.tools import jackhmmer
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from hapi import hapi_utils as uhapi




MAX_TEMPLATES = 4
MSA_CROP_SIZE = 2048
 
HHBLITS_AA_TO_ID = {'A': 0,'B': 2,'C': 1,'D': 2,'E': 3,'F': 4,'G': 5,'H': 6,'I': 7,'J': 20,'K': 8,'L': 9,'M': 10,'N': 11,
                        'O': 20,'P': 12,'Q': 13,'R': 14,'S': 15,'T': 16,'U': 1,'V': 17,'W': 18,'X': 20,'Y': 19,'Z': 3,'-': 21,}

TEMPLATE_FEATURES = {
      'template_aatype': np.float32,
      'template_all_atom_masks': np.float32,
      'template_all_atom_positions': np.float32,
      'template_domain_names': object,
      'template_sequence': object,
      'template_sum_probs': np.float32}



def preprocess_datapipeline_kwargs(
        data_pipeline,
        fasta_path=None, 
        pdbs_paths=None, 
        seqs=None,
        msa_output_dir=None
    ):

    if  isinstance(data_pipeline, DimerDualDataPipeline):
        assert pdbs_paths is not None, 'pdb should not be None for this pipeline'
        pipeline_kwargs = {
            'input_seqs': seqs,
            'msa_output_dir' : msa_output_dir,
            'input_pdbs_path' : pdbs_paths
        }
    elif isinstance(data_pipeline, PDBDataPipeline):
        assert pdbs_paths is not None, 'pdb should not be None for this pipeline'
        pipeline_kwargs = {
            'input_fasta_path' : fasta_path,
            'input_pdb_path' : pdbs_paths[0],
            'input_seq': seqs[0],
            'msa_output_dir' : msa_output_dir
        } 
    elif isinstance(data_pipeline, PDBEmptyDataPipeline):
        assert pdbs_paths is not None, 'pdb should not be None for this pipeline'
        pipeline_kwargs = {
            'input_pdb_path' : pdbs_paths[0],
        }
    elif isinstance(data_pipeline, EmptyDataPipeline):
        pipeline_kwargs = {
            'input_seq' : seqs[0]
        }
    else : # original pipelines of alphafold
        pipeline_kwargs = {
            'input_fasta_path' : fasta_path,
            'msa_output_dir' : msa_output_dir
        }
    return pipeline_kwargs



####################
# Monomer Pipelines
#######################
    

class PDBEmptyDataPipeline(object):
    """
    Monomer
    No MSA
    PDB as template
    """
    def __init__(self):
        super().__init__()
    # Note pdb should have the same basename as fasta
    def process(self, input_pdb_path: str) -> pipeline.FeatureDict:
        input_seqs = list(uhapi.convert_pdb_to_seqs(input_pdb_path).values())
        if len(input_seqs) != 1:
            raise ValueError(
                f'More than one input sequence found in {input_pdb_path}.')
        seq = input_seqs[0]
        seq_ids = [  HHBLITS_AA_TO_ID[seq[i]] for i in range(len(seq))  ]
        # compute features
        # seq features
        seq_feature_dict =  pipeline.make_sequence_features(seq, description="none", num_res=len(seq))
        # msa features
        msa_feature_dict = {}
        msa_feature_dict['msa'] = np.array([seq_ids],dtype='int32')
        msa_feature_dict['deletion_matrix_int'] = np.zeros((1,len(seq)),dtype='int32')
        msa_feature_dict['num_alignments'] = np.ones((len(seq)),dtype='int32')
        # template features
        residue_mask = np.ones((len(seq)),dtype='int32')
        template_features = uhapi.generate_template_features(input_pdb_path, residue_mask)
        
        
        feature_dict = {**seq_feature_dict, **msa_feature_dict, **template_features}  
        return feature_dict




class PDBDataPipeline:
  """
  Monomer
  With MSA
  PDB as template
  """
  def __init__(self,
               jackhmmer_binary_path,
               hhblits_binary_path,
               uniref90_database_path,
               mgnify_database_path,
               bfd_database_path,
               uniref30_database_path,
               small_bfd_database_path,
               template_searcher,
               template_featurizer,
               use_small_bfd,
               mgnify_max_hits=501,
               uniref_max_hits=10000,
               use_precomputed_msas=False
               ):
    """Initializes the data pipeline."""
    self._use_small_bfd = use_small_bfd
    self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=uniref90_database_path)
    if use_small_bfd:
      self.jackhmmer_small_bfd_runner = jackhmmer.Jackhmmer(
          binary_path=jackhmmer_binary_path,
          database_path=small_bfd_database_path)
    else:
      self.hhblits_bfd_uniref_runner = hhblits.HHBlits(
          binary_path=hhblits_binary_path,
          databases=[bfd_database_path, uniref30_database_path])
    self.jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=mgnify_database_path)
    self.template_searcher = template_searcher
    self.template_featurizer = template_featurizer
    self.mgnify_max_hits = mgnify_max_hits
    self.uniref_max_hits = uniref_max_hits
    self.use_precomputed_msas = use_precomputed_msas

  def process(self, input_fasta_path: str, input_pdb_path: str, input_seq: str , msa_output_dir: str) :
    """Runs alignment tools on the input sequence and creates features."""
    
    
    num_res = len(input_seq)
    uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
    jackhmmer_uniref90_result = pipeline.run_msa_tool(
        msa_runner=self.jackhmmer_uniref90_runner,
        input_fasta_path=input_fasta_path,
        msa_out_path=uniref90_out_path,
        msa_format='sto',
        use_precomputed_msas=self.use_precomputed_msas,
        max_sto_sequences=self.uniref_max_hits)
    mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')
    jackhmmer_mgnify_result = pipeline.run_msa_tool(
        msa_runner=self.jackhmmer_mgnify_runner,
        input_fasta_path=input_fasta_path,
        msa_out_path=mgnify_out_path,
        msa_format='sto',
        use_precomputed_msas=self.use_precomputed_msas,
        max_sto_sequences=self.mgnify_max_hits)

    msa_for_templates = jackhmmer_uniref90_result['sto']
    msa_for_templates = parsers.deduplicate_stockholm_msa(msa_for_templates)
    msa_for_templates = parsers.remove_empty_columns_from_stockholm_msa(
        msa_for_templates)

    if self.template_searcher.input_format == 'sto':
      pdb_templates_result = self.template_searcher.query(msa_for_templates)
    elif self.template_searcher.input_format == 'a3m':
      uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(msa_for_templates)
      pdb_templates_result = self.template_searcher.query(uniref90_msa_as_a3m)
    else:
      raise ValueError('Unrecognized template input format: '
                       f'{self.template_searcher.input_format}')

    pdb_hits_out_path = os.path.join(
        msa_output_dir, f'pdb_hits.{self.template_searcher.output_format}')
    with open(pdb_hits_out_path, 'w') as f:
      f.write(pdb_templates_result)

    uniref90_msa = parsers.parse_stockholm(jackhmmer_uniref90_result['sto'])
    mgnify_msa = parsers.parse_stockholm(jackhmmer_mgnify_result['sto'])

    pdb_template_hits = self.template_searcher.get_template_hits(
        output_string=pdb_templates_result, input_sequence=input_seq)

    if self._use_small_bfd:
      bfd_out_path = os.path.join(msa_output_dir, 'small_bfd_hits.sto')
      jackhmmer_small_bfd_result = pipeline.run_msa_tool(
          msa_runner=self.jackhmmer_small_bfd_runner,
          input_fasta_path=input_fasta_path,
          msa_out_path=bfd_out_path,
          msa_format='sto',
          use_precomputed_msas=self.use_precomputed_msas)
      bfd_msa = parsers.parse_stockholm(jackhmmer_small_bfd_result['sto'])
    else:
      bfd_out_path = os.path.join(msa_output_dir, 'bfd_uniref_hits.a3m')
      hhblits_bfd_uniref_result = pipeline.run_msa_tool(
          msa_runner=self.hhblits_bfd_uniref_runner,
          input_fasta_path=input_fasta_path,
          msa_out_path=bfd_out_path,
          msa_format='a3m',
          use_precomputed_msas=self.use_precomputed_msas)
      bfd_msa = parsers.parse_a3m(hhblits_bfd_uniref_result['a3m'])
    
    # hapi here
    residue_mask = np.ones((len(input_seq)),dtype='int32')
    template_features = uhapi.generate_template_features(input_pdb_path, residue_mask)
        

    sequence_features = pipeline.make_sequence_features(
        sequence=input_seq,
        description="none",
        num_res=num_res)

    msa_features = pipeline.make_msa_features((uniref90_msa, bfd_msa, mgnify_msa))


    return {**sequence_features, **msa_features, **template_features}
  



class EmptyDataPipeline(object):
    """
    Monomer
    NO MSA
    No template
    """
    def __init__(self):
        super().__init__()

    def process(self, 
        input_seq: str) -> pipeline.FeatureDict:
        
        seq_ids = [  HHBLITS_AA_TO_ID[input_seq[i]] for i in range(len(input_seq))  ]
        # compute features
        # input_seq features
        seq_feature_dict =  pipeline.make_sequence_features(input_seq, description="none", num_res=len(input_seq))
        # msa features
        msa_feature_dict = {}
        msa_feature_dict['msa'] = np.array([seq_ids],dtype='int32')
        msa_feature_dict['deletion_matrix_int'] = np.zeros((1,len(input_seq)),dtype='int32')
        msa_feature_dict['num_alignments'] = np.ones((len(input_seq)),dtype='int32')
        # template features
        template_features = {}
        for name in TEMPLATE_FEATURES:
            template_features[name] = np.array([], dtype=TEMPLATE_FEATURES[name])
            templates_results = TemplateSearchResult(features=template_features, errors=[], warnings=[])
            feature_dict = {**seq_feature_dict, **msa_feature_dict, **templates_results.features}  
        return feature_dict





#####################################
# Dimer pipeline
####################################"""

class DimerDualDataPipeline:  
    """Runs the alignment tools and assembles the input features."""
    def __init__(self, data_pipelineA, data_pipelineB, A_is_ligand: bool):
        self.data_pipelines_dict = {
            "A": data_pipelineA,
            "B": data_pipelineB,
        }
        self.A_is_ligand = A_is_ligand

    def _process_single_chain(
        self,
        chain_id: str,
        sequence: str,
        description: str,
        msa_output_dir: str,
        is_homomer_or_monomer: bool,
        pdb_path: str = None) -> pipeline.FeatureDict:
        """Runs the monomer pipeline on a single chain."""
        chain_fasta_str = f'>chain_{chain_id}\n{sequence}\n'
        
        with pipeline_multimer.temp_fasta_file(chain_fasta_str) as chain_fasta_path: 

            data_pipeline = self.data_pipelines_dict[chain_id]
            pipeline_kwargs = preprocess_datapipeline_kwargs(
                data_pipeline,
                fasta_path=chain_fasta_path,
                pdbs_paths=[pdb_path],
                msa_output_dir=msa_output_dir
                )
            chain_features = data_pipeline.process(**pipeline_kwargs)
        
        # We only construct the pairing features if there are 2 or more unique
        # sequences.
        if not is_homomer_or_monomer:
            seq_ids = [  HHBLITS_AA_TO_ID[sequence[i]] for i in range(len(sequence))  ]

            all_seq_msa_features = {
                'deletion_matrix_int_all_seq': np.zeros((1,len(seq_ids)),dtype='int32'),
                'msa_all_seq': np.array([seq_ids],dtype='int32'),
                'msa_species_identifiers_all_seq': np.array([''.encode()],dtype=object)
            }
            chain_features.update(all_seq_msa_features)
        return chain_features

    

    def process(self,
                input_seqs: List[str],
                msa_output_dir: str,
                input_pdbs_path:  List[str]) -> pipeline.FeatureDict:
        
      
        """single seq featurizer mode for multimer version"""
        input_descs = ['' for i in range(len(input_seqs)) ]
        chain_id_map = pipeline_multimer._make_chain_id_map(sequences=input_seqs,
                                         descriptions=input_descs)

        pdbs_dict = uhapi.adjust_pdb_chain_names(input_pdbs_path, self.A_is_ligand) 
        
        all_chain_features = {}
        sequence_features = {}
        is_homomer_or_monomer = len(set(input_seqs)) == 1
        for chain_id, fasta_chain in chain_id_map.items():
            if fasta_chain.sequence in sequence_features:
                all_chain_features[chain_id] = copy.deepcopy(
                    sequence_features[fasta_chain.sequence])
                continue
            chain_features = self._process_single_chain(
                chain_id=chain_id,
                sequence=fasta_chain.sequence,
                description=fasta_chain.description,
                msa_output_dir=msa_output_dir,
                is_homomer_or_monomer=is_homomer_or_monomer,
                pdb_path = pdbs_dict[chain_id] if pdbs_dict else None
            )
            chain_features = pipeline_multimer.convert_monomer_features(chain_features, chain_id=chain_id)
            all_chain_features[chain_id] = chain_features
            sequence_features[fasta_chain.sequence] = chain_features

        all_chain_features = pipeline_multimer.add_assembly_features(all_chain_features)

        np_example = feature_processing.pair_and_merge(
            all_chain_features=all_chain_features)
        
        # Pad MSA to avoid zero-sized extra_msa.
        np_example = pipeline_multimer.pad_msa(np_example, 512)

        return np_example
    
