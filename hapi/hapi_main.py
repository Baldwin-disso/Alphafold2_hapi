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
import os
import pickle
import random
import sys
import time
import random
from typing import Dict, Union
from pathlib import Path
from itertools import zip_longest
import numpy as np
import itertools
import shutil



from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import parsers
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import templates
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.model import config
from alphafold.model import data
from alphafold.model import model
from alphafold.relax import relax
from alphafold.common import protein 
from hapi import hapi_pipelines as phapi
from hapi import hapi_utils as uhapi
import glob


MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3

HHBLITS_AA_TO_ID = {'A': 0,'B': 2,'C': 1,'D': 2,'E': 3,'F': 4,'G': 5,'H': 6,'I': 7,'J': 20,'K': 8,'L': 9,'M': 10,'N': 11,
                              'O': 20,'P': 12,'Q': 13,'R': 14,'S': 15,'T': 16,'U': 1,'V': 17,'W': 18,'X': 20,'Y': 19,'Z': 3,'-': 21,}
          

TEMPLATE_FEATURES = {
      'template_aatype': np.float32,
      'template_all_atom_masks': np.float32,
      'template_all_atom_positions': np.float32,
      'template_domain_names': object,
      'template_sequence': object,
      'template_sum_probs': np.float32}

### Verify if the choices kwarg are suitable together 
def _check_flag(flag_name: str,
                other_flag_name: str,
                kwargs : dict,
                should_be_set: bool):
  
  if should_be_set != bool(kwargs[flag_name]):
    verb = 'be' if should_be_set else 'not be'
    raise ValueError(f'{flag_name} must {verb} set when running with '
                     f'"--{other_flag_name}={kwargs[other_flag_name]}".')
  
  
          

class AlphaFolder(object):
  """
  AlphaFolder class
  """
  def __init__(self, **kwargs):
   
    ######## Managing kwargs
    # get default kwargs 
    # update it with input kwargs and swap
    default_kwargs = self.get_default_kwargs()
    default_kwargs.update(kwargs)
    kwargs = default_kwargs

    # asserting kwargs are good
    missing_keys, unnecessary_keys = self.compare_kwargs_to_required_keys(kwargs)
    assert not(missing_keys), f"missing arguments : {missing_keys}"
    assert not(unnecessary_keys), f"too many arguments {unnecessary_keys}"
    
    
    # filter out path depending on the use of monomer or multimer models
    if kwargs["model_preset"] == "multimer":
      kwargs['pdb70_database_path'] = ''
    else: # monomer case
      kwargs['pdb_seqres_database_path'] = ''
      kwargs['uniprot_database_path'] = '' 

    # use kwargs to set attributes
    self.__dict__.update(kwargs) 

    if self.comments:
      print('\n instanciating object....')


    ########## alphafolder creation 

    if self.comments :
      print('\n\t - save kwargs as a dict')
    ''' then also defines members : data pipeline, model_runners, amber_relaxer, random_seed'''
  
    # checks if tools are installed (looking for the path of each)
    if self.comments :
      print('\n\t - checking if tools are installed')
    missing_tool = False
    for tool_name in (
        'jackhmmer', 'hhblits', 'hhsearch', 'hmmsearch', 'hmmbuild', 'kalign'):
      missing_tool = missing_tool or kwargs[tool_name + '_binary_path'] is None
    
    if missing_tool :
      raise ValueError(f'Could not find path to the "{tool_name}" binary. Make '
                        'sure it is installed on your system.')
    # position option of data
    if self.comments :
      print('\n\t - test if kwargs are good')
    use_small_bfd = kwargs['db_preset'] == 'reduced_dbs'
    _check_flag('small_bfd_database_path', 'db_preset', kwargs,
                should_be_set=use_small_bfd)
    _check_flag('bfd_database_path', 'db_preset', kwargs,
                should_be_set=not use_small_bfd)
    _check_flag('uniref30_database_path', 'db_preset', kwargs,
                should_be_set=not use_small_bfd)

    # specific database if multimer mode is used
    self.run_multimer_system = 'multimer' in kwargs['model_preset']
    _check_flag('pdb70_database_path', 'model_preset',  kwargs,
                should_be_set=not self.run_multimer_system)
    _check_flag('pdb_seqres_database_path', 'model_preset', kwargs,
                should_be_set=self.run_multimer_system)
    _check_flag('uniprot_database_path', 'model_preset', kwargs,
                should_be_set=self.run_multimer_system)

    if kwargs['model_preset'] == 'monomer_casp14':
      num_ensemble = 8
    else:
      num_ensemble = 1

    # set template searcher and featurizer base on multimer or not
    if self.comments :
      print('\n\t - set template searcher and featurizer base on multimer or not')
    if self.run_multimer_system:
      template_searcher = hmmsearch.Hmmsearch(
          binary_path=kwargs['hmmsearch_binary_path'],
          hmmbuild_binary_path=kwargs['hmmbuild_binary_path'],
          database_path=kwargs['pdb_seqres_database_path'])
      template_featurizer = templates.HmmsearchHitFeaturizer(
          mmcif_dir=kwargs['template_mmcif_dir'],
          max_template_date=kwargs['max_template_date'],
          max_hits=MAX_TEMPLATE_HITS,
          kalign_binary_path=kwargs['kalign_binary_path'],
          release_dates_path=None,
          obsolete_pdbs_path=kwargs['obsolete_pdbs_path'])
    else:
      template_searcher = hhsearch.HHSearch(
          binary_path=kwargs['hhsearch_binary_path'],
          databases=[kwargs['pdb70_database_path']])
      template_featurizer = templates.HhsearchHitFeaturizer(
          mmcif_dir=kwargs['template_mmcif_dir'],
          max_template_date=kwargs['max_template_date'],
          max_hits=MAX_TEMPLATE_HITS,
          kalign_binary_path=kwargs['kalign_binary_path'],
          release_dates_path=None,
          obsolete_pdbs_path=kwargs['obsolete_pdbs_path'])

    
    
    
    if self.pipeline_type == "full" and self.run_multimer_system: 
      if self.comments:
        print('\n pipeline_type = full and multimer')
      monomer_data_pipeline = pipeline.DataPipeline(
        jackhmmer_binary_path=kwargs['jackhmmer_binary_path'],
        hhblits_binary_path=kwargs['hhblits_binary_path'],
        uniref90_database_path=kwargs['uniref90_database_path'],
        mgnify_database_path=kwargs['mgnify_database_path'],
        bfd_database_path=kwargs['bfd_database_path'],
        uniref30_database_path=kwargs['uniref30_database_path'],
        small_bfd_database_path=kwargs['small_bfd_database_path'],
        template_searcher=template_searcher,
        template_featurizer=template_featurizer,
        use_small_bfd=use_small_bfd,
        use_precomputed_msas=kwargs['use_precomputed_msas'])
      
      num_predictions_per_model = kwargs['num_multimer_predictions_per_model']
      self.data_pipeline = pipeline_multimer.DataPipeline(
          monomer_data_pipeline=monomer_data_pipeline,
          jackhmmer_binary_path=kwargs['jackhmmer_binary_path'],
          uniprot_database_path=kwargs['uniprot_database_path'],
          use_precomputed_msas=kwargs['use_precomputed_msas'])
      
    elif self.pipeline_type == "full" and not self.run_multimer_system:
      num_predictions_per_model = 1
      if self.comments:
        print('\n pipeline_type = full and not multimer')
      self.data_pipeline = pipeline.DataPipeline(
        jackhmmer_binary_path=kwargs['jackhmmer_binary_path'],
        hhblits_binary_path=kwargs['hhblits_binary_path'],
        uniref90_database_path=kwargs['uniref90_database_path'],
        mgnify_database_path=kwargs['mgnify_database_path'],
        bfd_database_path=kwargs['bfd_database_path'],
        uniref30_database_path=kwargs['uniref30_database_path'],
        small_bfd_database_path=kwargs['small_bfd_database_path'],
        template_searcher=template_searcher,
        template_featurizer=template_featurizer,
        use_small_bfd=use_small_bfd,
        use_precomputed_msas=kwargs['use_precomputed_msas'])
      

    elif self.pipeline_type == "empty" and not self.run_multimer_system:
      num_predictions_per_model = 1
      if self.comments :
        print('\n pipeline_type = empty and not multimer')
      self.data_pipeline = phapi.EmptyDataPipeline()

      
    elif self.pipeline_type == "pdbempty" and not self.run_multimer_system:
      num_predictions_per_model = 1
      if self.comments :
        print('\n pipeline_type = pdbempty and not multimer')
      self.data_pipeline = phapi.PDBEmptyDataPipeline()

    elif self.pipeline_type == "pdb" and not self.run_multimer_system:
      num_predictions_per_model = 1
      if self.comments :
        print('\n pipeline_type = pdb and not multimer')
      self.data_pipeline = phapi.PDBDataPipeline(
        jackhmmer_binary_path=kwargs['jackhmmer_binary_path'],
        hhblits_binary_path=kwargs['hhblits_binary_path'],
        uniref90_database_path=kwargs['uniref90_database_path'],
        mgnify_database_path=kwargs['mgnify_database_path'],
        bfd_database_path=kwargs['bfd_database_path'],
        uniref30_database_path=kwargs['uniref30_database_path'],
        small_bfd_database_path=kwargs['small_bfd_database_path'],
        template_searcher=template_searcher,
        template_featurizer=template_featurizer,
        use_small_bfd=use_small_bfd,
        use_precomputed_msas=kwargs['use_precomputed_msas']
        )
      
    elif self.pipeline_type == 'symdocking' and self.run_multimer_system:
      num_predictions_per_model = kwargs['num_multimer_predictions_per_model']
      data_pipelineA =  phapi.PDBEmptyDataPipeline()
      data_pipelineB = phapi.PDBEmptyDataPipeline()
      self.data_pipeline = phapi.DimerDualDataPipeline(
        data_pipelineA=data_pipelineA,
        data_pipelineB=data_pipelineB,
        A_is_ligand=self.A_is_ligand
      )
    
    # define model runners
    if self.comments :
      print('\n\t - define model runners')
    self.model_runners = {}
    model_names = config.MODEL_PRESETS[kwargs['model_preset']]
    for model_name in model_names:
      model_config = config.model_config(model_name)
      if self.run_multimer_system:
        model_config.model.num_ensemble_eval = num_ensemble
      else:
        model_config.data.eval.num_ensemble = num_ensemble
      model_params = data.get_model_haiku_params(
          model_name=model_name, data_dir=kwargs['data_dir'])
      model_runner = model.RunModel(model_config, model_params)
      for i in range(num_predictions_per_model):
        self.model_runners[f'{model_name}_pred_{i}'] = model_runner
    if self.comments :
      print('Have {} models: {}'.format(
      len(self.model_runners),
      list(self.model_runners.keys())
      ))
    if self.comments :
      print('\n\t - set amber relaxer or not')
    if kwargs['run_relax']:
      self.amber_relaxer = relax.AmberRelaxation(
          max_iterations=RELAX_MAX_ITERATIONS,
          tolerance=RELAX_ENERGY_TOLERANCE,
          stiffness=RELAX_STIFFNESS,
          exclude_residues=RELAX_EXCLUDE_RESIDUES,
          max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
          use_gpu=kwargs['use_gpu_relax'])
    else:
      self.amber_relaxer = None
    if self.comments :
      print('\n\t - set random seed')
    if kwargs['random_seed'] is None:
      self.random_seed = random.randrange(sys.maxsize // len(self.model_runners))
    else:
      self.random_seed = kwargs['random_seed']
    if self.comments:
      print('Using random seed {} '.format(self.random_seed))


  @classmethod
  def compare_kwargs_to_required_keys(cls,kwargs):
    ''' return missing keys and unnecessary keys of input kwargs
        based on required_keys (provided by class method)
    '''
    required_keys = cls.get_required_keys()
    missing_keys =  [ k for k in required_keys if k not in kwargs ]
    unnecessary_keys = [ k for k in kwargs if k not in required_keys ]
    return missing_keys, unnecessary_keys

  @classmethod
  def get_required_keys(cls):
    '''return list of required keys
      {'user': {'a': None, 'b': None}, 'other': {'c': None, 'd': 3, 'e': True}}
       -> ['a', 'b', 'c', 'd', 'e']}
    '''
    required_keys_per_type =cls.get_required_keys_per_type()
    required_keys = list(itertools.chain(*required_keys_per_type.values()))
    return required_keys
  
  @classmethod 
  def get_required_keys_per_type(cls):
    ''' return list of keys per type
    {'user': {'a': None, 'b': None}, 'other': {'c': None, 'd': 3, 'e': True}}
    -> {'user': ['a', 'b'], 'other': ['c', 'd', 'e']}
    '''
    kwargs_template = cls.get_kwargs_template()
    required_keys_per_type = { k:list(v.keys()) for k,v in kwargs_template.items()  }
    return required_keys_per_type

  @classmethod 
  def get_default_kwargs(cls):
    '''return dictionnary of kwargs where value is not None 
    {'user': {'a': None, 'b': None}, 'other': {'c': None, 'd': 3, 'e': True}}
    -> {'d': 3, 'e': True}
    '''
    kwargs_template = cls.get_kwargs_template()
    default_kwargs = { k2:v2 for k in kwargs_template for (k2,v2) in kwargs_template[k].items()}
    return default_kwargs


  @classmethod
  def get_kwargs_template(cls):
    kwargs_template = {
      "settings_paths" :{
        "data_dir" : None,
        "metric_path" : None,
        "uniref90_database_path" : None,
        "mgnify_database_path" : None,
        "template_mmcif_dir" : None,
        "obsolete_pdbs_path" : None,
        "bfd_database_path" : None,
        "small_bfd_database_path" : None,
        "uniref30_database_path" : None,
        "uniprot_database_path" : None,
        "pdb70_database_path" : None,
        "pdb_seqres_database_path" : None,
        "jackhmmer_binary_path" : None,
        "hhblits_binary_path" : None,
        "hhsearch_binary_path" : None,
        "hmmsearch_binary_path" : None,
        "hmmbuild_binary_path" : None,
        "kalign_binary_path" : None
      },
      "settings_alphafolder":{
        #  provided by alphafolder json
        "pipeline_type" : "full",
        "models_out_of_fn" : [1], 
        "max_template_date" : "2050-01-01", 
        "use_gpu_relax" : True, 
        "random_seed" : 7753428609026882419, 
        "db_preset" : "full_dbs", 
        "model_preset" : "monomer",
        "benchmark" : False, 
        "num_multimer_predictions_per_model" : 5, 
        "use_precomputed_msas" : False, 
        "run_relax" : True, 
        "comments": True,
        "output_dir" : "./outputs",
        "cache_dir" : "./cache"
      },
    }
    return kwargs_template



  @classmethod
  def from_jsons(cls, af_json_path ,paths_json_path, **kwargs_override):
    # load jsons
    with open(af_json_path) as af_json_path:
      af_kwargs = json.load(af_json_path)
    with open(paths_json_path) as paths_json_path:
      paths_kwargs = json.load(paths_json_path)

    kwargs = {**af_kwargs, **paths_kwargs} # concatenate json kwargs
    kwargs.update(kwargs_override) # update json kwargs with kwargs_override

    # return instanciated AlphaFolder
    return cls(**kwargs)


  def to_jsons(self, af_json_path, paths_json_path):
    """Saves the instance's configuration to a JSON file."""
    required_keys_per_type = self.get_required_keys_per_type()
    af_kwargs = {k: self.__dict__[k] for k in required_keys_per_type["settings_alphafolder"]}
    paths_kwargs = {k: self.__dict__[k] for k in required_keys_per_type["settings_paths"]}
    with open(af_json_path, "w") as af_json_file:
      json.dump(af_kwargs, af_json_path, indent=4)
    with open(paths_json_path, "w") as paths_json_file:
      json.dump(paths_kwargs, paths_json_file, indent=4)
    
  
  # ideally only one fasta (even in multimer case)
  # but several pdbs (one per monomer)
  # the conversion happens here
  def _sync_inputs(
      self,
      in_fastas,
      in_name,
      in_seqs,
      in_pdbs=None,
    ):
    """
    This function is made to synchronize the inputs so that they are different representation of the same case/problem to solve
    It gather all the data, files, format them, and store them 

    I) if in_name is provided, the fonction verifies if this particular cases is already gathered and formatted to the place it is supposed to 
    (self.cache_dir/in_name). If that the case, it returns the data coming from that particular directory

    II) In the other cases, it attempts to format and gather the files, according to the following rules :

    - case 1 : seq and name are given -> fasta is generated (along with a temp fasta file filled with name and seq)
    - case 2 : only seqs are given : both fasta and name (random string) are generated 
    - case 3 : fasta are given -> name is randomly generated and seqs are loaded from these files, then stored to a unique fasta file
    - case 4 : fasta and name are given -> same than case 3  but name is overriden 
    - case 5 : pdbs : if unique it is split, if non unique, chains are verified 
    """    
    
    out_pdbs = None
    # case where name has a directory and a fasta path
    if in_name  and Path(self.cache_dir, in_name).exists() \
      and Path(self.cache_dir, in_name, in_name).with_suffix('.fasta').exists(): 

      # set name, set fasta and read fasta to get seqs
      out_name = in_name
      cache_dir = Path(self.cache_dir, out_name)
      out_fasta = str(Path(cache_dir, out_name).with_suffix('.fasta'))
      with open(out_fasta) as f:
          fasta_str = f.read()
      out_seqs, _ = parsers.parse_fasta(fasta_str)

      # pdb management
      # if pdb folder exists then use it to set out_pdbs
      # otherwise use pdb input and set it
      pdbs_path = Path(cache_dir, 'pdbs') 
      if pdbs_path.exists(): 
        out_pdbs = glob.glob(str(Path(pdbs_path,'*.pdb' )))
      else:
        out_pdbs = None
      
    elif in_seqs: # case directly using sequences
      # name : create random name if name is not provided
      out_name = (
        ''.join(['id_',uhapi.generate_random_string(12)]) 
        if in_name is None 
        else in_name
      )
      # seq 
      out_seqs = in_seqs 

      # fasta : creation and return
      Path(self.cache_dir, out_name).mkdir(parents=True, exist_ok=True)
      with open(str(Path(self.cache_dir, out_name, out_name).with_suffix('.fasta')), 'w') as f:
      
        if len (out_seqs) >= 2: # multimer case
          for i,seq in enumerate(out_seqs):
            f.write(">" + out_name + "_"  \
              + protein.PDB_CHAIN_IDS[i] + "\n" + seq + "\n")      
        else: # monomer case
          f.write(">" + out_name + "\n" + out_seqs[0] + "\n")
        out_fasta = f.name

    elif in_fastas:
      # path
      if isinstance(in_fastas, str) or  len(in_fastas) == 1: # only one fasta
        # out fasta is this file
        in_fasta = in_fastas if isinstance(in_fastas, str) else in_fastas[0]
        
        # out_name name is overriden if provided otherwise it's the file name
        out_name = (
                ''.join(['id_',uhapi.generate_random_string(12)]) 
                if in_name is None 
                else in_name
              )    

        # cache_dir creation
        cache_dir = Path(self.cache_dir, out_name)
        cache_dir.mkdir(exist_ok=True, parents=True)

        # out fasta    
        out_fasta = str(Path(cache_dir, out_name).with_suffix('.fasta'))
        shutil.copyfile(in_fasta, out_fasta)
      
        # sequences are those provided by the fasta file
        with open(out_fasta) as f:
          fasta_str = f.read()
        out_seqs, _ = parsers.parse_fasta(fasta_str)

      else: # several fastas
         # concatenate sequence in a unique fasta file
        out_name = (
          ''.join(['id_',uhapi.generate_random_string(12)]) 
          if in_name is None 
          else in_name
        )
        out_seqs = []
        descs = []
        for in_fasta in in_fastas:
          # gather all sequences and descriptions
          with open(in_fasta) as f:
            fasta_str = f.read()
          aux_seqs, aux_descs = parsers.parse_fasta(fasta_str)
          out_seqs.extend(aux_seqs)
          descs.extend(aux_descs)

        Path(self.cache_dir, out_name).mkdir(parents=True, exist_ok=True)       
        with open(str(Path(self.cache_dir, out_name, out_name).with_suffix('.fasta')),'w') as f:
          for desc,seq in zip(descs, out_seqs):
            f.write(">" + desc + "\n" + seq + "\n")        
        out_fasta = f.name  
    elif in_pdbs is not None:
      # name
      out_name = (
        ''.join(['id_',uhapi.generate_random_string(12)]) 
        if in_name is None 
        else in_name
      )
      cache_dir = Path(self.cache_dir, out_name)
      pdbs_folder = Path( cache_dir, 'pdbs')
      pdbs_folder.mkdir(parents=True, exist_ok=True)
      # pdb
      in_pdbs = [in_pdbs] if isinstance(in_pdbs,str) else in_pdbs 
      if len(in_pdbs) == 1:
        out_pdbs = uhapi.separate_pdb_multimer_file(in_pdbs[0], output_dir=str(pdbs_folder) )
      else:
        #uhapi.check_unique_chain_ids_and_single_chain_per_file(in_pdbs)
        if self.comments:
          print(f"warning : multiple pdbs are passed for case {out_name}. chains are gathered and rename in ascending order")
        out_pdbs = uhapi.separate_and_sort_chains_from_multiple_pdbs(in_pdbs, pdbs_folder)
        
      #  fasta
      seqs_dict = {k: v for out_pdb in out_pdbs for k, v in uhapi.convert_pdb_to_seqs(out_pdb).items()}
      ordered_seqs = [(k,seqs_dict[k]) for k in protein.PDB_CHAIN_IDS if k in seqs_dict ]
      # path : create temp fasta file
      with open(str(Path(cache_dir, out_name).with_suffix('.fasta')), 'w') as f:
        if len (ordered_seqs) >= 2: # multimer case
          for k,seq in ordered_seqs:
            f.write(">" + out_name + "_"  + k + "\n" + seq + '\n')      
        else: # monomer case
          f.write(">" + out_name  + "\n" + ordered_seqs[0][1] + '\n')
      out_fasta = f.name

      # seqs
      out_seqs = [ seq for (k,seq) in ordered_seqs   ]
           
    else:
        raise ValueError("No valid input provided.")
    # ensure we have multiple seqs only when multimer model is given
    return out_fasta,  out_pdbs, out_name, out_seqs # one fasta, several pdbs, one name, several out_seqs


  def _format_cases(self, unformated_strs):
    """
    return a list of cases (each case is a folding problem to perform)
    This a function made for predict_multi

    in Monomer mode 
      unformated_strs = [ 'str1', 'str2' , 'str3',  'str4'  ]
      -> unformated_strs = [ ['str1'], ['str2'] , ['str3'] , ['str4']  ]

    in Multimer mode 
      unformated_strs = [ 'str1', 'str2' , 'str3'  'str4'  ]
      -> unformated_strs = [ ['str1' , 'str2' ,'str3' , 'str4']  ]
      unformated_strs = [ 'str1',  'str2' , 'SEP' , 'str3' , 'str4', 'SEP' , 'str5' ]
      -> unformated_strs = [ ['str1' , 'str2'] , ['str3' , 'str4'], ['str5']  ]  
    """
    if unformated_strs == [] or unformated_strs is None :
      return []
    elif not self.run_multimer_system: # monomer mode
      assert not ('SEP' in  unformated_strs), 'do not use SEP in monomer mode'
      formated_strs = [[s] for s in unformated_strs]  
      return formated_strs  
    else: # multiple multimer case
      formated_strs = []
      if 'SEP' in unformated_strs:
        while 'SEP' in  unformated_strs:
          sep_index =  unformated_strs.index('SEP')
          formated_strs.append(unformated_strs[:sep_index])
          unformated_strs = unformated_strs[sep_index+1:]
      formated_strs.append(unformated_strs)
      return formated_strs



  def compute_features(
      self,
      fasta_path=None,
      pdbs_paths=None,
      case_name=None,
      seqs=None,
      ):
    """
    function that computes features required for alphafold DNN
    if features appears to already exist, they are loaded
    BEWARE : if name is not given (nor manually or in fasta file) 
    features are recomputed from scratch
    """
    # call _prepare_seq_data to handle different case for inputs
    if self.comments :
      print('computing or loading feature')
      print('\n\t -prepare input data')
    fasta_path, pdbs_paths, case_name, seqs = self._sync_inputs(fasta_path, case_name, seqs, in_pdbs=pdbs_paths)

    timings = {}
    # manage paths
    output_dir = Path(self.output_dir, case_name)
    output_dir.mkdir(exist_ok=True, parents=True)
    msa_output_dir = Path(output_dir, 'msas')
    msa_output_dir.mkdir(exist_ok=True, parents=True)
    

    # Get features.
    t_0 = time.time()
    
    pipeline_kwargs= phapi.preprocess_datapipeline_kwargs(
      self.data_pipeline,
      fasta_path=fasta_path,
      pdbs_paths=pdbs_paths,
      seqs=seqs,
      msa_output_dir=str(msa_output_dir)
      )
    
    # else: # compute the MSA and template
    feature_dict = self.data_pipeline.process(**pipeline_kwargs)
      
    # Write out features as a pickled dictionary if asked
    if self.comments:
      print('\n\t - saving features as a pickl dict')

    timings['features'] = time.time() - t_0
    if self.comments:
      print('returning feature dictionnary')
    return feature_dict, case_name






  def predict_structure(
    self,
    feature_dict,
    case_name=None,
  ):
    if self.comments :
      print("predicting structure")
 
    if self.comments :
      print("\n\t - starting model runner loop")
    output_dir = os.path.join(self.output_dir, case_name)
    unrelaxed_pdbs = {}
    relaxed_pdbs = {}
    relax_metrics = {}
    ranking_confidences = {}
    timings = {}
    query_prediction_results ={}
    # Run the models.
    num_models = len(self.model_runners)
    for model_index, (model_name, model_runner) in enumerate(self.model_runners.items()):
      if self.comments:
        print('\n\t #### Running on {}'.format(case_name))
      t_0 = time.time()
      if self.comments:
        print("\n\t - processing features")

      model_random_seed = model_index + int(self.random_seed) * num_models
      processed_feature_dict = model_runner.process_features(feature_dict, random_seed=model_random_seed)
      timings[f'process_features_{model_name}'] = time.time() - t_0

      t_0 = time.time()
      if self.comments :
        print("\n\t - call predict function from runner")
      prediction_result = model_runner.predict(processed_feature_dict,
                                              random_seed=model_random_seed)
      t_diff = time.time() - t_0
      timings[f'predict_and_compile_{model_name}'] = t_diff
      if self.comments:
        print(
            'Total JAX model {} on {} predict time (includes compilation time, see --benchmark): {}s'.format(
            model_name, case_name, t_diff)
        )
      if model_index in self.models_out_of_fn:
        query_prediction_results[model_name] = prediction_result

      if self.benchmark:
        if self.comments:
          print("\n\t - benchmark")
        t_0 = time.time()
        model_runner.predict(processed_feature_dict,
                            random_seed=model_random_seed)
        t_diff = time.time() - t_0
        timings[f'predict_benchmark_{model_name}'] = t_diff
        if self.comments:
          print(
              'Total JAX model {} on {} predict time (excludes compilation time): {}s'.format(
              model_name, case_name, t_diff)
          )

      plddt = prediction_result['plddt']
      ranking_confidences[model_name] = prediction_result['ranking_confidence']

      # Save the model outputs.
      if self.comments:
        print("\n\t - saving the model outputs")
      result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
      
      with open(result_output_path, 'wb') as f:
        pickle.dump(prediction_result, f, protocol=4)

      # Add the predicted LDDT in the b-factor column.
      # Note that higher predicted LDDT value means higher model confidence.
      plddt_b_factors = np.repeat(
          plddt[:, None], residue_constants.atom_type_num, axis=-1)
      unrelaxed_protein = protein.from_prediction(
          features=processed_feature_dict,
          result=prediction_result,
          b_factors=plddt_b_factors,
          remove_leading_feature_dimension=not model_runner.multimer_mode)

      unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)
      unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
      
      with open(unrelaxed_pdb_path, 'w') as f:
        f.write(unrelaxed_pdbs[model_name])

      if self.amber_relaxer:
        if self.comments:
          print("\n\t -run amber relaxation ")
        # Relax the prediction.
        t_0 = time.time()
        relaxed_pdb_str, _, violations = self.amber_relaxer.process(
            prot=unrelaxed_protein)
        relax_metrics[model_name] = {
            'remaining_violations': violations,
            'remaining_violations_count': sum(violations)
        }
        timings[f'relax_{model_name}'] = time.time() - t_0

        relaxed_pdbs[model_name] = relaxed_pdb_str

        # Save the relaxed PDB.
        relaxed_output_path = os.path.join(
            output_dir, f'relaxed_{model_name}.pdb')
        
        with open(relaxed_output_path, 'w') as f:
          f.write(relaxed_pdb_str)

    # Rank by model confidence and write out relaxed PDBs in rank order.
    if self.comments:
      print("\n\t - rank the model confidence ")
    ranked_order = []
    for idx, (model_name, _) in enumerate(
        sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)):
      ranked_order.append(model_name)
      ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
      
      with open(ranked_output_path, 'w') as f:
        if self.amber_relaxer:
          f.write(relaxed_pdbs[model_name])
        else:
          f.write(unrelaxed_pdbs[model_name])

    ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
    label = 'iptm+ptm' if 'iptm' in prediction_result else 'plddts'
    ranking_json_str = json.dumps(
            {label: ranking_confidences, 'order': ranked_order}, indent=4)
        
    
    with open(ranking_output_path, 'w') as f:
      f.write(ranking_json_str)

    if self.comments:
      print('Final timings for {}: {}'.format(case_name, timings))

    timings_output_path = os.path.join(output_dir, 'timings.json')
    
    with open(timings_output_path, 'w') as f:
      f.write(json.dumps(timings, indent=4))
    if self.amber_relaxer:
      relax_metrics_path = os.path.join(output_dir, 'relax_metrics.json')
       
      with open(relax_metrics_path, 'w') as f:
        f.write(json.dumps(relax_metrics, indent=4))
    if self.comments:
      print("\n\t -this runner is finished ")

    # define what should be returned by the function
    return unrelaxed_pdbs, relaxed_pdbs, query_prediction_results, relax_metrics, timings
  


  def predict(self,
      fasta_path=None,
      pdbs_paths=None,
      case_name=None,
      seqs=None,
      receptor_offset = None,
      hotspots = None
    ):
    if self.comments :
      print("\n Computing or loading features")
      
    feature_dict, case_name = self.compute_features(
      fasta_path=fasta_path,
      pdbs_paths=pdbs_paths,
      case_name=case_name,
      seqs=seqs,
      )
    if self.comments :
      print('\n\t -> Features loaded/computed ')
      print('\n predicting structure')
      
    unrelaxed_pdbs, relaxed_pdbs, query_prediction_results, relax_metrics, timings = self.predict_structure(
      feature_dict=feature_dict,
      case_name=case_name,
    )
    return unrelaxed_pdbs, relaxed_pdbs, query_prediction_results, relax_metrics, timings
  
  def predict_mutiple(
      self,
      fastas_paths=None,
      pdbs_paths=None,
      cases_names=None,
      sequences=None,
      hotspots = None,
      receptor_offset = None
    ):
    # format sequences
    

    fastas_paths = self._format_cases(fastas_paths)
    pdbs_paths = self._format_cases(pdbs_paths)
    sequences = self._format_cases(sequences)

    # run all the problem one after one another    
    results = []
    for (fasta_path,pdbs,case_name,seqs) in zip_longest(fastas_paths, pdbs_paths, cases_names, sequences):
      (unrelaxed_pdbs, 
       relaxed_pdbs, 
       query_prediction_results, 
       relax_metrics, 
       timings
      ) = self.predict(
        fasta_path=fasta_path,
        pdbs_paths=pdbs,
        case_name=case_name,
        seqs=seqs
      )

      results.append((unrelaxed_pdbs, 
       relaxed_pdbs, 
       query_prediction_results, 
       relax_metrics, 
       timings
      ))

    return(unrelaxed_pdbs, 
      relaxed_pdbs, 
      query_prediction_results, 
      relax_metrics, 
      timings
    )




if __name__ == '__main__':
  
  parser = argparse.ArgumentParser(
                    prog='alphafold_HAPI',
                    description='alphafold hacked API')
  
  # main script inputs
  parser.add_argument(
    '--fasta-files',
    type=str,
    nargs='+',
    default=[],
    help='Fasta files to process. Each fasta file contains one monomer or one multimer to process' 
  )

  parser.add_argument(
    '--pdb-files',
    type=str,
    nargs='+',
    default=[],
    help='pdb files to process' 
  )

  parser.add_argument(
    '--sequences',
    type=str,
    nargs='+',
    default=[],
    help='sequences of proteins to process (instead of fasta files). Sequences are considered as a N batches of N monomer to process\
      or one batch of 1 Multimer, dependeing on --preset-mode options. N batches of multimers can also be passed\
      using "SEP" between string. Example --sequences seq1 seq2 SEP seq3 seq4 seq5 SEP seq6 seq7 seq8 seq9 '
  )

  parser.add_argument(
    '--cases-names',
    type=str,
    nargs='+',
    default=[],
    help='replace fasta_names from original alphafold : it either override fasta_names if fasta_files are used as inputs, or gives names to sequences as if they were fastas' 
  )

  parser.add_argument(
    '--settings-alphafolder',
    type=str,
    required=True,
    help='required settings for alphafolder object'
  )

  parser.add_argument(
    '--settings-paths',
    type=str,
    required=True,
    help='required settings for paths'
  )

  # HAPI specific params
  input_kwargs = vars(parser.parse_args())


  # intercept fasta_files and only features option. Filter them from input_kwargs
  fasta_files = input_kwargs['fasta_files']
  pdb_files = input_kwargs['pdb_files']
  sequences =  input_kwargs['sequences'] 
  cases_names = input_kwargs['cases_names']
  settings_alphafolder = input_kwargs['settings_alphafolder']
  settings_paths = input_kwargs['settings_paths']


  alphafolder = AlphaFolder.from_jsons( af_json_path=settings_alphafolder ,paths_json_path=settings_paths)
    
  alphafolder.predict_mutiple(
    fastas_paths=fasta_files,
    pdbs_paths = pdb_files,
    cases_names=cases_names,
    sequences=sequences,
  )
   