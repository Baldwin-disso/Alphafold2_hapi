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
import pickle
import time
from typing import Dict, Union
from pathlib import Path
from itertools import zip_longest
import numpy as np
import pandas as pd
# alphafold imports
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import parsers
from alphafold.model import config
from alphafold.common import protein 
# hapi import 
from hapi.hapi_main import AlphaFolder
from hapi.hapi_main import _check_flag
from hapi import hapi_utils as uhapi
import re




#locally update model presets
config.MODEL_PRESETS.update({'multimer':('model_1_multimer_v3',) })
config.MODEL_PRESETS['monomer_casp14'] = config.MODEL_PRESETS['monomer']

class AlphaFolderPPI(AlphaFolder):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    assert  not self.run_relax or not self.no_output_pdbs, "The relax protocol can only be run if the pdb file is predicted"
    
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
        "use_checkpoint":True, 
        "A_is_ligand": True,
        "output_dir" : "./outputs",
        "cache_dir": "./cache",
        "metrics_subdir":"metrics/", 
        "array_task_id" : None,
        "save_metrics" : False,
        "no_output_pdbs" : False,
      }
    }
    return kwargs_template
  


  def write_metrics_csv(self, case_name, prediction_result, t_init,  hotspots=None, receptor_offset=None,    **kwargs ):
    headers = ["file_name", "binderseq", "binderlen", "pae_interaction_total", "pae_binder", "pae_target", "plddt_moy", "plddt_binder", "plddt_target", "ptm", "itpm","time_diff"]

    # only when hotspot exists
    if hotspots is not None:
      headers.append("pae_hotspot")
      headers.append("proxsite")


    csv_output = Path(self.output_dir, 'metrics.csv')
  
    if not Path(csv_output).is_file() or Path(csv_output).stat().st_size == 0:
        with open(str(csv_output), mode='w') as file:
            file.write(";".join(headers) + "\n")
            
    plddt = prediction_result.get('plddt', [])
    pae = prediction_result.get('predicted_aligned_error', [])
    ptm = prediction_result.get('ptm', [])
    iptm = prediction_result.get('iptm', [])
    t_diff = time.time() - t_init

    with open(str(Path(self.cache_dir, case_name, case_name).with_suffix('.fasta'))) as f:
      input_fasta_str = f.read()
      binderseq = parsers.parse_fasta(input_fasta_str)[0][0]

    binderlen = len(binderseq)
    pae_binder, pae_target, pae_interaction_total, pae_hotspot = uhapi.generate_pae_score(binderlen, pae, hotspots=hotspots, receptor_offset=receptor_offset)

    
    plddt_moy = np.mean(plddt)
    plddt_binder = np.mean( plddt[:binderlen] )
    plddt_target = np.mean( plddt[binderlen:] )
    
    # proxsite 
    if hotspots is not None: 
      proxsite = uhapi.generate_proxsite_score(binderlen, final_atom_positions=prediction_result['structure_module']['final_atom_positions'], hotspots=hotspots, receptor_offset=receptor_offset)
      
    
    row = [
        case_name,
        binderseq,
        binderlen,
        pae_interaction_total,
        pae_binder,
        pae_target,
        plddt_moy,
        plddt_binder,
        plddt_target,
        ptm,
        iptm,
        t_diff
    ]

    # only when hotspot exists
    if pae_hotspot is not None:
      row.append(pae_hotspot)
      row.append(proxsite)
    
    with open(str(csv_output), mode='a') as file:
        file.write(";".join(map(str, row)) + "\n")
    



  def save_metrics_fn(self, case_name, prediction_result, coordinates, sequences,  **kwargs):
    
    metric_folder = Path(self.output_dir, self.metrics_subdir)
    metric_folder.mkdir(exist_ok=True, parents=True)
    metric_path = str(Path(metric_folder, 'metrics_{}'.format(case_name) ).with_suffix('.pkl'))
    # creation of dict to save
    aux = uhapi.extract_L_R_format_from_str(case_name)
    if aux is not None:
      ligand, receptor = aux 
      data_to_save = {
        'protein': receptor,
        'ligand': ligand,
        'coordinates_NCAC': coordinates
      }
    else :
      sequences = list(sequences.values())
      data_to_save = {
        'ligand': 'A',
        'protein': 'B',
        'sq_ligand': sequences[0],
        'sq_protein': sequences[1],
        'coordinates_NCAC': coordinates
      }
    
   
    data_to_save['plddt'] = prediction_result.get('plddt', [])

    # multimer case
    if 'ptm' in prediction_result:
        data_to_save['distogram'] = prediction_result.get('distogram', [])
        data_to_save['predicted_aligned_error'] = prediction_result.get('predicted_aligned_error', [])
        data_to_save['ptm'] = prediction_result.get('ptm', [])
        data_to_save['iptm'] = prediction_result.get('iptm', [])

    with open(metric_path, 'wb') as file:
        pickle.dump(data_to_save, file, protocol=4) # for compatibility with PeTriBOX
        
    if self.comments:
      print(f"Saving metrics in .pkl, case : {case_name}")

  
  def checkpoint_filter(self,
    fastas_paths=None,
    pdbs_paths=None,
    cases_names=None,
    sequences=None,
    ):
       
    checkpoint_path = f"{self.cache_dir}/Checkpoint_{self.array_task_id}.csv" \
        if self.array_task_id is not None \
        else f"{self.cache_dir}/Checkpoint\.csv"
    if cases_names is not None and Path(checkpoint_path).exists() :
      # load 
      df = pd.read_csv(checkpoint_path)
      if not df.empty:
          fastas_paths, pdbs_paths, cases_names, sequences = \
            uhapi.remove_cases_after_checkpoints(df, fastas_paths , pdbs_paths, cases_names, sequences)
          if self.comments:
            print("The process will start from the last checkpoint")
    else:
      if self.comments:
        print("The checkpoint file does not exist or no cases names are provided. The process will start from the beginning.")
    return fastas_paths, pdbs_paths, cases_names, sequences 

  

  def predict_structure(
    self,
    feature_dict,
    case_name=None,
    hotspots=None,
    receptor_offset=None,
  ):
    t_init = time.time()
    if self.comments:
      print("predicting structure")
      print("\n\t - preparing input")
    
    if self.comments:
      print("\n\t - starting model runner loop")

 
    unrelaxed_pdbs = {}
    relaxed_pdbs = {}
    relax_metrics = {}
    ranking_confidences = {}
    timings = {}
    query_prediction_results ={}
    # Run the models.
    num_models = len(self.model_runners)
    for model_index, (model_name, model_runner) in enumerate(self.model_runners.items()):
      print('\n\t #### Running on {}'.format(case_name))
      print("\n\t - processing features")

      model_random_seed = model_index + int(self.random_seed) * num_models
      processed_feature_dict = model_runner.process_features(feature_dict, random_seed=model_random_seed)
      
      if self.comments:
        print("\n\t - call predict function from runner")
      prediction_result = model_runner.predict(processed_feature_dict,
                                              random_seed=model_random_seed)
     
      unrelaxed_protein = protein.from_prediction(
        features=processed_feature_dict,
        result=prediction_result,
        b_factors=None,
        remove_leading_feature_dimension=not model_runner.multimer_mode)
       
      self.write_metrics_csv(case_name, prediction_result, t_init, hotspots=hotspots, receptor_offset=receptor_offset)
      if self.save_metrics :
        seq = ''.join([residue_constants.restypes[i] for i in unrelaxed_protein.aatype])
        coordinates = uhapi.extract_canc_coordinates(unrelaxed_protein)
        seqs = uhapi.extract_sequences(unrelaxed_protein)
        self.save_metrics_fn(case_name, prediction_result, coordinates, seqs)
        
      if self.use_checkpoint: 
        t_diff = time.time() - t_init
        checkpoint_path = 'Checkpoint_{}.csv'.format(self.array_task_id) if self.array_task_id is not None else 'Checkpoint.csv'
        path = os.path.join(self.cache_dir, checkpoint_path)
        with open(path, mode='a') as file:
          file.write(f"{case_name}, {t_diff}\n")
          


      if not self.no_output_pdbs:
        unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)
        unrelaxed_pdb_path = os.path.join(self.output_dir, f'pdb_out/unrelaxed_{case_name}.pdb')
        if not os.path.exists(os.path.join(self.output_dir, f'pdb_out')):
                os.makedirs(os.path.join(self.output_dir, f'pdb_out'))
          
        with open(unrelaxed_pdb_path, 'w') as f:
          f.write(unrelaxed_pdbs[model_name])

        if self.amber_relaxer:
          print("\n\t - run amber relaxation ")
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

          #  Save the relaxed PDB.
          relaxed_output_path = os.path.join(
              self.output_dir, f'pdb_out/relaxed_{case_name}.pdb')
            
          with open(relaxed_output_path, 'w') as f:
            f.write(relaxed_pdb_str)
        
    print("\n\t ## Fin du model, runner finished ")
    return unrelaxed_pdbs, relaxed_pdbs, query_prediction_results, relax_metrics, timings



  def predict(self,
      fasta_path=None,
      pdb_path=None,
      case_name=None,
      seqs=None,
      receptor_offset = None,
      hotspots = None
      ):
         
      if self.comments:
        print("\n Computing or loading features")
      feature_dict, case_name = self.compute_features(
        fasta_path=fasta_path,
        pdbs_paths=pdb_path,
        case_name=case_name,
        seqs=seqs,
        )
      if self.comments:
        print('\n\t -> Features loaded/computed ')
        print('\n predicting structure')
      unrelaxed_pdbs, relaxed_pdbs, query_prediction_results, relax_metrics, timings = self.predict_structure(
        feature_dict=feature_dict,
        case_name=case_name,
        receptor_offset=receptor_offset,
        hotspots=hotspots
      )
      return unrelaxed_pdbs, relaxed_pdbs, query_prediction_results, relax_metrics, timings
  


    


  def predict_mutiple( 
      self,
      fastas_paths=None,
      pdbs_paths=None,
      cases_names=None,
      sequences=None,
      hotspots = None,
      receptor_offset = None,
      ):
  

    fastas_paths = self._format_cases(fastas_paths)
    pdbs_paths = self._format_cases(pdbs_paths)
    sequences = self._format_cases(sequences)
    if self.use_checkpoint:
      fastas_paths, pdbs_paths, cases_names, sequences = self.checkpoint_filter(fastas_paths, pdbs_paths, cases_names, sequences)
    
    for (fasta_path,pdb_path,case_name,seqs) in zip_longest(fastas_paths, pdbs_paths, cases_names, sequences):
      checkpoint_path = 'Checkpoint_{}.csv'.format(self.array_task_id) if self.array_task_id is not None else 'Checkpoint.csv'
      path = os.path.join(self.cache_dir, checkpoint_path)

      # HACK
      if len(pdb_path) > 1 and "L_301" in pdb_path[1]:
        print("L_301 is excluded of the process (contains an ambiguous amino acid)")
        if "R_" in pdb_path[0]:
          receptor_name = pdb_path[0].split('/')[-1].split('.')[0]
        name = f'{receptor_name}_L_301'
        with open(path, mode='a') as file:
          file.write(f"{name}, 0 \n")
        unrelaxed_pdbs, relaxed_pdbs, query_prediction_results, relax_metrics, timings = None, None, None, None, None
        continue
            
  
      (unrelaxed_pdbs, relaxed_pdbs, query_prediction_results, relax_metrics, timings) = self.predict(
        fasta_path=fasta_path, 
        pdb_path=pdb_path,
        case_name=case_name,
        seqs=seqs,
        receptor_offset=receptor_offset,
        hotspots=hotspots 
        )

    return(unrelaxed_pdbs, 
      relaxed_pdbs, 
      query_prediction_results, 
      relax_metrics, 
      timings
    )


