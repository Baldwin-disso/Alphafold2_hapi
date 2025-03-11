
![Hapi](https://drive.google.com/uc?id=1EdzB4D1r4zKl8iDayt1WJjJCL2OA-_gQ)

# HAPI - AlphaFold 2 Hacky API 

Here is the original repository of HAPI based on our [paper](https://www.biorxiv.org/content/10.1101/2025.03.05.641631v1), an **independant and unofficial** hacky API for [Deepmind AlphaFold2](https://github.com/google-deepmind/alphafold) allowing to:
- **Seemlessly use AlphaFold 2 model in any Python script** by the mean of the AlphaFolder Object that reads various sequences representations (string, fasta file path, pdb file path)
- **Swap between different types of DataPipelines** in the AlphaFold Model :
    - Original monomer Data Pipeline 
    - Original multimer Data Pipeline (running with multimer engine) 
    - empty pipeline (AlphaFold monomer without MSA or Structure Templates)
    - Pdb pipelines (using pdb files as Structure Templates)
    - Pdb empty pipelines (using pdb files as Structure Templates, No MSA)
    - Dimer pdb piplines (running with multimer engine and using pdb files as Structure Templates, made to output alphafold Scores for dimers and assess protein-protein interactions)
    - (and more to come)
- Use a specific Version of hapi, hapi-ppi to **assess Protein-Peptides interactions**

---

## **Install**

### **1. Install alphafold databases, binaries, weights and an alphafold compatible environnement**
To run Hapi you will need to install alphafold databases and tools and an alphafold compatible conda environnement.
For instance, follow the steps described at the [non-docker setup of alphafold](https://github.com/kalininalab/alphafold_non_docker) including the download of the alphafold databases, if you need to use the Data Pipelines that require them. 

### **2. Install hapi**

1. **Clone official deepmind AlphaFold 2 repos (if not done yet)**
    ```bash
    git clone https://github.com/google-deepmind/alphafold.git
    ```  

    Note that Hapi was tried and developped with alphafold 2.3.2

2. **Clone hapi repository**

    ```bash
    git clone git@github.com:Baldwin-disso/Alphafold2_hapi.git
    ```

3. **Copy alphafold subfolder into hapi root repository**

    ```bash
    cp -r alphafold-2.3.1/alphafold  Alphafold2_hapi/
    ```

4. **Activate previously installed AlphaFold 2 environnement and do a pip installation.**

    For instance, if you used the  [non-docker setup of alphafold](https://github.com/kalininalab/alphafold_non_docker):
    ```
    cd Alphafold2_hapi
    conda activate alphafold
    pip install -r requirements.txt
    pip install -e .
    ```       

### **3. (optional) Download MSAs of test examples**
For some of the included examples ran by the provided shell_scripts, hapi uses the full datapipelines that compute the msas.
As it is a long process, we optionnally provide the msas for quick testing.
Go to the repository (`cd Alphafold2_hapi`) then:

```bash
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nQvVKkGsHSNgu7hjxcqKxvTzelwxbGvp' -O  hapi_examples_msas.tar.gz
tar -xzvf hapi_examples_msas.tar.gz
cp -r hapi_examples_msas/outputs/ .
rm -r hapi_examples_msas && rm hapi_examples_msas.tar.gz
```


### **4. (Optional) Download Propedia dataset to reproduce the paper results**

In the alphafold hapi [paper](), we ran hapi-pi on a subset of the Propedia dataset. 
To reproduce these results, you will need our reformatted version of the propedia dataset [paper]().
To download it, make sure you are in the hapi repo (`cd Alphafold2_hapi`) and that you activate the alphafold environnement (`conda activate alphafold`) then install gdown:

```bash
pip install gdown
```

then run 

```bash
gdown https://drive.google.com/uc?id=1qUYTNw9zy0cf5mo6g1Nu34wZWIH5pNzy
tar -xzvf database.tar.gz
rm database.tar.gz
```

Alternatively, you can manually download the file from a web browser using the following link:

https://drive.google.com/file/d/1qUYTNw9zy0cf5mo6g1Nu34wZWIH5pNzy/view?usp=drive_link

and uncompress it directly in the hapi repository.

##  **Usage**

### **Prerequisite : create a .json path configuration file**

To use AlphaFolder, you will at least need to know and feed the path of the evolutionnary databases and binary from your system, a template is given in `Alphafold2_hapi/settings_paths/paths_templates.json` that you should copy 
and save as `Alphafold2_hapi/settings_paths/paths.json` after you filled it.




Alternatively, we provide  `path_configure.py` python script 
that allows to quickly create a json paths configuration. You only need to provide the root folder of the tools and databases
```bash
python hapi/generate_json_paths_file.py --bin-root <tools-path>  --db-root <databases-path>
```

For example  :

```bash
python hapi/generate_json_paths_file.py --bin-root /mydrive/myuser/anaconda3/envs/alphafold/bin/   --db-root /mydrive/myuser/data/alphafold_required_DB
```

Of course this assumes that binaries and databases respectively shares the same root folder. If that's not the case with you AlphaFold configuration, you will have to set these manually.





### **Instanciating the AlphaFolder Object:** 

For convenience, we provide a class method that instanciate an AlphaFolder object
based on your path json file and any of the alphafolder setting json configuration file provided in  `<Alphafold2_hapi_repo>/settings_alphafolder/`

For instance, to instanciate the AlphaFolder with multimer engine and the normal multimer data pipeline do the following:

```python
from hapi.hapi_main import AlphaFolder
# instanciating alphafold from jsons files
ovrd_kwargs = {} # set override kwargs here
alphafolder = AlphaFolder.from_jsons(
    af_json_path ="settings_alphafolder/alphafolder_multimer.json",
    paths_json_path = "settings_paths/paths.json",
    **ovrd_kwargs
)
```
with 
-  `af_json_path` : json file path containing the main options of AlphaFolder.
-  `paths_json_path` : json path file that defines the paths of the evolutionnary databases and binary (jackhammer etc) required for the pipelines. (**You have to configure it for  your system as described above**), 
- `ovrd_kwargs` allows you to override the values provided by the json files on this instance of alphafolder object.




Alternatively, you can display the required kwargs of AlphaFolder using the following class method:

```python
from hapi.hapi_main import AlphaFolder

default_kwargs = AlphaFolder.get_default_kwargs() 

print(default_kwargs)
```

and then instanciate  AlphaFolder after you modified the kwargs for your system: 

```python 
# your kwargs update
kwargs.update({'output_dir':"./my_output_directory"})
'''
   etc.
   ...
  
'''

# Instanciate AlphaFolder object
alphafolder = AlphaFolder(**default_kwargs)
```


### **Using the AlphaFolder object methods**

AlphaFolder is based on the python class `AlphaFolder`, that initialise alphafold when instanciated.
(How AlphaFolder is instanciated is explained below).

Once, instanciated, alphafolder can be used for inference using the following methods :
- `AlphaFolder.compute_features` to run the different datapipelines hapi proposes, and that return a `feature_dict`
object necessary to feed the neural network part of alphafold. 
It takes the following input arguments :
    - in_name : Name of the inference problem to do. It is an arbitrary name that does not need to be related to the inputs sequences of files.
    if let to None, a random name will be generated for this particular problem
    - in_pdbs or in_fastas or in_sequences : either of these arguments should be provided with a list of pdb path, fastas or sequences.
    note that only one of these arguments should be fed with data. The other 2 arguments have to be `None` which is the default value
    - if multiple pdbs or fastas are provided, hapi will consider that all the chains consists in a single multimer "problem", concatenating all chains from the files.

    This methods return a `feature_dict` and the `case_name` (name of the inference problem)
Note that when called for the first time, hapi will format the inputs and store files in a cache directory, that is indexed by the input_name. 

- `AlphaFolder.predict_structure` run the neural network part of AlphaFold (the Evoformer and the structure module).
it takes two argument as input : 
    - in_name : the name of the problem to solve
    - feature_dict : the feature dict create by `AlphaFolder.compute_features`

- `AlphaFolder.predict` : run consecutively `AlphaFolder.compute_features` and `AlphaFolder.predict_structure`

- `AlphaFolder.predict_multiple` : run sequentially several folding problems. Similarly to `predict` and `compute_features` methods, case name and sequences or fastas or pdbs can be provided. However, because several multimer cases can be passed, we use the string `'SEP'` as a separator to indicate hapi what files/sequences correspond to which case.
    For example :

    ```python
    # import and instanciate then :

    cases_name = ["case1", "case2"]
    sequences = [ "AAAA","MAAHKGA",  "SEP" , "AAAA", "MAAHKGAEHHHKAAEHHEQA", "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFI" ]


    alphafolder.predict_mutiple(
        sequences=sequences,
        cases_names=cases_name,
    )
    ```

    In this example, alphafolder inference is run 2 times with one dimer and one trimer :
    - "case1" dimer with the chains "AAAA" and "MAAHKGA".
    - "case2" trimer  with the chains "AAAA",  "MAAHKGAEHHHKAAEHHEQA" and "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFI".

    For fastas and pdbs, we use the same separator to separate the files used in each case :

    ```python
    # import and instanciate then:

    cases_names = ["pdbcase1", "pdbcase2"]
    pdbs_paths = [ "pdbs/bind_dummy_A.pdb", "pdbs/bind_dummy_B.pdb",  "SEP" , "pdbs/multimer_A.pdb", "pdbs/multimer_B.pdb" ]


    alphafolder.predict_mutiple(
        pdbs_paths=pdbs_paths,
        cases_names=cases_names,
    )
    ```

    In this second example, we have 2 cases :
    - pdbcase1 with the chains coming from "bind_dummy_A.pdb" and "bind_dummy_B.pdb".
    - pdbcase2 with the chains coming from "multimer_A.pdb", "multimer_B.pdb".

    Please Note that the pdb chains will likely be rename in the results. The correspondance of renamed chains can be found in the cache directory



###  **Bash examples**

Hapi can also be ran from command line or bash script. 

Examples are provided in `Alphafold2_hapi/shellscripts` which uses data 
provided in `Alphafold2_hapi/pdbs` and outputs in `Alphafold2_hapi/outputs`.

Note:
- `run_all.sh` will run sequentially all the shell scripts.
- `clean.sh` will clean the ouput directory without removing the msas subdirectories.
-  In order to run, you need both to install hapi and to set up `Alphafold2_hapi/settings_path/paths.json` as explained above.

### **Predict PPI**

AlphaFolder can be used as a quick PPI predictor taking benefits from the provided data  pipelines that don't require the long evolutionnary search.

For this, we also provide the `AlphfolderPPI` object that inherits from  the `AlphaFolder` class which can conveniently write metrics that have been shown to be correlated with the interaction in our paper.

This can be achieved via the `predict_ppi.py` script and an example is provided as `shell_scripts/example_ppi_prediction.sh`



### **Generate outputs to reproduce the paper results**
To reproduce the paper results, that analyse the power of the multimeric mode for PPI, you will likely  need to create a slurm array file 
to generate the hapi_ppi outputs in a embarrassingly parallel fashion. The input `--array-size` and  `--array-task-id` are the parameters that should respectively be fed with `$SLURM_ARRAY_TASK_ID` and `SLURM_ARRAY_TASK_COUNT`.


As a staring basis, We provide `Alphafold2_hapi/shell_scripts/generate_paper_outputs_part.sh` that will process a part of the data (about 1% of the output data).

To generate the full database produced in our paper, you will have to generate all the data defined by the 3 npz file, and you will need about 1.5 TB of storage


## **Cite this Work**

```
@article{connesson2025hapi,
  title={Boosting Protein-Protein Interaction Detection with AlphaFold Multimer and Transformers},
  author={Connesson, l{\'e}na and Krouk, Gabriel and Dumortier, Baldwin},
  journal={BioRxiv},
  pages={2025--03},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

## **License**

This project is licensed under the **Apache License 2.0**. See the `LICENSE` file for details.


This project contains modified code from AlphaFold 2, developed by DeepMind.
AlphaFold 2 is licensed under Apache 2.0. See [LICENSE](LICENSE) for more details.

- AlphaFold 2: [https://github.com/deepmind/alphafold](https://github.com/deepmind/alphafold)
- Apache 2.0 License: [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)



## **Contributing**

If you’d like to contribute, please submit an issue or a pull request. Any improvements are welcome!


## **Contributors**
- **Lena Connesson** - Main developement 
- **Baldwin Dumortier** - Main Developement
- **Gabriel Krouk** - Ideas and other contributions


## **References**

<a id="1">[1]</a> 
Bennett, N. R., Coventry, B., Goreshnik, I., Huang, B., Allen, A., Vafeados, D., ... & Baker, D. (2023). Improving de novo protein binder design with deep learning. Nature Communications, 14(1), 2625.

<a id="2">[2]</a> 
Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Tunyasuvunakool, K., ... & Hassabis, D. (2020). AlphaFold 2. Fourteenth Critical Assessment of Techniques for Protein Structure Prediction.







