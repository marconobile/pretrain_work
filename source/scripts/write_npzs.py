import os
import shutil

import pandas as pd
from rdkit import Chem
import numpy as np
from random import shuffle

from my_general_utils import *

# for dbg: "args": ["-csv", "/home/nobilm@usi.ch/pretrain_paper/targets_datasets/mu_opioid_receptor_data.csv", "--data_dir", "/storage_common/nobilm/pretrain_paper/TEST"],
parser_entries = [
  {'identifiers': ["-csv", '--csv_path'], 'type': str, 'help': 'The path to the csv containing smiles and labels', 'optional':True},
  {'identifiers': ['--data_dir'], 'type': str, 'help': 'The dir where to save npzs'},
]




def smi2npz(smi, npz_filepath, label, atom2int=None, all_path:str='', minimize:bool=True):
  '''
  writes smiles to npz format
  smi: input smiles
  npz_filepath: npz path/filename.npz
  atom2int: the mapping to go from atom symbol to class for 1hot/nn.embedding
  label, what to set as gt for this obs
  '''
  if not atom2int: assert (all_path != "")
  all_path_smiles = get_field_from_npzs(all_path, 'smiles')
  atom2int, _ = atom2int if atom2int else get_atom2int_int2atom(all_path_smiles)
  print(f'NUM_AtomTypes: {len(atom2int)}, {atom2int}')
  m = preprocess_mol(Chem.MolFromSmiles(smi), sanitize=False)
  pyg_m = mol2pyg(m, atom2int, minimize)
  pyg_m['y'] = label
  _f = lambda x: torch.tensor(x, dtype=torch.float32).reshape(1,1)
  pyg2npz(pyg_m, npz_filepath, f=_f)



