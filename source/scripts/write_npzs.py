import os

from pathlib import Path
import pandas as pd
from rdkit import Chem
import numpy as np
from random import shuffle

from my_general_utils import *
# from my_general_utils import (
#   MyArgPrsr,
#   preprocess_mol,
#   mol2pyg,
#   save_npz,
#   ls,
#   move_files_to_folder,
#   split_list,
# )

parser_entries = [
  {'identifiers': ["-csv", '--csv_path'], 'type': str, 'help': 'The path to the csv containing smiles and labels'},
  {'identifiers': ['--data_dir'], 'type': str, 'help': 'The dir where to save npzs'},
]

# "args": ["-csv", "/home/nobilm@usi.ch/pretrain_paper/targets_datasets/mu_opioid_receptor_data.csv", "--data_dir", "/storage_common/nobilm/pretrain_paper/TEST"],

def get_smiles_and_targets_from_csv(path):
  # the csv here processed have only 2 cols, where it is assumed that first col is smiles and second col is label
  dset = pd.read_csv(path)
  smi_key, target_key = list(dset.keys())
  smiles, y  = dset[smi_key].to_list(), dset[target_key].to_list()
  assert len(smiles) == len(y)
  return smiles, y

def get_data_folders(dir): return Path(dir)/'all', Path(dir)/'train', Path(dir)/'val'

def create_data_folders(dir):
  r'''if not present, create, if already present, raises: I want to be sure to do not remove good/large data by mistake'''
  all, train, val = get_data_folders(dir)
  for p in [all, train, val]: os.makedirs(p, exist_ok = False)
  return all, train, val

def write_npzs(args):
  all, _, _ = create_data_folders(args.data_dir)
  smiles, labels = get_smiles_and_targets_from_csv(args.csv_path)

  mols, ys = [], []
  atom_types = set('H')
  for smi, lbl in zip(smiles, labels):
    m = preprocess_mol(Chem.MolFromSmiles(smi))
    if m == None: continue
    for atom in Chem.RemoveHs(m).GetAtoms(): # RemoveHs returns mol without Hs, input m is *not modifed*
      atom_types.add(atom.GetSymbol())
    mols.append(m)
    ys.append(lbl)

  atom_types = list(atom_types)
  atom_types.sort()
  atom2int = {atom_type: i for i, atom_type in enumerate(atom_types)}
  print(f"NUM_AtomTypes: {len(atom2int)}, types: {atom2int}")

  pyg_mols = []
  for m, y in zip(mols, ys):
    pyg_m = mol2pyg(m, atom2int)
    if pyg_m == None: continue
    pyg_m.y = np.array(y, dtype=np.float32)
    pyg_mols.append(pyg_m)

  print(f"#Input mols {len(smiles)}, #mols left: {len(pyg_mols)}, #mols dropped in preprocessing: {len(smiles)-len(pyg_mols)}")
  save_npz(pyg_mols=pyg_mols, f=lambda y: y.reshape(1,1), folder_name=all)

def split_npz_wrt_label(filepaths):
    # split .npz filepaths wrt obs label
    positive_examples, negative_examples = [], []
    for file in filepaths:
        data = np.load(file)
        label = data['graph_labels'].item()
        if label == 0: negative_examples.append(file)
        elif label == 1: positive_examples.append(file)
        else: raise ValueError("graph_labels not in {0,1}")
    return positive_examples, negative_examples

def create_split_for_muOR(all_path, train_path, val_path):
  npzs_path = ls(all_path)
  shuffle(npzs_path)
  positive_examples, negative_examples = split_npz_wrt_label(npzs_path)
  shuffle(positive_examples)
  shuffle(negative_examples)

  val_positive_examples, train_positive_examples = positive_examples[:50], positive_examples[50:]
  val_negative_examples, train_negative_examples = negative_examples[:50], negative_examples[50:]
  assert len(val_positive_examples) + len(train_positive_examples) == len(positive_examples)
  assert len(val_negative_examples) +  len(train_negative_examples) == len(negative_examples)

  move_files_to_folder(train_path, train_positive_examples+train_negative_examples)
  move_files_to_folder(val_path, val_positive_examples+val_negative_examples)

def split_train_val_with_balanced_labels(all_path, train_path, val_path, perc=.1):
  npzs_path = ls(all_path)
  shuffle(npzs_path)
  positive_examples, negative_examples = split_npz_wrt_label(npzs_path)
  shuffle(positive_examples)
  shuffle(negative_examples)

  val_positive_examples, train_positive_examples = split_list(positive_examples, perc)
  val_negative_examples, train_negative_examples = split_list(negative_examples, perc)
  assert len(val_positive_examples) + len(train_positive_examples) == len(positive_examples)
  assert len(val_negative_examples) +  len(train_negative_examples) == len(negative_examples)

  move_files_to_folder(val_path, val_positive_examples+val_negative_examples)
  move_files_to_folder(train_path, train_positive_examples+train_negative_examples)

def opioid_dset_handling(args):
  # todo
  # 1) standardize atom2int
  # 2) atomicize functions to make this function easy!
  pass

def main():
  args = MyArgPrsr(parser_entries)
  # write_npzs(args)

  # all, train, val = get_data_folders(args.data_dir)
  # create_split_for_muOR(all, train, val)
  # split_train_val_with_balanced_labels(all, train, val)



if __name__ == '__main__':
  main()


# halicin: NUM_AtomTypes: 23, types: {'Al': 0, 'As': 1, 'Bi': 2, 'Br': 3, 'C': 4, 'Ca': 5, 'Cl': 6, 'Co': 7, 'F': 8, 'Fe': 9, 'H': 10, 'Hg': 11, 'I': 12, 'N': 13, 'O': 14, 'P': 15, 'Pb': 16, 'Pt': 17, 'S': 18, 'Sb': 19, 'Se': 20, 'Si': 21, 'Zn': 22}



# for opioid:
# unisco csv di test
# parso csv
# split in tran+val e test