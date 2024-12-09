import os
import shutil

from pathlib import Path
import pandas as pd
from rdkit import Chem
import numpy as np
from random import shuffle

from my_general_utils import *


parser_entries = [
  {'identifiers': ["-csv", '--csv_path'], 'type': str, 'help': 'The path to the csv containing smiles and labels', 'optional':True},
  {'identifiers': ['--data_dir'], 'type': str, 'help': 'The dir where to save npzs'},
]

# "args": ["-csv", "/home/nobilm@usi.ch/pretrain_paper/targets_datasets/mu_opioid_receptor_data.csv", "--data_dir", "/storage_common/nobilm/pretrain_paper/TEST"],

def get_smiles_and_targets_from_csv(path):
  '''the csv here processed have only 2 cols, where it is assumed that first col is smiles and second col is label'''
  dset = pd.read_csv(path)
  smi_key, target_key = list(dset.keys())
  smiles, y  = dset[smi_key].to_list(), dset[target_key].to_list()
  assert len(smiles) == len(y)
  return smiles, y

def get_data_folders(dir): return Path(dir)/'all', Path(dir)/'train', Path(dir)/'val', Path(dir)/'test'

def split_list(the_list, p1, p2, p3):
    assert abs(p1 + p2 + p3 - 1.0) < 1e-6, "Percentages must sum up to 1"

    total_length = len(the_list)
    len_a = int(total_length * p1)
    len_b = int(total_length * p2)
    len_c = total_length - len_a - len_b  # remaining elements go to c

    part_a = the_list[:len_a]
    part_b = the_list[len_a:len_a + len_b]
    part_c = the_list[len_a + len_b:]

    return part_a, part_b, part_c

def test_split_list():
    # Test case 1: 80%, 15%, 5%
    the_list = list(range(100))
    p1, p2, p3 = 0.8, 0.15, 0.05
    part_a, part_b, part_c = split_list(the_list, p1, p2, p3)

    assert len(part_a) == int(100 * p1), f"Expected {int(100 * p1)} elements in part_a, got {len(part_a)}"
    assert len(part_b) == int(100 * p2), f"Expected {int(100 * p2)} elements in part_b, got {len(part_b)}"
    assert len(part_c) == 100 - int(100 * p1) - int(100 * p2), f"Expected {100 - int(100 * p1) - int(100 * p2)} elements in part_c, got {len(part_c)}"

    # Test case 2: 80%, 20%, 0%
    the_list = list(range(100))
    p1, p2, p3 = 0.8, 0.2, 0.0
    part_a, part_b, part_c = split_list(the_list, p1, p2, p3)

    assert len(part_a) == int(100 * p1), f"Expected {int(100 * p1)} elements in part_a, got {len(part_a)}"
    assert len(part_b) == int(100 * p2), f"Expected {int(100 * p2)} elements in part_b, got {len(part_b)}"
    assert len(part_c) == 100 - int(100 * p1) - int(100 * p2), f"Expected {100 - int(100 * p1) - int(100 * p2)} elements in part_c, got {len(part_c)}"

    print("All tests passed!")

def create_data_folders(dir):
  r'''if not present, create, if already present, raises: I want to be sure to do not remove good/large data by mistake'''
  all_dir, train_dir, val_dir, test_dir = get_data_folders(dir)
  for p in [all_dir, train_dir, val_dir, test_dir]: os.makedirs(p, exist_ok = True)
  return all_dir, train_dir, val_dir, test_dir

def get_atom2int_respecting_split_tr_val_te(train_smiles:List[str], train_ys:List, val_smiles:List[str], val_ys:List, test_smiles:List[str], test_ys :List):
  splits = [(train_smiles, train_ys), (val_smiles, val_ys), (test_smiles, test_ys)]
  out_mols = [[] for _ in splits]
  atom_types = set('H')

  for i, l in enumerate(splits):
    for smi, lbl in zip(l[0], l[1]):
      m = preprocess_mol(Chem.MolFromSmiles(smi))
      if m == None: continue
      for atom in Chem.RemoveHs(m).GetAtoms(): atom_types.add(atom.GetSymbol()) # RemoveHs returns mol without Hs, input m is *not modifed!*
      out_mols[i].append((m, lbl))

  atom_types = list(atom_types)
  atom_types.sort()
  atom2int = {atom_type: i for i, atom_type in enumerate(atom_types)}

  def listify(idx:int):
    mols, labels = zip(*out_mols[idx])
    return  list(mols), list(labels)

  train_smiles, train_ys = listify(0)
  val_smiles, val_ys = listify(1)
  test_smiles, test_ys = listify(2)

  return atom2int, train_smiles, train_ys, val_smiles, val_ys, test_smiles, test_ys

def get_atom2int_int2atom(smiles):
  atom_types = set('H')
  for smi in smiles:
    m = preprocess_mol(Chem.MolFromSmiles(smi), sanitize=False)
    if m == None: continue
    for atom in Chem.RemoveHs(m).GetAtoms(): atom_types.add(atom.GetSymbol()) # RemoveHs returns mol without Hs, input m is *not modifed!*
  atom_types = list(atom_types)
  atom_types.sort()
  atom2int = {atom_type: i for i, atom_type in enumerate(atom_types)}
  return atom2int, {v:k for k,v in atom2int.items()}

def get_atom2int_respecting_split_tr_te(train_smiles:List[str], train_ys:List, test_smiles:List[str], test_ys :List):
  splits = [(train_smiles, train_ys), (test_smiles, test_ys)]
  out_mols = [[] for _ in splits]
  atom_types = set('H')

  for i, l in enumerate(splits):
    for smi, lbl in zip(l[0], l[1]):
      m = preprocess_mol(Chem.MolFromSmiles(smi), sanitize=False)
      if m == None: continue
      for atom in Chem.RemoveHs(m).GetAtoms(): atom_types.add(atom.GetSymbol()) # RemoveHs returns mol without Hs, input m is *not modifed!*
      out_mols[i].append((m, lbl))

  atom_types = list(atom_types)
  atom_types.sort()
  atom2int = {atom_type: i for i, atom_type in enumerate(atom_types)}

  def listify(idx:int):
    mols, labels = zip(*out_mols[idx])
    return  list(mols), list(labels)

  train_mols, train_ys = listify(0)
  test_mols, test_ys = listify(1)
  return atom2int, train_mols, train_ys, test_mols, test_ys

def write_npzs(args):
  all, _, _, _ = create_data_folders(args.data_dir)
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

  pyg_mols = mols2pyg_list(mols, ys, atom2int)

  print(f"#Input mols {len(smiles)}, #mols left: {len(pyg_mols)}, #mols dropped in preprocessing: {len(smiles)-len(pyg_mols)}")
  save_npz(pyg_mols=pyg_mols, f=lambda y: y.reshape(1,1), folder_name=all)

def mols2pyg_list(mols, ys, atom2int):
  pyg_mols = []
  for m, y in zip(mols, ys):
    pyg_m = mol2pyg(m, atom2int)
    if pyg_m == None: continue
    pyg_m.y = np.array(y, dtype=np.float32)
    pyg_mols.append(pyg_m)
  return pyg_mols

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

def split_train_val_with_balanced_labels(data_dir, perc=(.8, .1, .1)):

  all_path,train_path,val_path,test_path = create_data_folders(data_dir)
  npzs_path = ls(all_path)
  shuffle(npzs_path)
  positive_examples, negative_examples = split_npz_wrt_label(npzs_path)
  shuffle(positive_examples)
  shuffle(negative_examples)

  # split positive negs in the perc
  train_positive_examples, val_positive_examples, test_positive_examples = split_list(positive_examples, perc[0], perc[1], perc[2])
  train_negative_examples, val_negative_examples, test_negative_examples = split_list(negative_examples, perc[0], perc[1], perc[2])

  test_split_list()
  assert len(val_positive_examples) + len(train_positive_examples) + len(test_positive_examples) == len(positive_examples)
  assert len(val_negative_examples) +  len(train_negative_examples) + len(test_negative_examples) == len(negative_examples)

  train_data = train_positive_examples+train_negative_examples
  val_data = val_positive_examples+val_negative_examples
  test_data = test_positive_examples+test_negative_examples

  shuffle(train_data)
  shuffle(val_data)
  shuffle(test_data)

  move_files_to_folder(train_path, train_data)
  move_files_to_folder(val_path, val_data)
  if len(test_positive_examples) + len(test_negative_examples) != 0: move_files_to_folder(test_path, test_data)


def opioid_dset_handling(args, create_folders:bool=True):
  all, train, val, test = create_data_folders(args.data_dir) if create_folders else get_data_folders(args.data_dir)

  data_train_val = "/home/nobilm@usi.ch/pretrain_paper/targets_datasets/mu_opioid_receptor_data.csv"
  data_test = "/home/nobilm@usi.ch/pretrain_paper/targets_datasets/mu_opioid_receptor_test_data.csv"

  train_val_smiles, train_val_labels = get_smiles_and_targets_from_csv(data_train_val)
  test_smiles, test_labels = get_smiles_and_targets_from_csv(data_test)

  atom2int, train_mols, train_ys, test_mols, test_ys = get_atom2int_respecting_split_tr_te(train_val_smiles,
                                                                                           train_val_labels,
                                                                                           test_smiles,
                                                                                           test_labels)

  pyg_mols_train = mols2pyg_list(train_mols, train_ys, atom2int)
  pyg_mols_test = mols2pyg_list(test_mols, test_ys, atom2int)

  # saves all train and val npzs in all folder
  save_npz(pyg_mols=pyg_mols_train, f=lambda y: y.reshape(1,1), folder_name=all)
  split_train_val_with_balanced_labels(all, train, val)
  save_npz(pyg_mols=pyg_mols_test, f=lambda y: y.reshape(1,1), folder_name=test)

def parse_smiles_file(file_path):
  """
  Parses a .smiles file and returns a list of smiles str.
  :param file_path: Path to the .smiles file
  :return: List of smiles strings
  """
  smiles_list = []
  with open(file_path, 'r') as file:
    for line in file:
      smiles = line.strip()
      if smiles: smiles_list.append(smiles)
  return smiles_list

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

# def guacamol2npz(path, args):
#   smiles = parse_smiles_file(path)
#   create_folders = True
#   all, train, val, test = create_data_folders(args.data_dir) if create_folders else get_data_folders(args.data_dir)
#   # get_atom2int_respecting_split_tr_te(train_smiles:List[str], train_ys:List, test_smiles:List[str], test_ys :List)
# #   pyg_mols_test = mols2pyg_list(test_mols, test_ys, atom2int)
# #   save_npz(pyg_mols=pyg_mols_test, f=lambda y: y.reshape(1,1), folder_name=test)

def rm_and_recreate_dir(dir):
  if os.path.exists(dir): shutil.rmtree(dir)
  os.makedirs(dir)

def write_single_smiles():
  #! to write a single smiles 2 npz
  smi = "C=CCN1CC[C@]23[C@@H]4C(=O)CC[C@]2([C@H]1CC5=C3C(=C(C=C5)O)O4)O"
  # file = '/storage_common/nobilm/pretrain_paper/frad_descriptors/frad_descr_128/from_global_interaction/opioid/naloxone_not_minimized/naloxone'
  rm_and_recreate_dir(os.path.dirname(file))
  smi2npz(smi, file, label = 0, all_path='/storage_common/nobilm/pretrain_paper/merged/all', minimize=False)

  smi = "CCC(=O)N(C1CCN(CC1)CCC2=CC=CC=C2)C3=CC=CC=C3"
  # file = '/storage_common/nobilm/pretrain_paper/frad_descriptors/frad_descr_128/from_global_interaction/opioid/fentanil_not_minimized/fentanil'
  rm_and_recreate_dir(os.path.dirname(file))
  smi2npz(smi, file, label = 1, all_path='/storage_common/nobilm/pretrain_paper/merged/all', minimize=False)

  smi = 'C1=C(SC(=N1)SC2=NN=C(S2)N)[N+](=O)[O-]'
  # file = '/storage_common/nobilm/pretrain_paper/frad_descriptors/frad_descr_128/from_global_interaction/halicin/halicin_not_minimized/halicin'
  rm_and_recreate_dir(os.path.dirname(file))
  smi2npz(smi, file, label = 1, all_path='/storage_common/nobilm/pretrain_paper/merged/all', minimize=False)

  smi = 'C1CN(CCC12C3=CC=CC=C3NC(=O)O2)CCC4=CC=C(C=C4)C(F)(F)F'
  # file = '/storage_common/nobilm/pretrain_paper/frad_descriptors/frad_descr_128/from_global_interaction/baum/abaucin_not_minimized/abaucin'
  rm_and_recreate_dir(os.path.dirname(file))
  smi2npz(smi, file, label = 1, all_path='/storage_common/nobilm/pretrain_paper/merged/all', minimize=False)

def main():
  args = MyArgPrsr(parser_entries)

  # split_train_val_with_balanced_labels(args.data_dir)

  # write_npzs(args)

  # all, train, val, test = get_data_folders(args.data_dir)
  # create_split_for_muOR(all, train, val)
  # split_train_val_with_balanced_labels(all, train, val)

  #! get opioid data
  opioid_dset_handling(args, create_folders=False)



if __name__ == '__main__':
  main()


# halicin: NUM_AtomTypes: 23, types: {'Al': 0, 'As': 1, 'Bi': 2, 'Br': 3, 'C': 4, 'Ca': 5, 'Cl': 6, 'Co': 7, 'F': 8, 'Fe': 9, 'H': 10, 'Hg': 11, 'I': 12, 'N': 13, 'O': 14, 'P': 15, 'Pb': 16, 'Pt': 17, 'S': 18, 'Sb': 19, 'Se': 20, 'Si': 21, 'Zn': 22}
# with new preprocessing {'Al': 0, 'As': 1, 'Bi': 2, 'Br': 3, 'C': 4, 'Ca': 5, 'Cl': 6, 'Co': 7, 'F': 8, 'H': 9, 'Hg': 10, 'I': 11, 'N': 12, 'O': 13, 'P': 14, 'Pb': 15, 'Pt': 16, 'S': 17, 'Sb': 18, 'Se': 19, 'Si': 20, 'Zn': 21}
# baum: {'As': 0, 'Bi': 1, 'Br': 2, 'C': 3, 'Ca': 4, 'Cl': 5, 'Co': 6, 'F': 7, 'H': 8, 'Hg': 9, 'I': 10, 'N': 11, 'O': 12, 'P': 13, 'Pb': 14, 'Pt': 15, 'S': 16, 'Sb': 17, 'Se': 18, 'Si': 19, 'Zn': 20}
# opioid: {'Br': 0, 'C': 1, 'Cl': 2, 'F': 3, 'H': 4, 'I': 5, 'N': 6, 'O': 7, 'S': 8}


#! merged{'Al': 0, 'As': 1, 'Bi': 2, 'Br': 3, 'C': 4, 'Ca': 5, 'Cl': 6, 'Co': 7, 'F': 8, 'H': 9, 'Hg': 10, 'I': 11, 'N': 12, 'O': 13, 'P': 14, 'Pb': 15, 'Pt': 16, 'S': 17, 'Sb': 18, 'Se': 19, 'Si': 20, 'Zn': 21}




