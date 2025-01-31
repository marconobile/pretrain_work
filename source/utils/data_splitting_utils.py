
import deepchem as dc
# pip install deepchem
import os
import numpy as np
import pandas as pd
from pathlib import Path
from random import shuffle
from typing import Union, List



def scaffold_splitter(smile_list:List[str], save_dir, filepath_list:List[str]=None, frac_train: float = 0.8, frac_valid: float = 0.1, frac_test: float = 0.1, seed: int=42):
    '''https://deepchem.readthedocs.io/en/latest/api_reference/splitters.html#scaffoldsplitter
    https://deepchem.readthedocs.io/en/latest/api_reference/data.html#deepchem.data.DiskDataset.from_numpy

    save_dir – The directory to write this dataset to. If 'tmp' is specified, will use a default temporary directory instead.

    example usage:
    dir = '/storage_common/nobilm/pretrain_paper/guacamol/5k'
    smiles, filepaths = get_smiles_and_filepaths_from_valid_npz(dir)
    train_smiles,val_smiles,test_smiles=scaffold_splitter(smile_list=smiles, save_dir='tmp', filepath_list=filepaths)

    # TODO
    # 1) create folders train/val/test_smiles
    # 2) copy from target dir to created dirs wrt filepath
    '''
    if save_dir == 'tmp':
        save_dir = None

    _ = np.zeros(len(smile_list))
    # creation of a deepchem dataset with the smile codes in the ids field
    if filepath_list:
        dataset = dc.data.DiskDataset.from_numpy(X=filepath_list,y=_,w=_,ids=smile_list, data_dir=save_dir)
    else:
        dataset = dc.data.DiskDataset.from_numpy(X=_,y=_,w=_,ids=smile_list, data_dir=save_dir)

    scaffoldsplitter = dc.splits.ScaffoldSplitter()
    train_idxs, val_idxs, test_idxs = scaffoldsplitter.split(dataset, frac_train, frac_valid, frac_test, seed)

    # ugly but avoids N bool condition eval, keep same out List[Dict] format
    if filepath_list:
        train_smiles = [{'filepath':dataset.X[i], 'smiles':smile_list[i]} for i in train_idxs]
        val_smiles = [{'filepath':dataset.X[i], 'smiles':smile_list[i]} for i in val_idxs]
        test_smiles = [{'filepath':dataset.X[i], 'smiles':smile_list[i]} for i in test_idxs]
    else:
        train_smiles = [{'smiles':smile_list[i]} for i in train_idxs]
        val_smiles = [{'smiles':smile_list[i]} for i in val_idxs]
        test_smiles = [{'smiles':smile_list[i]} for i in test_idxs]

    return train_smiles,val_smiles,test_smiles


def get_smiles_and_targets_from_csv(path):
  '''the csv here processed have only 2 cols, where it is assumed that first col is smiles and second col is label'''
  assert path.endswith(".csv"), f"{path} is not a valid .csv file"
  dset = pd.read_csv(path)
  smi_key, target_key = list(dset.keys())
  smiles, y  = dset[smi_key].to_list(), dset[target_key].to_list()
  assert len(smiles) == len(y)
  return smiles, y


def parse_smiles_file(file_path):
  """
  Parses a .smiles file and returns a list of smiles str. (or a .txt where each line is a smile)
  :param file_path: Path to the .smiles file
  :return: List of smiles strings
  """
  smiles_list = []
  with open(file_path, 'r') as file:
    for line in file:
      smiles = line.strip()
      if smiles: smiles_list.append(smiles)
  return smiles_list


def get_data_folders(dir):
  return Path(dir)/'all', Path(dir)/'train', Path(dir)/'val', Path(dir)/'test'


def create_data_folders(dir):
  r'''if not present, create, if already present, raises: I want to be sure to do not remove good/large data by mistake'''
  all_dir, train_dir, val_dir, test_dir = get_data_folders(dir)
  for p in [all_dir, train_dir, val_dir, test_dir]: os.makedirs(p, exist_ok = True)
  return all_dir, train_dir, val_dir, test_dir


def split_list(the_list, p1, p2, p3):
  assert abs(p1 + p2 + p3 - 1.0) < 1e-6, "Percentages must sum up to 1"
  total_length = len(the_list)
  len_a = int(total_length * p1)
  len_b = int(total_length * p2)
  part_a = the_list[:len_a]
  part_b = the_list[len_a:len_a + len_b]
  part_c = the_list[len_a + len_b:]
  return part_a, part_b, part_c


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


# # TODO: fix this if needed
# def split_train_val_with_balanced_labels(data_dir, perc=(.8, .1, .1)):
#   all_path,train_path,val_path,test_path = create_data_folders(data_dir)
#   npzs_path = ls(all_path)
#   shuffle(npzs_path)
#   positive_examples, negative_examples = split_npz_wrt_label(npzs_path)
#   shuffle(positive_examples)
#   shuffle(negative_examples)
#   # split positive negs in the perc
#   train_positive_examples, val_positive_examples, test_positive_examples = split_list(positive_examples, perc[0], perc[1], perc[2])
#   train_negative_examples, val_negative_examples, test_negative_examples = split_list(negative_examples, perc[0], perc[1], perc[2])
#   test_split_list()
#   assert len(val_positive_examples) + len(train_positive_examples) + len(test_positive_examples) == len(positive_examples)
#   assert len(val_negative_examples) +  len(train_negative_examples) + len(test_negative_examples) == len(negative_examples)
#   train_data = train_positive_examples+train_negative_examples
#   val_data = val_positive_examples+val_negative_examples
#   test_data = test_positive_examples+test_negative_examples
#   shuffle(train_data)
#   shuffle(val_data)
#   shuffle(test_data)
#   move_files_to_folder(train_path, train_data)
#   move_files_to_folder(val_path, val_data)
#   if len(test_positive_examples) + len(test_negative_examples) != 0: move_files_to_folder(test_path, test_data)