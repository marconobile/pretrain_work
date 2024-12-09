from typing import List
import numpy as np
import drugtax
import re
import matplotlib.pyplot as plt

# conda install -yq -c rdkit rdkit
# pip install fcd_torch
# pip install drugtax
# conda install conda-forge::scikit-learn

from sklearn.metrics import (
  precision_recall_fscore_support,
  accuracy_score,
  precision_score,
  recall_score,
  confusion_matrix,
  roc_auc_score,
  balanced_accuracy_score,
  f1_score,
  ConfusionMatrixDisplay,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm
import torch.nn as nn

# some_file.py
from functools import partial
# from my_general_utils import MyArgPrsr
import e3nn
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from probing_utils import *

import sys
sys.path.insert(1, '/home/nobilm@usi.ch/pretrain_paper')
from my_general_utils import *
# from source.scripts.accuracy_utils import print_eval


class ChemNetEncoder():
  def __init__(self, device='cpu', n_jobs=1, batch_size=512):
    from fcd_torch import FCD
    self.fcd = FCD(device, n_jobs, batch_size)

  def encode(self, smiles_list: List[str]):
    self.encoded_smiles = self.fcd.get_predictions(smiles_list)
    return self.encoded_smiles

class DrugTaxEncoder():
  def encode(self, smiles_list: List[str]): return np.array([np.array(list(drugtax.DrugTax(smi).features.values())) for smi in smiles_list])

class TorchDLoader(Dataset):
  def __init__(self, x, y):
    self.smooth_gt = False
    self.x, self.y = x, y
  def __len__(self): return len(self.x)
  def __getitem__(self, idx):
    # if self.y[idx] == 0: self.y[idx] = self.y[idx] + .1
    # else: self.y[idx] = self.y[idx] - .1
    return self.x[idx], self.y[idx]

def train_mlp(mlp, x_train, y_train, x_test, y_test):
  print(f"With NN")
  train_dset = TorchDLoader(x_train, y_train)
  train_loader = DataLoader(train_dset, batch_size=16, shuffle=True) # sampler
  mlp.fit(train_loader)
  val_dset = TorchDLoader(x_test, y_test)
  mlp.compute_return_post_fit(val_dset, list(y_test))

def get_smiles_and_labels(path):
  get_smi_y = lambda x: (x.smiles, x.graph_labels)
  smiles, y = zip(*[get_smi_y(i) for i in get_field_from_npzs(path, ["smiles", "graph_labels"])])
  return smiles, y

###############################################################################################################

parser_entries = [{'identifiers': ["-task", "--task"], 'type': str}]
# python source/scripts/probing.py -task halicin

if __name__ == "__main__":
  args = MyArgPrsr(parser_entries)

  if args.task == 'opioid':
    train_path = '/storage_common/nobilm/pretrain_paper/opioid/not_minimized/replica_1/train'
    val_path = '/storage_common/nobilm/pretrain_paper/opioid/not_minimized/replica_1/val'
    # test_path = '/storage_common/nobilm/pretrain_paper/opioid/not_minimized/replica_1/test'
    test_path = '/storage_common/nobilm/pretrain_paper/opioid/single_smiles/inputs'
  elif args.task == 'halicin':
    train_path = '/storage_common/nobilm/pretrain_paper/halicin/not_minimized/replica_1/train'
    val_path = '/storage_common/nobilm/pretrain_paper/halicin/not_minimized/replica_1/val'
    test_path = '/storage_common/nobilm/pretrain_paper/halicin/single_smiles/inputs'
  elif args.task == 'baum':
    train_path = '/storage_common/nobilm/pretrain_paper/baumannii/not_minimized/replica_1/train'
    val_path = '/storage_common/nobilm/pretrain_paper/baumannii/not_minimized/replica_1/val'
    test_path = '/storage_common/nobilm/pretrain_paper/baumannii/single_smiles/inputs'
  else: raise ValueError(f"Task {args.task} not valid")

  TRAIN_smi, TRAIN_y = get_smiles_and_labels(train_path)
  VAL_smi, VAL_y = get_smiles_and_labels(val_path)
  TEST_smi, TEST_y = get_smiles_and_labels(test_path)

  chemnet_encoder = ChemNetEncoder(device='cuda:0')
  chemnet_encodings_train = chemnet_encoder.encode(TRAIN_smi)
  chemnet_encodings_val   = chemnet_encoder.encode(VAL_smi)
  chemnet_encodings_test  = chemnet_encoder.encode(TEST_smi)

  drugtax_encoder = DrugTaxEncoder()
  drugtax_encodings_train = drugtax_encoder.encode(TRAIN_smi)
  drugtax_encodings_val   = drugtax_encoder.encode(VAL_smi)
  drugtax_encodings_test  = drugtax_encoder.encode(TEST_smi)

  eval_chemnet = partial(fit_and_eval,
                          xt=chemnet_encodings_train, yt=TRAIN_y,
                          xv=chemnet_encodings_val, yv=VAL_y,
                          xtest=chemnet_encodings_test, ytest=TEST_y,
                        )
  eval_drugtax = partial(fit_and_eval,
                          xt=drugtax_encodings_train, yt=TRAIN_y,
                          xv=drugtax_encodings_val, yv=VAL_y,
                          xtest=drugtax_encodings_test,ytest=TEST_y,
                        )

  print("With chemnet_features")
  for m in MODELS: eval_chemnet(m)
  # train_mlp(NN(512, NUMBER_OF_EPOCHS), chemnet_encodings_train, TRAIN_y, chemnet_encodings_val, VAL_y)

  print("\n")
  print("With drugtax_features")
  for m in MODELS: eval_drugtax(m)
