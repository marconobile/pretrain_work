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
import sys
sys.path.insert(1, '/home/nobilm@usi.ch/pretrain_paper')
from my_general_utils import *

import h5py
from functools import partial
from source.scripts.accuracy_utils import print_eval
from my_general_utils import MyArgPrsr
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

# conda activate probing_venv

#############
# encoeders #
#############

class ChemNetEncoder():
  def __init__(self, device='cpu', n_jobs=1, batch_size=512):
    from fcd_torch import FCD
    self.fcd = FCD(device ,n_jobs ,batch_size)

  def encode(self, smiles_list: List[str]):
    self.encoded_smiles = self.fcd.get_predictions(smiles_list)
    return self.encoded_smiles

class DrugTaxEncoder():
  def encode(self, smiles_list: List[str]):
    return np.array([np.array(list(drugtax.DrugTax(smi).features.values())) for smi in smiles_list])

class MyAllegroEncoder(object):
  def encode(self, h5file):
    h5_file = h5py.File(h5file, 'r')
    obs, gt = [],[]
    for idx in range(len(h5_file)//2): #! // since the h5 contains both x and y
      obs_key, gt_key = f'observation_{idx}', f'ground_truth_{idx}'
      obs.append(np.array(h5_file[obs_key][()]))
      gt.append(np.array(h5_file[gt_key][()]))
    return np.array(obs), np.array(gt).squeeze()

##########
# models #
##########

class Model(object):
  '''every model must extend this'''
  def __init__(self, model): self.model = model
  def fit(self, x, y): raise NotImplementedError("Implement the way your model trains!")
  def get_preds(self, x): raise NotImplementedError("Implement the way your model predicts!")
  def get_probabilities(self, x): raise NotImplementedError("Implement the way your model gets normalized probs!")
  def compute_return_post_fit(self, x, gt): print_eval(gt, self.get_preds(x), self.get_probabilities(x))

class SklearnModel(Model):
  def __init__(self, model): super().__init__(model)
  def fit(self, x,y): self.model.fit(x, y)
  def get_preds(self, x): return self.model.predict(x)
  def get_probabilities(self, x): return self.model.predict_proba(x)[:,1]

###########
# pytorch #
###########

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class TorchDLoader(Dataset):
  def __init__(self, x, y):
    self.smooth_gt = True
    self.x, self.y = x, y
  def __len__(self): return len(self.x)
  def __getitem__(self, idx):
    # if self.y[idx] == 0: self.y[idx] = self.y[idx] + .1
    # else: self.y[idx] = self.y[idx] - .1
    return self.x[idx], self.y[idx]

def get_mlp(inp_dim):
  multiplier = 4
  return  nn.Sequential(
    nn.LayerNorm(inp_dim),
    nn.Linear(inp_dim, multiplier*inp_dim),
    nn.SiLU(),
    nn.LayerNorm(multiplier*inp_dim),
    nn.Linear(multiplier*inp_dim, 1),
)

def focal_loss(preds, targets):
  gamma = 0
  alpha = .9
  p = preds.sigmoid()
  ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, targets, reduction="none")

  p_t = p * targets + (1 - p) * (1 - targets)
  loss = ce_loss * ((1 - p_t) ** gamma)

  if alpha >= 0:
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss

  return loss.mean()

class NN(Model):
  def __init__(self, in_features:int, num_epochs:int):
    self.model = get_mlp(in_features)
    self.loss_fn = focal_loss #nn.BCEWithLogitsLoss()
    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=0.0)
    # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
    self.num_epochs = num_epochs

  def fit(self, dloader, device='cuda:0'):
    self.model.train()
    self.model.to(device)
    losses = []
    for epoch in range(self.num_epochs):
        for x,y in dloader:
            x = x.to(device)
            y = y.to(device)
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            # try: y_pred = self.model(x)
            # except: y_pred = self.model(x, x) # if tensor product
            loss = self.loss_fn(y_pred, y.unsqueeze(-1))# + second_loss(y_pred)
            loss.backward()
            self.optimizer.step()
            # print(epoch, self.optimizer.param_groups[0]["lr"], loss.item())
            losses.append(loss.item())
    plt.plot([i for i in range(len(losses))], losses)
    plt.savefig('train_loss.png')

  @torch.no_grad()
  def _get_probabilities(self, dset):
    self.model.to('cpu')
    return torch.tensor([self.model(torch.tensor(dset[i][0])).sigmoid() for i in range(len(dset))]).float()
    # try: return torch.tensor([self.model(torch.tensor(dset[i][0])).sigmoid() for i in range(len(dset))]).float()
    # except: return torch.tensor([
    #   self.model(torch.tensor(dset[i][0], dset[i][0])).sigmoid()
    #   for i in range(len(dset))
    # ]).float()

  def get_preds(self, dset):
    self.model.eval()
    return (self._get_probabilities(dset)>.5).float().tolist()

  def get_probabilities(self, dset): return self._get_probabilities(dset).tolist()
  def compute_return_post_fit(self, dset, gt): print_eval(gt, self.get_preds(dset), self.get_probabilities(dset))

def train_mlp(mlp, x_train, y_train, x_test, y_test):
  print(f"With NN")
  train_dset = TorchDLoader(x_train, y_train)
  train_loader = DataLoader(train_dset, batch_size=16, shuffle=True) # sampler
  mlp.fit(train_loader)
  val_dset = TorchDLoader(x_test, y_test)
  mlp.compute_return_post_fit(val_dset, list(y_test))

class TPPreprocessor():
  def __init__(self, irreps_in1):
    self.tp = e3nn.o3.FullTensorProduct(
      irreps_in1=irreps_in1,
      irreps_in2=irreps_in1,
      filter_ir_out=[e3nn.o3.Irrep(0, 1)],
      # irrep_normalization ({'component', 'norm'}) –
    )

  def __call__(self, x):
    x = torch.tensor(x)
    return self.tp(x,x)


class FFBlock(torch.nn.Module):
  def __init__(self, inp_size):
    super().__init__()
    self.block = nn.Sequential(
      nn.LayerNorm(inp_size),
      nn.Linear(inp_size, 4*inp_size),
      nn.SiLU(),
      # nn.LayerNorm(4*inp_size),
      nn.Linear(4*inp_size, inp_size)
    )
  def forward(self, x): return x+self.block(x)


class WeightedTP(e3nn.o3.FullyConnectedTensorProduct):
  def __init__(self, irreps_in1, irreps_out):
    super().__init__(
      irreps_in1=irreps_in1,
      irreps_in2=irreps_in1,
      irreps_out=irreps_out,
      internal_weights=True,
      irrep_normalization='component',
      path_normalization='element',
    )
    out_mul = int(re.search(r'\d+', irreps_out).group()) #only works if out is l=0 only
    # self.mlp = get_mlp(out_mul)

    self.mlp = nn.Sequential(
      FFBlock(out_mul),
      # FFBlock(out_mul),
      # FFBlock(out_mul),
      # FFBlock(out_mul),
      # FFBlock(out_mul),
      nn.SiLU(),
      nn.LayerNorm(out_mul),
      nn.Linear(out_mul, 1),
)

  def __call__(self, x):
    x = super().__call__(x,x) #!.sigmoid()
    return self.mlp(x) if self.mlp else x

parser_entries = [
  {'identifiers': ["-task", "--task"], 'type': str},
  {'identifiers': ["-train", "--train"], 'type': Path},
  {'identifiers': ["-val", "--val"], 'type': Path},
]

# python source/scripts/probing.py -task halicin -train /storage_common/nobilm/pretrain_paper/frad_descriptors/frad_descr_128/from_global_interaction/halicin/train_fingerprints_no_transf.h5 -val /storage_common/nobilm/pretrain_paper/frad_descriptors/frad_descr_128/from_global_interaction/halicin/single_smiles/halicin_single_smiles_merged.h5

if __name__ == "__main__":
  args = MyArgPrsr(parser_entries)
  train_fingerprints_path, val_fingerprints_path = args.train, args.val

  if args.task == 'opioid_val':
    train_path = '/storage_common/nobilm/pretrain_paper/muOpioid_correct_splits/train'
    val_path = '/storage_common/nobilm/pretrain_paper/muOpioid_correct_splits/test'
  # elif args.task == 'opioid_test':
  #   train_path = '/storage_common/nobilm/pretrain_paper/muOpioid_correct_splits/train'
  #   val_path = '/storage_common/nobilm/pretrain_paper/muOpioid_correct_splits/test'
  elif args.task == 'halicin_val':
    train_path = '/storage_common/nobilm/pretrain_paper/halicin/train'
    val_path = '/storage_common/nobilm/pretrain_paper/halicin/val'
  elif args.task == 'halicin_test':
    train_path = '/storage_common/nobilm/pretrain_paper/halicin/train'
    val_path = '/storage_common/nobilm/pretrain_paper/halicin/single_smiles/halicin'
  elif args.task == 'baum_val':
    train_path = '/storage_common/nobilm/pretrain_paper/baumannii/train'
    val_path = '/storage_common/nobilm/pretrain_paper/baumannii/val'
  elif args.task == 'baum_test':
    train_path = '/storage_common/nobilm/pretrain_paper/baumannii/train'
    val_path = '/storage_common/nobilm/pretrain_paper/baumannii/single_smiles/abaucin'
  else: raise ValueError(f"Task {args.task} not valid")


  # # OTHER ENCODERS
  train = get_field_from_npzs(train_path, ["smiles", "graph_labels"])
  val = get_field_from_npzs(val_path, ["smiles", "graph_labels"])
  get_smi_y = lambda x: (x.smiles, x.graph_labels)
  TRAIN_smi, TRAIN_y = zip(*[get_smi_y(i) for i in train])
  VAL_smi, VAL_y = zip(*[get_smi_y(i) for i in val])

  chemnet_encoder = ChemNetEncoder(device='cuda:0')
  chemnet_encodings_train = chemnet_encoder.encode(TRAIN_smi)
  chemnet_encodings_val = chemnet_encoder.encode(VAL_smi)

  drugtax_encoder = DrugTaxEncoder()
  drugtax_encodings_train = drugtax_encoder.encode(TRAIN_smi)
  drugtax_encodings_val = drugtax_encoder.encode(VAL_smi)

  allegro_encoder = MyAllegroEncoder()
  allegro_encodings_train, allegro_TRAIN_y = allegro_encoder.encode(train_fingerprints_path)
  allegro_encodings_val, allegro_VAL_y = allegro_encoder.encode(val_fingerprints_path)

  mul=32
  # tp = TPPreprocessor(f'{mul}x1o')
  # allegro_encodings_train = [np.array(tp(allegro_encodings_train[i][mul:])) for i in range(len(allegro_encodings_train))]
  # allegro_encodings_val = [np.array(tp(allegro_encodings_val[i][mul:])) for i in range(len(allegro_encodings_val))]

  # MODELS
  models = []
  models = [ # https://scikit-learn.org/1.5/auto_examples/classification/plot_classifier_comparison.html
    RandomForestClassifier(),
    # svm.SVC(probability=True),
    DecisionTreeClassifier(),
    GradientBoostingClassifier(),
    KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.025, random_state=42, probability=True),
    # SVC(gamma=2, C=1, random_state=42, probability=True),
    # AdaBoostClassifier(algorithm="SAMME", random_state=42),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis(),
    # GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
  ]

  def f(m, xt, yt, xv, yv):
    print(f"With {str(m)}")
    sklearn_model = SklearnModel(m)
    sklearn_model.fit(xt, yt)
    sklearn_model.compute_return_post_fit(xv, yv)

  eval_chemnet, eval_drugtax, eval_frad = None, None, None
  eval_chemnet = partial(f,  xt=chemnet_encodings_train, yt=TRAIN_y, xv=chemnet_encodings_val, yv=VAL_y)
  eval_drugtax = partial(f,  xt=drugtax_encodings_train, yt=TRAIN_y, xv=drugtax_encodings_val, yv=VAL_y)
  eval_frad =    partial(f,  xt=allegro_encodings_train, yt=allegro_TRAIN_y, xv=allegro_encodings_val, yv=allegro_VAL_y)

  if eval_chemnet:
    print("With chemnet_features")
    for m in models: eval_chemnet(m)
    train_mlp(NN(512, 100), chemnet_encodings_train, TRAIN_y, chemnet_encodings_val, VAL_y)
    print("#####################")
  if eval_drugtax:
    print("With drugtax_features")
    for m in models: eval_drugtax(m)
    print("#####################")
  if eval_frad:
    print("With allegro_features")
    for m in models: eval_frad(m)
    mlp = NN(allegro_encodings_train[0].shape[-1], 100) # default model

    # weighted tp
    # mul=384
    # allegro_encodings_train = [allegro_encodings_train[i][mul:] for i in range(len(allegro_encodings_train))]
    # allegro_encodings_val = [allegro_encodings_val[i][mul:] for i in range(len(allegro_encodings_val))]
    # wtp = WeightedTP(irreps_in1=f'{mul}x1o', irreps_out='512x0e')
    # mlp.model = wtp

    train_mlp(mlp, allegro_encodings_train, allegro_TRAIN_y, allegro_encodings_val, allegro_VAL_y)

#! take the dimensionality from molgps, chemnet, frad
#! molgps: Finetune: we use 2-layer
#! NNs with a hidden dimension of 256. For each experiment, when retraining this model, we set the
#! dropout rate to zero and train for 40 epochs using a batch size of 256 and a constant learning rate of
#! 0.0001.
#! probing: 2-layered 128 and train for 30 epochs with a batch size of 128, constantLr: 1.e-4

#! chemnet dims 512

#! test with 128, 256, 512 @ frad lvl
