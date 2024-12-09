import torch
# from typing import List
# import numpy as np
# import drugtax
# import re
# import matplotlib.pyplot as plt

#  conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
# pip install rdkit
# pip install fcd_torch
# pip install drugtax
# conda install conda-forge::scikit-learn
# conda install e3nn
# pip install -e .


# conda install -yq -c rdkit rdkit  # ?
# pip install fcd_torch
# pip install drugtax
# conda install conda-forge::scikit-learn
# conda install e3nn


# from sklearn.metrics import (
#   precision_recall_fscore_support,
#   accuracy_score,
#   precision_score,
#   recall_score,
#   confusion_matrix,
#   roc_auc_score,
#   balanced_accuracy_score,
#   f1_score,
#   ConfusionMatrixDisplay,
# )
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import svm
# import torch.nn as nn

# # some_file.py
import sys
sys.path.insert(1, '/home/nobilm@usi.ch/pretrain_paper')
from my_general_utils import *
from source.scripts.accuracy_utils import print_eval

# import e3nn

# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

NUMBER_OF_EPOCHS = 400

MODELS = [
  RandomForestClassifier(),
  SVC(probability=True),
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

def focal_loss(preds, targets):
  gamma = 2
  alpha = .65
  p = preds.sigmoid()
  ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, targets, reduction="none")

  p_t = p * targets + (1 - p) * (1 - targets)
  loss = ce_loss * ((1 - p_t) ** gamma)

  if alpha >= 0:
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss

  return loss.mean()

# def get_mlp(inp_dim):
#   multiplier = 4
#   return  torch.nn.Sequential(
#     torch.nn.LayerNorm(inp_dim),
#     torch.nn.Linear(inp_dim, multiplier*inp_dim),
#     torch.nn.SiLU(),
#     torch.nn.LayerNorm(multiplier*inp_dim),
#     torch.nn.Linear(multiplier*inp_dim, 1),
# )

def get_mlp(inp_dim):
  return  torch.nn.Sequential(
    torch.nn.Linear(inp_dim, 4*inp_dim),
    torch.nn.ReLU(),
    # FFBlock(inp_dim),
    # FFBlock(inp_dim),
    # FFBlock(inp_dim),
    # FFBlock(inp_dim),
    torch.nn.Linear(4*inp_dim, 1),
)

class NN(Model):
  def __init__(self, in_features:int, num_epochs:int):
    self.model = get_mlp(in_features)
    self.loss_fn =  torch.nn.BCEWithLogitsLoss() #  focal_loss #  #
    self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-5, weight_decay=5e-3)#torch.optim.AdamW(self.model.parameters(), lr=1e-5, weight_decay=0.0)
    # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
    self.num_epochs = num_epochs

  def fit(self, dloader, device='cuda:0'): # 'cuda:0'):
    self.model.train()
    self.model.to(device)
    # losses = []
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
            # losses.append(loss.item())
    # plt.plot([i for i in range(len(losses))], losses)
    # plt.savefig('train_loss.png')

  @torch.no_grad()
  def _get_probabilities(self, dset):
    self.model.to('cpu')
    return torch.tensor(
      [
        self.model(torch.tensor(dset[i][0])).sigmoid()
        for i in range(len(dset))
      ]
    ).float()

  def get_preds(self, dset):
    self.model.eval()
    return (self._get_probabilities(dset)>.5).float().tolist()

  def get_probabilities(self, dset): return self._get_probabilities(dset).tolist()
  def compute_return_post_fit(self, dset, gt):
    # assert isinstance(dset, list)
    # assert isinstance(gt, list)
    print_eval(gt, self.get_preds(dset), self.get_probabilities(dset))

class FFBlock(torch.nn.Module):
  def __init__(self, inp_size):
    super().__init__()
    self.block = torch.nn.Sequential(
      torch.nn.LayerNorm(inp_size),
      torch.nn.Linear(inp_size, 4*inp_size),
      torch.nn.SiLU(),
      # torch.nn.GELU(approximate='tanh'),
      # torch.nn.LayerNorm(4*inp_size),
      torch.nn.Linear(4*inp_size, inp_size)
    )
  def forward(self, x): return x+self.block(x)

def fit_and_eval(m, xt, yt, xv, yv, xtest=None, ytest=None):
    print(f"With {str(m)}")
    sklearn_model = SklearnModel(m)
    sklearn_model.fit(xt, yt)
    sklearn_model.compute_return_post_fit(xv, yv)
    sklearn_model.compute_return_post_fit(xtest, ytest) # and ytest



