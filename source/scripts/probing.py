from typing import List
import numpy as np
import drugtax

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

# some_file.py
import sys
sys.path.insert(1, '/home/nobilm@usi.ch/pretrain_paper')
from my_general_utils import *

import h5py
from functools import partial

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
  '''
  every model must extend this
  '''
  def __init__(self, model): self.model = model
  def fit(self, x, y): raise NotImplementedError("Implement the way your model trains!")
  def get_preds(self, x): raise NotImplementedError("Implement the way your model predicts!")
  def get_probabilities(self, x): raise NotImplementedError("Implement the way your model gets normalized probs!")
  def _compute_return_post_fit(self, x, gt):

    # preds = self.get_preds(x)
    # print("Confusion matrix: ")
    # print(confusion_matrix(gt, preds))
    # print("Accuracy score: ")
    # self.accuracy_score = accuracy_score(gt, preds)
    # print(self.accuracy_score)
    # print("ROC_AUC score: ")
    # self.roc_auc_score = roc_auc_score(gt, self.get_probabilities(x))
    # print(self.roc_auc_score)

    preds = self.get_preds(x)
    cm = confusion_matrix(gt, preds, labels=[False, True])
    tn, fp, fn, tp = cm.ravel()
    matrix_string = (
        f"Confusion Matrix:\n"
        f"                Predicted\n"
        f"                Positive     Negative\n"
        f"Actual Positive   TP: {tp}        FN: {fn}\n"
        f"       Negative   FP: {fp}        TN: {tn}\n"
    )
    print(matrix_string)
    _roc_auc_score = roc_auc_score(gt, self.get_preds(x))
    print('_roc_auc_score: ', _roc_auc_score)
    precision, recall, fscore, support = precision_recall_fscore_support(gt, preds)
    print('precision: ', precision)
    print('recall: ', recall)
    print('fscore: ', fscore)
    print('support: ', support)
    ba = balanced_accuracy_score(gt, preds)
    f1 = f1_score(gt, preds, average='binary')
    print('balanced accuracy: ', ba)
    print('f1 score: ', f1)

class MLP(Model):
  # todo
  pass

class SklearnModel(Model):
  def __init__(self, model): super().__init__(model)
  def fit(self, x,y): self.model.fit(x, y)
  def get_preds(self, x): return self.model.predict(x)
  def get_probabilities(self, x): return self.model.predict_proba(x)[:,1]



if __name__ == "__main__":

  train_path = "/storage_common/nobilm/pretrain_paper/muOpioid_correct_splits/train"
  val_path = "/storage_common/nobilm/pretrain_paper/muOpioid_correct_splits/test"

  train = get_field_from_npzs(train_path, ["smiles", "graph_labels"])
  val = get_field_from_npzs(val_path, ["smiles", "graph_labels"])

  get_smi_y = lambda x: (x.smiles, x.graph_labels)
  TRAIN_smi, TRAIN_y = zip(*[get_smi_y(i) for i in train])
  VAL_smi, VAL_y = zip(*[get_smi_y(i) for i in val])

  # ENCODERS
  chemnet_encoder = ChemNetEncoder()
  chemnet_encodings_train = chemnet_encoder.encode(TRAIN_smi)
  chemnet_encodings_val = chemnet_encoder.encode(VAL_smi)

  drugtax_encoder = DrugTaxEncoder()
  drugtax_encodings_train = drugtax_encoder.encode(TRAIN_smi)
  drugtax_encodings_val = drugtax_encoder.encode(VAL_smi)

  allegro_encoder = MyAllegroEncoder()
  allegro_encodings_train, allegro_TRAIN_y = allegro_encoder.encode('/storage_common/nobilm/pretrain_paper/muOpioid_correct_splits/train_fingerprints')
  allegro_encodings_val, allegro_VAL_y = allegro_encoder.encode('/storage_common/nobilm/pretrain_paper/muOpioid_correct_splits/test_fingerprints')

  # MODELS
  models = [RandomForestClassifier(), svm.SVC(probability=True), DecisionTreeClassifier(), GradientBoostingClassifier()]

  def f(m, xt, yt, xv, yv):
    print(f"With {str(m)}")
    sklearn_model = SklearnModel(m)
    sklearn_model.fit(xt, yt)
    sklearn_model._compute_return_post_fit(xv, yv)

  eval_chemnet = partial(f,  xt=chemnet_encodings_train, yt=TRAIN_y, xv=chemnet_encodings_val, yv=VAL_y)
  eval_drugtax = partial(f,  xt=drugtax_encodings_train, yt=TRAIN_y, xv=drugtax_encodings_val, yv=VAL_y)
  eval_frad =    partial(f,  xt=allegro_encodings_train, yt=allegro_TRAIN_y, xv=allegro_encodings_val, yv=allegro_VAL_y)

  print("With chemnet_features")
  for m in models: eval_chemnet(m)
  print("With drugtax_features")
  for m in models: eval_drugtax(m)
  print("With allegro_features")
  for m in models: eval_frad(m)

  exit()


#! take the dimensionality from molgps, chemnet, frad
#! molgps: Finetune: we use 2-layer
#! MLPs with a hidden dimension of 256. For each experiment, when retraining this model, we set the
#! dropout rate to zero and train for 40 epochs using a batch size of 256 and a constant learning rate of
#! 0.0001.
#! probing: 2-layered 128 and train for 30 epochs with a batch size of 128, constantLr: 1.e-4

#! chemnet dims 512

#! test with 128, 256, 512 @ frad lvl
