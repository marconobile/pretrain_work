from typing import List
import numpy as np
import drugtax

# conda install -yq -c rdkit rdkit
# pip install fcd_torch
# pip install drugtax
# conda install conda-forge::scikit-learn

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm

# some_file.py
import sys
sys.path.insert(1, '/home/nobilm@usi.ch/pretrain_paper')
from my_general_utils import *

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

class MyAllegroEncoder():
  # TODO
  pass


# TODO MODEL ENSAMBLING


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
    preds = self.get_preds(x)
    print("Confusion matrix: ")
    print(confusion_matrix(gt, preds))
    print("Accuracy score: ")
    self.accuracy_score = accuracy_score(gt, preds)
    print(self.accuracy_score)
    print("ROC_AUC score: ")
    self.roc_auc_score = roc_auc_score(gt, self.get_probabilities(x))
    print(self.roc_auc_score)



class MLP(Model):
  # todo
  pass

class SklearnModel(Model):
  def __init__(self, model): super().__init__(model)
  def fit(self, x,y): self.model.fit(x, y)
  def get_preds(self, x): return self.model.predict(x)
  def get_probabilities(self, x): return self.model.predict_proba(x)[:,1]



if __name__ == "__main__":

  train_path = "/storage_common/nobilm/pretrain_paper/opioid/train"
  val_path = "/storage_common/nobilm/pretrain_paper/opioid/val"

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

  # MODELS
  print("RF WITH CHEMNET ENCODINGS")
  sklearn_model = SklearnModel(RandomForestClassifier())
  sklearn_model.fit(chemnet_encodings_train, TRAIN_y)
  sklearn_model._compute_return_post_fit(chemnet_encodings_val, VAL_y)

  print("SVM WITH DRUGTAX ENCODINGS")
  sklearn_model.fit(drugtax_encodings_train, TRAIN_y)
  sklearn_model._compute_return_post_fit(drugtax_encodings_val, VAL_y)

  print("SVM WITH CHEMNET ENCODINGS")
  sklearn_model = SklearnModel(svm.SVC(probability=True))
  sklearn_model.fit(chemnet_encodings_train, TRAIN_y)
  sklearn_model._compute_return_post_fit(chemnet_encodings_val, VAL_y)

  print("SVM WITH DRUGTAX ENCODINGS")
  sklearn_model.fit(drugtax_encodings_train, TRAIN_y)
  sklearn_model._compute_return_post_fit(drugtax_encodings_val, VAL_y)

  print("CART WITH CHEMNET ENCODINGS")
  sklearn_model = SklearnModel(DecisionTreeClassifier())
  sklearn_model.fit(chemnet_encodings_train, TRAIN_y)
  sklearn_model._compute_return_post_fit(chemnet_encodings_val, VAL_y)

  print("CART WITH DRUGTAX ENCODINGS")
  sklearn_model.fit(drugtax_encodings_train, TRAIN_y)
  sklearn_model._compute_return_post_fit(drugtax_encodings_val, VAL_y)

  print("GRAD_BOOST WITH CHEMNET ENCODINGS")
  sklearn_model = SklearnModel(GradientBoostingClassifier())
  sklearn_model.fit(chemnet_encodings_train, TRAIN_y)
  sklearn_model._compute_return_post_fit(chemnet_encodings_val, VAL_y)

  print("GRAD_BOOST WITH DRUGTAX ENCODINGS")
  sklearn_model.fit(drugtax_encodings_train, TRAIN_y)
  sklearn_model._compute_return_post_fit(drugtax_encodings_val, VAL_y)





#! take the dimensionality from molgps, chemnet, frad
#! molgps: Finetune: we use 2-layer
#! MLPs with a hidden dimension of 256. For each experiment, when retraining this model, we set the
#! dropout rate to zero and train for 40 epochs using a batch size of 256 and a constant learning rate of
#! 0.0001.
#! probing: 2-layered 128 and train for 30 epochs with a batch size of 128, constantLr: 1.e-4

#! chemnet dims 512

#! test with 128, 256, 512 @ frad lvl
