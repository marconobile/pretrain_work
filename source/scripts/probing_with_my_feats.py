import sys
sys.path.insert(1, '/home/nobilm@usi.ch/pretrain_paper')
from sklearn.model_selection import PredefinedSplit
from sklearn.svm import SVR
from torch.utils.data import Dataset, DataLoader #, WeightedRandomSampler
import e3nn
from my_general_utils import *
from probing_utils import *
import torch.nn as nn
import numpy as np
import re
import os

def select_best_params_for_rf_with_test(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    import random

    train_ind = [-1 for i in range(len(X_train)-1)]
    val_ind = [0 for i in range(len(X_test)-1)]
    x = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    ps = PredefinedSplit(test_fold=np.concatenate((train_ind,val_ind)))

    param_grid = {
        'n_estimators': [5, 10, 20, 30 , 40, 50, 60, 80, 90, 100, 150, 200, 250, 300, 350,  400, 450, 500],
        'max_depth': [5, 10, 20, 30, 40, 50, 100, 150, 200, None],
        'min_samples_split': [2, 5, 10, 15, 20, 25],
        'min_samples_leaf': [1, 2, 4, 8, 16, 32],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy', 'log_loss']
    }

    l = []
    for i in range(5):
      seed = random.randint(1, 10000)
      rf = RandomForestClassifier(random_state=seed)
      grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=20, verbose=0,  cv=ps)
      grid_search.fit(x, y)
      l.append(f"Best parameters found in round {i}: {grid_search.best_params_}")
    return l

def select_best_params_for_rf(xt, yt):

  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import GridSearchCV

  # Define the parameter grid
  param_grid = {
      'n_estimators': [5, 10, 20, 30 , 40, 50, 60 , 100, 200, 300, 400, 500],
      'max_depth': [5, 10, 20, 30, 40, 50, None],
      'min_samples_split': [2, 5, 10, 15, 20, 25],
      'min_samples_leaf': [1, 2, 4, 8, 16, 32],
      'max_features': ['sqrt', 'log2', None],
      'bootstrap': [True, False],
      'criterion': ['gini', 'entropy', 'log_loss']
  }

  # Initialize the classifier
  rf = RandomForestClassifier()

  # Initialize GridSearchCV
  grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                            cv=3, n_jobs=-1, verbose=2)

  # Fit the model
  grid_search.fit(xt, yt)

  # Print the best parameters
  print(f"Best parameters found: {grid_search.best_params_}")

  # Use the best model
  best_rf = grid_search.best_estimator_
  return best_rf

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

    self.mlp = nn.Sequential(
      FFBlock(out_mul),
      nn.SiLU(),
      nn.LayerNorm(out_mul),
      nn.Linear(out_mul, 1),
)

  def __call__(self, x):
    x = super().__call__(x,x) #!.sigmoid()
    return self.mlp(x) if self.mlp else x

class NpzDataset(Dataset):
    def __init__(self, dir_path,
                 graph_lvl:bool=True,
                 prepend_local_scalars:bool=False
                ):
      self._get_data(dir_path)
      self.graph_lvl = graph_lvl

    def to_list(self):
      xs, ys = [], []
      for x, y in self:
        xs.append(x)
        ys.append(y)
      return xs, ys

    def _get_data(self, dir_path):
      self.data = []
      # Load all .npz files in the specified directory
      for file_name in os.listdir(dir_path):
        if file_name.endswith('.npz'):
          file_path = os.path.join(dir_path, file_name)
          npz_file = np.load(file_path)
          # Create a dictionary for each .npz file's content
          data_dict = {key: npz_file[key] for key in npz_file.files}
          data_dict['target'] = data_dict['target'].squeeze()
          self.data.append(data_dict)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
      # !this logic could be extracted at transform lvl
      if self.graph_lvl:
        data = self.data[idx]
        out = {}
        for k,v in data.items(): out[k] = v.sum(0)
        # if True: #self.smoothing:
        #   if out['target'] == 0: new_target = .1
        #   if out['target'] == 1: new_target = .9
      return out['global_scalar_features_only'], out['target']


###############################################################################################################

parser_entries = [
  {'identifiers': ["-train", "--train"], 'type': Path},
  {'identifiers': ["-val", "--val"], 'type': Path},
  {'identifiers': ["-test", "--test"], 'type': Path, 'optional': True},
]
# python source/scripts/probing_with_my_feats.py -train path -val path -test path

if __name__ == "__main__":
  print("With allegro_features")
  args = MyArgPrsr(parser_entries)
  train_dir, val_dir, test_dir = args.train, args.val,  args.test

  train_dset = NpzDataset(train_dir)
  val_dset = NpzDataset(val_dir)
  test_dset = NpzDataset(test_dir)

  train_loader = DataLoader(train_dset, batch_size=32, shuffle=True) # , num_workers=4, collate_fn=lambda x: to_device_transform(x))
  val_loader = DataLoader(val_dset, batch_size=1, shuffle=False) # , num_workers=4, collate_fn=lambda x: to_device_transform(x))
  test_loader = DataLoader(test_dset, batch_size=1, shuffle=False) # , num_workers=4, collate_fn=lambda x: to_device_transform(x))

  # print(train_dset[0][1])
  # for k,v in train_dset[0].items():
  #   print(k, v.shape)
  #   break
  # exit()

  # train mlp
  mlp = NN(in_features=384, num_epochs=NUMBER_OF_EPOCHS)
  mlp.fit(train_loader)

  # get data for eval
  def get_xy_as_lists_from_dloader(dloader):
    xs, ys = zip(*[(x,y) for x,y in dloader])
    ys = [y.numpy().tolist()[0] for y in list(ys)]
    return list(xs), ys

  print("with NN:")
  x_val, y_val = get_xy_as_lists_from_dloader(val_loader)
  print("Validation")
  mlp.compute_return_post_fit(x_val, y_val)
  x_test, y_test = get_xy_as_lists_from_dloader(test_loader)
  print("Test")
  mlp.compute_return_post_fit(x_test, y_test)



  # select_best_params_for_rf(xt, yt)
  # print(select_best_params_for_rf_with_test(xt, yt, xv, yv))
  # rf_kwargs = {'bootstrap': False, 'criterion': 'entropy', 'max_depth': 20, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
  # for smiles but bad = rf_kwargs =  {'bootstrap': True, 'criterion': 'gini', 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 5}
  # m = RandomForestClassifier(**rf_kwargs)
  # m = RandomForestClassifier(n_estimators=20)
  # fit_and_eval(m, xt, yt, xv, yv)


  # sklearn models
  xt, yt = train_dset.to_list()
  xv, yv = val_dset.to_list()
  xtest, ytest = test_dset.to_list()

  for m in MODELS: fit_and_eval(m, xt, yt, xv, yv, xtest, ytest)



# python source/scripts/probing_with_my_feats.py -train /storage_common/nobilm/pretrain_paper/halicin/not_minimized/replica_1/merge_384_lmax2/train_features -val /storage_common/nobilm/pretrain_paper/halicin/not_minimized/replica_1/merge_384_lmax2/val_features -test /storage_common/nobilm/pretrain_paper/halicin/single_smiles/merge_384_lmax2
# python source/scripts/probing_with_my_feats.py -train /storage_common/nobilm/pretrain_paper/baumannii/not_minimized/replica_1/merge_384_lmax2/train_features -val /storage_common/nobilm/pretrain_paper/baumannii/not_minimized/replica_1/merge_384_lmax2/val_features -test /storage_common/nobilm/pretrain_paper/baumannii/single_smiles/merge_384_lmax2
# python source/scripts/probing_with_my_feats.py -train /storage_common/nobilm/pretrain_paper/opioid/not_minimized/replica_1/merge_384_lmax2/train_features -val /storage_common/nobilm/pretrain_paper/opioid/not_minimized/replica_1/merge_384_lmax2/val_features -test /storage_common/nobilm/pretrain_paper/opioid/single_smiles/merge_384_lmax2









  # #  weighted tp
  # mul=384
  # allegro_encodings_train = [allegro_encodings_train[i][mul:] for i in range(len(allegro_encodings_train))]
  # allegro_encodings_val = [allegro_encodings_val[i][mul:] for i in range(len(allegro_encodings_val))]
  # wtp = WeightedTP(irreps_in1=f'{mul}x1o', irreps_out='512x0e')
  # mlp.model = wtp

  # train_mlp(mlp, allegro_encodings_train, allegro_TRAIN_y, allegro_encodings_val, allegro_VAL_y)
