Repo for : "Ligand Activity Prediciton Pretraining Large EGNN"

Storage common folder tree:

- models_backups

- tasks:
  1) muOpioid
  2) muOpioid_multiclass
  3) baumannii
  4) halicin


- single_smiles=.npzs of minimized/not_minimized smiles
- single_smiles: .npz of features obtained via model_i

- task
  - single_smiles
    - inputs: .npzs of original mols (both minimized and not minimized)
    - model_i: .npzs of inputs obtained via model_i
	- minimized/not_minimized
		- all: content=all .npzs of minimized/not_minimized mols
    - replica_i
      - train: content=train .npzs of minimized/not_minimized mols
      - val: content=val .npzs of minimized/not_minimized mols
      - test: content=test .npzs of minimized/not_minimized mols
      - model_i
        - train_features: .npz of features obtained via model_i
        - val_features: .npz of features obtained via model_i
        - test_features: .npz of features obtained via model_i



