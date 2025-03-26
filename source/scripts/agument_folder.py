
from source.utils import parse_csv
from source.utils.conforge_conformer_generation import smi2npz
from source.utils.data_splitting_utils import scaffold_splitter
from source.utils.npz_utils import get_field_from_npzs

#################################################
#!############ agument folder confs #############
#################################################

scaffold_splitting:bool = False # add option for label-based splitting
starting_path = '/storage_common/nobilm/pretrain_paper/guacamol/EXPERIMENTS/bace_with_safe/train'
save_dir = '/storage_common/nobilm/pretrain_paper/guacamol/EXPERIMENTS/bace_with_safe/train_agumented' # it gets created

out = get_field_from_npzs(starting_path)
smiles, ys, safe_counts = [], [], []
for npz_id, npz in enumerate(out):
    smiles.append(str(npz['smiles']))
    ys.append(float(npz['graph_labels']))
    safe_counts.append(npz['safe_count'])

# PARAMS
n_confs_to_keep:int=100
n_confs_to_generate:int=200
minRMSD:float=2.0
filter_via_dihedral_fingerprint:bool=True

# automatically generates 3d using CONFORGE
pyg_mols_saved, n_mols_skipped = smi2npz(
    save_dir=save_dir,
    smi_list=smiles,
    ys=ys,
    split=scaffold_splitting,
    n_confs_to_keep=n_confs_to_keep,
    n_confs_to_generate=n_confs_to_generate,
    safe_counts=safe_counts,
    filter_via_dihedral_fingerprint=filter_via_dihedral_fingerprint,
)

# Log the values of the variables to a file
log_file_path = f"{save_dir}/data_generation_log.txt"
with open(log_file_path, 'w') as log_file:
    log_file.write(f"scaffold_splitting: {scaffold_splitting}\n")
    log_file.write(f"n_confs_to_keep: {n_confs_to_keep}\n")
    log_file.write(f"n_confs_to_generate: {n_confs_to_generate}\n")
    log_file.write(f"minRMSD: {minRMSD}\n")
    log_file.write(f"filter_via_dihedral_fingerprint: {filter_via_dihedral_fingerprint}\n")
    log_file.write(f"Num of pyg_mols_saved: {len(pyg_mols_saved)}\n")
    log_file.write(f"n_mols_skipped: {n_mols_skipped}\n")
    log_file.write(f"Head 5 of pyg_mols_saved: {pyg_mols_saved[:5]}\n")
