from source.utils import parse_csv
from source.utils.conforge_conformer_generation import smi2npz
from source.utils.data_splitting_utils import scaffold_splitter
from source.utils.npz_utils import get_field_from_npzs

####################################################
#!########### first write npzs from csv ############
####################################################

# INPUT
path:str = '/home/nobilm@usi.ch/pretrain_paper/data/moelculenet/SAFEbbbp.csv' #SAFEesol.csv' #SAFEbace.csv'
save_dir:str = '/storage_common/nobilm/pretrain_paper/guacamol/EXPERIMENTS/bbbp' # no need to create this, it gets created

# Input processing, after SAFE this is fixed
out = parse_csv(path, [0,2,3]) # after SAFE this is fixed
smiles = out['smiles'] # after SAFE this is fixed
ys = out['ys'] # after SAFE this is fixed
safe_counts = out['num_of_chunks_in_mol'] # after SAFE this is fixed
assert len(smiles) == len(ys) == len(safe_counts)

# PARAMS
use_scaffold_splitting:bool = True # add option for label-based splitting
n_confs_to_keep:int=1
n_confs_to_generate:int=200
minRMSD:float=1.5
filter_via_dihedral_fingerprint:bool=False
fill_with_frad:bool=False #True, # wheter to fill or not the npz with n_confs_to_keep via frad

# automatically generates 3d using CONFORGE
pyg_mols_saved, n_mols_skipped = smi2npz(
    save_dir=save_dir,
    smi_list=smiles,
    ys=ys,
    split=use_scaffold_splitting,
    n_confs_to_keep=n_confs_to_keep,
    n_confs_to_generate=n_confs_to_generate,
    safe_counts=safe_counts,
    filter_via_dihedral_fingerprint=filter_via_dihedral_fingerprint,
    fill_with_frad=fill_with_frad,
)

if use_scaffold_splitting:
    scaffold_splitter(save_dir, 'tmp')

# Log the values of the variables to a file
log_file_path = f"{save_dir}/data_generation_log.txt"
with open(log_file_path, 'w') as log_file:
    log_file.write(f"use_scaffold_splitting: {use_scaffold_splitting}\n")
    log_file.write(f"n_confs_to_keep: {n_confs_to_keep}\n")
    log_file.write(f"n_confs_to_generate: {n_confs_to_generate}\n")
    log_file.write(f"minRMSD: {minRMSD}\n")
    log_file.write(f"filter_via_dihedral_fingerprint: {filter_via_dihedral_fingerprint}\n")
    log_file.write(f"Num of pyg_mols_saved: {len(pyg_mols_saved)}\n")
    log_file.write(f"n_mols_skipped: {n_mols_skipped}\n")
    log_file.write(f"fill_with_frad: {fill_with_frad}\n")
    log_file.write(f"Head 5 of pyg_mols_saved: {pyg_mols_saved[:5]}\n")

#################################################################################################
# !ONLY scaffold splitter
# proj_dir = '/storage_common/nobilm/pretrain_paper/guacamol/EXPERIMENTS/correct_bace_with_edges'
# scaffold_splitter(save_dir, 'tmp')
#################################################################################################

