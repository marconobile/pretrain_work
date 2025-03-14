from source.utils import parse_csv
from source.utils.conforge_conformer_generation import smi2npz
from source.utils.data_splitting_utils import scaffold_splitter
from source.utils.npz_utils import get_field_from_npzs

####################################################
#!########### first write npzs from csv ############
####################################################

# INPUT
path = '/home/nobilm@usi.ch/pretrain_paper/data/moelculenet/bace.csv'
out = parse_csv(path, [0,2])
out.keys()
smiles = out['mol']
ys = out['Class']

# PARAMS
save_dir:str = '/storage_common/nobilm/pretrain_paper/guacamol/EXPERIMENTS/bace_single_conf_with_fragsTEST'
scaffold_splitting:bool = True # add option for label-based splitting
n_confs_to_keep:int=2
n_confs_to_generate:int=10
minRMSD:float=1.5
filter_via_dihedral_fingerprint:bool=False

# automatically generates 3d using CONFORGE
pyg_mols_saved, n_mols_skipped = smi2npz(
    save_dir=save_dir,
    smi_list=smiles,
    ys=ys,
    split=scaffold_splitting,
    n_confs_to_keep=n_confs_to_keep,
    n_confs_to_generate=n_confs_to_generate,
)

if scaffold_splitting:
    scaffold_splitter(save_dir, 'tmp')


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



#################################################################################################
# !ONLY scaffold splitter
# proj_dir = '/storage_common/nobilm/pretrain_paper/guacamol/EXPERIMENTS/correct_bace_with_edges'
# scaffold_splitter(save_dir, 'tmp')
#################################################################################################


#################################################
#!############ agument folder confs #############
#################################################

# starting_path = '/storage_common/nobilm/pretrain_paper/guacamol/EXPERIMENTS/correct_bace_with_edges/train'
# save_dir = '/storage_common/nobilm/pretrain_paper/guacamol/EXPERIMENTS/correct_bace_with_edges/train_agumented'

# out = get_field_from_npzs(starting_path)
# smiles, ys = [], []
# for npz_id, npz in enumerate(out):
#     smiles.append(str(npz['smiles']))
#     ys.append(float(npz['graph_labels']))

# smi2npz(
#     save_dir=save_dir,
#     generate_confs=True,
#     smi_list=smiles,
#     ys=ys,
#     split=False,
#     n_confs_to_keep=10,
#     n_confs_to_generate=10,
# )