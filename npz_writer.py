from source.utils import parse_csv
from source.utils.conforge_conformer_generation import smi2npz
from source.utils.data_splitting_utils import scaffold_splitter
from source.utils.npz_utils import get_field_from_npzs

###################################################
############ first write npzs from csv ############
###################################################

# path = '/home/nobilm@usi.ch/pretrain_paper/data/moelculenet/bace.csv'
# out = parse_csv(path, [0,2])
# out.keys()
# smiles = out['mol']
# ys = out['Class']

# save_dir = '/storage_common/nobilm/pretrain_paper/guacamol/EXPERIMENTS/correct_bace_with_edges'
# smi2npz(
#     save_dir=save_dir,
#     generate_confs=True,
#     smi_list=smiles,
#     ys=ys,
#     split=True,
#     n_confs_to_keep=10,
#     n_confs_to_generate=10,
# )



#################################################################################################
# scaffold splitter
# proj_dir = '/storage_common/nobilm/pretrain_paper/guacamol/EXPERIMENTS/correct_bace_with_edges'
# scaffold_splitter(proj_dir, 'tmp')
#################################################################################################




################################################
############# agument folder confs #############
################################################

starting_path = '/storage_common/nobilm/pretrain_paper/guacamol/EXPERIMENTS/correct_bace_with_edges/train'
save_dir = '/storage_common/nobilm/pretrain_paper/guacamol/EXPERIMENTS/correct_bace_with_edges/train_agumented'

out = get_field_from_npzs(starting_path)
smiles, ys = [], []
for npz_id, npz in enumerate(out):
    smiles.append(str(npz['smiles']))
    ys.append(float(npz['graph_labels']))

smi2npz(
    save_dir=save_dir,
    generate_confs=True,
    smi_list=smiles,
    ys=ys,
    split=False,
    n_confs_to_keep=10,
    n_confs_to_generate=10,
)


starting_path = '/storage_common/nobilm/pretrain_paper/guacamol/EXPERIMENTS/correct_bace_with_edges/val'
save_dir = '/storage_common/nobilm/pretrain_paper/guacamol/EXPERIMENTS/correct_bace_with_edges/val_agumented'

out = get_field_from_npzs(starting_path)
smiles, ys = [], []
for npz_id, npz in enumerate(out):
    smiles.append(str(npz['smiles']))
    ys.append(float(npz['graph_labels']))

smi2npz(
    save_dir=save_dir,
    generate_confs=True,
    smi_list=smiles,
    ys=ys,
    split=False,
    n_confs_to_keep=10,
    n_confs_to_generate=10,
)