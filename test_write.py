from tqdm import tqdm
from rdkit import Chem as rdChem
from collections import OrderedDict
from copy import deepcopy

from source.utils.conforge_conformer_generation import generate_conformers, get_conformer_generator, rdkit_generate_conformers
from source.utils.mol2pyg import mols2pyg_list_with_targets, pyg2mol
from source.utils.mol_utils import get_energy, smi_reader_params, smi_writer_params, drop_disconnected_components,visualize_3d_mols, set_coords, preprocess_mol,has_steric_clashes,fix_conformer,minimum_atom_distance, get_rdkit_conformer
from source.utils.data_splitting_utils import create_data_folders, parse_smiles_file, get_smiles_and_targets_from_csv
from source.data_transforms._frad_transforms import frad

from source.utils.file_handling_utils import ls
from source.utils.npz_utils import get_field_from_npzs, save_npz
from source.utils.data_splitting_utils import scaffold_splitter

import time
import datetime


start_time = time.perf_counter()

# get data
input_data = '/home/nobilm@usi.ch/pretrain_paper/data/halicin_data.csv'
smiles,targets = get_smiles_and_targets_from_csv(input_data)

# create dir
dir = '/storage_common/nobilm/pretrain_paper/second_test'
all_dir, train_dir, val_dir, test_dir = create_data_folders(dir)

# set up conf gen
max_confs = 3 #100
n_confs_to_keep = 2 # num of confs to keep after generating max_confs
conf_generator = get_conformer_generator(max_confs=max_confs)

# todos:
# 1) add splitting via scaffold_splitter <- this takes smiles
# 2) parallelization
# 3) mol filtering?
# 4) logs
# 5) cast to script with mytinyargprs

# train_smiles,val_smiles,test_smiles = scaffold_splitter(smiles, 0.9, 0.1, 0.0)



i_save = 0
for s,y in tqdm(zip(smiles, targets), total=len(smiles)):
  s = drop_disconnected_components(s)
  conformers = generate_conformers(s, conf_generator, n_confs_to_keep)
  if not conformers: continue
  mol2pyg_kwargs = {"max_energy": max((get_energy(m) for m in conformers))}
  pyg_mol_confs = mols2pyg_list_with_targets(conformers, [s]*len(conformers), [y]*len(conformers), **mol2pyg_kwargs)
  i_save = save_npz(pyg_mol_confs, folder_name=all_dir, idx=i_save)


print("final i_save", i_save)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
# Convert elapsed time to hh:mm:ss format
td = datetime.timedelta(seconds=elapsed_time)
str_time = str(td)
print(f"Execution time: {str_time}")


