import os
from source.utils.mol2pyg import mol2pyg
from rdkit import Chem as rdChem
from source.utils.npz_utils import save_npz

input_sdf = '/storage_common/nobilm/pretrain_paper/PCQM4Mv2Dataset/pcqm4m-v2-with_safe_count/pcqm4m_train_safe_HLgap.sdf'
suppl = rdChem.SDMolSupplier(input_sdf)
print(f'initial len: {len(suppl)}')

save_dir = '/storage_common/nobilm/pretrain_paper/PCQM4Mv2Dataset/all' #'/storage_common/nobilm/pretrain_paper/PCQM4Mv2Dataset'
if os.path.exists(save_dir):
    raise FileExistsError(f"The directory '{save_dir}' already exists. Please provide a different directory or remove the existing one.")

pyg_mols_to_save = []
for i, mol in enumerate(suppl):
    pyg_mol = mol2pyg(mol, nsafe=mol.GetProp('safe_count'), homo_lumo_gap=mol.GetProp('homo_lumo_gap'))
    pyg_mols_to_save.append(pyg_mol)

save_npz(pyg_mols_to_save, folder_name=save_dir, split=False)
