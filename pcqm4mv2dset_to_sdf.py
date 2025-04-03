from rdkit import Chem as rdChem
import safe
from rdkit import Chem
from ogb.lsc import PCQM4Mv2Dataset


# there is a 1:1 match between these 2 files
dataset = PCQM4Mv2Dataset(root = '/storage_common/nobilm/pretrain_paper/PCQM4Mv2Dataset', only_smiles = True)
train_idxs = dataset.get_idx_split()['train'] # the only structures for which we have 3d

input_sdf = '/storage_common/nobilm/pretrain_paper/PCQM4Mv2Dataset/from_wget/pcqm4m-v2-train.sdf'
print(f"Reading {input_sdf}")
suppl = rdChem.SDMolSupplier(input_sdf)
print(f'Number of mols in {len(suppl)}:{input_sdf}')

output_sdf = '/storage_common/nobilm/pretrain_paper/PCQM4Mv2Dataset/pcqm4m-v2-with_safe_count/pcqm4m_train_safe_HLgap.sdf' # OUTPUT N: 2791546 after write to npz 2791546
writer = Chem.SDWriter(output_sdf)

for idx, mol in enumerate(suppl):

    if not idx in train_idxs:
        continue

    assert idx in train_idxs

    if mol is None:
        continue  # Skip invalid molecules
    try:
        smi = rdChem.MolToSmiles(mol)
        safe_smi = safe.encode(smi)
        safe_count = 1
        if '.' in safe_smi:
            safe_count = len(safe_smi.split('.'))
    except:
        # skip mol instead of providing misleading data
        continue

    safe_count = int(safe_count)
    assert safe_count is not None, "safe_count is NaN"
    mol.SetProp("safe_count", str(safe_count)) # these must be str to be set

    homo_lumo_gap = float(dataset[idx][1])
    assert homo_lumo_gap is not None, "homo_lumo_gap is NaN"
    mol.SetProp("homo_lumo_gap", str(homo_lumo_gap)) # these must be str to be set

    writer.write(mol)

writer.close()
print(f"Processed molecules written to {output_sdf}")









