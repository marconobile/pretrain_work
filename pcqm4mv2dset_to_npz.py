from rdkit import Chem as rdChem
import safe
from rdkit import Chem

input_sdf = '/storage_common/nobilm/pretrain_paper/PCQM4Mv2Dataset/from_wget/pcqm4m-v2-train.sdf'
print(f"Reading {input_sdf}")
suppl = rdChem.SDMolSupplier(input_sdf)
print(f'Number of mols in {len(suppl)}:{input_sdf}')

output_sdf = '/storage_common/nobilm/pretrain_paper/PCQM4Mv2Dataset/pcqm4m-v2-with_safe_count/pcqm4m_all_data_with_safe.sdf'
writer = Chem.SDWriter(output_sdf)

for idx, mol in enumerate(suppl):

    # #! ----------------------
    # if idx == 1000:
    #     break
    # #! ----------------------

    if mol is None:
        continue  # Skip invalid molecules
    try:
        smi = rdChem.MolToSmiles(mol)
        safe_smi = safe.encode(smi)
        safe_count = 1
        if '.' in safe_smi:
            safe_count = len(safe_smi.split('.'))
    except:
        # better to skip mol instead of providing misleading data
        continue

    mol.SetProp("safe_count", str(safe_count))
    writer.write(mol)

writer.close()
print(f"Processed molecules written to {output_sdf}")


#! NOW I NEED A smi2npz THAT LEAVES 3D UNALATERED







