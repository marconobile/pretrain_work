from rdkit import Chem as rdChem
import numpy as np

suppl = rdChem.SDMolSupplier('/storage_common/nobilm/pretrain_paper/PCQM4Mv2Dataset/from_wget/pcqm4m-v2-train.sdf')

atom_types_set = set()
atoms_per_mol = []

for mol in suppl:
    if mol is None:
        continue
    atom_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    atom_types_set.update(atom_numbers)
    atoms_per_mol.append(mol.GetNumAtoms())

print("Unique atom types:", atom_types_set)


atoms_per_mol = np.array(atoms_per_mol)
bincount_result = np.bincount(atoms_per_mol)

mean_atoms = np.mean(atoms_per_mol)
median_atoms = np.median(atoms_per_mol)
std_atoms = np.std(atoms_per_mol)
min_atoms = np.min(atoms_per_mol)
max_atoms = np.max(atoms_per_mol)

print("Descriptive statistics of atoms per molecule:")
print(f"Mean: {mean_atoms}")
print(f"Median: {median_atoms}")
print(f"Standard Deviation: {std_atoms}")
print(f"Min: {min_atoms}")
print(f"Max: {max_atoms}")

print("Bincount of atom counts:", bincount_result)


