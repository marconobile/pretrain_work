import torch
import numpy as np
from rdkit import Chem
from utils import apply_changes, GetDihedral


def do_not_nosify_mol(data):
  data.noise_target = torch.zeros_like(data.pos, dtype=torch.float32)
  return data


def nosify_mol(data):
    '''
    data: pyg data object
    return: pyg data object with new coords
    '''
    dihedral_noise_tau, coords_noise_tau = 2, 0.04
    mol = Chem.MolFromSmiles(str(data.smiles))
    mol = Chem.AddHs(mol, addCoords=True)
    conf = mol.GetConformer() #get_conformer(mol,max_attempts=10)

    # return data obj with noise equal to 0
    if conf == None: return do_not_nosify_mol(data)

    # apply dihedral noise
    try:
        # if good conformer try to apply noise using precomputed torsional idxs/angles
        rotable_bonds = data.rotable_bonds.tolist()
        if rotable_bonds:
            original_dihedral_angles_degrees = data.dihedral_angles_degrees
            # apply dihedral noise
            noised_dihedral_angles_degrees = original_dihedral_angles_degrees + np.random.normal(0, 1, size=original_dihedral_angles_degrees.shape) * dihedral_noise_tau
            apply_changes(mol, noised_dihedral_angles_degrees, rotable_bonds)
    except:
        # if idxs not good, recompute all
        rotable_bonds = get_torsions([mol])
        original_dihedral_angles_degrees = np.array([GetDihedral(conf, rot_bond) for rot_bond in rotable_bonds])
        # apply dihedral noise
        noised_dihedral_angles_degrees = original_dihedral_angles_degrees + np.random.normal(0, 1, size=original_dihedral_angles_degrees.shape) * dihedral_noise_tau
        apply_changes(mol, noised_dihedral_angles_degrees, rotable_bonds)

    # apply coords noise
    pos_after_dihedral_noise = conf.GetPositions()
    pos_noise_to_be_predicted = np.random.normal(0, 1, size=pos_after_dihedral_noise.shape) * coords_noise_tau

    # set in data object for training
    data.noise_target = torch.tensor(pos_noise_to_be_predicted, dtype=torch.float)
    data.pos = torch.tensor(pos_after_dihedral_noise + pos_noise_to_be_predicted, dtype=torch.float)

    return data










