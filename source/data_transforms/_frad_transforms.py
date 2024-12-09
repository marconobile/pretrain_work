import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Chem.rdchem import HybridizationType
from geqtrain.utils.torch_geometric import Data

from my_general_utils import (
  get_conformer,
  get_torsions,
  GetDihedral,
  create_log,
  append_line_to_log,
)

from rdkit.Geometry import Point3D


def SetDihedral(conf, atom_idx, new_vale):
    """
    Sets the value of a dihedral angle (torsion) in a molecule's conformation.

    This function modifies the dihedral angle defined by four atoms in the given molecule conformation to the specified value.

    Args:
        conf (RDKit Conformer): The conformation of the molecule.
        atom_idx (tuple): A tuple of four integers representing the indices of the atoms that define the dihedral angle.
        new_vale (float): The new value of the dihedral angle in degrees.

    """
    # rdMolTransforms.SetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)
    try:
        rdMolTransforms.SetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)
    except:
        rdMolTransforms.SetDihedralDeg(conf, atom_idx[1], atom_idx[0], atom_idx[2], atom_idx[3], new_vale)
        # print('--------- dihedral idxs were swapped ---------')

def apply_changes(mol, values, rotable_bonds):
    """
    Applies specified dihedral angle changes to a molecule based on the provided values for the dihedral angles.

    Args:
        mol (RDKit Mol): The original molecule to which the changes will be applied.
        values (list of float): A list of new values for the dihedral angles in degrees.
        rotable_bonds (list of tuple): A list of tuples, where each tuple contains four integers representing
                                       the indices of the atoms that define a rotatable bond.

    Returns:
        None: modifies inplace the input mol (RDKit Mol)
    """
    [SetDihedral(mol.GetConformer(), rotable_bonds[r], values[r].item()) for r in range(len(rotable_bonds))]

def nosify_mol(data):
    '''
    data: pyg data object
    return: pyg data object with new coords
    '''
    dihedral_noise_tau, coords_noise_tau = 2, 0.04
    mol = Chem.MolFromSmiles(str(data.smiles))
    mol = Chem.AddHs(mol)
    conf = get_conformer(mol,max_attempts=10)

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

def do_not_nosify_mol(data):
  data.noise_target = torch.zeros_like(data.pos, dtype=torch.float32)
  return data


def get_energy(mol):
  '''
  RDKit performs energy minimization of mol conformation using the Merck Molecular Force Field (MMFF94).
  kilocalories per mole (kcal/mol)  energies are reported in kcal mol-1.
  '''
  converged, energy = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=0)[0] # RETURNS: a list of (not_converged, energy) 2-tuples. If not_converged is 0 the optimization converged for that conformer.
  return energy


def minimize_energy(mol):
  '''
  RDKit performs energy minimization of mol conformation using the Merck Molecular Force Field (MMFF94).
  kilocalories per mole (kcal/mol)  energies are reported in kcal mol-1.
  mol is changed in place
  '''
  converged, energy = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=500)[0]
  return mol, energy

# class EnergyEvaluator():
#   def __init__(self):
#     self.rdkit_energies_not_minimized_log = create_log('./', name="rdkit_energies_not_minimized_log.txt")
#     self.rdkit_energies_minimized_log = create_log('./', name="rdkit_energies_minimized_log.txt")
#     self.energies_post_frad_log = create_log('./', name="energies_post_frad_log.txt")
#   def __call__(self,data) : pass

rdkit_energies_not_minimized_log = create_log('./', name="rdkit_energies_not_minimized_log.txt")
rdkit_energies_minimized_log = create_log('./', name="rdkit_energies_minimized_log.txt")
energies_post_frad_log = create_log('./', name="energies_post_frad_log.txt")

def nosify_mol_print_energies(data):
    '''
    data: pyg data object
    return: pyg data object with new coords
    '''
    dihedral_noise_tau, coords_noise_tau = 2, 0.04
    mol = Chem.MolFromSmiles(str(data.smiles))
    mol = Chem.AddHs(mol)
    conf = get_conformer(mol,max_attempts=10)

    # return data obj with noise equal to 0
    if conf == None: return do_not_nosify_mol(data)

    rdkit_conformer_energy = get_energy(mol)
    append_line_to_log(rdkit_energies_not_minimized_log, str(rdkit_conformer_energy))
    stable_energy = minimize_energy(mol)
    append_line_to_log(rdkit_energies_minimized_log, str(stable_energy))

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

    #! this is missing the coords noise, it consider only dihedrals
    post_frad = get_energy(mol)
    append_line_to_log(energies_post_frad_log, str(post_frad))

    return data



def optimize_coords(data):
  mol = Chem.AddHs(Chem.MolFromSmiles(str(data.smiles)))
  conf = get_conformer(mol,max_attempts=10)
  if conf == None: return data
  pre_opt_pos = conf.GetPositions()
  AllChem.MMFFOptimizeMoleculeConfs(
    mol,
    numThreads=1,
    # maxIters=500,
  )
  assert conf.GetPositions() != pre_opt_pos
  data.pos = torch.tensor(conf.GetPositions(), dtype=torch.float)
  return data



