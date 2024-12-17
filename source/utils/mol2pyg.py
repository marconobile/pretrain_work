import torch
import numpy as np
import warnings
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from source.utils.atom_encoding import periodic_table_group, periodic_table_period
from source.utils.mol_utils import smi_writer_params
from source.data_transforms.utils import get_torsions, GetDihedral
try:
  from geqtrain.utils.torch_geometric import Data
except ImportError:
  from torch_geometric.data import Data
  warnings.warn("Warning: using torch_geometric lib instead of geqtrain.utils.torch_geometric")


def mols2pyg_list_with_targets(mols, smiles, ys, **mol2pyg_kwargs):
  pyg_mols = []
  for m, s, y in zip(mols, smiles, ys):
    pyg_m = mol2pyg(m, s, **mol2pyg_kwargs)
    if pyg_m == None: raise ValueError('Error in casting mol to pyg') # continue
    pyg_m.y = np.array(y, dtype=np.float32)
    pyg_mols.append(pyg_m)
  return pyg_mols


def mol2pyg(mol, smi, max_energy:float=0.0):
    '''
    IMPO: do not trust smiles: given smi -> get MOL -> addHs -> do work. Then for any other place where you need to act on MOL, restart from input smi and repeat smi -> get MOL -> addHs -> do work
    IMPO: this does not set y

    returns: pyg data obj or None if some operations are not possible
    '''
    conf = mol.GetConformer() # ToDo: ok iff sensible conformer has been already generated and setted
    aromatic, is_in_ring, _hybridization, chirality = [], [], [], []
    pos,group, period = [], [], []
    num_atoms = mol.GetNumAtoms()
    adj_matrix = np.zeros((num_atoms, num_atoms), dtype=int)
    for i, atom in enumerate(mol.GetAtoms()):
      assert atom.GetIdx() == i

      for neighbor in atom.GetNeighbors():
        neighbor_idx = neighbor.GetIdx()
        adj_matrix[i, neighbor_idx] = 1
        adj_matrix[neighbor_idx, i] = 1

      group.append(periodic_table_group(atom))
      period.append(periodic_table_period(atom))

      positions = conf.GetAtomPosition(i)
      pos.append((positions.x, positions.y, positions.z))

      aromatic.append(1 if atom.GetIsAromatic() else 0)
      is_in_ring.append(1 if atom.IsInRing() else 0)
      # https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.ChiralType
      chirality.append(atom.GetChiralTag())

      hybridization = atom.GetHybridization()
      hybridization_value = 0
      if hybridization == HybridizationType.SP: hybridization_value = 1
      elif hybridization == HybridizationType.SP2: hybridization_value = 2
      elif hybridization == HybridizationType.SP3: hybridization_value = 3
      _hybridization.append(hybridization_value)

    rotable_bonds = get_torsions([mol])
    # TODO pass args as dict, where the dict is built as: k:v if v!=None, is there a better refactoring? Builder pattern?
    return Data(
      adj_matrix=torch.tensor(adj_matrix),
      group=torch.tensor(group),
      period=torch.tensor(period),
      pos=torch.tensor(pos, dtype=torch.float32),
      smiles=smi, #Chem.MolToSmiles(mol, smi_writer_params()),
      hybridization=torch.tensor(_hybridization, dtype=torch.long),
      is_aromatic=torch.tensor(aromatic, dtype=torch.long),
      is_in_ring=torch.tensor(is_in_ring, dtype=torch.long),
      chirality=torch.tensor(chirality, dtype=torch.long),
      rotable_bonds=torch.tensor(rotable_bonds, dtype=torch.long),
      dihedral_angles_degrees=torch.tensor(
          [GetDihedral(conf, rot_bond) for rot_bond in rotable_bonds], dtype=torch.float32),
      max_energy=max_energy, # max energy across all sampled conformers of input mol
    )