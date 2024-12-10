
import numpy as np
import warnings
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
import torch
from source.utils.rdkit_mol_utils import get_torsions, GetDihedral
try:
  from geqtrain.utils.torch_geometric import Data
except ImportError:
  from torch_geometric.data import Data
  warnings.warn("Warning: using torch_geometric lib instead of geqtrain.utils.torch_geometric")


def mols2pyg_list(mols, ys, atom2int):
  pyg_mols = []
  for m, y in zip(mols, ys):
    pyg_m = mol2pyg(m, atom2int)
    if pyg_m == None: continue
    pyg_m.y = np.array(y, dtype=np.float32)
    pyg_mols.append(pyg_m)
  return pyg_mols


def mol2pyg(mol, types, minimize:bool=True):
    '''
    either returns data pyg data obj or None if some operations are not possible
    IMPO: this does not set y
    '''
    conf = mol.GetConformer() # get_conformer(mol)
    if conf == None: return None

    # if minimize: mol = optimize_coords(mol, conf)

    type_idx, aromatic, is_in_ring, _hybridization, chirality = [], [], [], [], []
    for atom in mol.GetAtoms():

      type_idx.append(types[atom.GetSymbol()])
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
    return Data(
      z=torch.tensor(type_idx),
      pos=torch.tensor(conf.GetPositions(), dtype=torch.float32),
      smiles=Chem.MolToSmiles(mol),
      hybridization=torch.tensor(_hybridization, dtype=torch.long),
      is_aromatic=torch.tensor(aromatic, dtype=torch.long),
      is_in_ring=torch.tensor(is_in_ring, dtype=torch.long),
      chirality=torch.tensor(chirality, dtype=torch.long),
      rotable_bonds=torch.tensor(rotable_bonds, dtype=torch.long),
      dihedral_angles_degrees=torch.tensor(
          [GetDihedral(conf, rot_bond) for rot_bond in rotable_bonds], dtype=torch.float32),
    )