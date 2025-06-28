import torch
import numpy as np
import warnings
from rdkit import Chem as rdChem
from source.utils.mol_utils import get_molecule_fragments, smi_reader_params, set_coords, get_rdkit_conformer
from source.data_transforms.utils import get_torsions, GetDihedral
from rdkit.Chem import Descriptors
try:
    from geqtrain.utils.torch_geometric import Data
except ImportError:
    from torch_geometric.data import Data
    warnings.warn("Warning: using torch_geometric lib instead of geqtrain.utils.torch_geometric")

from source.utils.data_utils.featurizer import atom_to_feature_vector, possible_atomic_properties, possible_bond_properties, bond_to_feature_vector, allowable_features
from collections import defaultdict
from einops import repeat




def mols2pyg_list(mols:list, smiles:list, **mol2pyg_kwargs)-> list:
    pyg_mols = []
    for m, s in zip(mols, smiles):
        pyg_m = mol2pyg(m, s, **mol2pyg_kwargs)
        if pyg_m == None:
            raise ValueError('Error in casting mol to pyg') # continue
        pyg_mols.append(pyg_m)
    return pyg_mols


def mols2pyg_list_with_targets(mols, smiles, ys, **mol2pyg_kwargs):
    pyg_mols = []
    for m, s, y in zip(mols, smiles, ys):
        pyg_m = mol2pyg(m, s, **mol2pyg_kwargs)
        if pyg_m == None:
           raise ValueError('Error in casting mol to pyg') # continue
        pyg_m.y = np.array(y, dtype=np.float32)
        pyg_mols.append(pyg_m)
    return pyg_mols


# def mol2pyg(mol:rdChem.Mol, conf=None, use_rdkit_3d:bool=False, **kwargs) -> Data:
#     '''
#     IMPO: do not trust smiles: given smi -> get MOL -> addHs -> do work. Then for any other place where you need to act on MOL, restart from input smi and repeat smi -> get MOL -> addHs -> do work
#     IMPO: this does not set y
#     returns:
#         pyg data obj or None if some operations are not possible
#     '''
#     if mol is None:
#         print("Molecule is none")
#         return mol

#     mol = rdChem.RemoveHs(mol)
#     # assert all(atom.GetAtomicNum() != 1 for atom in mol.GetAtoms()), "Molecule contains hydrogen atoms"

#     if not use_rdkit_3d and mol.GetNumConformers() == 0:
#         raise ValueError("Molecule has no conformers")

#     if conf is None:
#         conf = get_rdkit_conformer(mol)
#     if conf is None:
#         print("Could not get rdkit Conformer obj for molecule")
#         return None
#     assert mol.GetNumConformers() != 0

#     num_atoms = mol.GetNumAtoms()
#     rotable_bonds = get_torsions(mol)
#     dihedral_angles_degrees = [GetDihedral(conf, rot_bond) for rot_bond in rotable_bonds]

#     adj_matrix = np.zeros((num_atoms, num_atoms), dtype=np.uint8)
#     atoms_features = defaultdict(list) # value[idx] -> feat for atom at idx
#     pos = []

#     rows, cols = [], []
#     bonds_features = defaultdict(list)
#     covalent_bonds = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()]

#     for i, atom in enumerate(mol.GetAtoms()):
#         assert atom.GetIdx() == i

#         # fill adj matrix
#         for neighbor in atom.GetNeighbors():
#             neighbor_idx = neighbor.GetIdx()
#             adj_matrix[i, neighbor_idx] = 1
#             adj_matrix[neighbor_idx, i] = 1

#         # coords, done as below to get 3d out of conf
#         positions = conf.GetAtomPosition(i)
#         pos.append((positions.x, positions.y, positions.z))

#         atom_feature = atom_to_feature_vector(atom)
#         for atomic_property in possible_atomic_properties:
#             atoms_features[atomic_property].append(atom_feature.get(atomic_property))

#         # build fully connected edge_index
#         for j in range(i + 1, num_atoms):
#             # if [i, j] not in edge_index and [j, i] not in edge_index:
#             rows += [i, j]
#             cols += [j, i]
#             is_a_covalent_bond = [i, j] in covalent_bonds or [j, i] in covalent_bonds
#             bond = mol.GetBondBetweenAtoms(i, j) if is_a_covalent_bond else None # GetBondBetweenAtoms returns same bond of ij and ji
#             bonds_feature = bond_to_feature_vector(bond)
#             for bond_property in possible_bond_properties:
#                 bonds_features[bond_property] += 2 * [bonds_feature.get(bond_property)] # can handle absent bond

#     # do permutation to sort wrt source idx as, eg:
#     # from:tensor([[0, 0, 0, 0, 1, 2, 3, 4],
#     #              [1, 2, 3, 4, 0, 0, 0, 0]])
#     # to:  tensor([[0, 0, 0, 0, 1, 2, 3, 4],
#     #              [1, 2, 3, 4, 0, 0, 0, 0]])
#     edge_index = torch.tensor([rows, cols], dtype=torch.long) # [2, E] each bidirectiona edge
#     perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()
#     edge_index = edge_index[:, perm]
#     bonds_features = {k:torch.tensor(v, dtype=torch.short)[perm] for k,v in bonds_features.items()} # k:tensor of shape (E)
#     assert edge_index.shape[-1] ==  num_atoms*(num_atoms-1)

#     fragmentdsIds_present, count_frags = get_molecule_fragments(mol) # binarized = presence, count = cumulative count
#     molecular_properties = {
#         "h_donors"    : Descriptors.NumHDonors(mol),
#         "h_acceptors" : Descriptors.NumHAcceptors(mol),
#         "count_frags" : count_frags,
#     }

#     # {"safe_count" : int}
#     # {"homo_lumo_gap" : float}
#     for k,v in kwargs.items():
#         molecular_properties[k]=float(v)

#     pos=torch.tensor(pos, dtype=torch.float32)
#     if pos.dim() == 2:
#         pos = pos.unsqueeze(0)

#     if edge_index.dim() == 2:
#         edge_index = edge_index.unsqueeze(0) # (2, E) -> (1, 2, E)
#         if pos.shape[0]>1:
#             edge_index = repeat(edge_index, 'b e d -> (repeat b) e d', repeat=pos.shape[0]) # 1 edge index for each conf

#     return Data(
#         adj_matrix=torch.tensor(adj_matrix, dtype=torch.short),
#         pos=pos,
#         rotable_bonds=torch.tensor(rotable_bonds, dtype=torch.short),
#         fragmentdsIds_present=torch.tensor(fragmentdsIds_present, dtype=torch.float32), # needed to be float
#         dihedral_angles_degrees=torch.tensor(dihedral_angles_degrees, dtype=torch.float32),
#         **{k: torch.tensor(v, dtype=torch.short) for k, v in atoms_features.items()},
#         edge_index=edge_index,
#         **bonds_features,
#         smiles=rdChem.MolToSmiles(mol, canonical=True),
#         **molecular_properties,
#     )

from source.scripts.fg_featurizer import FGFeaturizer
from rdkit import Chem as rdChem
from rdkit.Chem import rdMolTransforms

def mol2pyg(mol:rdChem.Mol, fg_featurizer:FGFeaturizer, **kwargs) -> Data:
    mol = rdChem.RemoveHs(mol)
    assert mol.GetNumConformers() == 1, "Only one conformer is supported"
    conf = rdChem.Conformer(mol.GetConformer())
    rdMolTransforms.CanonicalizeConformer(conf)

    num_atoms = mol.GetNumAtoms()
    rotable_bonds = get_torsions(mol)
    dihedral_angles_degrees = [GetDihedral(conf, rot_bond) for rot_bond in rotable_bonds]

    adj_matrix = np.zeros((num_atoms, num_atoms), dtype=np.uint8)
    atoms_features = defaultdict(list) # value[idx] -> feat for atom at idx
    pos = []

    rows, cols = [], []
    bonds_features = defaultdict(list)
    covalent_bonds = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()]

    if fg_featurizer is not None:
        fg_featurizer.process_mol(mol)
        fg_ids = []

    for i, atom in enumerate(mol.GetAtoms()):
        assert atom.GetIdx() == i

        # fill adj matrix
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            adj_matrix[i, neighbor_idx] = 1
            adj_matrix[neighbor_idx, i] = 1

        # coords, done as below to get 3d out of conf
        positions = conf.GetAtomPosition(i)
        pos.append((positions.x, positions.y, positions.z))

        atom_feature = atom_to_feature_vector(atom)
        for atomic_property in possible_atomic_properties:
            atoms_features[atomic_property].append(atom_feature.get(atomic_property))

        if fg_featurizer is not None:
            fg_ids.append(fg_featurizer.fg_per_atom(i))

        # build fully connected edge_index
        for j in range(i + 1, num_atoms):
            # if [i, j] not in edge_index and [j, i] not in edge_index:
            rows += [i, j]
            cols += [j, i]
            is_a_covalent_bond = [i, j] in covalent_bonds or [j, i] in covalent_bonds
            bond = mol.GetBondBetweenAtoms(i, j) if is_a_covalent_bond else None # GetBondBetweenAtoms returns same bond of ij and ji
            bonds_feature = bond_to_feature_vector(bond)
            for bond_property in possible_bond_properties:
                bonds_features[bond_property] += 2 * [bonds_feature.get(bond_property)] # can handle absent bond

    # do permutation to sort wrt source idx as, eg:
    # from:tensor([[0, 0, 0, 0, 1, 2, 3, 4],
    #              [1, 2, 3, 4, 0, 0, 0, 0]])
    # to:  tensor([[0, 0, 0, 0, 1, 2, 3, 4],
    #              [1, 2, 3, 4, 0, 0, 0, 0]])
    edge_index = torch.tensor([rows, cols], dtype=torch.long) # [2, E] each bidirectiona edge
    perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    bonds_features = {k:torch.tensor(v, dtype=torch.short)[perm] for k,v in bonds_features.items()} # k:tensor of shape (E)
    assert edge_index.shape[-1] ==  num_atoms*(num_atoms-1)

    fragmentdsIds_present, count_frags = get_molecule_fragments(mol) # binarized = presence, count = cumulative count
    molecular_properties = {
        "h_donors"    : Descriptors.NumHDonors(mol),
        "h_acceptors" : Descriptors.NumHAcceptors(mol),
        "count_frags" : count_frags,
    }

    # {"safe_count" : int}
    # {"homo_lumo_gap" : float}
    for k,v in kwargs.items():
        molecular_properties[k]=float(v)

    # add frame shape
    pos=torch.tensor(pos, dtype=torch.float32)
    if pos.dim() == 2:
        pos = pos.unsqueeze(0)

    if edge_index.dim() == 2:
        edge_index = edge_index.unsqueeze(0) # (2, E) -> (1, 2, E)
        if pos.shape[0]>1:
            edge_index = repeat(edge_index, 'b e d -> (repeat b) e d', repeat=pos.shape[0]) # 1 edge index for each conf

    fg_ids=torch.tensor(np.array(fg_ids), dtype=torch.float32).unsqueeze(0)

    return Data(
        adj_matrix=torch.tensor(adj_matrix, dtype=torch.short),
        pos=pos,
        rotable_bonds=torch.tensor(rotable_bonds, dtype=torch.short),
        fragmentdsIds_present=torch.tensor(fragmentdsIds_present, dtype=torch.float32), # needed to be float
        dihedral_angles_degrees=torch.tensor(dihedral_angles_degrees, dtype=torch.float32),
        **{k: torch.tensor(v, dtype=torch.short) for k, v in atoms_features.items()},
        fg_ids=fg_ids,
        edge_index=edge_index,
        **bonds_features,
        smiles=rdChem.MolToSmiles(mol, canonical=True),
        **molecular_properties,
    )
