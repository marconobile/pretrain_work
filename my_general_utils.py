import shutil
from typing import List, Union, Dict
import numpy as np
from os import listdir
from os.path import isfile, join
from types import SimpleNamespace
from pathlib import Path
import os
from rdkit import Chem
import warnings


#################
# GENERAL UTILS #
#################

def split_list(a_list, perc_train_data):
    assert (0.0 < perc_train_data <= 1.0)
    split = int(len(a_list) * perc_train_data)
    return a_list[:split], a_list[split:]

##########
# PARSER #
##########

class MyArgPrsr(object):
    r'''
    parser_entries = [{'identifiers': ["-p", '--path'], 'type': str, 'help': 'The path to the file'}]
    args = MyArgPrsr(parser_entries)
    path = args.path
    '''
    def __init__(self, parser_entries:List):
      from argparse import ArgumentParser
      parser = ArgumentParser()
      for el in parser_entries:
        if isinstance(el['identifiers'], list):
          for idntf in el['identifiers']: assert isinstance(idntf, str)
        elif isinstance(el['identifiers'], str): pass
        else: raise ValueError(f'identifiers not correct')
        assert isinstance(el['type'], type), f"type provided: {el['type']} is not a type"
        assert isinstance(el.get('help', ""), str), f"the help msg provided is not a str"
        if el.get('default', None): assert isinstance(el.get('default', None), el['type'])
        parser.add_argument(
            *el['identifiers'] if isinstance(el['identifiers'], list) else el['identifiers'], # if identifier are [-t, --tmp] then we can access it via self.args.tmp or self.args.t
            type=el['type'],           # data type used to interpret the inpt frm cmd line
            help=el.get('help', ''),
            default=el.get('default', None),
          )
      self.args = parser.parse_args()

    def __getattr__(self, arg: str): return getattr(self.args, arg)

#############
# DIR UTILS #
#############

def ls(dir): return [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]

def move_files_to_folder(dst_folder, files_to_move):
    out_filepaths = []
    for src_filepath in files_to_move:
        filename = os.path.basename(src_filepath)
        dst_filepath = join(dst_folder, filename)
        out_filepaths.append(dst_filepath)
        shutil.copy(src_filepath, dst_filepath)
    return out_filepaths


##################
# FILES HANDLING #
##################

def generate_file(path, filename):
    '''
    if path does not exist it is created
    if filename must have extension, default: '.txt'
    if file already exists it is overwritten

    args:
        - path directory where to save smiles list as txt
        - filename name of the file. By default it is created a txt file
    '''
    os.makedirs(path, exist_ok=True)
    path_to_file = os.path.join(path, filename)
    filename_ext = os.path.splitext(path_to_file)[-1].lower()
    if not filename_ext:
        path_to_file += '.txt'
    if os.path.isfile(path_to_file):
        try:
            os.remove(path_to_file)
        except OSError:
            raise f"{path_to_file} already existing and could not be removed"
    return path_to_file

def create_log(path, name="log.txt"):
    if not name.endswith(".txt"):
        name += ".txt"
    generate_file(path, name)
    return os.path.join(path, name)


def append_line_to_log(path_to_log, line):
    with open(path_to_log, "a") as log:
        log.write(line + "\n")


###############
# RDKIT UTILS #
###############

def get_field_from_npzs(path:str, field:Union[str, List]):
  if field == '*': return [np.load(npz) for npz in ls(path)]

  if isinstance(field, str): return [np.load(el)[field].item() for el in ls(path)]
  if not isinstance(field, List): raise ValueError(f'Unaccepted type for field, which is {type(field)}, but should be List or str ')

  out = []
  for npz in ls(path):
    data = np.load(npz)
    # data = data.files
    sn  = SimpleNamespace()
    for f in field: sn.__setattr__(f, data[f].item())
    sn.__setattr__("path", path)
    out.append(sn)
  return out

def preprocess_mol(m:Chem.Mol,
                  sanitize:bool=True,
                  addHs:bool=True,
                  drop_disconnected_components:bool=True
                ):
  if m == None: return None
  try:
    if addHs: m = Chem.AddHs(m)
    if drop_disconnected_components: m = max(Chem.GetMolFrags(m, asMols=True), key=lambda frag: frag.GetNumAtoms())
    if sanitize:
      error = Chem.SanitizeMol(m)
      if error: return None
  except: return None
  return m

def atom2int_int2Atom(mols) -> Dict:
    r'''returns dict[k]:v := atomSymbol:id'''
    atom_types = set('H')
    for mol in mols:
      for atom in Chem.RemoveHs(mol).GetAtoms(): # RemoveHs returns mol without Hs, input m is *not modifed*
          atom_types.add(atom.GetSymbol())
    atom_types = list(atom_types)
    atom_types.sort()
    return {atom_type: i for i, atom_type in enumerate(atom_types)}, {i:atom_type for i, atom_type in enumerate(atom_types)}

################
# FRAD related #
################

from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Chem.rdchem import HybridizationType
import torch

try:
  from geqtrain.utils.torch_geometric import Data
except ImportError:
  pass
  # from torch_geometric.data import Data
  # warnings.warn("Warning: using torch_geometric lib instead of geqtrain.utils.torch_geometric")

def get_conformer(mol, max_attempts:int=10):
    # TODO: create multithread version of this via https://www.rdkit.org/docs/source/rdkit.Chem.rdDistGeom.html#rdkit.Chem.rdDistGeom.EmbedMultipleConfs
    try:
        out = mol.GetConformer() # if fails mol needs to be embedded
        return out if out != -1 else None
    except:
        try:
            success = AllChem.EmbedMolecule(mol,
                                            useRandomCoords=True, # needs to be T
                                            useSmallRingTorsions=True,
                                            useMacrocycleTorsions=True,
                                            maxAttempts=max_attempts
                                          ) != -1
            if success:
                out = mol.GetConformer() # if mol embedding worked should be ok
                return out if out != -1 else None
        except: return None
    return None

def get_torsions(mol_list):
    """
    Extracts the torsion angles (dihedrals) from a list of molecules.

    This function identifies all the torsion angles in the given list of molecules and returns a list of these torsions.
    A torsion angle is defined by four atoms and is calculated based on the connectivity of these atoms in the molecule.

    Args:
        mol_list (list): A list of RDKit molecule objects.

    Returns:
        list: A list of tuples, where each tuple contains four integers representing the indices of the atoms
              that define a torsion angle in the molecule.

    """
    atom_counter = 0
    torsionList = []
    for m in mol_list:
        torsionSmarts = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
        torsionQuery = Chem.MolFromSmarts(torsionSmarts)
        matches = m.GetSubstructMatches(torsionQuery)
        for match in matches:
            idx2 = match[0]
            idx3 = match[1]
            bond = m.GetBondBetweenAtoms(idx2, idx3)
            jAtom = m.GetAtomWithIdx(idx2)
            kAtom = m.GetAtomWithIdx(idx3)
            for b1 in jAtom.GetBonds():
                if (b1.GetIdx() == bond.GetIdx()):
                    continue
                idx1 = b1.GetOtherAtomIdx(idx2)
                for b2 in kAtom.GetBonds():
                    if ((b2.GetIdx() == bond.GetIdx())
                            or (b2.GetIdx() == b1.GetIdx())):
                        continue
                    idx4 = b2.GetOtherAtomIdx(idx3)
                    # skip 3-membered rings
                    if (idx4 == idx1):
                        continue
                    # skip torsions that include hydrogens
                    #                     if ((m.GetAtomWithIdx(idx1).GetAtomicNum() == 1)
                    #                         or (m.GetAtomWithIdx(idx4).GetAtomicNum() == 1)):
                    #                         continue
                    if m.GetAtomWithIdx(idx4).IsInRing():
                        torsionList.append(
                            (idx4 + atom_counter, idx3 + atom_counter, idx2 + atom_counter, idx1 + atom_counter))
                        break
                    else:
                        torsionList.append(
                            (idx1 + atom_counter, idx2 + atom_counter, idx3 + atom_counter, idx4 + atom_counter))
                        break
                break

        atom_counter += m.GetNumAtoms()
    return torsionList

def GetDihedral(conf, atom_idx):
    """
    Retrieves the value of a dihedral angle (torsion) in a molecule's conformation.

    This function returns the current value of the dihedral angle defined by four atoms in the given molecule conformation.

    Args:
        conf (RDKit Conformer): The conformation of the molecule.
        atom_idx (tuple): A tuple of four integers representing the indices of the atoms that define the dihedral angle.

    Returns:
        float: The value of the dihedral angle in degrees.

    """
    return rdMolTransforms.GetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])

def mol2pyg(mol, types):
    '''
    either returns data pyg data obj or None if some operations are not possible
    IMPO: this does not set y
    '''
    conf = get_conformer(mol)
    if conf == None: return None

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

def save_npz(pyg_mols, f, folder_name=None, N=None, check=True):
    '''
    f: lambda function that defines how to treat the target value, examples:

        graph_labels = g['y'].numpy() # (1, N) # nb this is already unsqueezed
        then f is:
        f = lambda y: y.numpy()

        graph_labels = g['y'].reshape(1,1).numpy() # (1, N) # nb this is already unsqueezed
        then f is:
        f = lambda y: y.reshape(1,1).numpy()

    example of expected output shapes:
    coords
    (1, 5, 3)
    atom_types
    (5,)
    edge_index
    (1, 2, 20)
    edge_attr
    (1, 20, 5)
    graph_labels
    (1, 19)
    hybridization
    (5,)
    chirality
    (5,)
    is_aromatic
    (5,)
    is_in_ring
    (5,)
    '''
    N = N or len(pyg_mols)
    for idx in range(N):
        g = pyg_mols[idx]
        file = f'{folder_name}/mol_{idx}'

        coords = g['pos'].unsqueeze(0).numpy()  # (1, N, 3)
        # in general: if fixed field it must be (N,), else (1, N)
        atom_types = g['z'].numpy()
        # edge_index = g['edge_index'].unsqueeze(0).numpy() # (1, 2, E)
        # edge_attr = g['edge_attr'].unsqueeze(0).numpy() # (1, E, Edg_attr_dims)
        hybridization = g['hybridization'].numpy()  # (N, )
        chirality = g['chirality'].numpy()  # (N, )
        is_aromatic = g['is_aromatic'].numpy()  # (N, )
        is_in_ring = g['is_in_ring'].numpy()  # (N, )
        smiles = g['smiles']

        graph_labels = f(g['y'])  # (1, N) # nb this is already unsqueezed
        if check:  # this works iif all are fixed fields
            # coords
            # eg shape: (1, 66, 3)
            assert len(coords.shape) == 3
            B, N, D = coords.shape
            assert B == 1
            assert D == 3

            # atom_types
            # eg shape: (66,)
            assert len(atom_types.shape) == 1
            assert len(hybridization.shape) == 1
            assert len(chirality.shape) == 1
            assert len(is_aromatic.shape) == 1
            assert len(is_in_ring.shape) == 1

            assert atom_types.shape[0] == N
            assert hybridization.shape[0] == N
            assert chirality.shape[0] == N
            assert is_aromatic.shape[0] == N
            assert is_in_ring.shape[0] == N

            rotable_bonds = g['rotable_bonds'].numpy()
            dihedral_angles_degrees = g['dihedral_angles_degrees'].numpy()

            # graph_labels
            # eg shape: (1, 1)
            # assert len(graph_labels.shape) == 2
            # assert graph_labels.shape[0] == 1

        np.savez(
            file,
            coords=coords,
            atom_types=atom_types,
            # edge_index=edge_index, # if provided it must have a batch dimension
            # edge_attr=edge_attr,
            graph_labels=graph_labels,
            hybridization=hybridization,
            chirality=chirality,
            is_aromatic=is_aromatic,
            is_in_ring=is_in_ring,
            smiles=smiles,
            rotable_bonds=rotable_bonds,
            dihedral_angles_degrees=dihedral_angles_degrees,
        )