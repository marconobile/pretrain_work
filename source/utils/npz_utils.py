import torch
import numpy as np
from .file_handling_utils import ls
from typing import List, Union
import numpy as np
from types import SimpleNamespace


def get_field_from_npzs(path:str, field:Union[str, List]='*'):
  '''
  example usage:
  l = get_field_from_npzs(p)
  l[0][k] -> access to content
  '''
  npz_files = ls(path)
  if field == '*': return [np.load(npz) for npz in npz_files]
  possible_keys = [k for k in np.load(npz_files[0]).keys()]
  if field not in possible_keys: raise ValueError(f'{field} not in {possible_keys}')
  if isinstance(field, str): return [np.load(el)[field].item() for el in npz_files]
  if not isinstance(field, List): raise ValueError(f'Unaccepted type for field, which is {type(field)}, but should be List or str ')

  out = []
  for npz in ls(path):
    data = np.load(npz)
    sn  = SimpleNamespace()
    for f in field: sn.__setattr__(f, data[f].item())
    sn.__setattr__("path", path)
    out.append(sn)
  return out


def save_npz(pyg_mols, f, folder_name=None, N=None, check=True):
    '''
    no further processing of folder
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
    for idx in range(N): pyg2npz(pyg_mols[idx], f'{folder_name}/mol_{idx}', f)


def pyg2npz(g, file, f,  check:bool=True):
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

  graph_labels = f(g['y']) # (1, N) # nb this is already unsqueezed
  if isinstance(graph_labels, torch.Tensor): graph_labels = graph_labels.numpy()
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

