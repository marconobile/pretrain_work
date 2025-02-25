import os
import numpy as np
from .file_handling_utils import ls, silentremove
from typing import List, Union, Tuple
import numpy as np
from types import SimpleNamespace
from .data_splitting_utils import create_data_folders
from pathlib import Path
from numpy.lib.npyio import NpzFile


def get_field_from_npzs(path:str, field:Union[str, List]='*') -> List[NpzFile]:
  '''
  example usage:
  l = get_field_from_npzs(p)
  l[0][k] -> access to content
  '''
  is_single_npz = lambda path: os.path.splitext(path)[1].lower() == ".npz"
  npz_files = [path] if is_single_npz(path) else ls(path)
  if field == '*':
    return [np.load(npz) for npz in npz_files]
  possible_keys = (k for k in np.load(npz_files[0]).keys())
  if field not in possible_keys:
    raise ValueError(f'{field} not in {list(possible_keys)}')
  if isinstance(field, str):
    return [np.load(el)[field].item() for el in npz_files]
  if not isinstance(field, List):
    raise ValueError(f'Unaccepted type for field, which is {type(field)}, but should be List or str ')

  out = []
  for npz in ls(path):
    data = np.load(npz)
    sn  = SimpleNamespace()
    for f in field:
        sn.__setattr__(f, data[f].item())
    sn.__setattr__("path", path)
    out.append(sn)
  return out


def save_npz(pyg_mols, folder_name:str=None, N:int=None, idx:int=0):
    '''
    pyg_mols: list of pytorch geometric data objects
    f: func to apply to each g['y'], more below
    folder_name: name of the npz
    N how many npz to write out of the pyg_mols
    check: whether to check or not for shapes when saving the npz
    idx: int from where to start to save in f'{folder_name}/mol_{idx}'

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

    all_dir, _, _, _ = create_data_folders(folder_name)
    print('Saving npzs in: ', all_dir)
    if N:
        pyg_mols = pyg_mols[:N]

    for pyg_m in pyg_mols:
        save_pyg_as_npz(pyg_m, f'{all_dir}/mol_{idx}')
        idx +=1

    return idx


def save_pyg_as_npz(g, filepath:str|Path):

    # if fixed field it must be (N,), else (1, N)
    g['coords'] = g['pos'] if g['pos'].dim() == 3 else g['pos'].unsqueeze(0)  # (1, N, 3)
    g['graph_labels'] = g['y']

    # edge_index = g['edge_index'].unsqueeze(0) # (1, 2, E)
    # edge_attr = g['edge_attr'].unsqueeze(0) # (1, E, Edg_attr_dims)

    # Filter out the None values and downcast
    filtered_data = {}
    for k, v in g:
        if v is not None:
            try: v = v.numpy()
            except: pass
            filtered_data[k] = v

    np.savez(file=filepath, **filtered_data)
    test_npz_validity(filepath+".npz")


def get_smiles_and_filepaths_from_valid_npz(npz_dir:str|Path) -> Tuple[str, Path]:
    smiles, filepaths = [], []
    for npz_file in ls(npz_dir):
        if not test_npz_validity(npz_file):
            continue
        npz = get_field_from_npzs(npz_file)
        filepaths.append(Path(str(npz[0].zip.filename)))
        smiles.append(str(npz[0]['smiles']))
    assert len(smiles) == len(filepaths)
    return smiles, filepaths


def test_npz_validity(file):
    try:
        np.load(file, allow_pickle=True)
        return True
    except Exception as e:
        print(f'Error loading {file}: {e}. Removed {file}')
        silentremove(file)
        return False