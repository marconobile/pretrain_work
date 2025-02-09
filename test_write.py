from rdkit import Chem as rdChem
import torch
from source.utils.code_utils import TimeThis
from source.utils.conforge_conformer_generation import generate_conformers, get_conformer_generator
from source.utils.mol2pyg import mols2pyg_list_with_targets
from source.utils.mol_utils import get_energy, drop_disconnected_components
from source.utils.data_splitting_utils import create_data_folders, parse_smiles_file
from source.utils.npz_utils import save_pyg_as_npz

import multiprocessing as mp
from functools import partial
from multiprocessing import Queue

# global in this file
q = Queue() # queue to store failed molecules
num_threads = 20

def worker(smiles_target, save_dir, lock, shared_i_save):
  s, y = smiles_target
  s = drop_disconnected_components(s)
  conformers = generate_conformers(s, get_conformer_generator())
  if not conformers:
    q.put(smiles_target)
    return

  try:
    mol2pyg_kwargs = {"max_energy": max((get_energy(m) for m in conformers))}
  except rdChem.KekulizeException as e:
    print(f"KekulizeException: {e} for {s}")
    return

  pyg_mol_fixed_fields = mols2pyg_list_with_targets([conformers[0]], [s], [y], **mol2pyg_kwargs)[0]

  batched_pos = []
  for mol in conformers:
    pos = []
    conf = mol.GetConformer()
    for i, atom in enumerate(mol.GetAtoms()):
      positions = conf.GetAtomPosition(i)
      pos.append((positions.x, positions.y, positions.z))
    pos = torch.tensor(pos, dtype=torch.float32)
    batched_pos.append(pos)

  batched_pos = torch.stack(batched_pos)
  pyg_mol_fixed_fields.pos = batched_pos

  with lock:
    i_save = shared_i_save.value
    save_pyg_as_npz(pyg_mol_fixed_fields, f'{save_dir}/mol_{i_save}')
    shared_i_save.value += 1
    print(f"Saved file: {save_dir}/mol_{i_save}")


if __name__ == '__main__':
    input_data = '/home/nobilm@usi.ch/pretrain_paper/data/guacamol_v1_all.smiles' # test_guac_smiles
    # smiles, targets = get_smiles_and_targets_from_csv(input_data)
    smiles = parse_smiles_file(input_data)
    targets = [0] * len(smiles) # dummy target to match API

    # create dir
    dir = '/storage_common/nobilm/pretrain_paper/guacamol' #'/storage_common/nobilm/pretrain_paper/second_test'
    all_dir, train_dir, val_dir, test_dir = create_data_folders(dir)

    manager = mp.Manager()
    lock = manager.Lock()
    shared_i_save = manager.Value('i', 0)
    worker_ = partial(worker, save_dir=all_dir, lock=lock, shared_i_save=shared_i_save)

    with TimeThis():
        with mp.Pool(num_threads) as pool:
            tasks = [(s, y) for s, y in zip(smiles, targets)]
            results = list(pool.imap_unordered(worker_, tasks))
            # todo: check if cm already do the below
            # pool.close()  # Prevents any more tasks from being submitted to the pool
            # pool.join()   # Wait for the worker processes to terminate

    # process the failed molecules with rdkit
    print(f'Mols discarded {q.qsize()} out of {len(smiles)}')
    # while q.qsize() > 0: worker_(q.get())

