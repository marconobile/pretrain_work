import os
import shutil
import random
# import multiprocessing as mp
from functools import partial
from source.utils.parser import MyArgPrsr
from source.utils.code_utils import TimeThis
from source.utils.file_handling_utils import silentremove
# from tdqm import tdqm
from multiprocessing import Pool, cpu_count
from functools import partial


parser_entries = [
    {'identifiers': ["-p", '--path'], 'type': str,
        'help': 'Path from where to rm'},
    {'identifiers': ["-n", '--n'], 'type': int,
        'help': 'How many files to rm'},
    {'identifiers': ["-s", '--seed'], 'type': int,
        'help': 'seed', 'default': 42},
    # TODO add filter for ext
]

# example:
# python source/utils/rand_rm.py -p /storage_common/nobilm/pretrain_paper/guacamol/kek_err -n 50
# /storage_common/nobilm/pretrain_paper/guacamol/5k
if __name__ == "__main__":
    args = MyArgPrsr(parser_entries)
    path, n = args.path, args.n
    random.seed(args.seed)

    # Get list of files in path
    # if f.startswith("mol_") and f.endswith(".npz")]
    files = [f for f in os.listdir(path)]
    assert n < len(files), f"n must be less than the number of files in {path}"

    random.shuffle(files)
    selected_files = random.sample(files, n)

    def delete_batch(_path, batch):
        for file in batch:
            silentremove(os.path.join(_path, file))

    f = partial(delete_batch, path)

    batch_size = 100
    batches = [selected_files[i:i + batch_size]
               for i in range(0, len(selected_files), batch_size)]

    with Pool(processes=10) as pool:
        pool.map(f, batches)
