import os
import shutil
import random
from source.utils.parser import MyArgPrsr
import multiprocessing as mp
from functools import partial
from source.utils.code_utils import TimeThis
from multiprocessing import Pool, cpu_count

from itertools import islice
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm

parser_entries = [
  {'identifiers': ["-c", "--copy"], 'type': bool, 'optional': True},
  {'identifiers': ["-m", "--move"], 'type': bool, 'optional': True},
  {'identifiers': ["-source", '--source'], 'type': str, 'help': 'Source path'},
  {'identifiers': ["-dest", '--dest'], 'type': str, 'help': 'Destination path'},
  {'identifiers': ["-n", '--n'], 'type': int, 'help': 'How many files to move/copy','optional': True, 'default': -1},
  {'identifiers': ["-s", '--seed'], 'type': int, 'help': 'seed', 'default': 42},
  {'identifiers': ["-p", '--processes'], 'type': int, 'help': 'num processes', 'default': 0},
]

# example:
# python source/utils/rand_cp_mv_npzs.py -c True -source /storage_common/nobilm/pretrain_paper/guacamol/kek_err -d /storage_common/nobilm/pretrain_paper/guacamol/testing_guac -n 50
# python source/utils/rand_cp_mv_npzs.py -c True -source /storage_common/nobilm/pretrain_paper/guacamol/all -d /storage_common/nobilm/pretrain_paper/guacamol/one_million -n 1000000
# python source/utils/rand_cp_mv_npzs.py -c True -source /storage_common/nobilm/pretrain_paper/guacamol/100k -d /storage_common/nobilm/pretrain_paper/guacamol/5k -n 5000 -p 80

# python source/utils/rand_cp_mv_npzs.py -c True -source /storage_common/nobilm/pretrain_paper/guacamol/all -d /storage_common/nobilm/pretrain_paper/guacamol/500k -n 500000

if __name__ == "__main__":
    args = MyArgPrsr(parser_entries)
    copy, move = args.copy, args.move
    source,dest=args.source,args.dest
    n=args.n
    random.seed(args.seed)

    assert copy or move, "Either --copy or --move must be specified"
    assert not (copy and move), "Only one of --copy or --move can be specified"

    # Get list of files in source
    files = os.listdir(source)
    # files = [f for f in os.listdir(source) if f.startswith("mol_") and f.endswith(".npz")]
    if n == -1:
      n = len(files)
    else:
      assert n > 0, "n must be greater then 0"
      if len(files) < n: raise ValueError(f"Not enough files in {source} to move {n} files")
      # Shuffle and randomly select n files without replacement
      random.shuffle(files)
      selected_files = random.sample(files, n)

    # Copy/Move selected files to dest
    f = shutil.copy if copy else shutil.move

    user_input = input(f"Executing: {f}. Continue? (y/n): ").strip().lower()
    if user_input in ['no', 'n']:
      print("Operation aborted.")
      exit(0)
    print("Starting...")

    def apply_f(src_dst_tuple):
      src, dst = src_dst_tuple
      f(src, dst)

    ## print('num processes:', args.processes)
    ## with TimeThis():
    ##   with mp.Pool(args.processes) as pool: # TODO: this is probably inefficient since single tast is too fast, do it batched
    ##     tasks = [(os.path.join(source, file), os.path.join(dest, file)) for file in selected_files]
    ##     results = list(pool.imap_unordered(apply_f, tasks))
    ##     # todo: check if cm already do the below
    ##     # pool.close()  # Prevents any more tasks from being submitted to the pool
    ##     # pool.join()   # Wait for the worker processes to terminate

    # def f_batch(_source, _dest, batch):
    #   for file in batch:
    #     f(os.path.join(_source, file), os.path.join(_dest, file))

    # f_batched = partial(f_batch, source, dest)

    # batch_size = 1000
    # batches = [selected_files[i:i + batch_size] for i in range(0, len(selected_files), batch_size)]
    # with Pool(processes=cpu_count()) as pool:
    #   pool.map(f_batched, batches)

    tasks = [(os.path.join(source, file), os.path.join(dest, file)) for file in selected_files]
    with ProcessPoolExecutor() as executor:
      list(tqdm(executor.map(apply_f, tasks, chunksize=1000), total=len(tasks)))


