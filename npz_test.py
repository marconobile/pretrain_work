import numpy as np
from source.utils.file_handling_utils import ls, silentremove
from itertools import islice
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from source.utils.code_utils import TimeThis
from tqdm import tqdm
# ProcessPool is for CPU-bound tasks so you can benefit from multiple CPU.
# ThreadPoolExecutor is for IO-bound tasks so you can benefit from IO-wait.
# If you have many short-lived CPU-bound tasks use threads due to the overhead of creating and managing multiple processes,
# as the cost of creating a process can be significant compared to the amount of time spent actually executing the task


# https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor.map

# def batch(iterable, size):
#     """Yield successive batches of size 'size' from 'iterable'."""
#     it = iter(iterable)
#     while chunk := list(islice(it, size)):
#         yield chunk


# def process_batch_serial(batch, func):
#     """Apply 'func' to a batch of objects."""
#     for obj in batch:
#         func(obj)

def process_batched_parallel(obj_list, func, batch_size=100, max_workers=10):
    """
    Process objects in parallel in batches.

    Parameters:
        obj_list: List of objects to process.
            executor.submit(process_batch_wrapper, batched_objs, func)
        batch_size: Size of each batch.
        max_workers: Number of workers to use.
    """
        # for batched_objs in batch(obj_list, batch_size):
        #     executor.submit(process_batch_serial, batched_objs, func)
      #  For very long iterables, using a large value for chunksize can significantly improve performance
    with ThreadPoolExecutor() as executor: #ProcessPoolExecutor, ThreadPoolExecutor  max_workers=max_workers,  # process f args must be pickable
      list(tqdm(executor.map(test_npz, obj_list, chunksize=batch_size), total=len(obj_list)))


def test_npz(file):
  try:
    np.load(file, allow_pickle=True)
  except Exception as e:
    print(f'Error loading {file}: {e}. Removed {file}')
    silentremove(file)

if __name__ == "__main__":
  _dir = '/storage_common/nobilm/pretrain_paper/guacamol/500k/'
  with TimeThis("File count"):
    files = ls(_dir)
  print(len(files))
  with TimeThis("Process files"):
    process_batched_parallel(files, test_npz, batch_size=1000) #, max_workers=max(cpu_count()-10, 1))
  print(ls(_dir))
