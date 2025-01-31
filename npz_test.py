import numpy as np
from source.utils.file_handling_utils import ls, silentremove
from itertools import islice
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from source.utils.code_utils import TimeThis
from tqdm import tqdm
from source.utils.npz_utils import test_npz_validity
# ProcessPool is for CPU-bound tasks so you can benefit from multiple CPU.
# ThreadPoolExecutor is for IO-bound tasks so you can benefit from IO-wait.
# If you have many short-lived CPU-bound tasks use threads due to the overhead of creating and managing multiple processes,
# as the cost of creating a process can be significant compared to the amount of time spent actually executing the task


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
    with Pool(processes=max_workers) as pool:
        pool.map(func, obj_list, chunksize=batch_size)

if __name__ == "__main__":
    _dir = '/storage_common/nobilm/pretrain_paper/guacamol/100k'
    with TimeThis("File count"):
        files = ls(_dir)
    print(len(files))
    with TimeThis("Process files"):
        process_batched_parallel(files, test_npz_validity, batch_size=10000) #, max_workers=max(cpu_count()-10, 1))
#   print(len(ls(_dir)))
