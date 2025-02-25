import os
import numpy as np

p = '/storage_common/nobilm/pretrain_paper/guacamol/EXPERIMENTS/100k_random_split_first_test/train'
tot = 0
n_confs = []

with os.scandir(p) as entries:
    for entry in entries:
        if entry.is_file() and not entry.name.startswith('.'):
            n = np.load(os.path.join(p, entry.name))['coords'].shape[0]
            n_confs.append(n)
            tot += n

print(f"Number of conforms in {p}: {tot}")
print("Bincount of num of confs:")
print(np.bincount(n_confs))
