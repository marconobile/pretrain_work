import numpy as np
from source.utils.file_handling_utils import ls,silentremove
from tqdm import tqdm

_dir = '/storage_common/nobilm/pretrain_paper/guacamol/100k/'
# '/storage_common/nobilm/pretrain_paper/guacamol/one_million'
for f in tqdm(ls(_dir)):
  try:
    data = np.load(f, allow_pickle=True)
  except:
    print(f'removed {f}')
    silentremove(f)

# f = '/storage_common/nobilm/pretrain_paper/guacamol/100k/mol_748408.npz'
# data = np.load(f, allow_pickle=True)

