# TODO BRING THIS INSDE NZPWRITER
import os
from source.utils.npz_utils import get_field_from_npzs
from tqdm import tqdm

basepath = '/storage_common/nobilm/pretrain_paper/guacamol/EXPERIMENTS/bbbp/'
trainpath = basepath+'train'
out = get_field_from_npzs(trainpath)

i = 0
for el in tqdm(out, desc="Validating and cleaning train files"):
    if not 'bond_stereo' in el.keys():
        file = el.zip.filename
        os.remove(file)
        i += 1
        continue
    for k, v in el.items():
        pos_shape = el['coords'].shape[1]
        if k in ['adj_matrix',
                 'atomic_num',
                 'chirality',
                 'degree',
                 'formal_charge',
                 'group',
                 'hybridization',
                 'is_aromatic',
                 'is_in_ring',
                 'numH',
                 'number_radical_e',
                 'period']:
            if v.shape[0] != pos_shape:
                file = el.zip.filename
                os.remove(file)
                i += 1
                break

print('Removed from train:' , i)


valpath = basepath+'val'
out = get_field_from_npzs(valpath)

i = 0
for el in tqdm(out, desc="Validating and cleaning val files"):
    if not 'bond_stereo' in el.keys():
        file = el.zip.filename
        os.remove(file)
        i += 1
        continue
    for k, v in el.items():
        pos_shape = el['coords'].shape[1]
        if k in ['adj_matrix',
                 'atomic_num',
                 'chirality',
                 'degree',
                 'formal_charge',
                 'group',
                 'hybridization',
                 'is_aromatic',
                 'is_in_ring',
                 'numH',
                 'number_radical_e',
                 'period']:
            if v.shape[0] != pos_shape:
                file = el.zip.filename
                os.remove(file)
                i += 1
                break


print('Removed from val:' , i)