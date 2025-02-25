import torch
from rdkit import Chem as rdChem
from tqdm import tqdm
from source.utils.mol_utils import preprocess_mol
from source.utils.mol2pyg import mol2pyg
import numpy as np
from source.utils.npz_utils import save_npz, save_pyg_as_npz
import CDPL.Chem as CDPLChem
import CDPL.ConfGen as ConfGen



# load + scaffold split
from source.utils import parse_csv
from source.utils.mol_utils import drop_disconnected_components, preprocess_mol, visualize_3d_mols
from source.utils.mol2pyg import mols2pyg_list_with_targets
from source.utils.npz_utils import save_npz
from source.utils.conforge_conformer_generation import generate_conformers
from source.utils.data_splitting_utils import scaffold_splitter
from collections import defaultdict


csv = '/home/nobilm@usi.ch/pretrain_paper/data/moelculenet/bace.csv'
experimet_directory = '/storage_common/nobilm/pretrain_paper/guacamol/EXPERIMENTS/bace_NEW_ENCODING_TEST'

# parse csv
out = parse_csv(csv, [0,2])
smiles = out['mol']
ys = out['Class']

# def getSettings():
#     settings = ConfGen.ConformerGenerator()
#     settings.settings.setSamplingMode(1) # AUTO = 0; SYSTEMATIC = 1; STOCHASTIC = 2;
#     settings.settings.timeout = 360000000 #! this is core
#     settings.settings.minRMSD = 2.5 # for freesolv test #! this is core ; 2.5
#     settings.settings.energyWindow = 150000.0
#     settings.settings.setMaxNumOutputConformers(100) # mostly irrelevant?
#     return settings

filtered = defaultdict(list)
for s, y in zip(smiles, ys): 
    s = drop_disconnected_components(s)
    mol = preprocess_mol(rdChem.MolFromSmiles(s))
    if mol:
        conformers = generate_conformers(s, getSettings())
        if conformers:
            filtered['smiles'].append(s)
            filtered['mols'].append(conformers[0]) #! HERE IM TAKING ONLY LOWEST ENERGY CONF
            filtered['y'].append(y)


pyg_mol_fixed_fields = mols2pyg_list_with_targets(mols=filtered['mols'], smiles=filtered['smiles'], ys=filtered['y'])

save_npz(pyg_mol_fixed_fields, experimet_directory)
scaffold_splitter(experimet_directory, 'tmp')