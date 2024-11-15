import numpy as np
import os, shutil
from os import listdir
from os.path import isfile, join
from random import shuffle, seed
from pathlib import Path
from rdkit import Chem
import torch
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import HybridizationType
from itertools import combinations
from source.data_transforms._frad_transforms import apply_changes, get_torsions, GetDihedral, preprocess_mol, get_conformer
from typing import Dict, List
from geqtrain.utils.torch_geometric import Data

def pairwise_distances(coords):
    # Get all pairs of points
    indices = list(combinations(range(len(coords)), 2))
    max_distance = 0
    for i, j in indices:
        distance = torch.dist(coords[i], coords[j]).item()
        if distance > max_distance:
            max_distance = distance
    return max_distance

def max_intra_graph_distance(data_list):
    max_distance_overall = 0
    for data in data_list:
        coords = data.pos
        max_distance = pairwise_distances(coords)
        if max_distance > max_distance_overall:
            max_distance_overall = max_distance

    return max_distance_overall

def max_intra_graph_distance_list(data_list):
    max_distances = []
    for data in data_list:
        coords = data.pos
        max_distance = pairwise_distances(coords)
        max_distances.append(max_distance)

    max_distances.sort(reverse=True)
    return max_distances

def max_intra_graph_distance_list_with_idxs(data_list):
    max_distances_with_indices = []
    for idx, data in enumerate(data_list):
        coords = data.pos
        max_distance = pairwise_distances(coords)
        max_distances_with_indices.append((idx, max_distance))

    max_distances_with_indices.sort(key=lambda x: x[1], reverse=True)
    return max_distances_with_indices

def average_neighbors_within_cutoff(data_list, cutoff=20):
    total_neighbors_within_cutoff = 0
    total_nodes = 0

    for data in data_list:
        coords = data.pos
        num_nodes = coords.size(0)

        if num_nodes == 0:
            continue

        neighbors_within_cutoff = torch.zeros(num_nodes)

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    distance = torch.dist(coords[i], coords[j]).item()
                    if distance <= cutoff:
                        neighbors_within_cutoff[i] += 1

        total_neighbors_within_cutoff += neighbors_within_cutoff.sum().item()
        total_nodes += num_nodes

    if total_nodes == 0:
        return 0

    average_neighbors_within_cutoff_per_node = total_neighbors_within_cutoff / total_nodes
    return average_neighbors_within_cutoff_per_node

def check_if_all_sanitizable(smiles):
    mols = []
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        try:
            Chem.SanitizeMol(m)
        except:
            continue
        mols.append(m)

    assert len(mols) == len(smiles)
    return mols



def print_frad_npz(path):
    data = np.load(path)
    lst = data.files
    for item in lst:
        print(item)
        if item == 'smiles':
            print(data[item])
        else:
            print(data[item].shape)

def addHs_and_standardize_smiles(mol):
    '''
    mol: mol with or without Hs

    In the context of SMILES, "canonical" just means that if you provide the same molecule as input you will always get the same SMILES as output,
    i.e. the output SMILES is not dependent on the order of the input atoms or bonds.

    returns smiles with
    '''
    mol = Chem.AddHs(mol)

    smi = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
        allBondsExplicit=True,
        allHsExplicit=True,
    )
    #! Removing Hs is the default in MolFromSmiles, they API binding of pyrdkit to c++ does not expose the removeHs flag
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smi, params)
    return mol, smi # mol is exactly what smi is encoding, without ambiguity, while smi will always be the same

def print_smi_with_atom_idxs(smi):
    from io import BytesIO
    from rdkit.Chem import Draw

    m = Chem.MolFromSmiles(smi)
    # # do opearations
    # m = Chem.AddHs(m)
    # conf = get_conformer(m)
    # assert conf != None


    Draw.MolToImage(m, addAtomIndices=True)
    d2d = Draw.MolDraw2DCairo(500,500)
    dopts = d2d.drawOptions()
    dopts.addAtomIndices = True
    d2d.DrawMolecule(m)
    d2d.FinishDrawing()
    bio = BytesIO(d2d.GetDrawingText())
    return bio
