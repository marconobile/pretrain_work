'''
Using CONFORGE: https://pubs.acs.org/doi/10.1021/acs.jcim.3c00563
https://cdpkit.org/cdpl_python_tutorial/cdpl_python_tutorial.html#generating-conformer-ensembles
'''
import torch
import numpy as np
from source.utils.mol2pyg import mol2pyg
from source.utils.npz_utils import save_npz
from tqdm import tqdm
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def smi2npz(
    save_dir:str,
    smi_list:list,
    generate_confs:bool=True,
    ys:list|None=None,
    split:bool=True,
    minRMSD:float=1.5,
    n_confs_to_keep:int=1,
    n_confs_to_generate:int=10,
    write:bool=True,
    filter_via_dihedral_fingerprint:bool=False,
) -> None:

    '''given a smile, save associated npz in save_dir'''

    # handle the case of no targets
    ys_provided = ys is not None
    if ys is None:
        ys = [None] * len(smi_list)

    conforge_settings = getSettings(minRMSD = minRMSD, max_num_out_confs_to_generate=n_confs_to_generate) if generate_confs else None
    pyg_mols_to_save = []
    n_mols_skipped = 0
    print(f"Initial number of smiles {len(smi_list)}")
    for smi, y in tqdm(zip(smi_list, ys), total=len(smi_list), desc="Processing SMILES"):
        smi = drop_disconnected_components(smi)
        if generate_confs:
            conformers = generate_conformers(smi,
                                             conforge_settings,
                                             n_confs_to_keep,
                                             filter_via_dihedral_fingerprint=filter_via_dihedral_fingerprint)
            if not conformers:
                n_mols_skipped +=1
                continue
            pyg_mol = mol2pyg(conformers[0]) # set all non-pos-related fields
            if pyg_mol is None:
                n_mols_skipped +=1
                continue
            pyg_mol = set_conformer_in_pyg_mol(conformers, pyg_mol) # set pos of all confs
        else:
            raise ValueError("Better to use generate_confs=True")
            # mol = rdChem.MolFromSmiles(smi)
            # mol = preprocess_mol(mol, addHs=False)
            # pyg_mol = mol2pyg(mol, use_rdkit_3d=True)
            # if pyg_mol is None:
            #     n_mols_skipped +=1
            #     continue

        if ys_provided:
            pyg_mol.y = np.array(y, dtype=np.float32)
        pyg_mols_to_save.append(pyg_mol)

    print(f'number of mols skipped: {n_mols_skipped}')
    print(f"Final number of smiles {len(pyg_mols_to_save)}")
    if write:
        save_npz(pyg_mols_to_save, folder_name=save_dir, split=split)
    return pyg_mols_to_save, n_mols_skipped


############
# PRIVATE: #
############

import os
import sys
from rdkit import Chem as rdChem
import numpy as np
import tempfile
import CDPL.ConfGen as ConfGen
import CDPL.Chem as CDPLChem
from source.utils.mol_utils import drop_disconnected_components, get_dihedral_angles, preprocess_mol
from source.utils.rdkit_conformer_generation import *
from source.utils.file_handling_utils import silentremove
from einops import repeat

def set_conformer_in_pyg_mol(conformers, pyg_mol):
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
    pyg_mol.pos = batched_pos
    pyg_mol.edge_index = pyg_mol.edge_index.unsqueeze(0) # (2, E) -> (1, 2, E)
    pyg_mol.edge_index = repeat(pyg_mol.edge_index, 'b e d -> (repeat b) e d', repeat=batched_pos.shape[0]) # 1 edge index for each conf
    return pyg_mol


# mapping status codes to human readable strings
status_to_str = { ConfGen.ReturnCode.UNINITIALIZED                  : 'uninitialized',
                  ConfGen.ReturnCode.TIMEOUT                        : 'max. processing time exceeded',
                  ConfGen.ReturnCode.ABORTED                        : 'aborted',
                  ConfGen.ReturnCode.FORCEFIELD_SETUP_FAILED        : 'force field setup failed',
                  ConfGen.ReturnCode.FORCEFIELD_MINIMIZATION_FAILED : 'force field structure refinement failed',
                  ConfGen.ReturnCode.FRAGMENT_LIBRARY_NOT_SET       : 'fragment library not available',
                  ConfGen.ReturnCode.FRAGMENT_CONF_GEN_FAILED       : 'fragment conformer generation failed',
                  ConfGen.ReturnCode.FRAGMENT_CONF_GEN_TIMEOUT      : 'fragment conformer generation timeout',
                  ConfGen.ReturnCode.FRAGMENT_ALREADY_PROCESSED     : 'fragment already processed',
                  ConfGen.ReturnCode.TORSION_DRIVING_FAILED         : 'torsion driving failed',
                  ConfGen.ReturnCode.CONF_GEN_FAILED                : 'conformer generation failed' }


def generateConformationEnsembles(mol: CDPLChem.BasicMolecule, conf_gen: ConfGen.ConformerGenerator):
    """
    Generates a conformation ensemble for the argument molecule using the provided initialized ConfGen.ConformerGenerator instance.

    Parameters:
    - mol (CDPLChem.BasicMolecule): Molecule to generate a conformation ensemble for.
    - conf_gen (ConfGen.ConformerGenerator): Instance of the ConfGen.ConformerGenerator class.

    Returns:
    - int: Status code indicating the success of the conformation ensemble generation.
    - int: Number of generated conformers.
    """
    # prepare the molecule for conformer generation
    ConfGen.prepareForConformerGeneration(mol) #! in here all the preprocessing required for the conf gen is done automatically

    # generate the conformer ensemble
    status = conf_gen.generate(mol)
    num_confs = conf_gen.getNumConformers()

    # if sucessful, store the generated conformer ensemble as
    # per atom 3D coordinates arrays (= the way conformers are represented in CDPKit)
    if status == ConfGen.ReturnCode.SUCCESS or status == ConfGen.ReturnCode.TOO_MUCH_SYMMETRY:
        conf_gen.setConformers(mol)
    else:
        num_confs = 0

    return (status, num_confs)


def preprocess_smile(s):
    s = drop_disconnected_components(s)
    m = rdChem.MolFromSmiles(s)
    m = preprocess_mol(m)
    return rdChem.MolToSmiles(m)


def smi_to_cdpl_mol(s):
    s = drop_disconnected_components(s)
    m = rdChem.MolFromSmiles(s)
    m = preprocess_mol(m)
    s = rdChem.MolToSmiles(m)
    return CDPLChem.parseSMILES(s)


def getSettings(minRMSD:float, max_num_out_confs_to_generate:int = 100) -> ConfGen.ConformerGenerator:
    settings = ConfGen.ConformerGenerator()
    settings.settings.setSamplingMode(1) # AUTO = 0; SYSTEMATIC = 1; STOCHASTIC = 2;
    settings.settings.timeout = 36000
    settings.settings.minRMSD = minRMSD
    print(f'Using minRMSD = {settings.settings.minRMSD}')
    settings.settings.energyWindow = 150000.0
    settings.settings.setMaxNumOutputConformers(max_num_out_confs_to_generate) # -n in https://cdpkit.org/v1.1.1/cdpl_python_cookbook/confgen/gen_ensemble.html
    return settings


# def generate_conformers_rdkit(smi:str) -> list:
#     mol = rdChem.MolFromSmiles(smi, smi_reader_params())
#     mol, filtered_conformers = rdkit_generate_conformers(mol)
#     if not mol:
#         return []

#     with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
#         tmp_name = tmp_file.name + ".sdf"
#         rdkit_save_conformers_to_sdf(mol, tmp_name, filtered_conformers)
#         tmp_mol_ensemble = list(rdChem.rdmolfiles.SDMolSupplier(tmp_name, removeHs=False))
#         return load_conformers_from_sdf(tmp_mol_ensemble, nconfstokeep)


def generate_conformers(
    smi:str,
    conf_gen:ConfGen.ConformerGenerator,
    max_num_out_confs_to_keep:int,
    tmp_dir:str='/home/nobilm@usi.ch/pretrain_paper/tmp/',
    filter_via_dihedral_fingerprint:bool=False,
    removeHs:bool=True,
)-> list:
    '''
    tmp folder is used to write/read output of conforge, tmp file is delete right after
    removeHs:bool=True whether to remove Hs from output
    '''
    mol = CDPLChem.parseSMILES(smi)
    status, num_confs = generateConformationEnsembles(mol, conf_gen)
    print(f"{num_confs} for {smi}")
    if num_confs == 0:
        print(f"no conformers generated for {smi} via CONFORGE: {status_to_str[status]}")
        return []

    os.makedirs(tmp_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=tmp_dir, delete=True) as tmp_file:
        unique_filename = tmp_file.name + ".sdf"
        writer = CDPLChem.MolecularGraphWriter(unique_filename)
        if not writer.write(mol):
            writer.close()
            silentremove(unique_filename)
            sys.exit('Error: output of conformer ensemble for molecule %s failed' % smi)
        writer.close()
        mol_ensemble = list(rdChem.rdmolfiles.SDMolSupplier(unique_filename, removeHs=removeHs))
        silentremove(unique_filename)
        if filter_via_dihedral_fingerprint:
            return load_conformers_from_sdf(mol_ensemble, max_num_out_confs_to_keep)
        return mol_ensemble[:max_num_out_confs_to_keep]



def load_conformers_from_sdf(mol_ensamble:list, keep_n:int) -> list:
    mol_ensemble = [m for m in mol_ensamble if m and m.GetConformer() !=-1]
    if len(mol_ensemble) < keep_n:
        return mol_ensemble

    # Compute the dot product matrix between confs dihedrals-vec as descriptors
    dihedral_fingerprints = [get_dihedral_angles(mol) for mol in mol_ensemble]
    n = len(dihedral_fingerprints)
    dot_product_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dot_product_matrix[i, j] = np.dot(dihedral_fingerprints[i], dihedral_fingerprints[j])
            dot_product_matrix[j, i] = dot_product_matrix[i, j]

    # Get the indices of the N lowest values (most unique)
    sum_of_dot_prods = dot_product_matrix.sum(-1)
    sorted_indices = np.argsort(sum_of_dot_prods)
    lowest_indices = sorted_indices[:keep_n]
    return [mol_ensemble[i] for i in lowest_indices]