import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Chem.rdchem import HybridizationType
# from geqtrain.utils.torch_geometric import Data

def get_conformer(mol, max_attempts:int=10):
    try:
        out = mol.GetConformer() # if fails mol needs to be embedded
        return out if out != -1 else None
    except:
        try:
            # ps = AllChem.ETKDG()
            # ps.maxIterations = 10
            kwargs = {
                # "numConfs":1,
                "useRandomCoords":True, # this is important to be true
                "useSmallRingTorsions":True,
                "useMacrocycleTorsions":True,
                # "params":ps,
                "maxAttempts":max_attempts,
            }
            success = AllChem.EmbedMolecule(mol, **kwargs) != -1
            if success:
                out = mol.GetConformer() # if mol embedding works this should be ok
                return out if out != -1 else None
        except: return None
    return None

def preprocess_mol(m, sanitize=True):
    if m == None: return None
    try:
        m = Chem.AddHs(m)
        # drops any disconnected fragment
        fragments = Chem.GetMolFrags(m, asMols=True)
        m = max(fragments, key=lambda frag: frag.GetNumAtoms())
        if sanitize:
            error = Chem.SanitizeMol(m)
            if error: return None
    except: return None
    return m


def get_torsions(mol_list):
    """
    Extracts the torsion angles (dihedrals) from a list of molecules.

    This function identifies all the torsion angles in the given list of molecules and returns a list of these torsions.
    A torsion angle is defined by four atoms and is calculated based on the connectivity of these atoms in the molecule.

    Args:
        mol_list (list): A list of RDKit molecule objects.

    Returns:
        list: A list of tuples, where each tuple contains four integers representing the indices of the atoms
              that define a torsion angle in the molecule.

    """
    atom_counter = 0
    torsionList = []
    for m in mol_list:
        torsionSmarts = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
        torsionQuery = Chem.MolFromSmarts(torsionSmarts)
        matches = m.GetSubstructMatches(torsionQuery)
        for match in matches:
            idx2 = match[0]
            idx3 = match[1]
            bond = m.GetBondBetweenAtoms(idx2, idx3)
            jAtom = m.GetAtomWithIdx(idx2)
            kAtom = m.GetAtomWithIdx(idx3)
            for b1 in jAtom.GetBonds():
                if (b1.GetIdx() == bond.GetIdx()):
                    continue
                idx1 = b1.GetOtherAtomIdx(idx2)
                for b2 in kAtom.GetBonds():
                    if ((b2.GetIdx() == bond.GetIdx())
                            or (b2.GetIdx() == b1.GetIdx())):
                        continue
                    idx4 = b2.GetOtherAtomIdx(idx3)
                    # skip 3-membered rings
                    if (idx4 == idx1):
                        continue
                    # skip torsions that include hydrogens
                    #                     if ((m.GetAtomWithIdx(idx1).GetAtomicNum() == 1)
                    #                         or (m.GetAtomWithIdx(idx4).GetAtomicNum() == 1)):
                    #                         continue
                    if m.GetAtomWithIdx(idx4).IsInRing():
                        torsionList.append(
                            (idx4 + atom_counter, idx3 + atom_counter, idx2 + atom_counter, idx1 + atom_counter))
                        break
                    else:
                        torsionList.append(
                            (idx1 + atom_counter, idx2 + atom_counter, idx3 + atom_counter, idx4 + atom_counter))
                        break
                break

        atom_counter += m.GetNumAtoms()
    return torsionList

def SetDihedral(conf, atom_idx, new_vale):
    """
    Sets the value of a dihedral angle (torsion) in a molecule's conformation.

    This function modifies the dihedral angle defined by four atoms in the given molecule conformation to the specified value.

    Args:
        conf (RDKit Conformer): The conformation of the molecule.
        atom_idx (tuple): A tuple of four integers representing the indices of the atoms that define the dihedral angle.
        new_vale (float): The new value of the dihedral angle in degrees.

    """
    # rdMolTransforms.SetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)
    try:
        rdMolTransforms.SetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)
    except:
        rdMolTransforms.SetDihedralDeg(conf, atom_idx[1], atom_idx[0], atom_idx[2], atom_idx[3], new_vale)
        print('--------- dihedral idxs were swapped ---------')

def GetDihedral(conf, atom_idx):
    """
    Retrieves the value of a dihedral angle (torsion) in a molecule's conformation.

    This function returns the current value of the dihedral angle defined by four atoms in the given molecule conformation.

    Args:
        conf (RDKit Conformer): The conformation of the molecule.
        atom_idx (tuple): A tuple of four integers representing the indices of the atoms that define the dihedral angle.

    Returns:
        float: The value of the dihedral angle in degrees.

    """
    return rdMolTransforms.GetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])

def apply_changes(mol, values, rotable_bonds):
    """
    Applies specified dihedral angle changes to a molecule based on the provided values for the dihedral angles.

    Args:
        mol (RDKit Mol): The original molecule to which the changes will be applied.
        values (list of float): A list of new values for the dihedral angles in degrees.
        rotable_bonds (list of tuple): A list of tuples, where each tuple contains four integers representing
                                       the indices of the atoms that define a rotatable bond.

    Returns:
        None: modifies inplace the input mol (RDKit Mol)
    """
    [SetDihedral(mol.GetConformer(), rotable_bonds[r], values[r].item()) for r in range(len(rotable_bonds))]

# def transform_noise(data, position_noise_scale):
#     return data + np.random.normal(loc=0, scale=1, size=data.shape) * position_noise_scale

def nosify_mol(data):
    '''
    data: pyg data object
    return: pyg data object with new coords
    '''
    dihedral_noise_tau, coords_noise_tau = 2, 0.04
    mol = Chem.MolFromSmiles(str(data.smiles))
    mol = Chem.AddHs(mol)
    conf = get_conformer(mol)

    if conf == None:
        # do not change coords
        # return data obj with noise equal to 0
        data.noise_target = torch.zeros_like(data.pos, dtype=torch.float)
        print("bad mol:", str(data.smiles))
        return data

    # apply dihedral noise
    try:
        # if good conformer try to apply noise using precomputed torsional idxs/angles
        rotable_bonds = data.rotable_bonds.tolist()
        if rotable_bonds:
            original_dihedral_angles_degrees = data.dihedral_angles_degrees
            # apply dihedral noise
            noised_dihedral_angles_degrees = original_dihedral_angles_degrees + np.random.normal(0, 1, size=original_dihedral_angles_degrees.shape) * dihedral_noise_tau
            apply_changes(mol, noised_dihedral_angles_degrees, rotable_bonds)
    except:
        # if idxs not good, recompute all
        rotable_bonds = get_torsions([mol])
        original_dihedral_angles_degrees = np.array([GetDihedral(conf, rot_bond) for rot_bond in rotable_bonds])
        # apply dihedral noise
        noised_dihedral_angles_degrees = original_dihedral_angles_degrees + np.random.normal(0, 1, size=original_dihedral_angles_degrees.shape) * dihedral_noise_tau
        apply_changes(mol, noised_dihedral_angles_degrees, rotable_bonds)

    # apply coords noise
    pos_after_dihedral_noise = conf.GetPositions()
    pos_noise_to_be_predicted = np.random.normal(0, 1, size=pos_after_dihedral_noise.shape) * coords_noise_tau

    # set in data object for training
    data.noise_target = torch.tensor(pos_noise_to_be_predicted, dtype=torch.float)
    data.pos = torch.tensor(pos_after_dihedral_noise + pos_noise_to_be_predicted, dtype=torch.float)

    return data


def mol2pyg_with_noise(data):
    '''
    either returns data pyg data obj or None if some operations are not possible
    IMPO: this does not set y
    '''
    types={'Br': 0, 'C': 1, 'Cl': 2, 'F': 3, 'H': 4, 'I': 5, 'N': 6, 'O': 7, 'S': 8}
    mol = Chem.MolFromSmiles(str(data['smiles']))

    mol = preprocess_mol(mol)
    if mol == None:
      data.noise_target = torch.zeros_like(data['pos'])
      return data
    conf = get_conformer(mol)
    if conf == None:
      data.noise_target = torch.zeros_like(data['pos'])
      return data

    # now apply noise
    dihedral_noise_tau, coords_noise_tau = 2, 0.04
    rotable_bonds = get_torsions([mol])
    original_dihedral_angles_degrees = np.array([GetDihedral(conf, rot_bond) for rot_bond in rotable_bonds])

    # apply dihedral noise
    noised_dihedral_angles_degrees = original_dihedral_angles_degrees + np.random.normal(0, 1, size=original_dihedral_angles_degrees.shape) * dihedral_noise_tau
    apply_changes(mol, noised_dihedral_angles_degrees, rotable_bonds)

    # apply coords noise
    pos_after_dihedral_noise = conf.GetPositions()
    pos_noise_to_be_predicted = np.random.normal(0, 1, size=pos_after_dihedral_noise.shape) * coords_noise_tau

    # then do stuff
    type_idx, aromatic, is_in_ring, _hybridization, chirality = [], [], [], [], []
    for atom in mol.GetAtoms():

        type_idx.append(types[atom.GetSymbol()])
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        is_in_ring.append(1 if atom.IsInRing() else 0)
        # https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.ChiralType
        chirality.append(atom.GetChiralTag())

        hybridization = atom.GetHybridization()
        hybridization_value = 0
        if hybridization == HybridizationType.SP: hybridization_value = 1
        elif hybridization == HybridizationType.SP2: hybridization_value = 2
        elif hybridization == HybridizationType.SP3: hybridization_value = 3
        _hybridization.append(hybridization_value)

    rows, cols, edge_types = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rows += [start, end]
        cols += [end, start]

    edge_index = torch.tensor([rows, cols], dtype=torch.long)

    data.z=torch.tensor(type_idx)
    data.pos=torch.tensor(pos_after_dihedral_noise + pos_noise_to_be_predicted, dtype=torch.float32)
    data.hybridization=torch.tensor(_hybridization, dtype=torch.long)
    data.is_aromatic=torch.tensor(aromatic, dtype=torch.long)
    data.is_in_ring=torch.tensor(is_in_ring, dtype=torch.long)
    data.chirality=torch.tensor(chirality, dtype=torch.long)
    data.rotable_bonds=torch.tensor(rotable_bonds, dtype=torch.long)
    data.noise_target=torch.tensor(pos_noise_to_be_predicted, dtype=torch.float32)
    data.edge_index = edge_index
    return data



# def nosify_mol(data):
#     '''
#     data: pyg data object
#     return: pyg data object with new coords
#     '''
#     dihedral_noise_tau, coords_noise_tau = 2, 0.04

#     # params = Chem.SmilesParserParams()
#     # params.removeHs = False
#     mol = Chem.MolFromSmiles(str(data.smiles)) #, params)
#     # mol = preprocess_mol(mol)
#     mol
#     id:int = AllChem.EmbedMolecule(mol,
#                                    useRandomCoords=True,
#                                    enforceChirality = True,       # enforceChirality : enforce the correct chirality if chiral centers are present.
#                                    useExpTorsionAnglePrefs= True, # useExpTorsionAnglePrefs : impose experimental torsion angle preferences
#                                    useBasicKnowledge= True,       # useBasicKnowledge : impose basic knowledge such as flat rings
#                                    useMacrocycleTorsions= True,   # useMacrocycleTorsions : use additional torsion profiles for macrocycles
#                                 )
#     # https://greglandrum.github.io/rdkit-blog/posts/2024-07-28-confgen-and-intramolecular-hbonds.html
#     # https://www.rdkit.org/docs/RDKit_Book.html#conformer-generation
#     # https://www.rdkit.org/docs/source/rdkit.Chem.rdDistGeom.html#rdkit.Chem.rdDistGeom.EmbedMolecule
#     # https://www.rdkit.org/docs/source/rdkit.Chem.rdForceFieldHelpers.html
#     # https://www.rdkit.org/docs/source/rdkit.Chem.AllChem.html
#     #! https://www.rdkit.org/docs/GettingStartedInPython.html#working-with-3d-molecules


#     try:
#         rotable_bonds = data.rotable_bonds.tolist()
#         if rotable_bonds:
#             original_dihedral_angles_degrees = data.dihedral_angles_degrees
#             # apply dihedral noise
#             noised_dihedral_angles_degrees = original_dihedral_angles_degrees + np.random.normal(loc=0, scale=1, size=original_dihedral_angles_degrees.shape) * dihedral_noise_tau
#             apply_changes(mol, noised_dihedral_angles_degrees, rotable_bonds)
#     except:
#         print("issue found")
#         rotable_bonds = get_torsions([mol])
#         original_dihedral_angles_degrees = np.array([GetDihedral(mol.GetConformer(), rot_bond) for rot_bond in rotable_bonds])
#         # apply dihedral noise
#         noised_dihedral_angles_degrees = original_dihedral_angles_degrees + np.random.normal(loc=0, scale=1, size=original_dihedral_angles_degrees.shape) * dihedral_noise_tau
#         apply_changes(mol, noised_dihedral_angles_degrees, rotable_bonds)


#     # apply coords noise
#     pos_after_dihedral_noise = mol.GetConformer().GetPositions()
#     pos_noise_to_be_predicted = np.random.normal(loc=0, scale=1, size=pos_after_dihedral_noise.shape) * coords_noise_tau

#     data.noise_target = torch.tensor(pos_noise_to_be_predicted, dtype=torch.float) # todo, which field to set?
#     data.pos = torch.tensor(pos_after_dihedral_noise + pos_noise_to_be_predicted, dtype=torch.float)

#     return data


# def nosify_mol(data):
#     '''
#     data: pyg data object
#     return: pyg data object with new coords
#     '''
#     # cast to mol
#     mol = Chem.MolFromSmiles(str(data.smiles))
#     mol = Chem.AddHs(mol)

#     # apply dihedral noise
#     mol_tmp = apply_dihedral_noise(mol)
#     mol_tmp = Chem.AddHs(mol_tmp)

#     # get new atoms coords
#     coord_conf = mol_tmp.GetConformer()
#     pos = coord_conf.GetPositions()
#     pos = torch.tensor(pos, dtype=torch.float)
#     data.pos = pos # and set it into data object

#     # write assert here about different angles

#     # apply coord noise
#     data = apply_coord_noise(data) # this already sets noisified coords and sets pos_target into data obj

#     return data

