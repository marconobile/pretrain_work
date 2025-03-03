# modified from: https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py
from typing import List, Final
from source.utils.atom_encoding import periodic_table_group, periodic_table_period

# allowable multiple choice node and edge features
# dictionary of list with hard coded values
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'misc'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],

    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'ABSENT',
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
        'ABSENT',
    ],
    'possible_is_conjugated_list': [
        'False',
        'True',
        'ABSENT',
        ],
}

GROUP: Final[str] = 'group'
PERIOD: Final[str] = 'period'
ATOMIC_NUM: Final[str] = 'atomic_num'
CHIRALITY: Final[str] = 'chirality'
DEGREE: Final[str] = 'degree'
FORMAL_CHARGE: Final[str] = 'formal_charge'
NUM_H: Final[str] ='numH'
NUM_RADICAL_E: Final[str] = 'number_radical_e'
HYBRIDIZATION: Final[str] = 'hybridization'
IS_AROMATIC: Final[str] = 'is_aromatic'
IS_IN_RING: Final[str] = 'is_in_ring'

possible_atomic_properties: List[str] = [
    GROUP,
    PERIOD,
    ATOMIC_NUM,
    CHIRALITY,
    DEGREE,
    FORMAL_CHARGE,
    NUM_H,
    NUM_RADICAL_E,
    HYBRIDIZATION,
    IS_AROMATIC,
    IS_IN_RING,
]

BOND_TYPE: Final[str] = 'bond_type'
BOND_STEREO: Final[str] = 'bond_stereo'
BOND_IS_CONJUGATED: Final[str] = 'is_conjugated'

possible_bond_properties: List[str] = [
    BOND_TYPE,
    BOND_STEREO,
    BOND_IS_CONJUGATED,
]


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1
# # miscellaneous case
# i = safe_index(allowable_features['possible_atomic_num_list'], 'asdf')
# assert allowable_features['possible_atomic_num_list'][i] == 'misc'
# # normal case
# i = safe_index(allowable_features['possible_atomic_num_list'], 2)
# assert allowable_features['possible_atomic_num_list'][i] == 2

def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = {
            GROUP: periodic_table_group(atom),
            PERIOD: periodic_table_period(atom),
            ATOMIC_NUM: safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            CHIRALITY: safe_index(allowable_features['possible_chirality_list'], str(atom.GetChiralTag())), # https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.ChiralType
            DEGREE: safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()), # num of covalent bonds
            FORMAL_CHARGE: safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            NUM_H: safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            NUM_RADICAL_E: safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            HYBRIDIZATION: safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            IS_AROMATIC: allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            IS_IN_RING: allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
        }
    return atom_feature
# from rdkit import Chem
# mol = Chem.MolFromSmiles('Cl[C@H](/C=C/C)Br')
# atom = mol.GetAtomWithIdx(1)  # chiral carbon
# atom_feature = atom_to_feature_vector(atom)
# assert atom_feature == [5, 2, 4, 5, 1, 0, 2, 0, 0]


def get_atom_feature_dims():
    return list(map(len, [
        allowable_features['possible_atomic_num_list'],
        allowable_features['possible_chirality_list'],
        allowable_features['possible_degree_list'],
        allowable_features['possible_formal_charge_list'],
        allowable_features['possible_numH_list'],
        allowable_features['possible_number_radical_e_list'],
        allowable_features['possible_hybridization_list'],
        allowable_features['possible_is_aromatic_list'],
        allowable_features['possible_is_in_ring_list']
        ]))

def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    if bond is None:
        bond_feature = {
            BOND_TYPE:allowable_features['possible_bond_type_list'].index(allowable_features['possible_bond_type_list'][-1]),
            BOND_STEREO:allowable_features['possible_bond_stereo_list'].index(allowable_features['possible_bond_stereo_list'][-1]),
            BOND_IS_CONJUGATED:allowable_features['possible_is_conjugated_list'].index(allowable_features['possible_is_conjugated_list'][-1]),
        }
    else:
        bond_feature = {
            BOND_TYPE:safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
            BOND_STEREO:safe_index(allowable_features['possible_bond_stereo_list'], str(bond.GetStereo())),
            BOND_IS_CONJUGATED:safe_index(allowable_features['possible_is_conjugated_list'], str(bond.GetIsConjugated())),
            }
    return bond_feature
# uses same molecule as atom_to_feature_vector test
# bond = mol.GetBondWithIdx(2)  # double bond with stereochem
# bond_feature = bond_to_feature_vector(bond)
# assert bond_feature == [1, 2, 0]

def get_bond_feature_dims():
    return list(map(len, [
        allowable_features['possible_bond_type_list'],
        allowable_features['possible_bond_stereo_list'],
        allowable_features['possible_is_conjugated_list']
        ]))