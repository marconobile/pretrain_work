from rdkit import Chem
from mendeleev.fetch import fetch_table
import numpy as np
# pip install mendeleev

ptable = fetch_table("elements")
PATOMIC_NUMBERS = {row["symbol"]: row["atomic_number"] for _, row in ptable.iterrows()}
PGROUP_IDS = {row["symbol"]: row["group_id"] for _, row in ptable.iterrows()}
PPERIOD_IDS = {row["symbol"]: row["period"] for _, row in ptable.iterrows()}

# Atom representations feature_atom ∈ R N_atom × c.
# The input atom representation is a concatenation of one-hot encodings of element group index and period index for the given atom, which is embedded by a linear projection layer R:(18+7) → Rc


def one_of_k_encoding(x, allowable_set):
    """
    Maps inputs not in the allowable set to the last element.
    modified from https://github.com/XuhanLiu/NGFP
    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_atom_encoding(atom: Chem.rdchem.Atom):
    assert isinstance(atom, Chem.rdchem.Atom), f'atom must be of type Chem.rdchem.Atom but is {type(atom)}'
    # https://github.com/zrqiao/NeuralPLexer/blob/2c52b10d3094e836661dfecfa3be76f47dcdea7e/neuralplexer/data/molops.py#L257
    # Periodic table encoding
    encoding_list = (
        one_of_k_encoding(PGROUP_IDS[atom.GetSymbol()], list(range(1, 19)))
        + one_of_k_encoding(PPERIOD_IDS[atom.GetSymbol()], list(range(1, 6)))
        # + one_of_k_encoding(atom.GetDegree(), list(range(7)))
        # + one_of_k_encoding(
        #     atom.GetHybridization(),
        #     [
        #         Chem.rdchem.HybridizationType.SP,
        #         Chem.rdchem.HybridizationType.SP2,
        #         Chem.rdchem.HybridizationType.SP3,
        #         Chem.rdchem.HybridizationType.SP3D,
        #         Chem.rdchem.HybridizationType.SP3D2,
        #     ],
        # )
        # + [atom.GetIsAromatic()]
    )
    return np.array(encoding_list)