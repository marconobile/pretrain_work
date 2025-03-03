from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from typing import List, Tuple

def GetDihedral(conf:Chem.Conformer, atom_idx:int):
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


def get_torsions(m:Chem.Mol)->List[Tuple[int, int, int, int]]:
    """
    Extracts the torsion angles (dihedrals) from mol

    This function identifies all the torsion angles in the given molecule and returns the list of these torsions.
    A torsion angle is defined by four atoms and is calculated based on the connectivity of these atoms in the molecule.

    Args:
        mol_list (list): A RDKit molecule objects.

    Returns:
        list: A list of tuples, where each tuple contains four integers representing the indices of the atoms
              that define a torsion angle in the molecule.
    """
    torsionList = []
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
                if ((b2.GetIdx() == bond.GetIdx()) or (b2.GetIdx() == b1.GetIdx())):
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
                    torsionList.append((idx4, idx3, idx2, idx1))
                    break
                else:
                    torsionList.append((idx1, idx2, idx3, idx4))
                    break
            break

    return torsionList