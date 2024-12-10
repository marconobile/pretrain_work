from rdkit import Chem
from rdkit.Chem import rdMolTransforms


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


def SetDihedral(conf, atom_idx, new_val):
  """
  Sets the value of a dihedral angle (torsion) in a molecule's conformation.

  This function modifies the dihedral angle defined by four atoms in the given molecule conformation to the specified value.

  Args:
      conf (RDKit Conformer): The conformation of the molecule.
      atom_idx (tuple): A tuple of four integers representing the indices of the atoms that define the dihedral angle.
      new_val (float): The new value of the dihedral angle in degrees.

  """
  # rdMolTransforms.SetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_val)
  try:
    rdMolTransforms.SetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_val)
  except:
    rdMolTransforms.SetDihedralDeg(conf, atom_idx[1], atom_idx[0], atom_idx[2], atom_idx[3], new_val)
    # print('--------- dihedral idxs were swapped ---------')


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