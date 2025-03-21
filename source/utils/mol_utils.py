
import numpy as np
from rdkit import Chem as rdChem
from rdkit.Geometry import Point3D
from rdkit.Chem import AllChem, rdMolTransforms, rdMolAlign, GetPeriodicTable
import torch
from copy import deepcopy
import warnings



from rdkit.Chem import Fragments
from source.utils.rdkit_fragments import frag2num, num2frag

def get_molecule_fragments(mol:rdChem.Mol):

    def _get_fragments_dict(mol):
        """
        Get all fragment descriptors present in a molecule.

        Parameters:
        - mol: RDKit molecule object

        Returns:
        - dict: Dictionary of fragment descriptors and their counts
        """
        fragment_dict = {}
        for func in dir(Fragments):
            if func.startswith('fr_'):
                try:
                    count = getattr(Fragments, func)(mol)
                    if count > 0:
                        fragment_dict[func] = count
                except:
                    continue
        return fragment_dict

    fragment_dict = _get_fragments_dict(mol)
    fragmentsIds = []
    num_of_frags = 0
    for k, v in fragment_dict.items():
        fragmentsIds.append(frag2num[k])
        num_of_frags+=v

    binarizedFragmentsIds = np.zeros(86, dtype=int)
    binarizedFragmentsIds[fragmentsIds] = 1 # 1 if fragId present, 0 otherwise
    return binarizedFragmentsIds, num_of_frags


def pyg2mol(pyg, removeHs:bool):
    mol = rdChem.MolFromSmiles(pyg.smiles, smi_reader_params(removeHs))
    if removeHs:
        mol = rdChem.RemoveHs(mol)
    else:
        mol = rdChem.AddHs(mol)

    if pyg.pos[0].shape[0] != 1:
        out = set_coords(mol, pyg.pos[0])
    else:
        out = set_coords(mol, pyg.pos)

    return out


def atomic_number_to_symbol(atomic_number: int) -> str:
    """Convert atomic number to atomic symbol using RDKit."""
    return GetPeriodicTable().GetElementSymbol(atomic_number)


def compute_rmsd(mol1: rdChem.Mol, mol2: rdChem.Mol) -> float:
    """
    Compute the Root Mean Square Deviation (RMSD) between two RDKit molecules.

    Parameters:
    - mol1: First RDKit molecule (should have conformers).
    - mol2: Second RDKit molecule (should have conformers).

    Returns:
    - float: RMSD value between the aligned molecules.
    """
    # Ensure conformers are present
    if not mol1.GetNumConformers() or not mol2.GetNumConformers():
        raise ValueError("Both molecules must have conformers for RMSD calculation.")

    # Align molecules and compute RMSD
    rmsd = rdMolAlign.AlignMol(mol1, mol2)
    return rmsd


def optimize_conformers(molecule):
    # Optimize all conformers using MMFFOptimizeMoleculeConfs
    results = AllChem.MMFFOptimizeMoleculeConfs(molecule, numThreads=0, maxIters=200, mmffVariant='MMFF94', nonBondedThresh=100.0, ignoreInterfragInteractions=True)

    # Extract energies and convergence information
    energies = [result[1] for result in results]
    converged = [result[0] == 0 for result in results]

    return energies, converged


def get_rdkit_conformer(mol, max_attempts:int=10):
    '''
    if returns none: mols not embeddable
    '''
    # TODO: create multithread version of this via https://www.rdkit.org/docs/source/rdkit.Chem.rdDistGeom.html#rdkit.Chem.rdDistGeom.EmbedMultipleConfs
    try:
        conf = mol.GetConformer() # if fails mol needs to be embedded
        return conf if conf != -1 else None
    except:
        try:
            success = AllChem.EmbedMolecule(mol,
                                            useRandomCoords=True, # needs to be T
                                            useSmallRingTorsions=True,
                                            useMacrocycleTorsions=True,
                                            maxAttempts=max_attempts
                                          ) != -1
            if success:
                conf = mol.GetConformer() # if mol embedding worked should be ok
                return conf if conf != -1 else None
        except: return None
    return None


def smi_writer_params():
  '''to be used in Chem.MolToSmiles
  docs: https://www.rdkit.org/docs/cppapi/structRDKit_1_1SmilesWriteParams.html
  defaults/signature:
  bool 	doIsomericSmiles
  bool 	doKekule = false
  bool 	canonical = true
  bool 	cleanStereo = true
  bool 	allBondsExplicit = false
  bool 	allHsExplicit = false
  bool 	doRandom = false
  int 	rootedAtAtom = -1
  bool 	includeDativeBonds
  bool 	ignoreAtomMapNumbers = false
  '''
  writer_params = rdChem.SmilesWriteParams()
  writer_params.doIsomericSmiles = True
  writer_params.allHsExplicit = True
  return writer_params


def smi_reader_params(removeHs:bool):
  '''to be used in Chem.MolFromSmiles
  docs: https://www.rdkit.org/docs/cppapi/structRDKit_1_1v1_1_1SmilesParserParams.html
  defaults/signature:
  int 	debugParse = 0
  bool 	sanitize = true
  std::map< std::string, std::string > * 	replacements
  bool 	allowCXSMILES = true
  bool 	strictCXSMILES
  bool 	parseName = true
  bool 	removeHs = true
  bool 	skipCleanup = false
  '''
  params = rdChem.SmilesParserParams()
  params.removeHs = removeHs # if input smi did not have hs then parsed smi will not have hs
  params.parseName = False
  return params


def get_energy(mol):
  '''
  RDKit performs energy minimization of mol conformation using the Merck Molecular Force Field (MMFF94).
  kilocalories per mole (kcal/mol)  energies are reported in kcal mol-1.
  If not_converged is 0 the optimization converged for that conformer.
  '''
  numThreads = 1
  assert numThreads == 1 # mandatory otherwise not usable in pytorch transform
  not_converged, energy = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=0, numThreads=numThreads)[0] # return: list of (not_converged, energy) 2-tuples. If not_converged is 0 the optimization converged for that conformer.
  return energy


def minimize_energy(mol):
  '''
  RDKit performs energy minimization of mol conformation using the Merck Molecular Force Field (MMFF94).
  kilocalories per mole (kcal/mol)  energies are reported in kcal mol-1.
  mol is changed in place
  If not_converged is 0 the optimization converged for that conformer.
  '''
  # todo: add check on mol conformers: if mol has no conformers then this fuction cannot be executed
  not_converged = True
  attempts = 0
  mol = rdChem.AddHs(mol)
  while not_converged:
    not_converged, energy = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1)[0] # return: list of (not_converged, energy) 2-tuples. If not_converged is 0 the optimization converged for that conformer.
    attempts +=1
    if attempts == 10:
      warnings.warn("returning before reaching convergence")
      mol = rdChem.RemoveHs(mol)
      return mol, energy
  mol = rdChem.RemoveHs(mol)
  return mol, energy


def drop_disconnected_components(inpt:rdChem.Mol|str) -> rdChem.Mol|str:
    '''assumption: largest fragment is always the mol desired'''
    if isinstance(inpt, rdChem.Mol):
        return max(rdChem.GetMolFrags(inpt, asMols=True), key=lambda frag: frag.GetNumAtoms())
    elif isinstance(inpt, str):
        return max(inpt.split('.'), key=len)
    raise ValueError(f"inpt type {type(inpt)} not supported, it must be str or rdkit mol")


def preprocess_mol(m:rdChem.Mol,
                  sanitize:bool=True,
                  addHs:bool=True,
                  _drop_disconnected_components:bool=True
                ):
    if m == None:
        return None
    try:
        if addHs:
            m = rdChem.AddHs(m, addCoords=True)
        #! dropping is a choice, we could also merge fragments as shown in: https://youtu.be/uvhZBpdDjoM?si=8Ica5_KfwUHmyUIX&t=1455
        if _drop_disconnected_components:
            m = drop_disconnected_components(m)
        if sanitize:
            error = rdChem.SanitizeMol(m)
            if error:
                return None
    except:
        return None
    return m


def visualize_3d_mols(mols,
    drawing_style: str = 'stick',
    titles: list[str] = None,
    width:int=1500,
    height:int=400,
    grid:tuple=None,
  ):
    import py3Dmol
    from rdkit import Chem as rdChem
    if not grid: grid = (1, len(mols))
    drawing_style_options = [
        "line",   # Wire Model
        "cross",  # Cross Model
        "stick",  # Bar Model
        "sphere", # Space Filling Model
        "cartoon",# Display secondary structure in manga
    ]
    assert drawing_style in drawing_style_options, f"Invalid drawing style. Choose from {drawing_style_options}"
    if not isinstance(mols, list): mols = [mols]
    if titles is None: titles = ["" for _ in mols]  # Default empty titles if none provided
    assert len(titles) == len(mols), "Length of titles must match the number of molecules."

    p = py3Dmol.view(width=width, height=height, viewergrid=grid)
    nrows = grid[0]
    ncols = grid[1]


    for row_idx in range(nrows):
        for col_idx in range(ncols):
            mol_idx = row_idx * ncols + col_idx
            p.removeAllModels(viewer=(row_idx, col_idx))
            p.addModel(rdChem.MolToMolBlock(mols[mol_idx], confId=0), 'sdf', viewer=(row_idx, col_idx))
            p.setStyle({drawing_style: {}},  viewer=(row_idx, col_idx))
            if titles[mol_idx]: p.addLabel(titles[mol_idx],  viewer=(row_idx, col_idx)) # , {'position': {'x': 0, 'y': 1.5, 'z': 0}, 'backgroundColor': 'white', 'fontSize': 16}
    p.zoomTo()
    p.show()


def get_dihedral_indices(mol):
  # Find all dihedral angles (torsions)
  dihedralSmarts = '[!#1]~[!#1]~[!#1]~[!#1]'
  return [torsion for torsion in mol.GetSubstructMatches(rdChem.MolFromSmarts(dihedralSmarts))]


def get_dihedral_angles(mol):
  # Calculate all dihedral angles
  return np.array([
    rdMolTransforms.GetDihedralDeg(mol.GetConformer(), dihedral[0], dihedral[1], dihedral[2], dihedral[3])
    for dihedral in get_dihedral_indices(mol)
  ])


def set_coords(mol, coords: torch.tensor):
  new_mol = deepcopy(mol)
  conf = rdChem.rdchem.Conformer(new_mol.GetNumAtoms()) # create empty rdkit Conformer
  for i in range(new_mol.GetNumAtoms()):
    x,y,z = coords[i][0].item(), coords[i][1].item(), coords[i][2].item()
    conf.SetAtomPosition(i, Point3D(x,y,z))
  new_mol.RemoveAllConformers() # remove all present conformers
  new_mol.AddConformer(conf) # add conformer to mol
  return new_mol


#! the issue here is max_iterations (?)
def fix_conformer(mol, max_iterations=200, force_field='UFF', convergence_criteria=1e-6):
    """
    Minimize the steric clashes in the given molecule conformer using a force field.

    Parameters:
    - mol: RDKit molecule object with a 3D conformer.
    - max_iterations: Maximum number of iterations for the optimization.
    - force_field: The force field to use ('UFF' or 'MMFF').
    - convergence_criteria: The convergence criteria for the optimization.

    Returns:
    - mol: The optimized RDKit molecule object.

    Example usage
    Load an example molecule (here, we use a SMILES string and generate a 3D conformation)
    smiles = 'CCO'
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    # Fix the conformer
    fixed_mol = fix_conformer(mol)
    """
    if not mol.GetNumConformers():
        raise ValueError("The molecule does not have any conformers.")

    # Get the original positions of all atoms
    conf = mol.GetConformer()
    original_coords = np.array(conf.GetPositions())

    # Select the force field
    if force_field == 'UFF':
        ff = AllChem.UFFGetMoleculeForceField(mol)
    elif force_field == 'MMFF':
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props)
    else:
        raise ValueError("Unsupported force field. Use 'UFF' or 'MMFF'.")

    # Minimize the energy and fix clashes
    converged = ff.Minimize(maxIts=max_iterations)

    # Get the optimized positions of all atoms
    optimized_coords = np.array(conf.GetPositions())
    displacement = np.linalg.norm(optimized_coords - original_coords, axis=1).mean()

    print(f"Optimization {'converged' if converged == 0 else 'did not converge'} after {max_iterations} iterations.")
    print(f"Average displacement per atom: {displacement:.4f} Å")

    if displacement > convergence_criteria:
        print("Warning: The displacement is higher than the convergence criteria. Consider increasing the number of iterations.")

    return mol


def plot_mol_with_atom_idxs(mol):
    from rdkit.Chem.Draw import IPythonConsole
    IPythonConsole.drawOptions.addAtomIndices = True
    return mol