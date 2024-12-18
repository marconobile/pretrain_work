
import numpy as np
from rdkit import Chem as rdChem
from rdkit.Geometry import Point3D
from rdkit.Chem import AllChem, rdMolTransforms
import torch
from copy import deepcopy
import warnings


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


def smi_reader_params():
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
  params.removeHs = False # if input smi did not have hs then parsed smi will not have hs
  params.parseName = False
  return params


def get_energy(mol):
  '''
  RDKit performs energy minimization of mol conformation using the Merck Molecular Force Field (MMFF94).
  kilocalories per mole (kcal/mol)  energies are reported in kcal mol-1.
  If not_converged is 0 the optimization converged for that conformer.
  '''
  not_converged, energy = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=0)[0] # return: list of (not_converged, energy) 2-tuples. If not_converged is 0 the optimization converged for that conformer.
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
  while not_converged:
    not_converged, energy = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=2)[0] # return: list of (not_converged, energy) 2-tuples. If not_converged is 0 the optimization converged for that conformer.
    attempts +=1
    if attempts == 10:
      warnings.warn("returning before reaching convergence")
      return mol, energy
  return mol, energy


def drop_disconnected_components(inpt):
  '''
  the assumption here is that the largest fragment is always the mol desired
  '''
  if isinstance(inpt, rdChem.Mol): return max(rdChem.GetMolFrags(inpt, asMols=True), key=lambda frag: frag.GetNumAtoms())
  elif isinstance(inpt, str): return max(inpt.split('.'), key=len)
  raise ValueError(f"inpt type {type(inpt)} not supported, it must be str or rdkit mol")


def preprocess_mol(m:rdChem.Mol,
                  sanitize:bool=True,
                  addHs:bool=True,
                  _drop_disconnected_components:bool=True
                ):
  if m == None: return None
  try:
    if addHs: m = rdChem.AddHs(m, addCoords=True)
    #! dropping is a choice, we could also merge fragments as shown in: https://youtu.be/uvhZBpdDjoM?si=8Ica5_KfwUHmyUIX&t=1455
    if _drop_disconnected_components: m = drop_disconnected_components(m)
    if sanitize:
      error = rdChem.SanitizeMol(m)
      if error: return None
  except: return None
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
    for j in range(len(mols)):
        p.removeAllModels(viewer=(0, j))
        p.addModel(rdChem.MolToMolBlock(mols[j], confId=0), 'sdf', viewer=(0, j))
        p.setStyle({drawing_style: {}}, viewer=(0, j))
        if titles[j]: p.addLabel(titles[j], viewer=(0, j)) # , {'position': {'x': 0, 'y': 1.5, 'z': 0}, 'backgroundColor': 'white', 'fontSize': 16}
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