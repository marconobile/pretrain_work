
import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import AllChem, rdMolTransforms
import torch
from copy import deepcopy


# def get_conformer(mol, max_attempts:int=10):
#     # TODO: create multithread version of this via https://www.rdkit.org/docs/source/rdkit.Chem.rdDistGeom.html#rdkit.Chem.rdDistGeom.EmbedMultipleConfs
#     try:
#         out = mol.GetConformer() # if fails mol needs to be embedded
#         return out if out != -1 else None
#     except:
#         try:
#             success = AllChem.EmbedMolecule(mol,
#                                             useRandomCoords=True, # needs to be T
#                                             useSmallRingTorsions=True,
#                                             useMacrocycleTorsions=True,
#                                             maxAttempts=max_attempts
#                                           ) != -1
#             if success:
#                 out = mol.GetConformer() # if mol embedding worked should be ok
#                 return out if out != -1 else None
#         except: return None
#     return None


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
  writer_params = Chem.SmilesWriteParams()
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
  params = Chem.SmilesParserParams()
  params.removeHs = False
  params.parseName = False
  return params


def get_energy(mol):
  '''
  RDKit performs energy minimization of mol conformation using the Merck Molecular Force Field (MMFF94).
  kilocalories per mole (kcal/mol)  energies are reported in kcal mol-1.
  '''
  converged, energy = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=0)[0] # return: list of (not_converged, energy) 2-tuples. If not_converged is 0 the optimization converged for that conformer.
  return energy


def minimize_energy(mol):
  '''
  RDKit performs energy minimization of mol conformation using the Merck Molecular Force Field (MMFF94).
  kilocalories per mole (kcal/mol)  energies are reported in kcal mol-1.
  mol is changed in place
  '''
  converged, energy = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=500)[0] # return: list of (not_converged, energy) 2-tuples. If not_converged is 0 the optimization converged for that conformer.
  return mol, energy


def preprocess_mol(m:Chem.Mol,
                  sanitize:bool=True,
                  addHs:bool=True,
                  drop_disconnected_components:bool=True
                ):
  if m == None: return None
  try:
    if addHs: m = Chem.AddHs(m, addCoords=True)
    #! dropping is a choice, we could also merge fragments as shown in: https://youtu.be/uvhZBpdDjoM?si=8Ica5_KfwUHmyUIX&t=1455
    if drop_disconnected_components: m = max(Chem.GetMolFrags(m, asMols=True), key=lambda frag: frag.GetNumAtoms())
    if sanitize:
      error = Chem.SanitizeMol(m)
      if error: return None
  except: return None
  return m


def visualize_3d_mols(mols):
  import py3Dmol
  if not isinstance(mols, list): mols = [mols]
  p = py3Dmol.view(width=1500, height=400, viewergrid=(1,len(mols)))
  for j in range(len(mols)):
      p.removeAllModels(viewer=(0,j))
      p.addModel(Chem.MolToMolBlock(mols[j], confId=0), 'sdf', viewer=(0,j))
      p.setStyle({'stick':{}}, viewer=(0,j))
  p.zoomTo()
  p.show()


def get_dihedral_indices(mol):
  # Find all dihedral angles (torsions)
  dihedralSmarts = '[!#1]~[!#1]~[!#1]~[!#1]'
  return [torsion for torsion in mol.GetSubstructMatches(Chem.MolFromSmarts(dihedralSmarts))]


def get_dihedral_angles(mol):
  # Calculate all dihedral angles
  return np.array([
    rdMolTransforms.GetDihedralDeg(mol.GetConformer(), dihedral[0], dihedral[1], dihedral[2], dihedral[3])
    for dihedral in get_dihedral_indices(mol)
  ])


def set_coords(mol, coords: torch.tensor):
  new_mol = deepcopy(mol)
  conf = Chem.rdchem.Conformer(new_mol.GetNumAtoms()) # create empty rdkit Conformer
  for i in range(new_mol.GetNumAtoms()):
    x,y,z = coords[i][0].item(), coords[i][1].item(), coords[i][2].item()
    conf.SetAtomPosition(i, Point3D(x,y,z))
  new_mol.RemoveAllConformers() # remove all present conformers
  new_mol.AddConformer(conf) # add conformer to mol
  return new_mol