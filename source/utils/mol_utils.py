
from rdkit import Chem
from rdkit.Chem import AllChem


def get_energy(mol):
  '''
  RDKit performs energy minimization of mol conformation using the Merck Molecular Force Field (MMFF94).
  kilocalories per mole (kcal/mol)  energies are reported in kcal mol-1.
  '''
  converged, energy = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=0)[0] # RETURNS: a list of (not_converged, energy) 2-tuples. If not_converged is 0 the optimization converged for that conformer.
  return energy


def minimize_energy(mol):
  '''
  RDKit performs energy minimization of mol conformation using the Merck Molecular Force Field (MMFF94).
  kilocalories per mole (kcal/mol)  energies are reported in kcal mol-1.
  mol is changed in place
  '''
  converged, energy = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=500)[0]
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
