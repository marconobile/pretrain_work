'''
Using CONFORGE: https://pubs.acs.org/doi/10.1021/acs.jcim.3c00563
https://cdpkit.org/cdpl_python_tutorial/cdpl_python_tutorial.html#generating-conformer-ensembles
'''
import os
import sys
import rdkit
import numpy as np

import CDPL.Chem as Chem
import CDPL.ConfGen as ConfGen
import CDPL.Chem as CDPLChem

from source.utils.mol_utils import get_dihedral_angles


def generateConformationEnsembles(mol: Chem.BasicMolecule, conf_gen: ConfGen.ConformerGenerator):
  """
  Generates a conformation ensemble for the argument molecule using the provided initialized ConfGen.ConformerGenerator instance.

  Parameters:
  - mol (Chem.BasicMolecule): Molecule to generate a conformation ensemble for.
  - conf_gen (ConfGen.ConformerGenerator): Instance of the ConfGen.ConformerGenerator class.

  Returns:
  - int: Status code indicating the success of the conformation ensemble generation.
  - int: Number of generated conformers.
  """
  # prepare the molecule for conformer generation
  ConfGen.prepareForConformerGeneration(mol)

  # generate the conformer ensemble
  status = conf_gen.generate(mol)
  num_confs = conf_gen.getNumConformers()

  # if sucessful, store the generated conformer ensemble as
  # per atom 3D coordinates arrays (= the way conformers are represented in CDPKit)
  if status == ConfGen.ReturnCode.SUCCESS or status == ConfGen.ReturnCode.TOO_MUCH_SYMMETRY: conf_gen.setConformers(mol)
  else: num_confs = 0

  # return status code and the number of generated conformers
  return (status, num_confs)


def generate_conformers(smi, N:int=10):
  '''
  N: num of conformers to select
  '''
  def get_conformer_generator():
    # Settings
    max_time = 36000 # Max. allowed molecule processing time in seconds (default: 3600 sec)
    min_rmsd = 0.5 # Output conformer RMSD threshold (default: 0.5)
    e_window = 20.0 # Output conformer energy window (default: 20.0)
    max_confs = 100 # Max. output ensemble size (default: 100)

    # create and initialize an instance of the class ConfGen.ConformerGenerator which
    # will perform the actual conformer ensemble generation work
    conf_gen = ConfGen.ConformerGenerator()
    conf_gen.settings.timeout = max_time * 1000          # apply the -t argument
    conf_gen.settings.minRMSD = min_rmsd                 # apply the -r argument
    conf_gen.settings.energyWindow = e_window            # apply the -e argument
    conf_gen.settings.maxNumOutputConformers = max_confs # apply the -n argument
    return conf_gen

  mol = CDPLChem.parseSMILES(smi)
  try:
      # generate conformer ensemble for read molecule
      status, num_confs = generateConformationEnsembles(mol, get_conformer_generator())

      # output generated ensemble (if available)
      if num_confs > 0:
          tmp_name = "./tmp.sdf"
          writer = Chem.MolecularGraphWriter(tmp_name)
          if not writer.write(mol): sys.exit('Error: output of conformer ensemble for molecule %s failed' % smi)
  except Exception as e: sys.exit('Error: conformer ensemble generation or output for molecule %s failed: %s' % (smi, str(e)))

  writer.close()
  tmp_mol_ensemble = rdkit.Chem.rdmolfiles.SDMolSupplier(tmp_name, removeHs=False)
  tmp_mol_ensemble = list(tmp_mol_ensemble)

  # try to get conf, if error then drop the mol, careful with indices
  mol_ensemble = [m for m in tmp_mol_ensemble if m.GetConformer() !=-1]
  print(len(mol_ensemble))
  if len(mol_ensemble) < N: return mol_ensemble

  dihedral_fingerprints = [get_dihedral_angles(mol) for mol in mol_ensemble]

  # Compute the dot product matrix
  n = len(dihedral_fingerprints)
  dot_product_matrix = np.zeros((n, n))
  for i in range(n):
      for j in range(i + 1, n):
          dot_product_matrix[i, j] = np.dot(dihedral_fingerprints[i], dihedral_fingerprints[j])
          dot_product_matrix[j, i] = dot_product_matrix[i, j]

  # Get the indices of the 5 lowest values
  sum_of_dot_prods = dot_product_matrix.sum(-1)
  sorted_indices = np.argsort(sum_of_dot_prods)
  lowest_indices = sorted_indices[:N]

  # remove the just created sdf
  if os.path.exists(tmp_name): os.remove(tmp_name)

  return [mol_ensemble[i] for i in lowest_indices]