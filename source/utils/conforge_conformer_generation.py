'''
Using CONFORGE: https://pubs.acs.org/doi/10.1021/acs.jcim.3c00563
https://cdpkit.org/cdpl_python_tutorial/cdpl_python_tutorial.html#generating-conformer-ensembles
'''
import warnings
import os
import sys
from rdkit import Chem as rdChem
import numpy as np
import tempfile
import CDPL.Chem as Chem
import CDPL.ConfGen as ConfGen
import CDPL.Chem as CDPLChem
from source.utils.mol_utils import get_dihedral_angles, get_rdkit_conformer, minimize_energy, smi_reader_params
from source.utils.rdkit_conformer_generation import *
from globals import *


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


def get_conformer_generator(max_time:int=960000):
    '''
    Settings
    max_confs: Max. output ensemble size
    max_time:  Max. allowed molecule processing time in seconds (default: 3600 sec)
    min_rmsd:  Output conformer RMSD threshold (default: 0.5):
      if (conformer_rmsd_wrt_og < min_rmsd) then drop conformer; if not distant enough the drop
      increasing min_rmsd thus variance is increased
    e_window:  Output conformer energy window (default: 20.0)

    Create and initialize an instance of the class ConfGen.ConformerGenerator which
    will perform the actual conformer ensemble generation work

    The parameters min_rmsd, e_window, and max_confs are used to control the generated conformer ensemble's diversity, energy and size.
    https://cdpkit.org/v1.1.1/applications/confgen.html
    '''
    conf_gen = ConfGen.ConformerGenerator()
    conf_gen.settings.timeout = max_time
    conf_gen.settings.minRMSD = min_rmsd
    conf_gen.settings.energyWindow = e_window
    conf_gen.settings.maxNumOutputConformers = max_confs
    # conf_gen.settings.setSamplingMode(2)
    # AUTO       = 0;
    # SYSTEMATIC = 1;
    # STOCHASTIC = 2;
    return conf_gen


def generate_conformers_rdkit(smi:str):
  mol = rdChem.MolFromSmiles(smi, smi_reader_params())
  mol, filtered_conformers = rdkit_generate_conformers(mol)
  if not mol: return []

  with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
    tmp_name = tmp_file.name + ".sdf"
    rdkit_save_conformers_to_sdf(mol, tmp_name, filtered_conformers)
    tmp_mol_ensemble = list(rdChem.rdmolfiles.SDMolSupplier(tmp_name, removeHs=False))

  return load_conformers_from_sdf(tmp_mol_ensemble, n_confs_to_keep)



def generate_conformers(smi:str, conf_gen:ConfGen.ConformerGenerator):
  mol = CDPLChem.parseSMILES(smi)
  status, num_confs = generateConformationEnsembles(mol, conf_gen)
  if num_confs == 0:
    warnings.warn(f"no conformers generated for {smi} via CONFORGE: {status_to_str[status]}")
    return []

  with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
    tmp_name = tmp_file.name + ".sdf"
    writer = Chem.MolecularGraphWriter(tmp_name)
    if not writer.write(mol):
      writer.close()
      sys.exit('Error: output of conformer ensemble for molecule %s failed' % smi)
    writer.close()
    tmp_mol_ensemble = list(rdChem.rdmolfiles.SDMolSupplier(tmp_name, removeHs=False))

  return load_conformers_from_sdf(tmp_mol_ensemble, n_confs_to_keep)


def load_conformers_from_sdf(mol_ensamble, keep_n:int):
  mol_ensemble = [m for m in mol_ensamble if m.GetConformer() !=-1]
  if len(mol_ensemble) < keep_n: return mol_ensemble

  # Compute the dot product matrix between confs dihedrals-vec as descriptors
  dihedral_fingerprints = [get_dihedral_angles(mol) for mol in mol_ensemble]
  n = len(dihedral_fingerprints)
  dot_product_matrix = np.zeros((n, n))
  for i in range(n):
    for j in range(i + 1, n):
      dot_product_matrix[i, j] = np.dot(dihedral_fingerprints[i], dihedral_fingerprints[j])
      dot_product_matrix[j, i] = dot_product_matrix[i, j]

  # Get the indices of the N lowest values (most unique)
  sum_of_dot_prods = dot_product_matrix.sum(-1)
  sorted_indices = np.argsort(sum_of_dot_prods)
  lowest_indices = sorted_indices[:keep_n]
  return [mol_ensemble[i] for i in lowest_indices]