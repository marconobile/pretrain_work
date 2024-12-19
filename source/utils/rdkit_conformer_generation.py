from rdkit import Chem as rdChem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import SDWriter
from source.utils.mol_utils import get_rdkit_conformer
import warnings

def rdkit_generate_conformers(mol, num_conformers=10, prune_rms_thresh=0.5, energy_threshold=50.0):
    """
    Generate and optimize conformers for a molecule.

    Parameters:
    - mol: rdkit mol
    - num_conformers: Number of conformers to generate
    - prune_rms_thresh: RMS threshold for pruning conformers: If the RMSD between the new conformer and any existing conformer is less than the pruneRmsThresh value,
        the new conformer is discarded as it is considered redundant.
    - energy_threshold: Energy threshold for filtering conformers: if energy of mol is greater then energy_threshold + min energy of ensemble then discard mol

    Returns:
    - List of conformers (RDKit molecule objects)
    """

    # Add hydrogens
    mol = rdChem.AddHs(mol, addCoords=True)
    # Get conformer
    mol.RemoveAllConformers() # remove all present conformers
    conf = get_rdkit_conformer(mol)
    if not conf:
      warnings.warn(f"rdkit fallback failed aswell, dropping molecule")
      return None, None

    # Generate initial conformers
    params = AllChem.ETKDGv3()
    params.pruneRmsThresh = prune_rms_thresh
    AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)

    # Optimize conformers and calculate energies
    conformer_energies = []
    for conf_id in range(mol.GetNumConformers()): # TODO replace with MMFFOptimizeMoleculeConfs
        # Optimize the conformation
        AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)

        # Calculate the energy
        energy = AllChem.MMFFGetMoleculeForceField(mol, confId=conf_id).CalcEnergy()
        conformer_energies.append((conf_id, energy))

    if not conformer_energies: return None, None
    # Filter out high-energy conformers
    min_energy = min(energy for _, energy in conformer_energies)
    filtered_conformers = [conf_id for conf_id, energy in conformer_energies if energy - min_energy < energy_threshold]
    return mol, filtered_conformers

def rdkit_save_conformers_to_sdf(mol, filename, filtered_conformers:list=[]):
    """
    Save conformers to an SDF file.

    Parameters:
    - mol: RDKit molecule object
    - conformer_ids: List of conformer IDs to save
    - filename: Output SDF file name
    """
    writer = SDWriter(filename)
    if filtered_conformers:
      for id in filtered_conformers:
        writer.write(mol, confId=id)
    else: writer.write(mol)
    writer.close()


# conformer_energies, converged = optimize_conformers(mol)
#   if not conformer_energies: return None, None

#   # Filter out high-energy conformers
#   min_energy = min(conformer_energies)
#   filtered_conformers = [i for i, energy in enumerate(conformer_energies) if energy - min_energy < energy_threshold]