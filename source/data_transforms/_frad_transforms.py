import torch
import numpy as np
from rdkit import Chem
from source.utils.mol_utils import smi_reader_params, get_energy, set_coords


def do_not_nosify_mol(data):
  data.noise_target = torch.zeros_like(data.pos, dtype=torch.float32)
  return data


def rotate_around_bond(coords, j, k, angle_diff_rad, atoms_to_rotate):
    """
    angle_diff_rad MUST be in RADIANS, since trigonometric functions (torch.sin, torch.cos) expect radians.

    Rotate the selected atoms (atoms_to_rotate) around the axis defined by atoms j and k
    to change the dihedral angle by angle_diff_rad (in radians).

    coords: Nx3 torch tensor
    j, k: indices of the atoms defining the rotation axis
    angle_diff_rad: the angle by which to rotate (in radians)
    atoms_to_rotate: indices of atoms to be rotated
    """
    p_j = coords[j]
    p_k = coords[k]

    # Define rotation axis (unit vector)
    axis = p_k - p_j
    axis = axis / torch.norm(axis)

    # Rodrigues' rotation formula
    ax, ay, az = axis
    c = torch.cos(angle_diff_rad)
    s = torch.sin(angle_diff_rad)
    R = torch.stack([
        torch.stack([c + ax*ax*(1-c),      ax*ay*(1-c) - az*s,  ax*az*(1-c) + ay*s]),
        torch.stack([ay*ax*(1-c) + az*s,   c + ay*ay*(1-c),     ay*az*(1-c) - ax*s]),
        torch.stack([az*ax*(1-c) - ay*s,   az*ay*(1-c) + ax*s,  c + az*az*(1-c)])
    ])

    for atom_idx in atoms_to_rotate:
        vec = coords[atom_idx] - p_j
        vec_rot = torch.matmul(R, vec)
        coords[atom_idx] = vec_rot + p_j
    return coords


def find_downstream_atoms(adjacency_matrix, j, k):
    """
    Given an adjacency matrix and a bond (j, k), find the downstream atoms starting from k
    after removing the bond (j, k). We assume that "downstream" means all atoms reachable
    from k without going back through j.

    adjacency_matrix: NxN boolean or 0/1 torch tensor
    j, k: indices of the pivot bond atoms
    """
    N = adjacency_matrix.size(0)
    # Copy adjacency to avoid permanent changes
    adj_copy = adjacency_matrix.clone()

    # Remove the j-k bond to prevent going "upstream"
    adj_copy[j, k] = False
    adj_copy[k, j] = False

    # BFS starting from k
    visited = torch.zeros(N, dtype=torch.bool)
    queue = [k]
    visited[k] = True

    while queue:
        current = queue.pop(0)
        # Explore neighbors
        neighbors = torch.nonzero(adj_copy[current], as_tuple=False).flatten()
        for nbr in neighbors:
            nbr = nbr.item()
            if not visited[nbr]:
                visited[nbr] = True
                queue.append(nbr)

    downstream_atoms = torch.nonzero(visited, as_tuple=False).flatten().tolist()
    return downstream_atoms


def set_dihedral_angle(coords, i, j, k, l, current_angle_deg, desired_angle_deg, adjacency_matrix):
  """
  Set the dihedral angle defined by atoms (i, j, k, l) to the desired_angle_deg.
  This function modifies coords in place.

  Using the adjacency_matrix to find all downstream atoms from the bond (j, k).
  """
  # Convert the difference to radians before performing the rotation
  angle_diff = desired_angle_deg - current_angle_deg
  angle_diff_rad = torch.deg2rad(torch.tensor(angle_diff, dtype=coords.dtype, device=coords.device))

  # Find downstream atoms
  downstream_atoms = find_downstream_atoms(adjacency_matrix, j, k)

  # Rotate all downstream atoms
  coords = rotate_around_bond(coords, j, k, angle_diff_rad, downstream_atoms)
  return coords


def apply_all_dihedrals(coords, rotable_bonds, original_dihedral_angles_degrees, desired_dihedral_angles_degrees, adjacency_matrix):
  """Apply all desired dihedral angles to the coordinates."""
  for (i, j, k, l), old_angle, new_angle in zip(rotable_bonds, original_dihedral_angles_degrees, desired_dihedral_angles_degrees):
    coords = set_dihedral_angle(coords, i, j, k, l, old_angle, new_angle, adjacency_matrix)
  return coords


def apply_dihedral_noise(data, scale):
  '''sample and apply dihedral noise'''
  rotable_bonds = data.rotable_bonds.tolist()
  if not rotable_bonds: return data
  original_dihedral_angles_degrees = data.dihedral_angles_degrees

  # sample and clip to [-180.0, +180.0]
  noise = np.random.normal(0, scale, size=original_dihedral_angles_degrees.shape)
  noise = np.clip(noise, a_min=-180.0, a_max=180.0)

  noised_dihedral_angles_degrees = original_dihedral_angles_degrees + noise
  coords = apply_all_dihedrals(data.pos,
                                rotable_bonds,
                                original_dihedral_angles_degrees,
                                noised_dihedral_angles_degrees.float(),
                                data.adj_matrix,
                              )
  return coords


def apply_coords_noise(coords, coords_noise_tau):
  '''sample and return coords noise'''
  coords_noise_to_be_predicted = np.random.normal(0, 1, size=coords.shape) * coords_noise_tau
  update_coords = torch.tensor(coords + coords_noise_to_be_predicted, dtype=torch.float)
  return update_coords, coords_noise_to_be_predicted


def frad(data, coords_noise_tau=0.04, dihedral_scale=40.0, k=2): #! impo hyperparams here
  #! this function modifies data inplace
  #TODO: make multiscale via partial
  while True:
    # sample from starting-conformer
    coords = apply_dihedral_noise(data, dihedral_scale)
    mol = Chem.MolFromSmiles(data.smiles, smi_reader_params()) # to be kept inside the while to avoid "energy" locks
    mol = set_coords(mol, coords)
    # if energy above T: resample
    if get_energy(mol) <= k*data.max_energy: break

  update_coords, coords_noise_to_be_predicted = apply_coords_noise(coords, coords_noise_tau)
  data.pos = update_coords
  data.noise_target = torch.tensor(coords_noise_to_be_predicted, dtype=torch.float)
  return data