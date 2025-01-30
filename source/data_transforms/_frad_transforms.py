import torch
import numpy as np
from scipy.spatial.distance import pdist
from copy import deepcopy

from geqtrain.data.AtomicData import AtomicData

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
        torch.stack([c + ax*ax*(1-c),      ax*ay *
                    (1-c) - az*s,  ax*az*(1-c) + ay*s]),
        torch.stack([ay*ax*(1-c) + az*s,   c + ay *
                    ay*(1-c),     ay*az*(1-c) - ax*s]),
        torch.stack([az*ax*(1-c) - ay*s,   az*ay *
                    (1-c) + ax*s,  c + az*az*(1-c)])
    ])

    # for atom_idx in atoms_to_rotate:
    #     vec = coords[atom_idx] - p_j
    #     vec_rot = torch.matmul(R.float(), vec)
    #     coords[atom_idx] = vec_rot + p_j
    coords[atoms_to_rotate] = (coords[atoms_to_rotate] - p_j) @ R.float().T + p_j


def find_downstream_atoms(adjacency_matrix, j, k):
    """
    Given an adjacency matrix and a bond (j, k), find the downstream atoms starting from k
    after removing the bond (j, k). We assume that "downstream" means all atoms reachable
    from k without going back through j.

    adjacency_matrix: NxN boolean or 0/1 torch tensor
    j, k: indices of the pivot bond atoms
    """
    N = adjacency_matrix.shape[0]

    # Remove the j-k bond to prevent going "upstream"
    adjacency_matrix[j, k] = False
    adjacency_matrix[k, j] = False

    # BFS starting from k
    visited = torch.zeros(N, dtype=torch.bool)
    queue = [k]
    visited[k] = True

    while queue:
        current = queue.pop(0)
        # Explore neighbors
        neighbors = torch.nonzero(torch.tensor(adjacency_matrix[current]), as_tuple=False).flatten()
        for nbr in neighbors:
            nbr = nbr.item()
            if not visited[nbr]:
                visited[nbr] = True
                queue.append(nbr)

    # downstream_atoms
    return torch.nonzero(visited, as_tuple=False).flatten().tolist()


def set_dihedral_angle_(coords, j, k, current_angle_deg, desired_angle_deg, adjacency_matrix):
    """
    Set the dihedral angle defined by atoms (i, j, k, l) to the desired_angle_deg.
    This function modifies coords in-place.
    Using the adjacency_matrix to find all downstream atoms from the bond (j, k).
    """
    # Convert the difference to radians before performing the rotation, since Rodrigues' rotation formula in rotate_around_bond(...) requires angle in radiants
    angle_diff_rad = torch.deg2rad(torch.tensor(desired_angle_deg - current_angle_deg))
    # find all atoms moved by the single rotation of torsional
    downstream_atoms = find_downstream_atoms(adjacency_matrix, j, k)
    rotate_around_bond(coords, j, k, angle_diff_rad, downstream_atoms)

# https://openkim.org/files/MO_959249795837_003/LennardJones612_UniversalShifted.params : 2**(1/6) * 0.5523570 = 0.62
def apply_dihedral_noise_(data:AtomicData, dihedral_scale: float = 20.0, min_interatomic_dist_required: float = 0.62):
    '''
    dihedral_scale: std of norm dist from which to sample the noise to add at torsional angles
    '''
    if not data.rotable_bonds.size:
        return data

    original_dihedral_angles_degrees = data.dihedral_angles_degrees
    noise = np.random.normal(0, dihedral_scale, size=original_dihedral_angles_degrees.shape)
    desired_dihedral_angles_degrees = original_dihedral_angles_degrees + noise

    og_pos = data.pos.clone()
    for (_, j, k, _), old_angle, new_angle in zip(data.rotable_bonds, original_dihedral_angles_degrees, desired_dihedral_angles_degrees):
        set_dihedral_angle_(data.pos, j, k, old_angle, new_angle, data.adj_matrix)

    if np.min(pdist(data.pos, 'euclidean')) <= min_interatomic_dist_required:
        data.pos = og_pos # undo


def apply_coords_noise_(data:AtomicData):
    '''sample and return coords noise
    std of noise is unformly sampled between min:float=0.004, max:float=0.3
    '''
    data.noise_target = torch.from_numpy(np.random.normal(0, 1, size=data.pos.shape) * np.random.uniform(0.004, 0.3)).to(torch.float32)
    data.pos += data.noise_target

def frad(data:AtomicData, add_coords_noise: bool = True):

    data = deepcopy(data) #! in-place op to the data obj persist thru dloader iterations
    apply_dihedral_noise_(data)

    if not add_coords_noise:
        data.noise_target = torch.zeros(data.pos, dtype=torch.float32)
        return data

    apply_coords_noise_(data)
    return data
