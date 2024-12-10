import torch
from itertools import combinations


def pairwise_distances(coords):
    # Get all pairs of points
    indices = list(combinations(range(len(coords)), 2))
    max_distance = 0
    for i, j in indices:
        distance = torch.dist(coords[i], coords[j]).item()
        if distance > max_distance:
            max_distance = distance
    return max_distance


def max_intra_graph_distance(data_list):
    max_distance_overall = 0
    for data in data_list:
        coords = data.pos
        max_distance = pairwise_distances(coords)
        if max_distance > max_distance_overall:
            max_distance_overall = max_distance

    return max_distance_overall


def max_intra_graph_distance_list(data_list):
    max_distances = []
    for data in data_list:
        coords = data.pos
        max_distance = pairwise_distances(coords)
        max_distances.append(max_distance)

    max_distances.sort(reverse=True)
    return max_distances


def max_intra_graph_distance_list_with_idxs(data_list):
    max_distances_with_indices = []
    for idx, data in enumerate(data_list):
        coords = data.pos
        max_distance = pairwise_distances(coords)
        max_distances_with_indices.append((idx, max_distance))

    max_distances_with_indices.sort(key=lambda x: x[1], reverse=True)
    return max_distances_with_indices


def average_neighbors_within_cutoff(data_list, cutoff=20):
    total_neighbors_within_cutoff = 0
    total_nodes = 0

    for data in data_list:
        coords = data.pos
        num_nodes = coords.size(0)

        if num_nodes == 0:
            continue

        neighbors_within_cutoff = torch.zeros(num_nodes)

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    distance = torch.dist(coords[i], coords[j]).item()
                    if distance <= cutoff:
                        neighbors_within_cutoff[i] += 1

        total_neighbors_within_cutoff += neighbors_within_cutoff.sum().item()
        total_nodes += num_nodes

    if total_nodes == 0:
        return 0

    average_neighbors_within_cutoff_per_node = total_neighbors_within_cutoff / total_nodes
    return average_neighbors_within_cutoff_per_node