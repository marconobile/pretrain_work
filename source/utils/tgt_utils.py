# taken from:
# https://github.com/shamim-hussain/tgt/blob/master/lib/training_schemes/pcqm/commons.py
import json
import os.path as osp
import torch
import torch.nn.functional as F

@torch.jit.script
def coords2dist(coords):
    return torch.norm(coords.unsqueeze(-2) - coords.unsqueeze(-3), dim=-1)

def add_coords_noise(coords, noise_level:float=0.2, noise_smoothing:float=1.0):
    noise = coords.new(coords.size()).normal_(0, noise_level)
    dist_mat = coords2dist(coords)
    smooth_mat = torch.softmax(-dist_mat/noise_smoothing, -1) 
    noise = torch.matmul(smooth_mat, noise)
    new_coords = coords + noise
    return new_coords, noise


def discrete_dist(dist, num_bins, range_bins):
    # Converts continuous distances to discrete bins
    # Clamps to valid bin range
    # Example: For 512 bins and 8 Å range, a 4 Å distance becomes bin 255 (i.e. half-way)
    dist = dist * ((num_bins - 1) / range_bins) # ((num_bins - 1) / range_bins): computes how many bins are in 1 angstrom
    dist = dist.long().clamp(0, num_bins - 1)
    return dist

def round_to_closest_power_of_2(n):
    """
    Rounds the input integer to the closest power of 2.
    If two powers of 2 are equally distant, returns the higher one.

    Args:
        n: A positive integer

    Returns:
        The closest power of 2 to n
    """
    # Handle edge cases
    if n <= 0:
        raise ValueError("Input must be a positive integer")
    if n == 1:
        raise ValueError("Cannot have only 1 bin")

    # Find the power of 2 immediately lower than or equal to n
    lower_power = 1
    while lower_power * 2 <= n:
        lower_power *= 2

    # Find the power of 2 immediately higher than n
    higher_power = lower_power * 2

    # Determine which power of 2 is closer to n
    if higher_power - n < n - lower_power:
        return int(higher_power)
    else:
        return int(lower_power)


class DiscreteDistLoss:
    def __init__(
            self,
            func_name: str,
            params: dict = {},
            **kwargs,
        ):
        '''
        num_bins: Number of discrete bins to categorize distances (default=512 as in https://github.com/shamim-hussain/tgt/tree/master/configs/pcqm)
        rmax: The maximum distance range in Ångströms (default=8 Å from same source above)
        '''
        self.params = params
        for key, value in kwargs.items():
            setattr(self, key, value)
        for key, value in params.items():
            setattr(self, key, value)
        self.func_name = "DiscreteDistLoss"
        self.num_bins = round_to_closest_power_of_2(self.rmax/self.resolution)
        self.desired_resolution = self.resolution
        self.actual_resolution = self.rmax/self.num_bins

    def __call__(
            self,
            pred: dict,
            ref: dict,
            key: str,
            mean: bool = True,
            **kwargs,
        ):
        '''
        dist_logits: Model's predicted logits for distance bins (shape: [batch_size, num_edges, num_bins])
        dist_targ: Target distances (from coords2dist, shape: [batch_size, num_edges])
        mask: Edge mask indicating which distances to consider (shape: [batch_size, num_edges])
        '''
        dist_logits = pred[key]
        dist_targ = coords2dist(ref['pos']) # on pos matrix(N,3)
        edge_index = pred['edge_index']
        dist_targ = dist_targ[edge_index[0], edge_index[1]]
        dist_targ = discrete_dist(dist_targ, self.num_bins, self.rmax)
        return F.cross_entropy(dist_logits, dist_targ, reduction="mean" if mean else "none")



class BinsProcessor:
    def __init__(self, path,
                 shift_half=True,
                 zero_diag=True,):
        self.path = path
        self.shift_half = shift_half
        self.zero_diag = zero_diag

        self.data_path = osp.join(path, 'data')
        self.meta_path = osp.join(path, 'meta.json')

        with open(self.meta_path, 'r') as f:
            self.meta = json.load(f)

        self.num_samples = self.meta['num_samples']
        self.num_bins = self.meta['num_bins']
        self.range_bins = self.meta['range_bins']

        self.bin_size = self.range_bins / (self.num_bins-1)

    def bins2dist(self, bins):
        bins = bins.float()
        if self.shift_half:
            bins = bins + 0.5
        dist = bins * self.bin_size
        dist = dist + dist.transpose(-2,-1)
        if self.zero_diag:
            dist = dist * (1 - torch.eye(dist.size(-1),
                                         dtype=dist.dtype,
                                         device=dist.device))
        return dist


# ORIGINAL
# class DiscreteDistLoss:
#     def __init__(
#             self,
#             func_name: str,
#             params: dict = {},
#             num_bins:int=256,
#             range_bins:float=8.0,
#             **kwargs,
#         ):
#         '''
#         num_bins: Number of discrete bins to categorize distances (default=512 as in https://github.com/shamim-hussain/tgt/tree/master/configs/pcqm)
#         range_bins: The maximum distance range in Ångströms (default=8 Å from same source above)
#         '''
#         self.params = params
#         for key, value in kwargs.items():
#             setattr(self, key, value)
#         self.func_name = "DiscreteDistLoss"

#         self.num_bins = num_bins
#         self.range_bins = range_bins

#     def __call__(
#             self,
#             pred: dict,
#             ref: dict,
#             key: str,
#             mean: bool = True,
#             **kwargs,
#         ):
#         '''
#         dist_logits: Model's predicted logits for distance bins (shape: [batch_size, num_edges, num_bins])
#         dist_targ: Target distances (from coords2dist, shape: [batch_size, num_edges])
#         mask: Edge mask indicating which distances to consider (shape: [batch_size, num_edges])
#         '''
#         dist_logits = pred[key]
#         dist_targ = coords2dist(ref['pos']) # on pos matrix(N,3)
#         edge_index = pred['edge_index']
#         dist_targ = dist_targ[edge_index[0], edge_index[1]]
#         dist_targ = discrete_dist(dist_targ, self.num_bins, self.range_bins)
#         return F.cross_entropy(dist_logits, dist_targ, reduction="mean" if mean else "none")