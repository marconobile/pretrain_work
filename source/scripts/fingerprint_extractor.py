# import h5py
# from torch_scatter import scatter
from geqtrain.data import AtomicDataDict
import torch
import numpy as np
from pathlib import Path

class DescriptorWriter(object):
    #! I MUST HAVE BS = 1 TO WORK
    def __init__(self, multiplicity:int, out_dir:Path):
      self.multiplicity = multiplicity
      self.id = 0
      self.out_dir = out_dir

    @torch.no_grad()
    def __call__(self, pbar, out, ref_data, **kwargs):
      '''
        the out is the data obj modified inplace by the nn
        nb local interaction atm supports only scalar features
        feats are saved at node-lvl
      '''
      #! assert len(out['batch'].unique().tolist())==1

      local_scalar_features_only = out[AtomicDataDict.NODE_ATTRS_KEY][...,:self.multiplicity].cpu().numpy()
      global_scalar_features_only = out[AtomicDataDict.NODE_OUTPUT_KEY][...,:self.multiplicity].cpu().numpy()
      global_equivariant_features_only = out[AtomicDataDict.NODE_OUTPUT_KEY][...,self.multiplicity:].cpu().numpy()
      target = ref_data[AtomicDataDict.GRAPH_OUTPUT_KEY].cpu().numpy()
      path_to_file = self.out_dir / f"mol_{self.id}"
      self.id+=1
      np.savez(
        path_to_file,
        local_scalar_features_only=local_scalar_features_only,
        global_scalar_features_only=global_scalar_features_only,
        global_equivariant_features_only=global_equivariant_features_only,
        target=target,
      )

