import h5py
from torch_scatter import scatter
from geqtrain.data import AtomicDataDict
import torch

class DescriptorWriter(object):
    def __init__(self,
                 feat_dim:int=32,
                 normalize_wrt_atom_count:bool=False,
                 only_scalars:bool=True,
                 cat_local_features:bool=False
                ):

      self.only_scalars = only_scalars
      self.normalize_wrt_atom_count = normalize_wrt_atom_count
      self.feat_dim = feat_dim
      self.context_aware_interaction_field = AtomicDataDict.NODE_OUTPUT_KEY
      self.cat_local_features = cat_local_features
      self.local_interaction = AtomicDataDict.NODE_ATTRS_KEY

      self.out_field = AtomicDataDict.GRAPH_OUTPUT_KEY
      self.observations, self.gt = [] , []

    @torch.no_grad()
    def __call__(self, pbar, out, ref_data, **kwargs):
      '''
      the inpt is the data obj modified by the nn
      '''
      if self.only_scalars:
        # self.feat_dim nb feat_dim is latent_dim
        graph_feature = scatter(out[self.context_aware_interaction_field][...,:self.feat_dim], index = out['batch'], dim=0)
        if self.cat_local_features:
          from_local = scatter(out[self.local_interaction][...,:self.feat_dim], index = out['batch'], dim=0)
          graph_feature =  torch.cat((from_local, graph_feature), dim=-1)
      else:
        # self.feat_dim + 3* self.feat_dim nb feat_dim is latent_dim
        graph_feature = scatter(out[self.context_aware_interaction_field], index = out['batch'], dim=0)


      if self.normalize_wrt_atom_count:
          _, counts = out['batch'].unique(return_counts=True)
          graph_feature /=counts

      self.observations.append(graph_feature.cpu())
      self.gt.append(out[self.out_field].cpu())

    def write_batched_obs_to_file(self, n_batches, filename:str='./dataset.h5'):
      # todo, here also write smile inside h5
      dset_id = 0
      with h5py.File(filename, 'w') as h5_file:
        for batch_idx in range(n_batches):
          obs_batch, gt_batch = self.observations[batch_idx], self.gt[batch_idx]
          for obs_idx in range(obs_batch.shape[0]): # expected bs first
            obs, gt = obs_batch[obs_idx], gt_batch[obs_idx]
            h5_file.create_dataset(f'observation_{dset_id}', data=obs.numpy())
            h5_file.create_dataset(f'ground_truth_{dset_id}', data=gt.numpy())
            dset_id+=1

