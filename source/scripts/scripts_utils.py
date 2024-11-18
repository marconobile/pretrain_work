# for AccuracyMetric
from sklearn.metrics import (
  precision_recall_fscore_support,
  confusion_matrix,
  roc_auc_score,
  balanced_accuracy_score,
  f1_score,
)


# for DescriptorWriter
import h5py
from torch_scatter import scatter
from geqtrain.data import AtomicDataDict


def print_eval(gt, preds, logits):
  cm = confusion_matrix(gt, preds, labels=[False, True])
  tn, fp, fn, tp = cm.ravel()
  matrix_string = (
      f"Confusion Matrix:\n"
      f"                Predicted\n"
      f"                Positive     Negative\n"
      f"Actual Positive   TP: {tp}        FN: {fn}\n"
      f"       Negative   FP: {fp}        TN: {tn}\n"
  )
  print(matrix_string)
  print(f"Correcttly classified: {tn+tp}, misclassified: {fp+fn}")
  _roc_auc_score = roc_auc_score(gt, logits)
  print('_roc_auc_score: ', _roc_auc_score)
  precision, recall, fscore, support = precision_recall_fscore_support(gt, preds)
  print('precision: ', precision)
  print('recall: ', recall)
  print('fscore: ', fscore)
  print('support: ', support)
  ba = balanced_accuracy_score(gt, preds)
  f1 = f1_score(gt, preds, average='binary')
  print('balanced accuracy: ',ba)
  print('f1 score: ', f1)


class AccuracyMetric(object):
    def __init__(self, key:str):
        '''key: key to be taken from ref_data'''
        self.key = key
        self._gts_list, self._preds_list, self._logits_list = [], [], []

    def __call__(self, pbar, out, ref_data, **kwargs):
        target = ref_data[self.key].cpu().bool()
        _logits = out[self.key].sigmoid()
        prediction = (_logits>.5).cpu().bool()
        self._gts_list += target.squeeze().tolist()
        self._logits_list += _logits.squeeze().tolist()
        self._preds_list += prediction.squeeze().tolist()

    def print_current_result(self): print_eval(self._gts_list, self._preds_list, self._logits_list)

class DescriptorWriter(object):
    def __init__(self, feat_dim:int=32, normalize_wrt_atom_count:bool=False):
      self.normalize_wrt_atom_count = normalize_wrt_atom_count
      self.feat_dim = feat_dim
      self.field = AtomicDataDict.NODE_FEATURES_KEY
      self.out_field = AtomicDataDict.GRAPH_OUTPUT_KEY
      self.observations, self.gt = [] , []

    def __call__(self, pbar, out, ref_data, **kwargs):
      '''
      the inpt is the data obj modified by the nn
      '''
      graph_feature = scatter(out[self.field][...,:self.feat_dim], index = out['batch'], dim=0)

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

