# for AccuracyMetric
from sklearn.metrics import (
  precision_recall_fscore_support,
  confusion_matrix,
  roc_auc_score,
  balanced_accuracy_score,
  f1_score,
)

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

