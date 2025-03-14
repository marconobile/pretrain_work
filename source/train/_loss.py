""" Adapted from https://github.com/mir-group/nequip
"""
import torch.nn
import torch.nn.functional as F
from torch_scatter import scatter_mean # scatter
from torcheval.metrics import BinaryAccuracy as TorchBinaryAccuracy


def ensemble_predictions_and_targets(predictions, targets, ensemble_indices):
    ''' checks whether field has already been ensembled, if not, ensembles it using ensemble_indices'''
    unique_ensembles = torch.unique(ensemble_indices)

    # if unique_ensembles.dim() != predictions.dim(): #TODO: this was written since it could be that it breaks if BS=1; to be checked
    #     predictions = predictions.unsqueeze(0)
    #     targets = targets.unsqueeze(0)

    # ensemble predictions
    is_input_already_ensembled = unique_ensembles.shape[0] == predictions.shape[0]
    if not is_input_already_ensembled:
        predictions = scatter_mean(predictions, ensemble_indices)

    # ensemble targets
    is_output_already_ensembled = unique_ensembles.shape[0] == targets.shape[0]
    if not is_output_already_ensembled:
        targets = scatter_mean(targets, ensemble_indices) # acts just as selection and ordering wrt unique_ensembles

    return predictions, targets


# class EnsembleLoss:
#     def __call__(
#         self,
#         pred:dict,
#         ref:dict,
#         predictions:torch.Tensor,
#         targets:torch.Tensor,
#     ):
#         assert 'ensemble_index' in pred
#         assert 'ensemble_index' in ref
#         predictions_ensembled, targets_ensembled = ensemble_predictions_and_targets(predictions, targets, pred['ensemble_index'])
#         return predictions_ensembled, targets_ensembled


class FocalLossBinaryAccuracy:
    def __init__(
        self,
        func_name: str,
        params: dict = {},
        **kwargs,
    ):
        '''
        alpha is a number between 0 and 1
        If alpha is 0.25, the loss for positive examples (target is 1) is multiplied by 0.25,
        and the loss for negative examples (target is 0) is multiplied by 0.75 (since 1-0.25=0.75).
        Effect: less weight to positive class and more weight to negative class, useful when the positive class is over-represented.

        gamma: purpose: focus more on hard-to-classify examples by reducing the relative loss for well-classified examples.
        higher gamma higher focus to hard-to-classify examples i.e. examples on which the net is not so confident.
        scalses up the loss value when the net is not confident in the correct class
        '''

        self.func_name = "FocalLossBinaryAccuracy"
        self.alpha: float = params.get('alpha', 0.85)
        self.gamma: float = params.get('gamma', 2)
        assert (0<self.alpha and self.alpha<1)

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        **kwargs,
    ):
        logits = pred[key].squeeze()
        targets_binary = ref[key].squeeze()
        p = logits.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets_binary, reduction="none")

        p_t = p * targets_binary + (1 - p) * (1 - targets_binary)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets_binary + (1 - self.alpha) * (1 - targets_binary)
            loss = alpha_t * loss

        return loss.mean() if mean else loss


class BinaryAccuracy:
    def __init__(
        self,
        func_name: str,
        params: dict = {},
        **kwargs,
    ):
        self.params = params
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.func_name = "BinaryAccuracy"
        # self.treshold_for_positivity = .5
        self.metric = TorchBinaryAccuracy()

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        **kwargs,
    ):
        if mean:
            raise(f"{__class__.__name__} cannot be used as loss function for training")

        logits = pred[key].squeeze()
        targets_binary = ref[key].squeeze()
        self.metric.update(logits.softmax(-1).argmax(-1),targets_binary)
        acc = self.metric.compute()
        self.metric.reset() # reset at each batch
        return acc.to(logits.device)

        # if 'ensemble_index' in pred:
        #     assert 'ensemble_index' in ref
        #     logits, targets_binary = ensemble_predictions_and_targets(logits, targets_binary, pred['ensemble_index'])

        # binarized_predictions = (logits.sigmoid()<self.treshold_for_positivity).float().reshape(*targets_binary.shape)
        # return torch.abs(targets_binary - binarized_predictions)


class EnsembleBCEWithLogitsLoss:
    def __init__(
        self,
        func_name: str,
        params: dict = {},
        **kwargs,
    ):
        self.params = params
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.func_name = "EnsembleBCEWithLogitsLoss"
        self.treshold_for_positivity = .5

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        **kwargs,
    ):
        logits = pred[key].squeeze()
        targets_binary = ref[key].squeeze()

        if 'ensemble_index' in pred:
            assert 'ensemble_index' in ref
            logits, targets_binary = ensemble_predictions_and_targets(logits, targets_binary, pred['ensemble_index'])

        return F.binary_cross_entropy_with_logits(
            logits,
            targets_binary,
            reduction="mean" if mean else "none"
        )


class OLDBinaryAccuracy:
    def __init__(
        self,
        func_name: str,
        params: dict = {},
        **kwargs,
    ):
        self.params = params
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.func_name = "BinaryAccuracy"
        self.treshold_for_positivity = .5

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        **kwargs,
    ):
        if mean:
            raise(f"{__class__.__name__} cannot be used as loss function for training")

        logits = pred[key].squeeze()
        targets_binary = ref[key].squeeze()

        # if 'ensemble_index' in pred:
        #     assert 'ensemble_index' in ref
        #     logits, targets_binary = ensemble_predictions(logits, targets_binary, pred['ensemble_index'])
        try:
            binarized_predictions = (logits.sigmoid()<self.treshold_for_positivity).float().reshape(*targets_binary.shape)
        except:
            binarized_predictions = (logits.sigmoid()<self.treshold_for_positivity).float()
        return torch.abs(targets_binary - binarized_predictions)