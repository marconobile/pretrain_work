""" Adapted from https://github.com/mir-group/nequip
"""

import logging
import inspect
import torch.nn

# import torch.functional as F

from typing import Dict
from importlib import import_module
from torch_scatter import scatter, scatter_mean

import torcheval
from torcheval.metrics.functional import binary_precision

from geqtrain.utils import instantiate_from_cls_name
from geqtrain.data import AtomicDataDict

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
        key: str, # key to be used to select element in DataDict
        mean: bool = True,
        **kwargs,
    ):
        inputs = pred[key].squeeze()
        targets = ref[key].squeeze()
        p = inputs.sigmoid()
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if mean: loss = loss.mean()
        return loss


class BinaryAccuracy:
    def __init__(
        self,
        func_name: str,
        params: dict = {},
        **kwargs,
    ):
        self.func_name = "BinaryAccuracy"
        self.treshold_for_positivity = .5

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str, # key to be used to select element in DataDict
        mean: bool = True,
        **kwargs,
    ):
        if mean:
            raise(f"{__class__.__name__} cannot be used as loss function for training")

        targets = ref[key]
        predictions = (pred[key].sigmoid()<self.treshold_for_positivity).float().reshape(*targets.shape)
        return torch.abs(targets - predictions)

        # accuracy = torcheval.metrics.functional.binary_accuracy(predictions.squeeze(), targets.squeeze())
        # return torch.full(targets.shape, accuracy, device=targets.device)

class Precision:
    def __init__(
        self,
        func_name: str,
        params: dict = {},
    ):
        self.treshold_for_positivity = .5

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str, # key to be used to select element in DataDict
        mean: bool = True,
    ):
        if mean:
            raise(f"{__class__.__name__} cannot be used as loss function for training")

        targets = ref[key]
        predictions = (pred[key].sigmoid()>self.treshold_for_positivity).float()
        # precision = torcheval.metrics.functional.binary_precision(predictions.squeeze(), targets.squeeze())
        # return torch.full(targets.shape, precision, device=targets.device)
        accuracy = torcheval.metrics.functional.binary_accuracy(predictions.squeeze(), targets.squeeze())
        return torch.full(targets.shape, accuracy, device=targets.device)

class Recall:
    def __init__(
        self,
        func_name: str,
        params: dict = {},
    ):
        self.treshold_for_positivity = .5

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str, # key to be used to select element in DataDict
        mean: bool = True,
    ):
        if mean:
            raise(f"{__class__.__name__} cannot be used as loss function for training")

        targets = ref[key].bool()
        predictions = (pred[key].sigmoid()>self.treshold_for_positivity).bool()
        recall = torcheval.metrics.functional.binary_recall(predictions.squeeze(), targets.squeeze())
        return torch.full(targets.shape, recall, device=targets.device)

class F1:
    def __init__(
        self,
        func_name: str,
        params: dict = {},
    ):
        self.treshold_for_positivity = .5

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str, # key to be used to select element in DataDict
        mean: bool = True,
    ):
        if mean:
            raise(f"{__class__.__name__} cannot be used as loss function for training")

        targets = ref[key]
        predictions = (pred[key].sigmoid()>self.treshold_for_positivity).float()
        f1 = torcheval.metrics.functional.binary_f1_score(predictions.squeeze(), targets.squeeze())
        return torch.full(targets.shape, f1, device=targets.device)

