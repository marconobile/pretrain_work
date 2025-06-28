""" Adapted from https://github.com/mir-group/nequip
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean # scatter
from torcheval.metrics import BinaryAccuracy as TorchBinaryAccuracy, BinaryAUROC
from torchmetrics.classification import BinaryHingeLoss


def ensemble_predictions_and_targets(predictions, targets, ensemble_indices):
    ''' checks whether field has already been ensembled, if not, ensembles it using ensemble_indices'''
    unique_ensembles = torch.unique(ensemble_indices)

    # if unique_ensembles.dim() != predictions.dim(): #TODO: this was written since it could be that it breaks if BS=1; to be checked
    #     predictions = predictions.unsqueeze(0)
    #     targets = targets.unsqueeze(0)

    # ensemble predictions
    if predictions.shape == torch.Size([]) and unique_ensembles.shape[0] == 1:
        is_input_already_ensembled = True
    else:
        is_input_already_ensembled = unique_ensembles.shape[0] == predictions.shape[0]

    if not is_input_already_ensembled:
        predictions = scatter_mean(predictions, ensemble_indices)

    # ensemble targets
    if targets.shape == torch.Size([]) and unique_ensembles.shape[0] == 1:
        is_output_already_ensembled = True
    else:
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
        self.treshold_for_positivity = .5
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

        predicted_label = (logits.sigmoid()>self.treshold_for_positivity ).float()
        if targets_binary.dim() == 0: # if bs = 1
            targets_binary = targets_binary.unsqueeze(0)
            predicted_label = predicted_label.unsqueeze(0)

        self.metric.update(predicted_label, targets_binary)
        acc = self.metric.compute()
        self.metric.reset() # reset at each batch
        return acc.to(logits.device)


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
        assert 'ensemble_index' in pred
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

        try:
            binarized_predictions = (logits.sigmoid()<self.treshold_for_positivity).float().reshape(*targets_binary.shape)
        except:
            binarized_predictions = (logits.sigmoid()<self.treshold_for_positivity).float()
        return torch.abs(targets_binary - binarized_predictions)


class BinaryAUROCMetric:
    def __init__(
        self,
        func_name: str='BinaryAUROC',
        params: dict = {},
        **kwargs,
    ):
        self.params = params
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.func_name = "BinaryAUROC"
        self.treshold_for_positivity = .5
        self.metric = BinaryAUROC()

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

        if targets_binary.dim() == 0: # if bs = 1
            targets_binary = targets_binary.unsqueeze(0)
            logits = logits.unsqueeze(0)

        self.metric.update(logits, targets_binary)
        rocauc = self.metric.compute()
        self.metric.reset() # reset at each batch
        return rocauc.to(logits.device)


class RMSELoss:
    def __init__(
        self,
        func_name: str='RMSE',
        params: dict = {},
        **kwargs,
    ):
        self.func_name = 'RMSE'
        self.params = params
        self.mse = nn.MSELoss() # reduction is by default mean
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        **kwargs,
    ):
        preds = pred[key]
        targets = ref[key]
        loss = torch.sqrt(self.mse(targets, preds))
        if mean: return torch.mean(loss)
        return loss


class HingeBinaryLoss:
    def __init__(
        self,
        func_name: str='Hinge',
        params: dict = {},
        **kwargs,
    ):
        self.func_name = 'Hinge'
        self.params = params
        self.bhl = BinaryHingeLoss(squared=True)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        **kwargs,
    ):
        '''
        Example from BinaryHingeLoss source code:
        # preds = torch.tensor([0.25, 0.25, 0.55, 0.75, 0.75])
        # target = torch.tensor([0, 0, 1, 1, 1])
        # bhl = BinaryHingeLoss()
        # print(bhl(preds, target), bhl(preds, target) == torch.tensor(0.6900))
        # bhl = BinaryHingeLoss(squared=True)
        # print(bhl(preds, target),torch.tensor(0.6905))
        '''
        preds = pred[key].squeeze()
        targets = ref[key].squeeze()
        self.bhl.to(preds.device)
        loss = self.bhl(preds, targets)
        if mean: return torch.mean(loss)
        return loss


class RegressionEnsemble:
    def __init__(
        self,
        func_name: str,
        params: dict = {},
        **kwargs,
    ):
        self.params = params
        for key, value in kwargs.items():
            setattr(self, key, value)
        for key, value in params.items():
            setattr(self, key, value)

        self.rms = False
        self.func_name = 'RegressionEnsemble' # needs to be equal to class name
        if self.aggregation == 'EnsembleMSE' or 'EnsembleRMSE':
            self.loss_func = nn.MSELoss()
            if self.aggregation == 'EnsembleRMSE':
                self.rms = True
        elif self.aggregation == 'EnsembleL1':
            self.loss_func = nn.L1Loss()
        else:
            raise ValueError(f"func_name: {self.aggregation} not valid, it must be in EnsembleMSE or EnsembleL1")

        self.func_name = "func_name"

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        **kwargs,
    ):
        predictions = pred[key].squeeze()
        targets = ref[key].squeeze()
        assert 'ensemble_index' in pred
        assert 'ensemble_index' in ref
        ensembled_predictions, ensembled_targets = ensemble_predictions_and_targets(predictions, targets, pred['ensemble_index'])

        loss = self.loss_func(
            ensembled_predictions,
            ensembled_targets,
        )
        if mean:
            loss = loss.mean()

        return torch.sqrt(loss) if self.rms else loss



class EnsembleBinaryAUROCMetric:
    def __init__(
        self,
        func_name: str='EnsembleBinaryAUROCMetric',
        params: dict = {},
        **kwargs,
    ):
        self.params = params
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.func_name = "EnsembleBinaryAUROCMetric"
        self.treshold_for_positivity = .5
        self.metric = BinaryAUROC()

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

        logits = pred[key].squeeze()
        targets_binary = ref[key].squeeze()
        assert 'ensemble_index' in pred
        assert 'ensemble_index' in ref
        logits, targets_binary = ensemble_predictions_and_targets(logits, targets_binary, pred['ensemble_index'])

        if targets_binary.dim() == 0: # if bs = 1
            targets_binary = targets_binary.unsqueeze(0)
            logits = logits.unsqueeze(0)

        self.metric.update(logits, targets_binary)
        rocauc = self.metric.compute()
        self.metric.reset() # reset at each batch
        return rocauc.to(logits.device)




class EnforceGraphEmbOrthogonalityLoss:
    def __init__(
        self,
        func_name: str,
        params: dict = {},
        **kwargs,
    ):
        """
        Diversity regularizer based on normalized cosine-distance:
        encourages embeddings in the same group to point in different directions.

        Args:
            eps: small constant to avoid numerical issues.
        """
        self.eps: float = 1e-6
        self.params = params
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.func_name = "EnforceGraphEmbOrthogonalityLoss"

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        **kwargs,
    ):
        """
        Args:
            A: Tensor of shape (N, K) – your embeddings.
            B: LongTensor of shape (N,) – integer group IDs for each row of A.
        Returns:
            loss: scalar; negative average pairwise cosine-distance across groups.
        """
        A = pred["graph_features"]
        B = pred["ensemble_index"]
        device = A.device
        unique_groups = B.unique()
        group_losses = []

        for g in unique_groups:
            idx = (B == g).nonzero(as_tuple=True)[0]
            M = idx.numel()
            if M <= 1:
                continue

            X = A[idx]                     # (M, K)
            # Normalize to unit length
            X_norm = F.normalize(X, p=2, dim=1, eps=self.eps)  # (M, K)

            # Compute pairwise cosine similarities: (M, M)
            sims = X_norm @ X_norm.t()     # cos(x_i, x_j)
            # Clip for numerical stability
            sims = sims.clamp(-1 + self.eps, 1 - self.eps)

            # Mask out self-similarities on the diagonal
            mask = ~torch.eye(M, dtype=torch.bool, device=device)
            pairwise_sims = sims[mask].view(M, M - 1)

            # Convert to distance: 1 - cos_sim
            pairwise_dist = 1.0 - pairwise_sims
            group_losses.append(pairwise_dist.mean())

        if not group_losses:
            return torch.tensor(0., device=device)

        # Negative because we *maximize* diversity (distance)
        return -torch.stack(group_losses).mean()
