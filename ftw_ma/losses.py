import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import DiceLoss as Dice


def _as_long_index(t):
    return t.long() if t.dtype != torch.long else t


class BinaryTverskyFocalLoss(nn.Module):
    """
    Binary focal Tversky loss: FTL = (1 - TI) ** gamma
    where TI = TP / (TP + alpha * FP + beta * FN)
    Reference: https://arxiv.org/abs/1810.07842
    """
    def __init__(self, smooth=1.0, alpha=0.7, gamma=1.33):
        super().__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = 1 - alpha
        self.gamma = gamma

    def forward(self, predict, target):
        predict = predict.contiguous().view(-1)
        target = target.contiguous().view(-1)

        tp = (predict * target).sum()
        fp = ((1 - target) * predict).sum()
        fn = (target * (1 - predict)).sum()

        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fn + self.beta * fp + self.smooth
        )
        return torch.pow(1 - tversky_index, 1 / self.gamma)


class TverskyFocalLoss(nn.Module):
    def __init__(self, mode="multiclass", from_logits=True,
                 smooth=1.0, alpha=0.7, gamma=1.33,
                 weight=None, ignore_index=-100, reduction="sum"):
        super().__init__()
        self.mode = mode
        self.from_logits = from_logits
        self.ignore_index = ignore_index
        self.weight = weight
        self.reduction = reduction
        self.tversky = BinaryTverskyFocalLoss(
            smooth=smooth, alpha=alpha, gamma=gamma
        )

    def forward(self, y_pred, y_true):
        device = y_pred.device

        # Convert logits to probabilities
        if self.from_logits:
            if self.mode == "binary":
                y_pred = torch.sigmoid(y_pred)
        
        # Always apply softmax in multiclass (match losses.py)
        if self.mode == "multiclass":
            y_pred = F.softmax(y_pred, dim=1)

        # --- Binary mode ---
        if self.mode == "binary":
            # [N,1,H,W] -> [N,H,W]
            if y_pred.ndim == 4 and y_pred.shape[1] == 1:
                y_pred = y_pred.squeeze(1)
            if y_true.ndim == 4 and y_true.shape[1] == 1:
                y_true = y_true.squeeze(1)

            if self.ignore_index is not None:
                valid_mask = y_true != self.ignore_index
                y_pred = y_pred[valid_mask]
                y_true = y_true[valid_mask].float()
            else:
                y_true = y_true.float()

            return self.tversky(y_pred, y_true)
        
        # --- Multiclass mode ---
        elif self.mode == "multiclass":
            nclass = y_pred.shape[1]
            valid_mask = (y_true != self.ignore_index)
            safe_target = _as_long_index(y_true.masked_fill(~valid_mask, 0))
            y_true_oh = F.one_hot(
                safe_target, num_classes=nclass
            ).permute(0, 3, 1, 2).contiguous()
            valid_float = valid_mask.unsqueeze(1).type_as(y_pred)

            if self.weight is None:
                weight = torch.full((nclass,), 1.0 / nclass, device=device)
            else:
                weight = torch.as_tensor(
                    self.weight, dtype=torch.float32, device=device
                )

            total_loss = 0.0
            for i in range(nclass):
                loss_i = self.tversky(
                    y_pred[:, i] * valid_float,
                    y_true_oh[:, i] * valid_float,
                )
                total_loss += weight[i] * loss_i

            if self.reduction == "mean":
                total_loss /= nclass
            return total_loss

class TverskyFocalCELoss(nn.Module):
    """
    Combination of Tversky Focal Loss and Cross Entropy Loss.

    Args:
        loss_weight (Tensor): class weights for CE loss
        tversky_weight (float): weight for Tversky loss
        tversky_smooth, tversky_alpha, tversky_gamma
        ignore_index (int): class index to ignore
    """
    def __init__(self, loss_weight=None, tversky_weight=0.5, tversky_smooth=1,
                 tversky_alpha=0.7, tversky_gamma=1.33, ignore_index=-100):
        super().__init__()
        self.loss_weight = loss_weight
        self.tversky_weight = tversky_weight
        self.tversky_smooth = tversky_smooth
        self.tversky_alpha = tversky_alpha
        self.tversky_gamma = tversky_gamma
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        tversky = TverskyFocalLoss(
            mode="multiclass",
            from_logits=True,
            smooth=self.tversky_smooth,
            alpha=self.tversky_alpha,
            gamma=self.tversky_gamma,
            weight=self.loss_weight,
            ignore_index=self.ignore_index,
            reduction="sum"  # Match losses.py behavior (no averaging)
        )
        ce_weight = None
        if self.loss_weight is not None:
            if isinstance(self.loss_weight, list):
                ce_weight = torch.tensor(self.loss_weight, dtype=torch.float32,
                                         device=predict.device)
            elif isinstance(self.loss_weight, torch.Tensor):
                ce_weight = self.loss_weight.to(device=predict.device,
                                                dtype=torch.float32)
        ce = nn.CrossEntropyLoss(weight=ce_weight, 
                                 ignore_index=self.ignore_index)
        return self.tversky_weight * tversky(predict, target) + \
               (1 - self.tversky_weight) * ce(predict, target)


class LocallyWeightedTverskyFocalLoss(TverskyFocalLoss):
    """
    Tversky Focal Loss weighted by inverse label frequency in current batch.
    """
    def forward(self, y_pred, y_true):
        device = y_pred.device
        nclass = y_pred.shape[1] if self.mode == "multiclass" else 1

        valid_mask = (y_true != self.ignore_index)
        target_safe = _as_long_index(y_true.masked_fill(~valid_mask, 0))
        if self.mode == "multiclass":
            unique, counts = torch.unique(target_safe[valid_mask], 
                                          return_counts=True)
        else:
            unique, counts = torch.unique(target_safe, return_counts=True)

        weight = torch.ones(nclass, device=device) * 1e-5
        if unique.numel() > 0:
            ratio = counts.float() / valid_mask.sum().float()
            w = (1.0 / ratio)
            w = w / w.sum()
            for i, idx in enumerate(unique):
                weight[int(idx.item())] = w[i].to(device, dtype=torch.float32)

        self.weight = weight
        return super().forward(y_pred, y_true)

class LocallyWeightedTverskyFocalCELoss(nn.Module):
    """
    Combination of Tversky Focal Loss and Cross Entropy Loss weighted by 
    inverse label frequency in current batch.
    """
    def __init__(self, ignore_index=-100, **kwargs):
        super().__init__()
        self.ignore_index = ignore_index
        self.kwargs = kwargs

    def forward(self, predict, target):
        device = predict.device
        nclass = predict.shape[1]
        
        # Calculate local weights
        valid_mask = (target != self.ignore_index)
        target_safe = _as_long_index(target.masked_fill(~valid_mask, 0))
        unique, counts = torch.unique(target_safe[valid_mask], 
                                      return_counts=True)
        
        weight = torch.ones(nclass, device=device) * 1e-5
        if unique.numel() > 0:
            ratio = counts.float() / valid_mask.sum().float()
            w = (1.0 / ratio)
            w = w / w.sum()
            for i, idx in enumerate(unique):
                weight[int(idx.item())] = w[i].to(device, dtype=torch.float32)
        
        # Use calculated weights in TverskyFocalCELoss
        loss_fn = TverskyFocalCELoss(
            loss_weight=weight, 
            ignore_index=self.ignore_index, 
            **self.kwargs
        )
        return loss_fn(predict, target)

# from bakeoff
class logCoshDice(Dice):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32) \
                if class_weights is not None else None
        )
        
    def aggregate_loss(self, loss: torch.Tensor) -> torch.Tensor:
        if self.class_weights is not None:
            weights = self.class_weights.to(loss.device)
            loss = (loss * weights) / weights.sum()

        loss = loss.mean()
        loss = torch.log(torch.cosh(loss))
        return loss