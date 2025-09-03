# from fieldmapper (S. Khallaghi and R. Abedi), with corrections for 
# ignore_index handling
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.core.debugger import set_trace

def _as_long_index(t: torch.Tensor) -> torch.Tensor:
    """Return a LongTensor suitable for one_hot/indexing."""
    if t.dtype != torch.long:
        return t.long()
    return t

class BinaryTverskyFocalLoss(nn.Module):
    '''
    Pytorch versiono of tversky focal loss proposed in paper
    'A novel focal Tversky loss function and improved Attention U-Net for 
    lesion segmentation' (https://arxiv.org/abs/1810.07842)

    Params:

        smooth (float): 
            A float number to smooth loss, and avoid NaN error, default: 1
        alpha (float): 
            Hyperparameters alpha, paired with (1 - alpha) to shift emphasis to
            improve recall
        gamma (float): 
            Tversky index, default: 1.33
        predict (torch.tensor): 
            Predicted tensor of shape [N, C, *]
        target (torch.tensor): 
            Target tensor either in shape [N,*] or of same shape with predict

    Returns:
        Loss tensor

    '''

    def __init__(self, smooth=1, alpha=0.7, gamma=1.33):
        super(BinaryTverskyFocalLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = 1 - self.alpha
        self.gamma = gamma

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], \
            "predict & target batch size do not match"

        # no reduction, same as original paper
        predict = predict.contiguous().view(-1)
        target = target.contiguous().view(-1)

        num = (predict * target).sum() + self.smooth
        den = (predict * target).sum() + \
            self.alpha * ((1 - predict) * target).sum() + \
            self.beta * (predict * (1 - target)).sum() + self.smooth
        loss = torch.pow(1 - num/den, 1 / self.gamma)

        return loss

class TverskyFocalLoss(nn.Module):
    '''

    Tversky focal loss

    Params:
        weight (torch.tensor): 
            Weight array of shape [num_classes,]
        ignore_index (int): 
            Class index to ignore
        predict (torch.tensor): 
            Predicted tensor of shape [N, C, *]
        target (torch.tensor): 
            Target tensor either in shape [N,*] or of same shape with predict 
            other args pass to BinaryTverskyFocalLoss

    Returns:
        same as BinaryTverskyFocalLoss

    '''
    def __init__(self, weight=None, ignore_index=-100, **kwargs):
        super(TverskyFocalLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        device = predict.device
        nclass = predict.shape[1]
        if predict.shape == target.shape:
            target_oh = target
            valid = (target_oh.sum(dim=1, keepdim=True) > 0).float()

        elif len(predict.shape) == 4:
            valid = (target != self.ignore_index)
            # safe_target = target.masked_fill(~valid, 0)
            safe_target = _as_long_index(target.masked_fill(~valid, 0))
            target_oh = (F.one_hot(safe_target, num_classes=nclass)
                         .permute(0, 3, 1, 2)
                         .contiguous())
            valid = valid.unsqueeze(1).float()

        else:
            raise ValueError(f"The shapes of 'predict' and 'target' are "\
                             f"incompatible.")

        tversky = BinaryTverskyFocalLoss(**self.kwargs)
        total_loss = 0
        
        if self.weight is None:
            self.weight = torch.Tensor([1. / nclass] * nclass).to(device)
        else:
            if isinstance(self.weight, list):
                self.weight = (torch.tensor(self.weight, dtype=torch.float64)
                               .to(device))
        
        predict = F.softmax(predict, dim=1)

        for i in range(nclass):
            tversky_loss = tversky(predict[:, i] * valid, 
                                   target_oh[:, i] * valid)
            assert self.weight.shape[0] == nclass, \
                f"Expect weight shape [{nclass}], get[{self.weight.shape[0]}]"
            tversky_loss *= self.weight[i]
            total_loss += tversky_loss
            
        return total_loss

class LocallyWeightedTverskyFocalLoss(nn.Module):
    r"""
    Tversky focal loss weighted by inverse of label frequency calculated 
    locally based on the input batch.

    Params:
        ignore_index (int): 
            Class index to ignore
        predict (torch.tensor): 
            Predicted tensor of shape [N, C, *]
        target (torch.tensor): 
            Target tensor either in shape [N,*] or of same shape with predict
            other args pass to BinaryTverskyFocalLoss
    Returns:
        same as TverskyFocalLoss
    """
    def __init__(self, ignore_index=-100, **kwargs):
        super(LocallyWeightedTverskyFocalLoss, self).__init__()
        self.kwargs = kwargs
        self.ignore_index = ignore_index

    def calculate_weights(self, target, num_class):

        valid = (target != self.ignore_index)
        unique, unique_counts = torch.unique(target[valid], return_counts=True)
        # ratio = unique_counts.float() / valid.sum().float()
        # weight = (1. / ratio) / torch.sum(1. / ratio)

        # loss_weight = torch.ones(num_class, device=target.device) * 0.00001
        # for i in range(len(unique)):
        #     loss_weight[unique[i]] = weight[i]

        # return loss_weight
        # ensure integer indices for indexing (suggested fix from GPT-5 mini)
        unique = unique.to(torch.long)
        ratio = unique_counts.float() / valid.sum().float()

        weight = (1.0 / ratio) / torch.sum(1.0 / ratio)

        loss_weight = torch.ones(num_class, device=target.device) * 1e-5
        # use Python ints (or .item()) when indexing to avoid dtype/device issues
        for i, idx in enumerate(unique):
            loss_weight[int(idx.item())] = weight[i]

        return loss_weight        

    def forward(self, predict, target):
        num_class = predict.shape[1]
        # Calculate the weights based on the current batch
        loss_weight = self.calculate_weights(target, num_class)

        # Initialize the loss
        loss_fn = TverskyFocalLoss(weight=loss_weight, 
                                   ignore_index=self.ignore_index, 
                                   **self.kwargs)
        
        return loss_fn(predict, target)


class TverskyFocalCELoss(nn.Module):
    """
        Combination of tversky focal loss and cross entropy loss though 
        summation

        Params:
            loss_weight (tensor): 
                a manual rescaling weight given to each class. If given, has to 
                be a Tensor of size C
            tversky_weight (float): 
                Weight on tversky focal loss for the summation, while weight on 
                cross entropy loss
            is (1 - tversky_weight)
            tversky_smooth (float): 
                A float number to smooth tversky focal loss, and avoid NaN 
                error, default: 1
            tversky_alpha (float):
            tversky_gamma (float):
            ignore_index (int): 
                Class index to ignore

        Returns:
            Loss tensor

    """

    def __init__(self, loss_weight=None, tversky_weight=0.5, tversky_smooth=1, 
                 tversky_alpha=0.7, tversky_gamma=1.33, ignore_index=-100):
        super(TverskyFocalCELoss, self).__init__()
        self.loss_weight = loss_weight
        self.tversky_weight = tversky_weight
        self.tversky_smooth = tversky_smooth
        self.tversky_alpha = tversky_alpha
        self.tversky_gamma = tversky_gamma
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], \
            "predict & target batch size do not match"

        tversky = TverskyFocalLoss(
            weight=self.loss_weight, ignore_index=self.ignore_index, 
            smooth=self.tversky_smooth, alpha=self.tversky_alpha, 
            gamma=self.tversky_gamma
        )
        ce = nn.CrossEntropyLoss(weight=self.loss_weight, 
                                 ignore_index=self.ignore_index)
        loss = self.tversky_weight * tversky(predict, target) + \
            (1 - self.tversky_weight) * ce(predict, target)

        return loss

class LocallyWeightedTverskyFocalCELoss(nn.Module):
    """
        Combination of tversky focal loss and cross entropy loss weighted by 
        inverse of label frequency

        Params:
            ignore_index (int): 
                Class index to ignore
            predict (torch.tensor): 
                Predicted tensor of shape [N, C, *]
            target (torch.tensor): 
                Target tensor either in shape [N,*] or of same shape with 
                predict other args pass to DiceCELoss, excluding loss_weight

        Returns:
            Same as TverskyFocalCELoss

    """

    def __init__(self, ignore_index=-100, **kwargs):
        super(LocallyWeightedTverskyFocalCELoss, self).__init__()
        self.ignore_index =  ignore_index
        self.kwargs = kwargs

    def forward(self, predict, target):
        # get class weights
        valid = (target != self.ignore_index)
        unique, unique_counts = torch.unique(target[valid], return_counts=True)
        ratio = unique_counts.float() / valid.sum().float()
        weight = (1. / ratio) / torch.sum(1. / ratio)

        lossWeight = torch.ones(predict.shape[1], 
                                device=predict.device) * 0.00001
        for i in range(len(unique)):
            lossWeight[unique[i]] = weight[i]

        loss = TverskyFocalCELoss(loss_weight=lossWeight, **self.kwargs)

        return loss(predict, target)
