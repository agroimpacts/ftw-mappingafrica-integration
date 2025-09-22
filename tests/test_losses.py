import torch
import pytest

from ftw_ma.losses import (
    LocallyWeightedTverskyFocalLoss,
    TverskyFocalLoss,
    BinaryTverskyFocalLoss,
)

def make_tensors(device):
    torch.manual_seed(0)
    N, C, H, W = 2, 3, 8, 8
    logits = torch.randn(N, C, H, W, device=device, dtype=torch.float32, 
                         requires_grad=True)
    targets = torch.randint(0, C, (N, H, W), device=device, dtype=torch.long)
    return logits, targets

# dynamically select devices available on this machine
_devices = ["cpu"]
if torch.cuda.is_available():
    _devices.append("cuda")
if getattr(torch.backends, "mps", None) is not None \
    and torch.backends.mps.is_available():
    _devices.append("mps")

@pytest.mark.parametrize("device", _devices)
def test_tversky_losses_basic(device):
    device = torch.device(device)
    logits, targets = make_tensors(device)

    # Locally weighted Tversky
    local_loss_fn = LocallyWeightedTverskyFocalLoss(
        ignore_index=None, smooth=1.0, alpha=0.7, gamma=1.33
    )
    local_loss = local_loss_fn(logits, targets)
    assert torch.isfinite(local_loss).all(), \
        "LocallyWeightedTverskyFocalLoss returned non-finite value"
    assert float(local_loss.item()) > 0.0, \
        "LocallyWeightedTverskyFocalLoss returned zero or negative loss"

    # Per-class Tversky focal loss
    tv_loss_fn = TverskyFocalLoss(weight=None, ignore_index=None, smooth=1.0, 
                                  alpha=0.7, gamma=1.33)
    tv_loss = tv_loss_fn(logits, targets)
    assert torch.isfinite(tv_loss).all(), \
        "TverskyFocalLoss returned non-finite value"
    assert float(tv_loss.item()) > 0.0, \
        "TverskyFocalLoss returned zero or negative loss"

    # Binary Tversky on a single channel (use class 1 vs rest as a binary task)
    # Build a pseudo-binary prediction and target
    prob_class1 = torch.softmax(logits, dim=1)[:, 1, :, :].detach()
    prob_class1 = prob_class1.to(device=device)
    bin_target = (targets == 1).long().to(device=device)
    bin_loss_fn = BinaryTverskyFocalLoss(smooth=1.0, alpha=0.7, gamma=1.33)
    bin_loss = bin_loss_fn(prob_class1, bin_target)
    assert torch.isfinite(bin_loss).all(), \
        "BinaryTverskyFocalLoss returned non-finite value"
    assert float(bin_loss.item()) > 0.0, \
        "BinaryTverskyFocalLoss returned zero or negative loss"

if __name__ == "__main__":
    # quick local run
    pytest.main([__file__, "-q"])