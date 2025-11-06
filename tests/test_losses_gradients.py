import torch
from ftw_ma import losses

def test_gradients_binary():
    """Ensure gradients flow correctly"""
    pred = torch.rand(10, 10, requires_grad=True)
    target = torch.rand(10, 10)
    
    loss_fn = losses.BinaryTverskyFocalLoss()
    loss = loss_fn(pred, target)
    loss.backward()
    
    assert pred.grad is not None, "No gradient computed"
    assert not torch.isnan(pred.grad).any(), "NaN in gradients"
    assert not torch.isinf(pred.grad).any(), "Inf in gradients"
    print(
        f"✅ Gradient stats: "\
        f"mean={pred.grad.mean():.6f}, std={pred.grad.std():.6f}"
    )


def test_gradients_multiclass():
    """Ensure multiclass gradients are sane"""
    pred = torch.randn(2, 3, 8, 8, requires_grad=True)
    target = torch.randint(0, 3, (2, 8, 8))
    
    loss_fn = losses.TverskyFocalLoss(mode="multiclass", from_logits=True)
    loss = loss_fn(pred, target)
    loss.backward()
    
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()
    print(f"✅ Multiclass gradient stats: mean={pred.grad.mean():.6f}")


if __name__ == "__main__":
    test_gradients_binary()
    test_gradients_multiclass()