import torch
import torch.nn.functional as F
from ftw_ma import losses

def test_binary_tversky_perfect_prediction():
    """Perfect prediction should give loss ≈ 0"""
    loss_fn = losses.BinaryTverskyFocalLoss(smooth=1e-5, alpha=0.5, gamma=1.0)
    
    # Perfect prediction: pred=target
    pred = torch.ones(10, 10)
    target = torch.ones(10, 10)
    
    loss = loss_fn(pred, target)
    
    # With perfect prediction: TP=100, FP=0, FN=0
    # TI = (100 + smooth) / (100 + 0 + 0 + smooth) ≈ 1
    # Loss = (1 - 1)^(1/gamma) ≈ 0
    print(f"Perfect prediction loss: {loss.item():.6f}")
    assert loss.item() < 0.01, f"Expected ~0, got {loss.item()}"


def test_binary_tversky_worst_prediction():
    """Worst prediction should give loss ≈ 1"""
    loss_fn = losses.BinaryTverskyFocalLoss(smooth=1e-5, alpha=0.5, gamma=1.0)
    
    # Worst prediction: pred = 1 - target
    pred = torch.zeros(10, 10)
    target = torch.ones(10, 10)
    
    loss = loss_fn(pred, target)
    
    # TP=0, FP=0, FN=100
    # TI = smooth / (0 + 0.5*0 + 0.5*100 + smooth) ≈ 0
    # Loss = (1 - 0)^1 = 1
    print(f"Worst prediction loss: {loss.item():.6f}")
    assert 0.99 < loss.item() < 1.01, f"Expected ~1, got {loss.item()}"


def test_multiclass_tversky_manual():
    """Manually verify multiclass calculation"""
    
    # Simple 2x2 image, 2 classes
    pred = torch.tensor([[
        [[0.9, 0.1],   # High conf class 0
         [0.9, 0.1]],
        [[0.1, 0.9],   # High conf class 1
         [0.1, 0.5]]   # Mixed
    ]], dtype=torch.float32)  # [1, 2, 2, 2]
    
    target = torch.tensor([[
        [0, 1],
        [0, 1]
    ]], dtype=torch.long)  # [1, 2, 2] - needs batch dimension
    
    alpha = 0.5
    gamma = 1.0
    smooth = 1.0
    
    # Our implementation
    loss_fn = losses.TverskyFocalLoss(
        mode="multiclass",
        alpha=alpha,
        gamma=gamma,
        smooth=smooth,
        from_logits=False,
        reduction="sum",
    )
    our_loss = loss_fn(pred, target)

    # Manual calculation: compute per-class binary Tversky Focal Loss
    pred_norm = F.softmax(pred, dim=1)

    # Build one-hot target [1, 2, 2, 2]
    target_oh = (
        F.one_hot(target, num_classes=2)
        .permute(0, 3, 1, 2)
        .float()
    )

    binary_loss_fn = losses.BinaryTverskyFocalLoss(
        alpha=alpha, gamma=gamma, smooth=smooth
    )

    nclass = 2
    manual_loss = 0.0

    for i in range(nclass):
        # Extract per-class predictions and targets
        pred_class = pred_norm[:, i]  # [1, 2, 2]
        target_class = target_oh[:, i]  # [1, 2, 2]

        # Compute binary loss for this class
        class_loss = binary_loss_fn(pred_class, target_class)

        # Weight by 1/nclass (default equal weighting)
        manual_loss += (1.0 / nclass) * class_loss

        print(
            f"Class {i} loss: {class_loss.item():.6f}"
        )

    print(f"\nOur multiclass loss:  {our_loss.item():.6f}")
    print(f"Manual calculated:    {manual_loss.item():.6f}")
    print(
        f"Difference:           {abs(our_loss.item() - manual_loss.item()):.6e}"
    )
    print(f"Softmax predictions:\n{pred_norm}")

    # They should match within floating-point precision
    assert abs(our_loss.item() - manual_loss.item()) < 1e-5, (
        "Multiclass loss doesn't match manual calculation: "
        f"diff={abs(our_loss.item() - manual_loss.item()):.6e}"
    )

def test_ce_composite_sanity():
    """Verify CE composite is weighted average of Tversky and CE"""
    tversky_weight = 0.3
    loss_fn = losses.TverskyFocalCELoss(
        tversky_weight=tversky_weight,
        tversky_alpha=0.5,
        tversky_gamma=1.0
    )
    
    pred = torch.randn(2, 3, 4, 4)
    target = torch.randint(0, 3, (2, 4, 4))
    
    composite_loss = loss_fn(pred, target)
    
    # Manually compute components
    tversky_fn = losses.TverskyFocalLoss(
        mode="multiclass", alpha=0.5, gamma=1.0, 
        from_logits=True, reduction="sum"
    )
    ce_fn = torch.nn.CrossEntropyLoss()
    
    tversky_loss = tversky_fn(pred, target)
    ce_loss = ce_fn(pred, target)
    
    expected = tversky_weight * tversky_loss + (1 - tversky_weight) * ce_loss
    
    print(f"Composite: {composite_loss.item():.6f}")
    print(f"Expected:  {expected.item():.6f}")
    print(f"Diff:      {abs(composite_loss.item() - expected.item()):.6e}")
    
    assert abs(composite_loss.item() - expected.item()) < 1e-5, \
        "Composite loss doesn't match weighted sum"


if __name__ == "__main__":
    print("=== Testing Perfect Prediction ===")
    test_binary_tversky_perfect_prediction()
    
    print("\n=== Testing Worst Prediction ===")
    test_binary_tversky_worst_prediction()
    
    print("\n=== Testing Manual Multiclass ===")
    test_multiclass_tversky_manual()
    
    print("\n=== Testing CE Composite ===")
    test_ce_composite_sanity()
    
    print("\n✅ All canonical tests passed!")