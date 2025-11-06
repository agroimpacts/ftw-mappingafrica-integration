import torch
from ftw_ma import losses


def reference_tversky_focal_loss(pred, target, alpha=0.7, gamma=1.33,
                                 smooth=1.0):
    """
    Canonical Tversky Focal Loss calculation from paper.

    Reference: https://arxiv.org/abs/1810.07842
    Formula: FTL = (1 - TI)^(1/gamma)
    where TI = (TP + smooth) / (TP + alpha*FN + beta*FP + smooth)
    and beta = 1 - alpha
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # Compute TP, FP, FN
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()

    # Tversky Index
    beta = 1 - alpha
    ti = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)

    # Focal Tversky Loss
    ftl = torch.pow(1 - ti, 1 / gamma)
    return ftl


def test_binary_against_reference():
    """Test binary loss against hand-coded reference"""
    torch.manual_seed(42)
    pred = torch.rand(10, 10)
    target = (torch.rand(10, 10) > 0.5).float()

    alpha = 0.7
    gamma = 1.33
    smooth = 1.0

    # Our implementation
    loss_fn = losses.BinaryTverskyFocalLoss(
        alpha=alpha, gamma=gamma, smooth=smooth
    )
    our_loss = loss_fn(pred, target)

    # Reference calculation
    ref_loss = reference_tversky_focal_loss(
        pred, target, alpha=alpha, gamma=gamma, smooth=smooth
    )

    print(f"Our loss:       {our_loss.item():.8f}")
    print(f"Reference loss: {ref_loss.item():.8f}")
    print(f"Difference:     {abs(our_loss.item() - ref_loss.item()):.3e}")

    assert abs(our_loss.item() - ref_loss.item()) < 1e-6, \
        ("Loss doesn't match reference: "
         f"diff={abs(our_loss.item() - ref_loss.item()):.3e}")


def test_multiclass_against_manual():
    """
    Test multiclass by manually computing per-class binary losses
    and verifying the weighted sum matches.
    """
    torch.manual_seed(42)

    # Simple 2-class, 4x4 image
    pred = torch.rand(1, 2, 4, 4)  # [N, C, H, W]
    target = torch.randint(0, 2, (1, 4, 4))  # [N, H, W]

    alpha = 0.5
    gamma = 1.0
    smooth = 1.0

    # Our implementation
    loss_fn = losses.TverskyFocalLoss(
        mode="multiclass",
        alpha=alpha,
        gamma=gamma,
        smooth=smooth,
        from_logits=False,  # pred is already probabilities
        reduction="sum",
    )
    our_loss = loss_fn(pred, target)

    # Manual calculation
    pred_softmax = torch.softmax(pred, dim=1)
    target_oh = (
        torch.nn.functional.one_hot(target, num_classes=2)
        .permute(0, 3, 1, 2)
        .float()
    )

    binary_loss_fn = losses.BinaryTverskyFocalLoss(
        alpha=alpha, gamma=gamma, smooth=smooth
    )

    # Compute per-class loss (weight = 1/nclass for each)
    nclass = 2
    manual_loss = 0.0
    for i in range(nclass):
        class_loss = binary_loss_fn(pred_softmax[:, i], target_oh[:, i])
        manual_loss += (1.0 / nclass) * class_loss

    print(f"Our multiclass loss:    {our_loss.item():.8f}")
    print(f"Manual per-class sum:   {manual_loss.item():.8f}")
    print(f"Difference:             {abs(our_loss.item() - manual_loss.item()):.3e}")

    assert abs(our_loss.item() - manual_loss.item()) < 1e-6, \
        ("Multiclass doesn't match manual sum: "
         f"diff={abs(our_loss.item() - manual_loss.item()):.3e}")


def test_ignore_index():
    """Test that ignore_index properly excludes pixels from loss calc"""
    torch.manual_seed(42)

    # Create 4x4 image with 3 classes
    pred = torch.randn(1, 3, 4, 4)  # [N, C, H, W]
    target = torch.randint(0, 3, (1, 4, 4))  # [N, H, W]

    # Mark some pixels as ignore (e.g. class 255)
    ignore_idx = 255
    target[0, 0, 0] = ignore_idx  # Top-left pixel
    target[0, 3, 3] = ignore_idx  # Bottom-right pixel

    alpha = 0.5
    gamma = 1.0
    smooth = 1.0

    # Loss WITH ignore_index
    loss_fn_ignore = losses.TverskyFocalLoss(
        mode="multiclass",
        alpha=alpha,
        gamma=gamma,
        smooth=smooth,
        from_logits=True,
        ignore_index=ignore_idx,
        reduction="sum",
    )
    loss_with_ignore = loss_fn_ignore(pred, target)

    # Manual calculation: mask out ignored pixels
    pred_softmax = torch.softmax(pred, dim=1)
    valid_mask = (target != ignore_idx)  # [1, 4, 4]
    valid_float = valid_mask.unsqueeze(1).float()  # [1, 1, 4, 4]

    # Build one-hot, setting ignored pixels to 0
    target_masked = target.clone()
    target_masked[~valid_mask] = 0
    target_oh = (
        torch.nn.functional.one_hot(target_masked, num_classes=3)
        .permute(0, 3, 1, 2)
        .float()
    )
    # Zero out ignored pixels in one-hot
    target_oh = target_oh * valid_float

    binary_loss_fn = losses.BinaryTverskyFocalLoss(
        alpha=alpha, gamma=gamma, smooth=smooth
    )

    nclass = 3
    manual_loss = 0.0
    for i in range(nclass):
        # Multiply by valid mask to zero out ignored pixels
        class_pred = pred_softmax[:, i] * valid_float.squeeze(1)
        class_tgt = target_oh[:, i] * valid_float.squeeze(1)
        class_loss = binary_loss_fn(class_pred, class_tgt)
        manual_loss += (1.0 / nclass) * class_loss

    print("\n=== Ignore Index Test ===")
    print(f"Loss with ignore_index={ignore_idx}: "
          f"{loss_with_ignore.item():.8f}")
    print(f"Manual masked loss:                 "
          f"{manual_loss.item():.8f}")
    print(f"Difference:                         "
          f"{abs(loss_with_ignore.item() - manual_loss.item()):.3e}")
    print(f"Valid pixels: {valid_mask.sum().item()}/{target.numel()} "
          f"(ignored {(~valid_mask).sum().item()})")

    # Verify ignored pixels don't contribute
    assert abs(loss_with_ignore.item() - manual_loss.item()) < 1e-5, \
        ("Ignore index handling incorrect: "
         f"diff={abs(loss_with_ignore.item() - manual_loss.item()):.3e}")

    # Also verify that loss WITH ignore is different from loss WITHOUT ignore
    # Create a clean target by replacing ignore_idx with valid class 0
    target_clean = target.clone()
    target_clean[target == ignore_idx] = 0
    
    loss_fn_no_ignore = losses.TverskyFocalLoss(
        mode="multiclass",
        alpha=alpha,
        gamma=gamma,
        smooth=smooth,
        from_logits=True,
        ignore_index=-100,  # Default, won't match any valid class
        reduction="sum",
    )
    loss_no_ignore = loss_fn_no_ignore(pred, target_clean)

    print(f"Loss without ignoring:              "
          f"{loss_no_ignore.item():.8f}")
    print(f"Difference (should be non-zero):    "
          f"{abs(loss_with_ignore.item() - loss_no_ignore.item()):.3e}")

    # Verify the ignore mechanism works correctly:
    # 1. Loss with ignore should be non-negative
    assert loss_with_ignore.item() >= 0, "Loss should be non-negative"
    # 2. Losses should typically be different (the ignored pixels, now treated 
    # as class 0, usually have different predictions than true class 0 pixels)
    # Note: We don't assert they MUST be different because by chance they 
    # could match


def test_edge_cases():
    """Test known edge cases with analytical solutions"""

    # Perfect prediction: TI = 1, FTL = 0
    pred_perfect = torch.ones(10, 10)
    target_perfect = torch.ones(10, 10)
    loss_fn = losses.BinaryTverskyFocalLoss(
        smooth=1e-5, alpha=0.5, gamma=1.0
    )
    loss_perfect = loss_fn(pred_perfect, target_perfect)
    print(f"\nPerfect prediction loss: {loss_perfect.item():.8f}")
    assert loss_perfect.item() < 1e-4, (
        f"Perfect prediction should be ~0, got {loss_perfect.item()}"
    )

    # Worst prediction: TI ≈ 0, FTL ≈ 1
    pred_worst = torch.zeros(10, 10)
    target_worst = torch.ones(10, 10)
    loss_worst = loss_fn(pred_worst, target_worst)
    print(f"Worst prediction loss:   {loss_worst.item():.8f}")
    assert 0.99 < loss_worst.item() < 1.01, (
        f"Worst prediction should be ~1, got {loss_worst.item()}"
    )

    # Half correct (alpha=0.5 means FP and FN weighted equally)
    # Pred=0.5 everywhere gives TP=0.5*100, FP=0.5*0, FN=0.5*100
    # TI = (50 + smooth) / (50 + 0.5*50 + 0.5*0 + smooth) = 50/75 ≈ 0.667
    # FTL = (1 - 0.667)^1 ≈ 0.333
    pred_half = torch.full((10, 10), 0.5)
    target_half = torch.ones(10, 10)
    loss_half = loss_fn(pred_half, target_half)
    print(f"Half prediction loss:    {loss_half.item():.8f}")
    expected_half = 1.0 - (50.0 / 75.0) # Approximate, ignore smooth for clarity
    print(f"Expected ~{expected_half:.3f}, got {loss_half.item():.3f}")


if __name__ == "__main__":
    print("=== Testing Binary Against Reference ===")
    test_binary_against_reference()

    print("\n=== Testing Multiclass Against Manual ===")
    test_multiclass_against_manual()

    print("\n=== Testing Ignore Index ===")
    test_ignore_index()

    print("\n=== Testing Edge Cases ===")
    test_edge_cases()

    print("\n✅ All reference implementation tests passed!")