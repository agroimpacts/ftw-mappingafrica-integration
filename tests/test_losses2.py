import torch
from ftw_ma import lossesorig as lossesorig
from ftw_ma import losses as losses

batch_mode = True  # Set to False to run single-batch tests

if batch_mode:
    torch.manual_seed(42)

    def compare_losses(old_loss_fn, new_loss_fn, mode="binary", n_batches=3):
        """Compare two loss functions over random inputs."""
        # Pre-generate all batches with consistent seed
        torch.manual_seed(42)
        batches = []
        for i in range(n_batches):
            if mode == "binary":
                # Old binary loss expects [N,H,W] for both pred and target
                preds = torch.rand(32, 128, 128)
                targets = torch.randint(0, 2, (32, 128, 128)).float()
            elif mode == "multiclass":
                preds = torch.randn(32, 3, 128, 128)
                targets = torch.randint(0, 3, (32, 128, 128))
            else:
                raise ValueError(f"Unknown mode: {mode}")
            batches.append((preds.clone().requires_grad_(True), targets))

        for i, (preds, targets) in enumerate(batches):
            # For binary mode with new loss, add channel dim back
            if mode == "binary":
                preds_new = preds.unsqueeze(1)  # [N,H,W] → [N,1,H,W]
            else:
                preds_new = preds
                
            old_loss = old_loss_fn(preds, targets)
            new_loss = new_loss_fn(preds_new, targets)

            diff = (new_loss - old_loss).abs().item()
            print(f"[{mode}] Batch {i}: Old={old_loss.item():.6f}, "
                f"New={new_loss.item():.6f}, Diff={diff:.6e}")


    # ============================
    # Binary comparison
    # ============================
    print("=== Binary Tversky Focal Loss ===")
    old_bin = lossesorig.BinaryTverskyFocalLoss(alpha=0.5, gamma=1.0)
    new_bin = losses.TverskyFocalLoss(
        mode="binary", alpha=0.5, gamma=1.0, from_logits=False
    )
    compare_losses(old_bin, new_bin, "binary")


    # ============================
    # Multiclass Tversky Focal Loss
    # ============================
    print("\n=== Multiclass Tversky Focal Loss ===")
    old_multi = lossesorig.TverskyFocalLoss(alpha=0.5, gamma=1.0)
    new_multi = losses.TverskyFocalLoss(
        mode="multiclass", alpha=0.5, gamma=1.0, 
    )
    compare_losses(old_multi, new_multi, "multiclass")


    # ============================
    # Locally Weighted Tversky Focal Loss
    # ============================
    print("\n=== Locally Weighted Tversky Focal Loss ===")
    old_local = lossesorig.LocallyWeightedTverskyFocalLoss(alpha=0.5, gamma=1.0)
    new_local = losses.LocallyWeightedTverskyFocalLoss(
        mode="multiclass", alpha=0.5, gamma=1.0, #reduction="sum"
    )
    compare_losses(old_local, new_local, "multiclass")


    # ============================
    # Tversky Focal + CE Loss
    # ============================
    print("\n=== Tversky Focal + CE Loss ===")
    old_ce = lossesorig.TverskyFocalCELoss(
        tversky_weight=0.5, tversky_alpha=0.5, tversky_gamma=1.0
    )
    new_ce = losses.TverskyFocalCELoss(
        tversky_weight=0.5, tversky_alpha=0.5, tversky_gamma=1.0
    )
    compare_losses(old_ce, new_ce, "multiclass")


    # ============================
    # Locally Weighted Tversky Focal + CE Loss
    # ============================
    print("\n=== Locally Weighted Tversky Focal + CE Loss ===")
    old_local_ce = lossesorig.LocallyWeightedTverskyFocalCELoss(
        tversky_weight=0.5, tversky_alpha=0.5, tversky_gamma=1.0
    )
    new_local_ce = losses.LocallyWeightedTverskyFocalCELoss(
        tversky_weight=0.5, tversky_alpha=0.5, tversky_gamma=1.0
    )
    compare_losses(old_local_ce, new_local_ce, "multiclass")

else:
    torch.manual_seed(42)

    def generate_synthetic_data(batch_size, n_classes, height, width, 
                                mode="multiclass"):
        if mode == "binary":
            preds = torch.rand(batch_size, 1, height, width)
            targets = (torch.rand(batch_size, height, width) > 0.5).float()
        elif mode == "multiclass":
            preds = torch.rand(batch_size, n_classes, height, width)
            targets = torch.randint(0, n_classes, (batch_size, height, width))
        else:
            raise ValueError("Invalid mode")
        return preds, targets

    def compare_losses(old_loss_fn, new_loss_fn, preds, targets, 
                       mode="multiclass"):
        old_loss = old_loss_fn(preds, 
                               targets.float() if mode=="binary" else targets)
        new_loss = new_loss_fn(preds, targets)
        print(f"[{mode}] Old: {old_loss.item():.6f}, "
              f"New: {new_loss.item():.6f}, "
              f"Diff: {abs(old_loss.item() - new_loss.item()):.6e}")

    if __name__ == "__main__":
        batch_size = 2
        height, width = 32, 32
        n_classes = 3

        # -------------------------
        # Binary test
        # -------------------------
        print("=== Binary ===")
        preds_bin, targets_bin = generate_synthetic_data(batch_size, 1, height, 
                                                         width, mode="binary")
        old_bin_loss = lossesorig.BinaryTverskyFocalLoss()
        new_bin_loss = losses.TverskyFocalLoss(
            mode="binary", 
            from_logits=False, 
            reduction="sum")
        compare_losses(old_bin_loss, new_bin_loss, preds_bin, targets_bin, 
                       mode="binary")

        # -------------------------
        # Multiclass test
        # -------------------------
        print("\n=== Multiclass ===")
        preds_mc, targets_mc = generate_synthetic_data(
            batch_size, n_classes, 
            height, width, mode="multiclass"
        )
        old_mc_loss = lossesorig.TverskyFocalLoss()
        new_mc_loss = losses.TverskyFocalLoss(
            mode="multiclass", 
            from_logits=False, 
            reduction="sum"
        )
        compare_losses(old_mc_loss, new_mc_loss, preds_mc, targets_mc, 
                       mode="multiclass")
