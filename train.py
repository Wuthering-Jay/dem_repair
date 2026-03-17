"""Train the first-stage supervised DEM repair MVP."""

from __future__ import annotations

import argparse
from typing import Any

import torch
from torch.utils.data import DataLoader

from datasets import RandomFlip
from losses import curvature_loss, grad_loss, hole_l1_loss, valid_l1_loss
from metrics import compute_dem_metrics
from utils.config_utils import load_yaml, resolve_path
from utils.io_utils import ensure_dir, save_json
from utils.runtime_utils import (
    build_dataset,
    build_generator,
    denormalize_tensor,
    load_checkpoint,
    save_prediction_preview,
)
from utils.seed_utils import get_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the supervised DEM repair MVP.")
    parser.add_argument("--config", default="configs/train/train_supervised.yaml", help="Path to the training config file.")
    return parser.parse_args()


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor fields to the selected device."""
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def compute_loss_dict(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    """Compute the supervised MVP loss components."""
    stage1_hole = hole_l1_loss(outputs["pred_stage1"], target, mask)
    final_hole = hole_l1_loss(outputs["pred_final"], target, mask)
    valid = valid_l1_loss(outputs["pred_final"], target, mask)
    grad = grad_loss(outputs["pred_final"], target, mask)
    curv = curvature_loss(outputs["pred_final"], target, mask)
    total = (
        float(loss_weights.get("hole_l1", 1.0)) * (final_hole + (0.5 * stage1_hole))
        + float(loss_weights.get("valid_l1", 0.1)) * valid
        + float(loss_weights.get("grad", 0.2)) * grad
        + float(loss_weights.get("curvature", 0.0)) * curv
    )
    return {
        "total": total,
        "hole_stage1": stage1_hole,
        "hole_final": final_hole,
        "valid_l1": valid,
        "grad": grad,
        "curvature": curv,
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, Any] | None]:
    """Run validation and return aggregated metrics plus one preview batch."""
    model.eval()
    metric_sums = {"mae": 0.0, "rmse": 0.0, "mask_mae": 0.0, "mask_rmse": 0.0, "slope_rmse": 0.0}
    total_samples = 0
    preview: dict[str, Any] | None = None

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["input_tensor"], batch["coarse_dem"], batch["mask"])
        prediction = denormalize_tensor(outputs["pred_final"], batch["dem_mean"], batch["dem_std"])
        target = denormalize_tensor(batch["gt_dem"], batch["dem_mean"], batch["dem_std"])
        batch_metrics = compute_dem_metrics(prediction, target, batch["mask"])
        batch_size = batch["input_tensor"].shape[0]
        for key, value in batch_metrics.items():
            metric_sums[key] += value * batch_size
        total_samples += batch_size

        if preview is None:
            preview = {
                "batch": batch,
                "outputs": outputs,
            }

    if total_samples == 0:
        return {}, preview
    return {key: value / total_samples for key, value in metric_sums.items()}, preview


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_config: dict[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler | None:
    name = train_config.get("lr_scheduler")
    if not name:
        return None
    epochs = int(train_config.get("epochs", 100))
    min_lr = float(
        train_config.get("lr_scheduler_min_lr", 1e-6))
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=min_lr,
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, epochs // 3),
            gamma=0.3,
        )
    print(f"[warn] unknown lr_scheduler={name}, "
          f"using constant LR")
    return None


def main() -> None:
    args = parse_args()
    train_config = load_yaml(args.config)
    dataset_config = load_yaml(
        train_config["dataset_config"])
    model_config = load_yaml(
        train_config["generator_config"])

    seed_everything(int(train_config.get("seed", 42)))
    device = get_device(
        str(train_config.get("device", "cuda")))

    train_dataset = build_dataset(
        dataset_config,
        split=dataset_config["train_split"],
        transform=RandomFlip(),
    )
    val_dataset = build_dataset(
        dataset_config,
        split=dataset_config["val_split"],
    )
    if len(train_dataset) == 0:
        print("No training samples found. "
              "Run scripts/build_samples.py first.")
        return

    batch_size = int(train_config.get("batch_size", 2))
    num_workers = int(train_config.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    ) if len(val_dataset) > 0 else None

    in_ch = len(dataset_config["required_channels"])
    model = build_generator(
        model_config, in_channels=in_ch,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 2e-4)),
        weight_decay=float(
            train_config.get("weight_decay", 0.0)),
    )
    scheduler = _build_scheduler(optimizer, train_config)

    output_root = resolve_path(
        train_config.get("output_dir", "outputs"))
    checkpoint_dir = ensure_dir(output_root / "checkpoints")
    log_dir = ensure_dir(output_root / "logs")
    figure_dir = ensure_dir(output_root / "figures")
    loss_weights = train_config.get("loss_weights", {})
    total_epochs = int(train_config.get("epochs", 100))
    save_every = int(train_config.get("save_every", 10))
    vis_every = int(train_config.get("visualize_every", 5))
    grad_clip = float(
        train_config.get("grad_clip_norm", 1.0))

    # ---- Resume from checkpoint ----
    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_metric = float("inf")
    resume_path = train_config.get("resume_from")
    if resume_path:
        ckpt = load_checkpoint(
            model, resume_path, device, optimizer)
        if ckpt:
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_metric = float(
                ckpt.get("best_metric", float("inf")))
            history = ckpt.get("history", [])
            if scheduler and "scheduler_state" in ckpt:
                scheduler.load_state_dict(
                    ckpt["scheduler_state"])
            print(f"[resume] from {resume_path}, "
                  f"epoch={start_epoch}")

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        batch_count = 0
        last_preview: dict[str, Any] | None = None

        total_batches = len(train_loader)
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch["input_tensor"],
                batch["coarse_dem"],
                batch["mask"],
            )
            losses = compute_loss_dict(
                outputs, batch["gt_dem"],
                batch["mask"], loss_weights,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip,
            )
            optimizer.step()

            epoch_loss_sum += float(
                losses["total"].detach().cpu())
            batch_count += 1
            last_preview = {
                "batch": batch, "outputs": outputs,
            }

            if batch_count % 100 == 0 or \
                    batch_count == total_batches:
                avg = epoch_loss_sum / batch_count
                print(
                    f"  epoch {epoch}/{total_epochs} "
                    f"batch {batch_count}/{total_batches}"
                    f" loss={avg:.4f}",
                    flush=True,
                )

        if scheduler is not None:
            scheduler.step()

        train_loss = epoch_loss_sum / max(batch_count, 1)
        val_metrics: dict[str, float] = {}
        val_preview: dict[str, Any] | None = None
        if val_loader is not None:
            val_metrics, val_preview = evaluate(
                model, val_loader, device)

        cur_lr = optimizer.param_groups[0]["lr"]
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": cur_lr,
            **val_metrics,
        }
        history.append(record)
        save_json(
            log_dir / "train_history.json",
            {"history": history},
        )

        # ---- Visualization ----
        if epoch % vis_every == 0 and last_preview:
            save_prediction_preview(
                figure_dir / f"train_e{epoch:03d}.png",
                partial_dem=last_preview["batch"][
                    "partial_dem"],
                coarse_dem=last_preview["batch"][
                    "coarse_dem"],
                prediction=last_preview["outputs"][
                    "pred_final"],
                target=last_preview["batch"]["gt_dem"],
                mask=last_preview["batch"]["mask"],
                dem_mean=last_preview["batch"]["dem_mean"],
                dem_std=last_preview["batch"]["dem_std"],
            )
        if epoch % vis_every == 0 and val_preview:
            save_prediction_preview(
                figure_dir / f"val_e{epoch:03d}.png",
                partial_dem=val_preview["batch"][
                    "partial_dem"],
                coarse_dem=val_preview["batch"][
                    "coarse_dem"],
                prediction=val_preview["outputs"][
                    "pred_final"],
                target=val_preview["batch"]["gt_dem"],
                mask=val_preview["batch"]["mask"],
                dem_mean=val_preview["batch"]["dem_mean"],
                dem_std=val_preview["batch"]["dem_std"],
            )

        # ---- Checkpointing ----
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": (
                scheduler.state_dict()
                if scheduler else None),
            "best_metric": best_metric,
            "history": history,
            "train_config": train_config,
            "dataset_config": dataset_config,
            "model_config": model_config,
        }
        torch.save(
            checkpoint, checkpoint_dir / "latest.pt")
        if epoch % save_every == 0:
            torch.save(
                checkpoint,
                checkpoint_dir / f"epoch_{epoch:03d}.pt",
            )

        sel = float(
            val_metrics.get("mask_rmse", train_loss))
        if sel < best_metric:
            best_metric = sel
            torch.save(
                checkpoint, checkpoint_dir / "best.pt")

        print(
            f"[train] epoch={epoch}/{total_epochs} "
            f"loss={train_loss:.4f} "
            f"lr={cur_lr:.2e} "
            f"val_rmse="
            f"{val_metrics.get('mask_rmse', float('nan')):.4f}"
        )

    print(f"training completed. "
          f"checkpoints saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()
