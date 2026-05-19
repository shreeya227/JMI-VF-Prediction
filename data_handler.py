"""
Training script for Harvard-GF OCT 3D TDS regression.

Features:
  - patient-level train/validation split
  - validation-selected checkpoint
  - held-out test evaluation
  - baseline / demographic_only / full AFF ablations
  - race or Hispanic ethnicity attribute
"""

import os
import sys
import json
import argparse
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(".")

from src.data_handler import Harvard_GF, create_patient_level_split, save_split, load_split

try:
    from src.model import FairResNet3D_R18_Attn, ResNet3D_Baseline
except Exception as exc:
    raise ImportError(
        "Could not import FairResNet3D_R18_Attn and ResNet3D_Baseline from src.model. "
        "Make sure src/model.py exists and the models have severity removed from their "
        "constructors and forward() functions."
    ) from exc

try:
    from src import logger
except Exception:
    logger = None


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic helps reproducibility.
    # benchmark=True can improve speed but may introduce non-determinism in some cases.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_train_dir(data_dir: str) -> str:
    """
    If data_dir has a Train subfolder, use it.
    Otherwise assume data_dir itself contains training .npz files.
    """
    train_dir = os.path.join(data_dir, "Train")
    if os.path.isdir(train_dir):
        return train_dir
    return data_dir


def build_param_groups_for_fair3d(model: nn.Module, base_lr: float):
    """
    Build optimizer parameter groups.

    Shared parameters get group_id=-1.
    subgroup-specific calibration layers get group_id=0,1,...

    This supports AFF learning-rate updates on group-specific layers only.
    """
    group_specific_modules = []

    if hasattr(model, "group_specific_layers") and isinstance(model.group_specific_layers, nn.ModuleList):
        for gid, module in enumerate(model.group_specific_layers):
            group_specific_modules.append((gid, module))

    specific_param_ids = set()
    for _, module in group_specific_modules:
        for p in module.parameters(recurse=True):
            specific_param_ids.add(id(p))

    shared_params = [p for p in model.parameters() if id(p) not in specific_param_ids]

    param_groups = [{"params": shared_params, "group_id": -1, "lr": base_lr}]

    for gid, module in group_specific_modules:
        param_groups.append({"params": module.parameters(), "group_id": gid, "lr": base_lr})

    return param_groups


def build_model(args, device):
    out_dim = 52
    num_groups = 2 if args.attribute_type == "hispanic" else 3

    if args.ablation == "baseline":
        # Baseline model does not use demographic attribute or AFF.
        model = ResNet3D_Baseline(
            in_channels=1,
            out_dim=out_dim,
            pretrained_backbone=False,
        )
    else:
        # demographic_only and full use demographic attribute.
        # Severity has been removed.
        model = FairResNet3D_R18_Attn(
            in_channels=1,
            out_dim=out_dim,
            num_groups=num_groups,
            attr_emb_dim=args.attr_emb_dim,
            pretrained_backbone=False,
        )

    return model.to(device)


def call_model(model, x, attr, ablation: str):
    """
    Central model-call helper so severity is never used.
    """
    if ablation == "baseline":
        return model(x)
    return model(x, attr.long())


def update_aff_learning_rates(
    optimizer: torch.optim.Optimizer,
    group_mae: Dict[int, float],
    base_lr: float,
    lr_worst: float,
    max_boost: float,
    sigmoid_midpoint: float,
    sigmoid_slope: float,
):
    """
    AFF update.

    Applies learning-rate scaling only to the subgroup-specific parameter group
    corresponding to the worst-performing group.

    Shared parameter group has group_id=-1 and always keeps base_lr.
    """
    if not group_mae:
        return None

    worst_group = max(group_mae, key=group_mae.get)
    best_group = min(group_mae, key=group_mae.get)
    disparity = float(group_mae[worst_group] - group_mae[best_group])

    # Rebased sigmoid:
    # raw sigmoid centered at midpoint, then rebased so disparity=0 maps near 0.
    raw = 1.0 / (1.0 + np.exp(-sigmoid_slope * (disparity - sigmoid_midpoint)))
    raw_zero = 1.0 / (1.0 + np.exp(-sigmoid_slope * (0.0 - sigmoid_midpoint)))
    scale01 = (raw - raw_zero) / max(1e-8, (1.0 - raw_zero))
    scale01 = float(np.clip(scale01, 0.0, 1.0))

    boosted_lr = lr_worst * (1.0 + (max_boost - 1.0) * scale01)
    boosted_lr = min(boosted_lr, lr_worst * max_boost)

    for pg in optimizer.param_groups:
        gid = pg.get("group_id", -1)

        if gid == -1:
            pg["lr"] = base_lr
        elif gid == worst_group:
            pg["lr"] = boosted_lr
        else:
            pg["lr"] = base_lr

    return {
        "worst_group": int(worst_group),
        "best_group": int(best_group),
        "mae_disparity": disparity,
        "scale01": scale01,
        "boosted_lr": boosted_lr,
    }


def compute_overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attrs: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    attrs = np.asarray(attrs).reshape(-1)
    results = {}

    for g in sorted(np.unique(attrs)):
        mask = attrs == g
        if mask.sum() == 0:
            continue

        mse = mean_squared_error(y_true[mask], y_pred[mask])
        results[int(g)] = {
            "n": int(mask.sum()),
            "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_true[mask], y_pred[mask])),
        }

    return results


def compute_mae_disparity(group_metrics: Dict[int, Dict[str, float]]) -> float:
    if not group_metrics:
        return 0.0
    maes = [v["mae"] for v in group_metrics.values()]
    return float(max(maes) - min(maes))


def equity_scaled_scores(
    overall_metrics: Dict[str, float],
    group_metrics: Dict[int, Dict[str, float]],
    alpha: float = 1.0,
) -> Dict[str, float]:
    """
    Equity-scaled scores:
      ES-MAE = overall_MAE - alpha * average absolute group MAE deviation
      ES-RMSE = overall_RMSE - alpha * average absolute group RMSE deviation

    These are scores, not conventional errors.
    Higher values mean smaller disparity penalty relative to the aggregate error.
    """
    if not group_metrics:
        return {"es_mae": np.nan, "es_rmse": np.nan}

    group_maes = np.asarray([v["mae"] for v in group_metrics.values()], dtype=float)
    group_rmses = np.asarray([v["rmse"] for v in group_metrics.values()], dtype=float)

    mae_penalty = np.mean(np.abs(group_maes - overall_metrics["mae"]))
    rmse_penalty = np.mean(np.abs(group_rmses - overall_metrics["rmse"]))

    return {
        "es_mae": float(overall_metrics["mae"] - alpha * mae_penalty),
        "es_rmse": float(overall_metrics["rmse"] - alpha * rmse_penalty),
    }


def run_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    ablation: str,
    train: bool,
):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    n_samples = 0

    preds = []
    targets = []
    attrs = []
    mds = []

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for batch in loader:
            # Dataset returns: oct_volume, target, attr, md
            x, y, attr, md = batch

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            attr = attr.to(device, non_blocking=True).long()

            pred = call_model(model, x, attr, ablation)

            # MSE over 52 TDS points.
            loss_tensor = criterion(pred, y)
            loss = loss_tensor.mean()

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            bs = x.shape[0]
            total_loss += loss.item() * bs
            n_samples += bs

            preds.append(pred.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
            attrs.append(attr.detach().cpu().numpy())
            mds.append(md.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    attrs = np.concatenate(attrs, axis=0).reshape(-1)
    mds = np.concatenate(mds, axis=0)

    overall = compute_overall_metrics(targets, preds)
    groups = compute_group_metrics(targets, preds, attrs)
    disparity = compute_mae_disparity(groups)
    es = equity_scaled_scores(overall, groups)

    metrics = {
        "loss": float(total_loss / max(1, n_samples)),
        **overall,
        "mae_disparity": disparity,
        **es,
        "group_metrics": groups,
    }

    outputs = {
        "pred": preds,
        "gt": targets,
        "attr": attrs,
        "md": mds,
    }

    return metrics, outputs


def print_metrics(prefix: str, metrics: Dict):
    print(
        f"{prefix}: "
        f"loss={metrics['loss']:.4f}, "
        f"MAE={metrics['mae']:.4f}, "
        f"RMSE={metrics['rmse']:.4f}, "
        f"MSE={metrics['mse']:.4f}, "
        f"R2={metrics['r2']:.4f}, "
        f"MAE disparity={metrics['mae_disparity']:.4f}, "
        f"ES-MAE={metrics['es_mae']:.4f}, "
        f"ES-RMSE={metrics['es_rmse']:.4f}"
    )

    for g, gm in metrics["group_metrics"].items():
        print(
            f"  Group {g}: n={gm['n']}, "
            f"MAE={gm['mae']:.4f}, RMSE={gm['rmse']:.4f}, R2={gm['r2']:.4f}"
        )


def save_checkpoint(path, model, optimizer, epoch, val_metrics, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_mae": val_metrics["mae"],
            "val_mse": val_metrics["mse"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "args": vars(args),
        },
        path,
    )


def load_best_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


def save_npz_outputs(path, outputs, metrics, best_epoch=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        test_pred=outputs["pred"],
        test_gt=outputs["gt"],
        test_attr=outputs["attr"],
        test_md=outputs["md"],
        best_epoch=best_epoch,
        test_mae=metrics["mae"],
        test_mse=metrics["mse"],
        test_rmse=metrics["rmse"],
        test_r2=metrics["r2"],
        test_mae_disparity=metrics["mae_disparity"],
        test_es_mae=metrics["es_mae"],
        test_es_rmse=metrics["es_rmse"],
    )


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attrs: np.ndarray,
    n_bootstrap: int = 100,
    seed: int = 123,
):
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]

    rows = []
    for _ in range(n_bootstrap):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        yt = y_true[idx]
        yp = y_pred[idx]
        at = attrs[idx]

        overall = compute_overall_metrics(yt, yp)
        groups = compute_group_metrics(yt, yp, at)
        disparity = compute_mae_disparity(groups)
        es = equity_scaled_scores(overall, groups)

        rows.append({
            "mae": overall["mae"],
            "rmse": overall["rmse"],
            "r2": overall["r2"],
            "mae_disparity": disparity,
            "es_mae": es["es_mae"],
            "es_rmse": es["es_rmse"],
        })

    summary = {}
    for key in rows[0].keys():
        values = np.asarray([r[key] for r in rows], dtype=float)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "ci_low": float(np.percentile(values, 2.5)),
            "ci_high": float(np.percentile(values, 97.5)),
        }

    return summary


def resolve_test_dir(args):
    if args.test_data_dir is not None:
        return args.test_data_dir

    test_dir = os.path.join(args.data_dir, "Test")
    if os.path.isdir(test_dir):
        return test_dir

    raise ValueError("Please provide --test_data_dir or use data_dir with a Test subfolder.")


def parse_args():
    parser = argparse.ArgumentParser(description="Harvard-GF 3D ResNet training without severity")

    # Paths
    parser.add_argument("--data_dir", default="/medailab/medailab/shilab/Harvard-GF", type=str)
    parser.add_argument("--test_data_dir", default=None, type=str)
    parser.add_argument("--result_dir", default="./results", type=str)
    parser.add_argument("--split_json", default=None, type=str)
    parser.add_argument("--perf_file", default="performance.csv", type=str)

    # Hardware/reproducibility
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--workers", default=8, type=int)

    # Data/model
    parser.add_argument("--model_type", default="efficientnet", type=str)
    parser.add_argument("--task", default="tds", type=str)
    parser.add_argument("--image_size", default=200, type=int)
    parser.add_argument("--modality_types", default="oct_bscans_3d", type=str)
    parser.add_argument("--attribute_type", default="race", choices=["race", "hispanic"], type=str)
    parser.add_argument("--ablation", default="full", choices=["baseline", "demographic_only", "full"], type=str)
    parser.add_argument("--attr_emb_dim", default=128, type=int)

    # Training
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--drop_path", default=0.3, type=float)
    parser.add_argument("--patience", default=5, type=int)

    # Patient-level split
    parser.add_argument("--val_size", default=300, type=int)
    parser.add_argument("--patient_id_key", default=None, type=str)
    parser.add_argument("--allow_filename_patient_id", default=True, type=str2bool)

    # Oversampling
    parser.add_argument("--oversample_minority_factor", default=1, type=int)

    # AFF
    parser.add_argument("--lr_best", default=5e-5, type=float)
    parser.add_argument("--lr_worst", default=6e-5, type=float)
    parser.add_argument("--rank_update_K", default=2, type=int)
    parser.add_argument("--sigmoid_midpoint", default=0.25, type=float)
    parser.add_argument("--sigmoid_slope", default=6.0, type=float)
    parser.add_argument("--max_boost", default=1.25, type=float)

    # Bootstrap
    parser.add_argument("--bootstrap_repeat_times", default=100, type=int)

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    if logger is not None:
        logger.configure(dir=args.result_dir, log_suffix="train")

    set_random_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    print("\n========== Configuration ==========")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print(f"device: {device}")
    print("===================================\n")

    train_data_dir = get_train_dir(args.data_dir)
    test_data_dir = resolve_test_dir(args)

    print(f"Train/validation data directory: {train_data_dir}")
    print(f"Test data directory: {test_data_dir}")

    # ------------------------------------------------------------
    # Patient-level train/validation split
    # ------------------------------------------------------------
    if args.split_json is None:
        split_json = os.path.join(
            args.result_dir,
            f"patient_level_split_{args.attribute_type}_seed{args.seed}.json",
        )
    else:
        split_json = args.split_json

    if os.path.exists(split_json):
        print(f"Loading existing split: {split_json}")
        split = load_split(split_json)
    else:
        print("Creating patient-level split...")
        split = create_patient_level_split(
            data_dir=train_data_dir,
            attribute_type=args.attribute_type,
            val_size=args.val_size,
            seed=args.seed,
            patient_id_key=args.patient_id_key,
            allow_filename_fallback=args.allow_filename_patient_id,
        )
        save_split(split, split_json)

    train_files = split["train_files"]
    val_files = split["val_files"]

    # ------------------------------------------------------------
    # Datasets/loaders
    # ------------------------------------------------------------
    train_dataset = Harvard_GF(
        train_data_dir,
        modality_type=args.modality_types,
        task=args.task,
        resolution=args.image_size,
        attribute_type=args.attribute_type,
        transform=None,
        file_list=train_files,
        oversample_minority_factor=args.oversample_minority_factor,
    )

    val_dataset = Harvard_GF(
        train_data_dir,
        modality_type=args.modality_types,
        task=args.task,
        resolution=args.image_size,
        attribute_type=args.attribute_type,
        transform=None,
        file_list=val_files,
        oversample_minority_factor=1,
    )

    test_dataset = Harvard_GF(
        test_data_dir,
        modality_type=args.modality_types,
        task=args.task,
        resolution=args.image_size,
        attribute_type=args.attribute_type,
        transform=None,
        file_list=None,
        oversample_minority_factor=1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"\nFinal train samples: {len(train_dataset)}")
    print(f"Final validation samples: {len(val_dataset)}")
    print(f"Final test samples: {len(test_dataset)}")

    # ------------------------------------------------------------
    # Model/optimizer/loss
    # ------------------------------------------------------------
    model = build_model(args, device)
    criterion = nn.MSELoss(reduction="none")

    if args.ablation == "baseline":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        param_groups = build_param_groups_for_fair3d(model, base_lr=args.lr_best)
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=args.lr_best,
            weight_decay=args.weight_decay,
        )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    best_val_mae = float("inf")
    best_epoch = -1
    bad_epochs = 0
    best_ckpt_path = os.path.join(args.result_dir, "best_checkpoint.pth")

    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n================ Epoch {epoch}/{args.epochs} ================")

        train_metrics, train_outputs = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            ablation=args.ablation,
            train=True,
        )

        val_metrics, val_outputs = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            ablation=args.ablation,
            train=False,
        )

        print_metrics("TRAIN", train_metrics)
        print_metrics("VAL", val_metrics)

        # AFF update every K epochs for full model only.
        aff_info = None
        if args.ablation == "full" and epoch % args.rank_update_K == 0:
            group_mae = {
                int(g): float(m["mae"])
                for g, m in train_metrics["group_metrics"].items()
            }
            aff_info = update_aff_learning_rates(
                optimizer=optimizer,
                group_mae=group_mae,
                base_lr=args.lr_best,
                lr_worst=args.lr_worst,
                max_boost=args.max_boost,
                sigmoid_midpoint=args.sigmoid_midpoint,
                sigmoid_slope=args.sigmoid_slope,
            )

            if aff_info is not None:
                print(
                    "AFF update: "
                    f"worst_group={aff_info['worst_group']}, "
                    f"best_group={aff_info['best_group']}, "
                    f"disparity={aff_info['mae_disparity']:.4f}, "
                    f"scale={aff_info['scale01']:.4f}, "
                    f"boosted_lr={aff_info['boosted_lr']:.8f}"
                )

        row = {
            "epoch": epoch,
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "train_mae_disparity": train_metrics["mae_disparity"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "val_mae_disparity": val_metrics["mae_disparity"],
            "aff_info": aff_info,
        }
        history.append(row)

        # Checkpoint selection by internal validation MAE only.
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            bad_epochs = 0

            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_metrics=val_metrics,
                args=args,
            )
            print(f"Saved new best checkpoint: epoch={epoch}, val_mae={best_val_mae:.4f}")
        else:
            bad_epochs += 1
            print(f"No validation improvement. bad_epochs={bad_epochs}/{args.patience}")

        if bad_epochs >= args.patience:
            print("Early stopping triggered.")
            break

    with open(os.path.join(args.result_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ------------------------------------------------------------
    # Final test evaluation using validation-selected checkpoint
    # ------------------------------------------------------------
    print("\n================ Final TEST evaluation ================")
    ckpt = load_best_checkpoint(model, best_ckpt_path, device)
    best_epoch = ckpt.get("epoch", best_epoch)

    test_metrics, test_outputs = run_epoch(
        model=model,
        loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        ablation=args.ablation,
        train=False,
    )

    print_metrics("TEST", test_metrics)

    test_npz_path = os.path.join(args.result_dir, "test_pred_gt_best_model.npz")
    save_npz_outputs(test_npz_path, test_outputs, test_metrics, best_epoch=best_epoch)
    print(f"Saved test predictions to: {test_npz_path}")

    # ------------------------------------------------------------
    # Bootstrap uncertainty
    # ------------------------------------------------------------
    print("\nRunning bootstrap uncertainty estimation...")
    boot = bootstrap_confidence_intervals(
        y_true=test_outputs["gt"],
        y_pred=test_outputs["pred"],
        attrs=test_outputs["attr"],
        n_bootstrap=args.bootstrap_repeat_times,
        seed=args.seed,
    )

    boot_path = os.path.join(args.result_dir, "bootstrap_summary.json")
    with open(boot_path, "w") as f:
        json.dump(boot, f, indent=2)

    print(f"Saved bootstrap summary to: {boot_path}")
    print("\nBootstrap summary:")
    for k, v in boot.items():
        print(
            f"  {k}: mean={v['mean']:.4f}, std={v['std']:.4f}, "
            f"95% CI=[{v['ci_low']:.4f}, {v['ci_high']:.4f}]"
        )

    print("\n✅ Training and final test evaluation complete.")


if __name__ == "__main__":
    main()
