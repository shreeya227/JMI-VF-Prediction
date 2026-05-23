"""
Test-only evaluation script.
Loads best_checkpoint.pth and runs final test evaluation.
Saves test predictions to test_pred_gt_best_model.npz.
Run with same args as training but skips all training/validation data loading.
"""

import os
import argparse
import random
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys
sys.path.append(".")

from src.modules import *
from src.data_handler import Harvard_GF
from src.model import FairResNet3D_R18_Attn
from src.model import ResNet3D_Baseline
from src import logger


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)


def reduce_loss_per_sample(loss_tensor, task):
    if task == "tds":
        if loss_tensor.dim() == 1:
            return loss_tensor
        return loss_tensor.mean(dim=1)
    if loss_tensor.dim() == 0:
        return loss_tensor.unsqueeze(0)
    return loss_tensor.view(loss_tensor.shape[0], -1).mean(dim=1)


def build_param_groups_for_fair3d(model, base_lr):
    group_specific = []
    if hasattr(model, "group_specific_layers") and isinstance(model.group_specific_layers, nn.ModuleList):
        for gid, gmod in enumerate(model.group_specific_layers):
            group_specific.append((gid, gmod))

    specific_param_ids = set()
    for _, gmod in group_specific:
        for p in gmod.parameters(recurse=True):
            specific_param_ids.add(id(p))

    shared_params = [p for p in model.parameters() if id(p) not in specific_param_ids]
    param_groups = [{"params": shared_params, "group_id": -1, "lr": base_lr}]
    for gid, gmod in group_specific:
        param_groups.append({"params": gmod.parameters(), "group_id": gid, "lr": base_lr})
    return param_groups


parser = argparse.ArgumentParser(description="Test-Only Evaluation")
parser.add_argument("--gpu_id", default=0, type=int)
parser.add_argument("--seed", default=123, type=int)
parser.add_argument("--result_dir", default="./results", type=str)
parser.add_argument("--test_data_dir", default="/medailab/medailab/shilab/Harvard-GF/Test", type=str)
parser.add_argument("--model_type", default="efficientnet", type=str)
parser.add_argument("--task", default="tds", type=str)
parser.add_argument("--image_size", default=200, type=int)
parser.add_argument("--modality_types", default="oct_bscans_3d", type=str)
parser.add_argument("--attribute_type", default="race", type=str)
parser.add_argument("--num_classes", default=2, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--workers", default=8, type=int)
parser.add_argument("--bootstrap_repeat_times", default=100, type=int)
parser.add_argument("--ablation", default="full", type=str)
parser.add_argument("--lr_best", default=5e-5, type=float)
parser.add_argument("--drop_path", default=0.3, type=float)


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    set_random_seed(args.seed)
    os.makedirs(args.result_dir, exist_ok=True)
    logger.configure(dir=args.result_dir, log_suffix="test_only")

    print(f"Running test-only evaluation for: {args.result_dir}")
    print(f"attribute_type: {args.attribute_type}, ablation: {args.ablation}")

    # ── Test dataset only ──────────────────────────────────────────
    test_havo_dataset = Harvard_GF(
        args.test_data_dir,
        modality_type=args.modality_types,
        task=args.task,
        resolution=args.image_size,
        attribute_type=args.attribute_type,
        transform=None,
        oversample_minority_factor=1,
    )

    test_dataset_loader = torch.utils.data.DataLoader(
        test_havo_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Test samples: {len(test_havo_dataset)}")

    # ── Build model ────────────────────────────────────────────────
    out_dim = 52  # TDS task
    criterion = nn.MSELoss(reduction="none")

    if args.ablation == "baseline":
        model = ResNet3D_Baseline(
            in_channels=1, out_dim=out_dim,
            num_severity=3, severity_emb_dim=64,
            pretrained_backbone=False,  # no pretrain needed for inference
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_best)
    else:
        num_groups = 2 if args.attribute_type == "hispanic" else 3
        model = FairResNet3D_R18_Attn(
            in_channels=1, out_dim=out_dim,
            num_groups=num_groups, attr_emb_dim=128,
            num_severity=3, severity_emb_dim=64,
            pretrained_backbone=False,
        )
        param_groups = build_param_groups_for_fair3d(model, base_lr=args.lr_best)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr_best)

    model = model.to(device)

    # ── Load best checkpoint ───────────────────────────────────────
    ckpt_path = os.path.join(args.result_dir, "best_checkpoint.pth")
    best_ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    print(f"Loaded checkpoint: epoch={best_ckpt.get('epoch')}, "
          f"val_mae={best_ckpt.get('val_mae', 'N/A'):.4f}")

    # ── Test evaluation ────────────────────────────────────────────
    model.eval()
    preds, gts, attrs, mds, severities = [], [], [], [], []

    with torch.no_grad():
        for input, target, attr, md, severity in test_dataset_loader:
            input = input.to(device)
            target = target.to(device)
            severity = severity.long().to(device)

            if attr.dim() == 1:
                attr = attr.unsqueeze(1)
            attr = attr.float().to(device)

            pred = model(input, attr[:, 0].long(), severity)

            preds.append(pred.cpu().numpy())
            gts.append(target.cpu().numpy())
            attrs.append(attr.cpu().numpy())
            mds.append(md.numpy())
            severities.append(severity.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    attrs = np.concatenate(attrs, axis=0).astype(int)
    mds = np.concatenate(mds, axis=0)
    severities = np.concatenate(severities, axis=0).astype(int)

    test_mae = mean_absolute_error(gts, preds)
    test_mse = mean_squared_error(gts, preds)
    test_r2 = r2_score(gts, preds)
    test_rmse = np.sqrt(test_mse)

    print(f"\nFinal TEST: MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, "
          f"MSE={test_mse:.4f}, R2={test_r2:.4f}")

    gids = attrs[:, 0] if attrs.ndim > 1 else attrs
    test_group_mae, test_group_mse, test_group_r2 = {}, {}, {}

    for g in np.unique(gids):
        mask = gids == g
        test_group_mae[int(g)] = mean_absolute_error(gts[mask], preds[mask])
        test_group_mse[int(g)] = mean_squared_error(gts[mask], preds[mask])
        test_group_r2[int(g)] = r2_score(gts[mask], preds[mask])
        print(f"  Group {int(g)} - MAE: {test_group_mae[int(g)]:.4f}, "
              f"MSE: {test_group_mse[int(g)]:.4f}, R2: {test_group_r2[int(g)]:.4f}")

    mae_values = list(test_group_mae.values())
    test_mae_disparity = max(mae_values) - min(mae_values)
    print(f"MAE Disparity: {test_mae_disparity:.4f}")

    # ── Save predictions ───────────────────────────────────────────
    test_save_path = os.path.join(args.result_dir, "test_pred_gt_best_model.npz")
    np.savez(
        test_save_path,
        test_pred=preds,
        test_gt=gts,
        test_attr=attrs,
        test_md=mds,
        best_epoch=best_ckpt.get("epoch"),
        test_mae=test_mae,
        test_mse=test_mse,
        test_r2=test_r2,
    )
    print(f"\nSaved test predictions to: {test_save_path}")

    # ── Bootstrap ──────────────────────────────────────────────────
    print("Running bootstrap...")
    (test_mse_b, test_mae_b, test_r2_b,
     test_es_mse, test_es_mae, test_es_r2,
     test_mse_by_attrs, test_mae_by_attrs, test_r2_by_attrs,
     test_between_group_disparity,
     test_mse_std, test_es_mse_std, test_mae_std, test_es_mae_std,
     test_mse_by_attrs_std, test_between_group_disparity_std) = bootstrap_regression_performance(
        preds, gts, attrs,
        bootstrap_repeat_times=args.bootstrap_repeat_times)

    print(f"\nBootstrap results (mean ± std):")
    print(f"  MAE:  {test_mae_b:.4f} ± {test_mae_std:.4f}")
    print(f"  RMSE: {np.sqrt(test_mse_b):.4f} ± {np.sqrt(test_mse_std):.4f}")
    print(f"  R2:   {test_r2_b:.4f}")
    print(f"  ES-MAE: {test_es_mae[0]:.4f}")
    print(f"  ES-RMSE: {test_es_mse[0]:.4f}")

    if test_mae_by_attrs is not None and len(test_mae_by_attrs) > 0:
        print(f"\nBootstrap group MAE (mean ± std):")
        for gi, (m, s) in enumerate(zip(test_mae_by_attrs[0], test_mse_by_attrs_std[0])):
            print(f"  Group {gi}: MAE={m:.4f}")

    print("\n Test-only evaluation complete.")
