
import os
import json
import time
import random
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torchvision.transforms.functional as TF


from src.data_handler import Harvard_DR_Fairness
from src import logger


from model import FairResNet3D_R18_Attn



def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class Random3DOctAugmentation:
    """
    Applies small, realistic per-slice perturbations while preserving slice order:
      - in-plane rotation: +/- rotation_range degrees
      - translation: up to translate_range * (H/W)
      - intensity scaling: 1 +/- intensity_range
      - gaussian noise: N(0, noise_std)
    Input expected: Tensor/array shaped [S, H, W] or [C, H, W]
    """
    def __init__(
        self,
        rotation_range=10,
        translate_range=0.05,
        intensity_range=0.10,
        noise_std=0.02,
    ):
        self.rotation_range = rotation_range
        self.translate_range = translate_range
        self.intensity_range = intensity_range
        self.noise_std = noise_std

    def __call__(self, volume):
        if not isinstance(volume, torch.Tensor):
            volume = torch.tensor(volume, dtype=torch.float32)

        # volume: [S, H, W]
        angle = random.uniform(-self.rotation_range, self.rotation_range)

        # translate in pixels based on H/W
        max_dx = self.translate_range * volume.shape[-2]
        max_dy = self.translate_range * volume.shape[-1]
        translations = (
            float(random.uniform(-max_dx, max_dx)),
            float(random.uniform(-max_dy, max_dy)),
        )

        # intensity scale
        factor = 1.0 + random.uniform(-self.intensity_range, self.intensity_range)

        out_slices = []
        for s in range(volume.shape[0]):
            img = volume[s]  # [H, W]

            # TF.rotate/affine expect [C,H,W]
            img = TF.rotate(img.unsqueeze(0), angle).squeeze(0)
            img = TF.affine(
                img.unsqueeze(0),
                angle=0.0,
                translate=translations,
                scale=1.0,
                shear=[0.0, 0.0],
            ).squeeze(0)

            img = img * factor

            if self.noise_std > 0:
                img = img + torch.randn_like(img) * self.noise_std

            out_slices.append(img)

        return torch.stack(out_slices, dim=0)  # [S, H, W]



# AFF (optimizer-level, train-only)

def rebased_sigmoid_clip(x: float, mid: float = 0.25, k: float = 6.0) -> float:
    """
    Map x >= 0 to [0,1] with a sigmoid that's 'zeroed' at 0 and crosses ~0.5 near `mid`.
    """
    sx = 1.0 / (1.0 + np.exp(-k * (x - mid)))
    s0 = 1.0 / (1.0 + np.exp(-k * (0.0 - mid)))
    denom = max(1e-12, 1.0 - s0)
    return float(np.clip((sx - s0) / denom, 0.0, 1.0))


def build_param_groups_for_fair3d(model: nn.Module, base_lr: float):
    """
    Creates param groups for AFF:
      - group_id = -1 => shared (all params except group-specific calibration)
      - group_id = 0..G-1 => per-group calibration params (LR-boosted by AFF)
    """
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
        param_groups.append({"params": list(gmod.parameters()), "group_id": gid, "lr": base_lr})
    return param_groups


def compute_group_mae_and_residuals(preds, targets, gids):
    """
    preds/targets: [N, 52], gids: [N]
    Returns:
      group_mae: dict gid->MAE
      mu_res: dict gid->mean residual vector (targets - preds) (52,)
    """
    group_mae = {}
    mu_res = {}
    for g in np.unique(gids):
        m = gids == g
        if m.sum() > 0:
            group_mae[int(g)] = float(np.mean(np.abs(preds[m] - targets[m])))
            mu_res[int(g)] = np.mean(targets[m] - preds[m], axis=0)
    return group_mae, mu_res


def aff_update_from_train_epoch(
    optimizer,
    train_preds,
    train_gts,
    train_attrs,
    attr_col: int,
    base_lr: float,
    lr_worst: float,
    mid: float = 0.25,
    k: float = 6.0,
    max_boost: float = 2.0,
    min_group_n: int = 10,
):
    """
    AFF update based ONLY on training-set predictions (prevents test leakage).
    Updates optimizer param groups that have `group_id` (0..G-1).
    """
    gids = train_attrs[:, attr_col].astype(int)

    valid = []
    for g in np.unique(gids):
        if (gids == g).sum() >= min_group_n:
            valid.append(int(g))
    if len(valid) < 2:
        return None, None, 0.0, 0.0, 1.0, {}

    group_mae, mu_res = compute_group_mae_and_residuals(train_preds, train_gts, gids)
    group_mae = {g: group_mae[g] for g in valid if g in group_mae}
    if len(group_mae) < 2:
        return None, None, 0.0, 0.0, 1.0, {}

    best_gid = min(group_mae, key=group_mae.get)
    worst_gid = max(group_mae, key=group_mae.get)

    vf_gap_raw = float(np.mean(np.abs(mu_res[worst_gid] - mu_res[best_gid])))

    clipped = rebased_sigmoid_clip(max(0.0, vf_gap_raw), mid=mid, k=k)
    boost_factor = min(1.0 + clipped, max_boost)
    boosted_lr = lr_worst * boost_factor

    new_lrs = {}
    for pg in optimizer.param_groups:
        gid = pg.get("group_id", None)
        if gid is None:
            continue
        if gid == worst_gid:
            pg["lr"] = boosted_lr
            new_lrs[int(gid)] = boosted_lr
        else:
            pg["lr"] = base_lr
            new_lrs[int(gid)] = base_lr

    return best_gid, worst_gid, vf_gap_raw, clipped, boost_factor, new_lrs


# Train / Eval

def train_one_epoch(model, optimizer, scaler, loader, device, criterion):
    model.train()
    loss_vals = []

    preds_all, gts_all, attrs_all = [], [], []

    for (x, y, attr, md) in loader:  # md is ignored (no severity)
        x = x.to(device)
        y = y.to(device)

        if isinstance(attr, list):
            attr = torch.stack(attr, dim=1)
        attr = attr.float().to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            # model signature: model(x, group_id) or model(x, attr[:,0])
            pred = model(x, attr[:, 0])
            # criterion is MSELoss(reduction='none'); average over outputs + batch
            loss = criterion(pred, y).mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_vals.append(loss.item())

        preds_all.append(pred.detach().cpu().numpy())
        gts_all.append(y.detach().cpu().numpy())
        attrs_all.append(attr.detach().cpu().numpy())

    preds_all = np.concatenate(preds_all, axis=0)
    gts_all = np.concatenate(gts_all, axis=0)
    attrs_all = np.concatenate(attrs_all, axis=0).astype(int)

    mse = mean_squared_error(gts_all, preds_all)
    mae = mean_absolute_error(gts_all, preds_all)
    r2 = r2_score(gts_all, preds_all)

    return float(np.mean(loss_vals)), mse, mae, r2, preds_all, gts_all, attrs_all


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_vals = []

    preds_all, gts_all, attrs_all = [], [], []

    for (x, y, attr, md) in loader:
        x = x.to(device)
        y = y.to(device)

        if isinstance(attr, list):
            attr = torch.stack(attr, dim=1)
        attr = attr.float().to(device)

        with torch.cuda.amp.autocast():
            pred = model(x, attr[:, 0])
            loss = criterion(pred, y).mean()

        loss_vals.append(loss.item())
        preds_all.append(pred.detach().cpu().numpy())
        gts_all.append(y.detach().cpu().numpy())
        attrs_all.append(attr.detach().cpu().numpy())

    preds_all = np.concatenate(preds_all, axis=0)
    gts_all = np.concatenate(gts_all, axis=0)
    attrs_all = np.concatenate(attrs_all, axis=0).astype(int)

    mse = mean_squared_error(gts_all, preds_all)
    mae = mean_absolute_error(gts_all, preds_all)
    r2 = r2_score(gts_all, preds_all)

    return float(np.mean(loss_vals)), mse, mae, r2, preds_all, gts_all, attrs_all



def parse_args():
    p = argparse.ArgumentParser("VF Fair Training (AFF, train-only)")

    p.add_argument("--data_dir", type=str, required=True, help="root folder containing train/ and test/")
    p.add_argument("--result_dir", type=str, default="./results_aff")

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)

    p.add_argument("--image_size", type=int, default=200)
    p.add_argument("--task", type=str, default="tds")  # expects 52-dim targets
    p.add_argument("--modality_type", type=str, default="oct_bscans_3d")

    p.add_argument("--attribute_type", type=str, default="race")  # dataset determines attr ordering
    p.add_argument("--need_balance", type=str2bool, default=False)
    p.add_argument("--dataset_proportion", type=float, default=1.0)

    # Optim
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--step_size", type=int, default=30)
    p.add_argument("--gamma", type=float, default=0.1)

    # AFF
    p.add_argument("--use_aff", type=str2bool, default=True)
    p.add_argument("--attr_col", type=int, default=0, help="which column in attr tensor to use for subgrouping (race=0 in your dataset)")
    p.add_argument("--rank_update_K", type=int, default=1, help="update AFF every K epochs")
    p.add_argument("--lr_best", type=float, default=5e-5, help="base lr to set for non-worst groups during AFF update")
    p.add_argument("--lr_worst", type=float, default=8e-5, help="baseline worst lr before boost")
    p.add_argument("--aff_mid", type=float, default=0.25)
    p.add_argument("--aff_k", type=float, default=6.0)
    p.add_argument("--aff_max_boost", type=float, default=2.0)
    p.add_argument("--aff_min_group_n", type=int, default=10)

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    set_random_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger.configure(dir=args.result_dir, log_suffix="train")
    logger.log(f"Seed: {args.seed}")
    logger.log(f"Device: {device}")

    with open(os.path.join(args.result_dir, "args_train.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Augmentation
    train_transform = Random3DOctAugmentation(
        rotation_range=10, translate_range=0.05, intensity_range=0.10, noise_std=0.02
    )

    # Datasets
    train_ds = Harvard_DR_Fairness(
        data_path=os.path.join(args.data_dir, "train"),
        subset="train",
        modality_type=args.modality_type,
        task=args.task,
        resolution=args.image_size,
        attribute_type=args.attribute_type,
        needBalance=args.need_balance,
        dataset_proportion=args.dataset_proportion,
        transform=train_transform,
    )

    test_ds = Harvard_DR_Fairness(
        data_path=os.path.join(args.data_dir, "test"),
        subset="test",
        modality_type=args.modality_type,
        task=args.task,
        resolution=args.image_size,
        attribute_type=args.attribute_type,
        needBalance=False,
        dataset_proportion=1.0,
        transform=None,  # NO AUG in eval
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    logger.log(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")

    model = FairResNet3D_R18_Attn(
        in_channels=1,
        out_dim=52,
        attr_dim=1,
        attr_emb_dim=128,
        pretrained_backbone=True,
        num_groups=3,
        group_col=0,
        use_group_calib=True,
        use_feature_norm=True,
    ).to(device)

    # Loss
    criterion = nn.MSELoss(reduction="none")

    # Optimizer with group_id param groups (for AFF)
    param_groups = build_param_groups_for_fair3d(model, base_lr=args.lr)
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    scaler = torch.cuda.amp.GradScaler()

    best_test_loss = float("inf")
    best_epoch = -1

    for epoch in range(args.epochs):
        t0 = time.time()

        tr_loss, tr_mse, tr_mae, tr_r2, tr_preds, tr_gts, tr_attrs = train_one_epoch(
            model, optimizer, scaler, train_loader, device, criterion
        )

        te_loss, te_mse, te_mae, te_r2, te_preds, te_gts, te_attrs = evaluate(
            model, test_loader, device, criterion
        )

        # Step scheduler (non-AFF)
        scheduler.step()

        
        if args.use_aff and ((epoch + 1) % args.rank_update_K == 0):
            best_gid, worst_gid, vf_gap_raw, clipped, boost_factor, new_lrs = aff_update_from_train_epoch(
                optimizer=optimizer,
                train_preds=tr_preds,
                train_gts=tr_gts,
                train_attrs=tr_attrs,
                attr_col=args.attr_col,
                base_lr=args.lr_best,
                lr_worst=args.lr_worst,
                mid=args.aff_mid,
                k=args.aff_k,
                max_boost=args.aff_max_boost,
                min_group_n=args.aff_min_group_n,
            )

            if worst_gid is not None:
                logger.log(
                    f"[AFF train-only] best={best_gid} worst={worst_gid} "
                    f"gap={vf_gap_raw:.4f} clipped={clipped:.4f} boost={boost_factor:.3f} lrs={new_lrs}"
                )

        # Track best on test loss (reporting only; no AFF uses test)
        if te_loss < best_test_loss:
            best_test_loss = te_loss
            best_epoch = epoch
            ckpt_path = os.path.join(args.result_dir, "best_checkpoint.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_loss": te_loss,
                    "test_mae": te_mae,
                    "test_mse": te_mse,
                    "test_r2": te_r2,
                    "args": vars(args),
                },
                ckpt_path,
            )

            np.savez(
                os.path.join(args.result_dir, "pred_gt_best_epoch.npz"),
                test_pred=te_preds,
                test_gt=te_gts,
                test_attr=te_attrs,
                epoch=epoch,
                test_mae=te_mae,
                test_mse=te_mse,
                test_r2=te_r2,
            )

        # Log
        logger.logkv("epoch", epoch)
        logger.logkv("train_loss", round(tr_loss, 6))
        logger.logkv("train_mse", round(tr_mse, 6))
        logger.logkv("train_mae", round(tr_mae, 6))
        logger.logkv("train_r2", round(tr_r2, 6))
        logger.logkv("test_loss", round(te_loss, 6))
        logger.logkv("test_mse", round(te_mse, 6))
        logger.logkv("test_mae", round(te_mae, 6))
        logger.logkv("test_r2", round(te_r2, 6))
        logger.logkv("best_epoch", best_epoch)
        logger.logkv("best_test_loss", round(best_test_loss, 6))
        logger.dumpkvs()

        dt = time.time() - t0
        logger.log(
            f"Epoch {epoch:03d} | "
            f"train MAE {tr_mae:.4f} R2 {tr_r2:.4f} | "
            f"test MAE {te_mae:.4f} R2 {te_r2:.4f} | "
            f"time {dt:.1f}s"
        )

    logger.log(f"Done. Best epoch: {best_epoch}, best test loss: {best_test_loss:.6f}")


if __name__ == "__main__":
    main()
