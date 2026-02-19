import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_handler import Harvard_DR_Fairness
from src.modules import FairResNet3D_R18_Attn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Seed for variability

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def rebased_sigmoid_clip(x, mid=0.25, k=6.0):
    sx = 1.0 / (1.0 + np.exp(-k * (x - mid)))
    s0 = 1.0 / (1.0 + np.exp(-k * (0.0 - mid)))
    return np.clip((sx - s0) / (1.0 - s0 + 1e-12), 0.0, 1.0)

def compute_group_mae(preds, targets, group_ids):
    group_mae = {}
    for g in np.unique(group_ids):
        mask = group_ids == g
        if mask.sum() > 1:
            group_mae[int(g)] = mean_absolute_error(targets[mask], preds[mask])
    return group_mae

def adjust_aff_learning_rate(
    optimizer,
    vf_gap,
    worst_gid,
    base_lr,
    worst_lr,
    max_boost=2.0
):
    clipped = rebased_sigmoid_clip(vf_gap)
    boost_factor = min(1.0 + clipped, max_boost)
    boosted_lr = worst_lr * boost_factor

    for pg in optimizer.param_groups:
        gid = pg.get("group_id", None)
        if gid is None:
            continue
        if gid == worst_gid:
            pg["lr"] = boosted_lr
        else:
            pg["lr"] = base_lr

    return clipped, boost_factor


# Training

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    preds, gts, attrs = [], [], []

    for x, y, attr in loader:
        x = x.to(device)
        y = y.to(device)
        attr = attr.to(device)

        optimizer.zero_grad()
        out = model(x, attr[:, 0])
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        preds.append(out.detach().cpu().numpy())
        gts.append(y.cpu().numpy())
        attrs.append(attr.cpu().numpy())

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    attrs = np.concatenate(attrs)

    return preds, gts, attrs

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    preds, gts, attrs = [], [], []

    for x, y, attr in loader:
        x = x.to(device)
        y = y.to(device)
        attr = attr.to(device)

        out = model(x, attr[:, 0])
        preds.append(out.cpu().numpy())
        gts.append(y.cpu().numpy())
        attrs.append(attr.cpu().numpy())

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    attrs = np.concatenate(attrs)

    mse = mean_squared_error(gts, preds)
    mae = mean_absolute_error(gts, preds)
    r2 = r2_score(gts, preds)

    return mse, mae, r2, preds, gts, attrs


class Random3DOctAugmentation:
    def __init__(self, rotation=5, translate=0.05, intensity=0.10, noise_std=0.02):
        self.rotation = rotation
        self.translate = translate
        self.intensity = intensity
        self.noise_std = noise_std

    def __call__(self, volume):
        if not isinstance(volume, torch.Tensor):
            volume = torch.tensor(volume, dtype=torch.float32)

        angle = random.uniform(-self.rotation, self.rotation)
        dx = random.uniform(-self.translate, self.translate) * volume.shape[1]
        dy = random.uniform(-self.translate, self.translate) * volume.shape[2]

        factor = 1.0 + random.uniform(-self.intensity, self.intensity)

        augmented = []
        for slice_idx in range(volume.shape[0]):
            img = volume[slice_idx]
            img = torch.nn.functional.rotate(img.unsqueeze(0), angle).squeeze(0)
            img = torch.nn.functional.affine(
                img.unsqueeze(0),
                angle=0,
                translate=[dx, dy],
                scale=1.0,
                shear=0
            ).squeeze(0)

            img = img * factor
            img += torch.randn_like(img) * self.noise_std
            augmented.append(img)

        return torch.stack(augmented)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr_worst", type=float, default=8e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Dataset
    train_set = Harvard_GF(os.path.join(args.data_dir, "train"))
    test_set  = Harvard_GF(os.path.join(args.data_dir, "test"))

    train_transform = Random3DOctAugmentation(rotation_range=10, translate_range=0.05, intensity_range=0.10, noise_std=0.02)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, transform=train_transform)
    test_loader  = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size)

    # Model
    model = FairResNet3D_R18_Attn(
        in_channels=1,
        out_dim=52,
        attr_dim=1,
        attr_emb_dim=128,
        pretrained_backbone=True
    ).to(device)

    # Optimizer
    param_groups = [
        {"params": model.shared_layers.parameters(), "group_id": -1, "lr": args.lr},
        {"params": model.group_specific_layers[0].parameters(), "group_id": 0, "lr": args.lr},
        {"params": model.group_specific_layers[1].parameters(), "group_id": 1, "lr": args.lr},
        {"params": model.group_specific_layers[2].parameters(), "group_id": 2, "lr": args.lr},
    ]

    optimizer = AdamW(param_groups)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = nn.MSELoss()

    # Training Loop
    for epoch in range(args.epochs):

        train_preds, train_gts, train_attrs = train_one_epoch(
            model, train_loader, optimizer, criterion
        )

        mse, mae, r2, preds, gts, attrs = evaluate(
            model, test_loader, criterion
        )

        print(f"Epoch {epoch}: MAE={mae:.4f}, R2={r2:.4f}")

        # -------- AFF --------
        group_mae = compute_group_mae(preds, gts, attrs[:, 0])
        if len(group_mae) >= 2:
            best_gid  = min(group_mae, key=group_mae.get)
            worst_gid = max(group_mae, key=group_mae.get)

            mask_best  = attrs[:, 0] == best_gid
            mask_worst = attrs[:, 0] == worst_gid

            mu_best  = np.mean(gts[mask_best]  - preds[mask_best], axis=0)
            mu_worst = np.mean(gts[mask_worst] - preds[mask_worst], axis=0)

            vf_gap = np.mean(np.abs(mu_worst - mu_best))

            clipped, boost = adjust_aff_learning_rate(
                optimizer,
                vf_gap,
                worst_gid,
                base_lr=args.lr,
                worst_lr=args.lr_worst
            )

            print(f"AFF: gap={vf_gap:.4f}, boost={boost:.3f}")

        scheduler.step()