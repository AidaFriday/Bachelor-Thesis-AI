# train_lfw_arcface_cpu.py
import math, argparse, time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

# ----------------------------
# ArcFace head (classification)
# ----------------------------
class ArcMarginProduct(nn.Module):
    def __init__(self, in_feat, out_feat, s=64.0, m=0.50, easy_margin=False):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_feat, in_feat))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb, label):
        emb = F.normalize(emb, p=2, dim=1)
        W = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(emb, W).clamp(-1, 1)
        sine = torch.sqrt((1.0 - cosine ** 2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        onehot = torch.zeros_like(cosine)
        onehot.scatter_(1, label.view(-1, 1), 1.0)
        logits = onehot * phi + (1.0 - onehot) * cosine
        logits *= self.s
        return logits

# ----------------------------
# Backbone: ResNet-18 → embedding
# ----------------------------
class FaceNet(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=False):
        super().__init__()
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.embed = nn.Linear(in_features, embedding_dim, bias=False)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.embed(x)
        x = self.bn(x)
        return x

# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def build_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

def _maybe_fix_nested_root(root):
    inner = Path(root) / "lfw-deepfunneled"
    if inner.exists() and inner.is_dir():
        subdirs = [p for p in inner.iterdir() if p.is_dir()]
        if len(subdirs) > 0:
            return str(inner)
    return root

def split_by_identity(dataset: ImageFolder, train_id_frac=0.8, min_images_per_id=2, seed=42):
    by_class = defaultdict(list)
    for idx, (_, cls) in enumerate(dataset.samples):
        by_class[cls].append(idx)

    valid_classes = [c for c, idxs in by_class.items() if len(idxs) >= min_images_per_id]
    if len(valid_classes) == 0:
        return [], [], [], []

    rng = np.random.default_rng(seed)
    rng.shuffle(valid_classes)
    split_point = max(1, int(len(valid_classes) * train_id_frac))
    train_classes = set(valid_classes[:split_point])
    val_classes   = set(valid_classes[split_point:]) or set(valid_classes[-1:])

    train_idx, val_idx = [], []
    for c, idxs in by_class.items():
        if c in train_classes: train_idx.extend(idxs)
        elif c in val_classes: val_idx.extend(idxs)
    return train_idx, val_idx, sorted(list(train_classes)), sorted(list(val_classes))

class RemapSubset(Dataset):
    """
    Pickle-safe dataset wrapper that:
      - selects a subset of ImageFolder by indices
      - remaps class ids via a provided dict (for train set)
    """
    def __init__(self, base: ImageFolder, indices, class_remap=None):
        self.base = base
        self.indices = list(indices)
        self.class_remap = class_remap  # dict old->new or None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        if self.class_remap is not None:
            y = self.class_remap[y]
        return x, y

def make_loaders(img_size=112, batch_size=64, root="./archive/lfw-deepfunneled",
                 train_id_frac=0.8, min_images_per_id=2, max_train_ids=None,
                 num_workers=0, seed=42):
    tfm = build_transforms(img_size)
    root = _maybe_fix_nested_root(root)
    full_dataset = ImageFolder(root=root, transform=tfm)
    if len(full_dataset) == 0:
        raise RuntimeError(f"No images under {root}")

    train_idx, val_idx, train_classes, val_classes = split_by_identity(
        full_dataset, train_id_frac=train_id_frac, min_images_per_id=min_images_per_id, seed=seed
    )

    # Optionally cap train identities
    if train_classes and max_train_ids is not None and len(train_classes) > max_train_ids:
        rng = np.random.default_rng(seed)
        keep = set(rng.choice(train_classes, size=max_train_ids, replace=False))
        train_idx = [i for i in train_idx if full_dataset.samples[i][1] in keep]
        train_classes = sorted(list(keep))

    if len(train_idx) == 0:
        # Fallback to random image split (so you can still run)
        print("[WARN] Identity split empty; falling back to random image split.")
        n_total = len(full_dataset)
        n_train = max(1, int(0.8 * n_total))
        n_val = n_total - n_train
        gen = torch.Generator().manual_seed(seed)
        train_subset, val_subset = torch.utils.data.random_split(full_dataset, [n_train, n_val], generator=gen)
        # Approximate num classes (not exact under random split)
        num_train_classes = len(full_dataset.classes)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
        val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False,num_workers=num_workers, pin_memory=False)
        print(f"[INFO] Random split: train images={len(train_subset)}, val images={len(val_subset)}")
        return full_dataset, train_loader, val_loader, num_train_classes, list(range(len(val_subset)))

    # Build class remap table (old id -> [0..num_train_classes-1])
    class_remap = {old: new for new, old in enumerate(train_classes)}
    train_set = RemapSubset(full_dataset, train_idx, class_remap=class_remap)
    val_set   = RemapSubset(full_dataset, val_idx, class_remap=None)

    print(f"[INFO] Train identities: {len(train_classes)} | Train images: {len(train_idx)}")
    print(f"[INFO] Val   identities: {len(val_classes)} | Val   images: {len(val_idx)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    num_train_classes = len(train_classes)
    return full_dataset, train_loader, val_loader, num_train_classes, val_idx

# ---------- Build verification pairs from VAL identities ----------
def build_verification_pairs(dataset: ImageFolder, val_indices, n_pos=2000, n_neg=2000, seed=42):
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for idx in val_indices:
        _, c = dataset.samples[idx]
        by_class[c].append(idx)

    classes = [c for c, idxs in by_class.items() if len(idxs) >= 2]
    if len(classes) == 0:
        raise RuntimeError("No validation identities with >=2 images to make positive pairs.")

    pos_pairs = []
    for _ in range(n_pos):
        c = rng.choice(classes)
        i1, i2 = rng.choice(by_class[c], size=2, replace=False)
        pos_pairs.append((i1, i2, 1))

    neg_pairs = []
    val_classes = list(by_class.keys())
    for _ in range(n_neg):
        c1, c2 = rng.choice(val_classes, size=2, replace=False)
        i1 = rng.choice(by_class[c1])
        i2 = rng.choice(by_class[c2])
        neg_pairs.append((i1, i2, 0))

    return pos_pairs + neg_pairs

@torch.no_grad()
def evaluate_verification(model, dataset: ImageFolder, pairs, batch_size=128, device="cpu"):
    model.eval()
    uniq = sorted({i for (i,_,_) in pairs} | {j for (_,j,_) in pairs})
    imgs = torch.stack([dataset[inx][0] for inx in uniq], dim=0).to(device)

    embs = []
    for s in range(0, len(imgs), batch_size):
        embs.append(model(imgs[s:s+batch_size]))
    embs = F.normalize(torch.cat(embs, dim=0), p=2, dim=1)

    idx_to_row = {idx: r for r, idx in enumerate(uniq)}
    sims, labels = [], []
    for i, j, lab in pairs:
        e1 = embs[idx_to_row[i]]
        e2 = embs[idx_to_row[j]]
        sims.append(float(torch.sum(e1 * e2)))
        labels.append(lab)

    sims = np.array(sims); labels = np.array(labels)
    auc = roc_auc_score(labels, sims)
    thresholds = np.linspace(-1, 1, 4001)
    accs = []
    for th in thresholds:
        preds = (sims >= th).astype(int)
        accs.append(accuracy_score(labels, preds))
    k = int(np.argmax(accs))
    return auc, accs[k], thresholds[k]

# ---------- Train / Eval ----------
def train_epoch(model, head, loader, optimizer, device):
    model.train(); head.train()
    loss_meter, correct, total = 0.0, 0, 0
    for imgs, targets in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device); targets = targets.to(device)
        optimizer.zero_grad()
        emb = model(imgs)
        logits = head(emb, targets)
        loss = F.cross_entropy(logits, targets)
        loss.backward(); optimizer.step()

        loss_meter += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        total += imgs.size(0)
    return loss_meter/total, correct/total

@torch.no_grad()
def eval_cls(model, head, loader, device):
    model.eval(); head.eval()
    loss_meter, correct, total = 0.0, 0, 0
    for imgs, targets in tqdm(loader, desc="Eval", leave=False):
        imgs = imgs.to(device)
        targets = targets.to(device)

        emb = model(imgs)
        # Instead of ArcFace head, do cosine sim against head.weight
        emb = F.normalize(emb, p=2, dim=1)
        W = F.normalize(head.weight, p=2, dim=1)
        logits = F.linear(emb, W) * head.s  # no margin

        # Clamp labels to head size to avoid OOB
        valid = targets < logits.shape[1]
        if valid.sum() == 0:
            continue
        logits = logits[valid]
        targets = targets[valid]

        loss = F.cross_entropy(logits, targets)
        loss_meter += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == targets).sum().item()
        total += imgs.size(0)
    return (loss_meter / max(1,total)), (correct / max(1,total))


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="./archive/lfw-deepfunneled")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--arc-s", type=float, default=64.0)
    parser.add_argument("--arc-m", type=float, default=0.50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--max-train-ids", type=int, default=400)
    parser.add_argument("--pairs-per-type", type=int, default=2000)
    parser.add_argument("--min-images-per-id", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 on Windows to avoid pickling issues")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")

    dataset, train_loader, val_loader, num_train_classes, val_indices = make_loaders(
        img_size=args.img_size,
        batch_size=args.batch_size,
        root=args.data_root,
        train_id_frac=0.8,
        min_images_per_id=args.min_images_per_id,
        max_train_ids=args.max_train_ids,
        num_workers=args.num_workers,
        seed=args.seed
    )

    model = FaceNet(embedding_dim=args.embedding_dim, pretrained=args.pretrained_backbone).to(device)
    head  = ArcMarginProduct(in_feat=args.embedding_dim, out_feat=num_train_classes,
                             s=args.arc_s, m=args.arc_m).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    # verification pairs from VAL identities
    try:
        pairs = build_verification_pairs(dataset, val_indices,
                                         n_pos=args.pairs_per_type, n_neg=args.pairs_per_type,
                                         seed=args.seed)
        have_pairs = True
    except RuntimeError as e:
        print(f"[WARN] No verification pairs: {e}")
        have_pairs = False

    best_auc = 0.0
    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, head, train_loader, optimizer, device)
        va_loss, va_acc = eval_cls(model, head, val_loader, device)
        if have_pairs:
            auc, best_acc, best_th = evaluate_verification(model, dataset, pairs,
                                                           batch_size=max(64, args.batch_size),
                                                           device=device)
        else:
            auc, best_acc, best_th = float("nan"), float("nan"), float("nan")
        dt = time.time() - t0

        print(f"[Epoch {epoch:02d}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.3f} "
              f"VERIF: AUC={auc:.4f} bestACC={best_acc:.3f} @thr={best_th:.3f} "
              f"({dt:.1f}s)")

        if have_pairs and auc > best_auc:
            best_auc = auc
            torch.save({
                "model": model.state_dict(),
                "head": head.state_dict(),
                "args": vars(args),
                "num_train_classes": num_train_classes,
                "best_auc": best_auc
            }, ckpt_dir / "model_best.pt")
            print(f"  ↳ Saved new best checkpoint (AUC {best_auc:.4f})")

    print("Done.")
    return history


if __name__ == "__main__":
    main()
