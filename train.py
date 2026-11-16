"""Train DINO-WT on synthetic multi-focus pairs generated from a folder of clean images.

Example:
    python train.py --config configs/default.yaml \
        --train_root /data/coco/train2017 --val_root /data/coco/val2017 --out runs/dinowt
"""
import argparse
import copy
import csv
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dinowt.data import SyntheticMFIFDataset
from dinowt.losses import DinoWTLoss, ssim
from dinowt.models import build_model
from dinowt.utils import (AverageMeter, apply_overrides, ensure_dir, get_device, load_config, seed_everything)


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    psnr_m, ssim_m = AverageMeter(), AverageMeter()
    for ia, ib, gt in loader:
        ia, ib, gt = ia.to(device), ib.to(device), gt.to(device)
        out = model(ia, ib)["fused"].clamp(0, 1)
        mse = torch.mean((out - gt) ** 2, dim=(1, 2, 3)).clamp_min(1e-10)
        psnr_m.update((10 * torch.log10(1.0 / mse)).mean().item(), ia.shape[0])
        ssim_m.update(ssim(out, gt).item(), ia.shape[0])
    return {"val_psnr": psnr_m.avg, "val_ssim": ssim_m.avg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--train_root")
    ap.add_argument("--val_root")
    ap.add_argument("--out", default="runs/dinowt")
    ap.add_argument("--resume", default=None, help="path to last.pt to resume from")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                    help="override config values, e.g. --set train.epochs=10 train.batch_size=4")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    if args.train_root: cfg["data"]["train_root"] = args.train_root
    if args.val_root: cfg["data"]["val_root"] = args.val_root
    seed_everything(cfg.get("seed", 42))
    device = get_device()
    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    out_dir = ensure_dir(args.out)
    d, t = cfg["data"], cfg["train"]
    size = cfg["model"]["img_size"]

    train_ds = SyntheticMFIFDataset(d["train_root"], size, True, tuple(d["sigma_range"]), hist_eq=d.get("hist_eq", False))
    val_ds = SyntheticMFIFDataset(d["val_root"], size, False, tuple(d["sigma_range"]), seed=cfg.get("seed", 42),
                                  max_images=d.get("val_max_images"), hist_eq=d.get("hist_eq", False))
    train_dl = DataLoader(train_ds, t["batch_size"], shuffle=True, num_workers=d["num_workers"],
                          pin_memory=device.type == "cuda", drop_last=True)
    val_dl = DataLoader(val_ds, t["batch_size"], shuffle=False, num_workers=d["num_workers"])
    print(f"train images: {len(train_ds)} | val images: {len(val_ds)} | device: {device} | amp: {use_amp}")

    model = build_model(cfg["model"]).to(device)
    encoder = copy.deepcopy(model.backbone) if cfg["loss"].get("use_perceptual", True) else None
    if encoder is not None:
        encoder.set_frozen(-1)
    loss_fn = DinoWTLoss(model.wavelet, encoder, cfg["loss"]).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params) / 1e6
    n_all = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"trainable params: {n_train:.2f}M / total {n_all:.2f}M")
    opt = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"], eta_min=t["min_lr"])
    scaler = make_scaler(use_amp)

    start_epoch, best = 0, -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_epoch, best = ck["epoch"] + 1, ck.get("best", -1.0)
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    log_path = os.path.join(out_dir, "log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "loss", "val_psnr", "val_ssim", "lr", "time_s"])

    for epoch in range(start_epoch, t["epochs"]):
        model.train()
        loss_fn.train()
        meter, t0 = AverageMeter(), time.time()
        pbar = tqdm(train_dl, desc=f"epoch {epoch + 1}/{t['epochs']}")
        for it, (ia, ib, gt) in enumerate(pbar):
            ia, ib, gt = ia.to(device), ib.to(device), gt.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(ia, ib)
                loss, terms = loss_fn(out["fused"], gt, out)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, t["grad_clip"])
            scaler.step(opt)
            scaler.update()
            meter.update(loss.item(), ia.shape[0])
            if it % t["log_every"] == 0:
                pbar.set_postfix({k: f"{v.item():.4f}" for k, v in terms.items()})
        sched.step()

        metrics = validate(model, val_dl, device)
        lr = opt.param_groups[0]["lr"]
        print(f"[epoch {epoch + 1}] loss={meter.avg:.4f} val_psnr={metrics['val_psnr']:.2f} "
              f"val_ssim={metrics['val_ssim']:.4f} lr={lr:.2e}")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{meter.avg:.5f}", f"{metrics['val_psnr']:.3f}",
                                    f"{metrics['val_ssim']:.4f}", f"{lr:.3e}", f"{time.time() - t0:.1f}"])

        if metrics["val_psnr"] > best:
            best = metrics["val_psnr"]
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "best": best},
                       os.path.join(out_dir, "best.pt"))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "config": cfg, "epoch": epoch, "best": best}, os.path.join(out_dir, "last.pt"))
    print(f"done. best val PSNR = {best:.2f} dB  ->  {os.path.join(out_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
