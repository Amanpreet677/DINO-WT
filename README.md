# DINO-WT: Frequency-Guided Self-Supervised Vision Transformer with Learnable Wavelet Representation for Multi-Focus Image Fusion

PyTorch implementation of **DINO-WT** (Amanpreet, Vinay Kukreja - Chitkara University).
The model fuses two multi-focus images into one all-in-focus image by combining

1. **DINOv2 ViT-B/14** semantic tokens (first 8 blocks frozen, last 4 fine-tuned),
2. a **learnable wavelet transform** (Haar-initialised, trainable 2x2 filter bank) that gives LL/LH/HL/HH sub-bands and a normalised high-frequency prior `Wf`,
3. **frequency-guided attention** that biases transformer attention with `Wf`,
4. a **lightweight decoder + inverse learnable wavelet** reconstruction,
5. a composite loss: pixel (L1) + SSIM + frequency + DINO-perceptual + edge (Eq. 27-32).

```
IA, IB ──► preprocess ──► learnable wavelet ──► Wf (frequency prior) ──┐
                     └──► DINOv2 tokens ───────────────────────────────┴─► frequency-guided attention
                                                                              │
                            fused image ◄── pixel + wavelet-domain fusion ◄── decoder (decision map M)
```

## Repository layout

```
DINO-WT/
├── configs/
│   ├── default.yaml          # paper settings (AdamW 1e-4, wd 1e-5, bs 8, 100 epochs, 224x224)
│   └── tiny_smoke.yaml       # tiny backbone, no downloads - for testing the pipeline on CPU
├── dinowt/
│   ├── models/
│   │   ├── wavelet.py        # LearnableWavelet (analysis, synthesis, frequency prior)
│   │   ├── backbone.py       # DINOv2 wrapper with partial freezing
│   │   ├── attention.py      # FrequencyGuidedAttention
│   │   ├── decoder.py        # LightweightDecoder (fusion head)
│   │   └── dinowt.py         # full model + build_model()
│   ├── losses.py             # composite loss, differentiable SSIM
│   ├── data.py               # synthetic MFIF dataset (COCO/VOC) + real pairs (Lytro/MFFW)
│   ├── preprocess.py         # optional ECC registration + histogram equalisation
│   ├── metrics.py            # Qabf, SSIM, VIF, EN, FMI, AG, SF, MI, PSNR, RMSE
│   └── utils.py
├── train.py   evaluate.py   infer.py
├── tests/
├── requirements.txt   LICENSE   .gitignore
```

## Installation

```bash
git clone https://github.com/<your-username>/DINO-WT.git
cd DINO-WT
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

DINOv2 weights are fetched through `torch.hub` (`facebookresearch/dinov2`) on first use, so internet access is needed once.
For offline machines, clone <https://github.com/facebookresearch/dinov2> and set `model.hub_dir` in the config
(and download the `dinov2_vitb14_pretrain.pth` weights into the torch hub cache).

## Data

| Purpose | Dataset | Expected layout |
|---|---|---|
| Training / synthetic testing | COCO, PASCAL VOC 2012 | any folder of images, e.g. `train2017/`, `VOC2012/JPEGImages/`. Pairs are generated on the fly with spatially varying Gaussian defocus. |
| Real testing | Lytro, MFFW | flat folder with `*-A.jpg` / `*-B.jpg` (e.g. `lytro-01-A.jpg`), **or** sub-folders `A/` and `B/` with matching file names. |

## Train

```bash
python train.py --config configs/default.yaml \
    --train_root /path/to/coco/train2017 \
    --val_root   /path/to/coco/val2017 \
    --out runs/dinowt
```

* Any config value can be overridden: `--set train.epochs=20 train.batch_size=4 model.pretrained=true`
* Resume: `--resume runs/dinowt/last.pt`
* Outputs: `runs/dinowt/best.pt` (best val PSNR), `last.pt`, `log.csv`.

Quick CPU sanity check without downloads (needs any folder with a few images):

```bash
python train.py --config configs/tiny_smoke.yaml --train_root some_images --val_root some_images --out runs/smoke
```

## Evaluate (10 metrics)

```bash
# real pairs (Lytro / MFFW)
python evaluate.py --ckpt runs/dinowt/best.pt --data real --root data/Lytro --out results/lytro --save_images

# synthetic pairs with ground truth
python evaluate.py --ckpt runs/dinowt/best.pt --data synthetic --root /path/to/coco/val2017 --out results/coco
```

Writes `per_image.csv` and `summary.json` and prints the mean of every metric.
Without ground truth (Lytro / MFFW) the reference-based metrics (SSIM, VIF, PSNR, RMSE) are averaged over the two source images.

## Inference

```bash
python infer.py --ckpt runs/dinowt/best.pt --a near.jpg --b far.jpg --out fused.png --save_mask
python infer.py --ckpt runs/dinowt/best.pt --folder data/Lytro --out results/lytro_fused
```

Images of any size are supported (internally resized to a multiple of 14, then resized back).

```python
import torch
from dinowt.utils import load_model_from_checkpoint
model, _ = load_model_from_checkpoint("runs/dinowt/best.pt", "cuda")
fused, mask = model.fuse(img_a, img_b)        # tensors (1,3,H,W) in [0,1]
```

## Tests

```bash
pytest -q
```

## Implementation notes (where the code interprets the paper)

* **Frequency-guided attention.** The two images' tokens form one sequence; queries come from DINO tokens, a learned embedding of `Wf` is added to the key/value inputs and every head adds `lambda_h * Wf` to its logits (additive bias, `lambda_h` learnable). Residual: `F_out = F_s + lambda * F_f` (Eq. 21).
* **Reconstruction.** The decoder predicts a decision map `M`; the image is fused in the pixel domain (Eq. 45) and in the wavelet domain (`M` applied to sub-bands, then `Psi^-1`). A learnable gate mixes both (initially 50/50).
* **Loss.** Eq. 32 (five terms) is used, with fixed weights from `configs/default.yaml`, plus a wavelet perfect-reconstruction term so the learnable filters stay a valid transform pair. The paper describes the weights as learnable and Algorithm 1 lists a four-term loss; here they are fixed hyper-parameters.
* **Metrics.** `Qabf` follows Xydeas & Petrovic; `VIF` is pixel-domain VIFp; `FMI` is a simplified histogram-based variant. Exact numbers can differ slightly from other toolboxes.
* Preprocessing registration / histogram equalisation are optional (`data.hist_eq`, `RealMFIFDataset(register=..., hist_eq=...)`).

## Citation

```bibtex
@article{amanpreet2026dinowt,
  title   = {DINO-WT: Frequency-Guided Self-Supervised Vision Transformer with Learnable Wavelet Representation for Multi-Focus Image Fusion},
  author  = {Amanpreet and Kukreja, Vinay},
  year    = {2026}
}
```

## License

MIT - see [LICENSE](LICENSE).
