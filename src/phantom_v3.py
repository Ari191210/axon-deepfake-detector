"""
PHANTOM v3.1 — World-Class Deepfake Detector (axon-state compatible)
==============================================

v3.1 PATCH NOTES (fixes on top of v3.0 — each was a run-breaking or
eval-corrupting bug):

  [P-1] CRITICAL / CHECKPOINT: stage-2 checkpoints only saved params with
        requires_grad=True. Stage 2 freezes `proj`, so CKPT_FINAL was saved
        WITHOUT the projection weights — Stage-3 "honest evaluation" and all
        inference then ran with a RANDOM projection. Checkpoints now persist
        every non-backbone parameter plus trainable backbone params.
  [P-2] CRITICAL / MULTI-SCALE: `_ms_proj` was lazily created inside
        forward() — after the optimizer, EMA, and set_stage() were built.
        It was therefore never trained, never in EMA, recreated with fresh
        random weights on every DataParallel replica each step, and missing
        at checkpoint load. Multi-scale now uses timm's
        `forward_intermediates` (single backbone pass, DP-safe) and a
        properly registered `ms_proj` built in __init__ with identity init.
  [P-3] CRITICAL / LEAKAGE: hard-negative mining mined from VAL_ITEMS and
        added them to the training pool — training directly on the held-out
        generators, corrupting the cross-generator AUC. Mining now draws
        from a train-only sample.
  [P-4] OOM RETRY: the retry rebuilt `dl` but the for-loop kept iterating
        the OLD iterator at the old batch size, so the retry never took
        effect. Rewritten with an explicit iterator that is actually
        replaced on OOM.
  [P-5] EWC: AXON_EWC.pth stores {"fisher": {...}, "optima": {...}} — the
        v3.0 loader iterated the top-level dict and crashed on
        `dict.shape`. Loader now parses both AXON and flat formats, and
        anchors the penalty at the AXON optima when present.
  [P-6] FREQUENCY BRANCH: the "DCT" branch was AdaptiveAvgPool2d(8x8) — a
        64-pixel thumbnail with no frequency content at all. Replaced with
        a real FFT log-magnitude branch (computed in fp32 outside autocast).
  [P-7] AXON PATHS: champion auto-detect looked in "champions/"; AXON's
        registry lives in "registry/". Both are checked now.

Everything else (two-stage training, cross-generator split with 2-class
guarantee, leakage hashing, bootstrap CI, calibration, robustness curves,
abstention, resume) is unchanged from v3.0.

USAGE
  Smoke test : python phantom_v3.py --smoke-test
  Full run   : python phantom_v3.py --input /data --work /output
  Inference  : python phantom_v3.py --inference /path/to/image.jpg --ckpt /path/to/model.pth
"""

# ============================================================================
#  IMPORTS
# ============================================================================
import os, sys, gc, json, math, time, random, hashlib, io, glob, argparse
import traceback, warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

import numpy as np
from PIL import Image, ImageFile, ImageFilter, ImageDraw, ImageEnhance
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    from torchvision import transforms
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torchvision"])
    from torchvision import transforms

try:
    import timm
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm>=1.0.3"])
    import timm

from sklearn.metrics import roc_auc_score, accuracy_score

# ============================================================================
#  AMP COMPATIBILITY SHIM
# ============================================================================
def _make_autocast():
    try:
        from torch.amp import autocast as _ac
        def autocast(dtype=None, enabled=True):
            return _ac("cuda", dtype=dtype or torch.float16, enabled=enabled)
        return autocast
    except ImportError:
        from torch.cuda.amp import autocast as _ac
        def autocast(dtype=None, enabled=True):
            return _ac(dtype or torch.float16, enabled=enabled)
        return autocast

def _make_grad_scaler(enabled=True):
    try:
        from torch.amp import GradScaler as _GS
        try:
            return _GS("cuda", enabled=enabled)
        except TypeError:
            return _GS(enabled=enabled)
    except ImportError:
        from torch.cuda.amp import GradScaler as _GS
        return _GS(enabled=enabled)

autocast = _make_autocast()
GradScaler = _make_grad_scaler

# ============================================================================
#  REPRODUCIBILITY
# ============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_OK = torch.cuda.is_available()


def _bf16_supported():
    try:
        return (AMP_OK
                and torch.cuda.get_device_capability(0)[0] >= 8
                and torch.cuda.is_bf16_supported())
    except Exception:
        return False


USE_BF16 = _bf16_supported()
DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
USE_SCALER = AMP_OK and not USE_BF16


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def safe_float(x, default=0.0) -> float:
    try:
        xf = float(x)
        return xf if math.isfinite(xf) else default
    except (TypeError, ValueError):
        return default


# ============================================================================
#  CONFIGURATION
# ============================================================================
class CFG:
    # ---- paths ----
    INPUT_DIR: str = "/kaggle/input"
    WORK_DIR: str = "/kaggle/working"
    AXON_STATE_DIR: str = ""
    REPLAY_PATH: str = ""
    WARMSTART_CKPT: str = ""
    EWC_PATH: str = ""
    EWC_LAMBDA: float = 0.0
    CKPT_STAGE1: str = f"{WORK_DIR}/phantom_stage1.pth"
    CKPT_FINAL: str = f"{WORK_DIR}/phantom_final.pth"
    CKPT_RESUME: str = f"{WORK_DIR}/phantom_resume.pth"
    CACHE_JSON: str = f"{WORK_DIR}/phantom_items.json"
    REPORT_JSON: str = f"{WORK_DIR}/phantom_report.json"

    # ---- model ----
    BACKBONE: str = "vit_base_patch14_dinov2.lvd142m"
    IMG_SIZE: int = 224
    PROBE_DIM: int = 256
    HEAD_DROPOUT: float = 0.35
    USE_SPECTRAL_NORM: bool = True
    MULTI_SCALE: bool = True
    MS_N_BLOCKS: int = 4              # intermediate blocks to aggregate
    FREQUENCY_BRANCH: bool = True     # P-6: real FFT log-magnitude branch

    # ---- cross-generator holdout ----
    HOLDOUT_SRC_FRAC: float = 0.25
    VAL_CAP_PER_SRC: int = 3000
    TRAIN_POOL: int = 60000
    PER_SRC_CAP_FRAC: float = 0.25

    # ---- data loading ----
    BATCH_SIZE: int = 32
    EVAL_BATCH_SIZE: int = 64
    NUM_WORKERS: int = 4

    # ---- stage 1: GENERALIZE ----
    S1_EPOCHS: int = 6
    S1_STEPS_PER_EPOCH: int = 700
    S1_LR: float = 1e-4

    # ---- stage 2: ADAPT (guarded) ----
    S2_EPOCHS: int = 3
    S2_STEPS_PER_EPOCH: int = 300
    S2_LR: float = 2e-5

    # ---- optimizer / scheduler ----
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP: float = 1.0
    WALL_CLOCK_H: float = 8.5

    # ---- losses ----
    FOCAL_GAMMA: float = 2.0
    FOCAL_SMOOTH: float = 0.05
    SUPCON_LAMBDA: float = 0.1
    SUPCON_TEMP: float = 0.07
    CONSIST_LAMBDA: float = 0.1
    EMB_NOISE_STD: float = 0.03
    FREQ_LAMBDA: float = 0.05

    # ---- data augmentation ----
    SBI_PROB: float = 0.25
    JPEG_PROB: float = 0.4
    DOWNUP_PROB: float = 0.3

    # ---- test-time augmentation ----
    TTA_ENABLED: bool = True

    # ---- EMA ----
    EMA_DECAY: float = 0.999

    # ---- logging / checkpointing ----
    LOG_EVERY: int = 100
    RESUME_EVERY_STEPS: int = 100

    # ---- evaluation ----
    BOOTSTRAP_N: int = 1000
    ECE_BINS: int = 15
    ABSTAIN_TARGET_COVERAGE: float = 0.90

    # ---- leakage detection ----
    LEAK_HASH_GRID: int = 8
    LEAK_HAMMING_THRESH: int = 5

    # ---- hard-negative mining (P-3: train-only pool) ----
    HARD_NEG_MINING: bool = True
    HARD_NEG_K: int = 64
    HARD_NEG_EVERY: int = 1
    HARD_NEG_MINE_POOL: int = 6000    # sample of TRAIN items to mine from

    # ---- data discovery ----
    IMG_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    REAL_TAGS: frozenset = frozenset({
        "real", "genuine", "authentic", "original", "ffhq",
        "celeba", "pristine", "flickr", "faces"
    })
    FAKE_TAGS: frozenset = frozenset({
        "fake", "synthetic", "deepfake", "generated", "ai", "gan",
        "diffusion", "sdxl", "midjourney", "stylegan", "dalle",
        "cifake", "synthesis"
    })
    AMBIGUOUS_HINTS: Tuple[str, ...] = (
        "140k", "real and fake", "real-and-fake", "real vs fake", "fake vs real"
    )
    SOURCE_OVERRIDES: Dict[str, int] = {
        "celeba": 0, "ffhq": 0, "flickr": 0,
        "chatgpt": 1, "gemini": 1, "gravex": 1,
    }
    SKIP_HINTS: Tuple[str, ...] = ("phantom", "brain", "project33", "axon")


def _gpu_memory_gb() -> float:
    if not AMP_OK:
        return 0.0
    try:
        return min(
            torch.cuda.get_device_properties(i).total_memory / (1024**3)
            for i in range(torch.cuda.device_count())
        )
    except Exception:
        return 0.0

_GPU_MEM_GB = _gpu_memory_gb()
if _GPU_MEM_GB > 0 and _GPU_MEM_GB < 20:
    CFG.EVAL_BATCH_SIZE = 32
    log(f"  GPU mem={_GPU_MEM_GB:.1f}GB/T4 -> eval_batch=32")
elif _GPU_MEM_GB >= 40:
    CFG.EVAL_BATCH_SIZE = 128
    log(f"  GPU mem={_GPU_MEM_GB:.1f}GB -> eval_batch=128")


def _resolve_axon_paths():
    if CFG.AXON_STATE_DIR and os.path.isdir(CFG.AXON_STATE_DIR):
        if not CFG.REPLAY_PATH:
            rp = os.path.join(CFG.AXON_STATE_DIR, "AXON_REPLAY.json")
            if os.path.exists(rp):
                CFG.REPLAY_PATH = rp
        if not CFG.WARMSTART_CKPT:
            # P-7: AXON stores its champions under "registry/"; also check
            # "champions/" and the state dir itself for portability.
            for subdir in ("registry", "champions", "."):
                idx_path = os.path.join(CFG.AXON_STATE_DIR, subdir, "index.json")
                if not os.path.exists(idx_path):
                    continue
                try:
                    with open(idx_path) as f:
                        idx = json.load(f)
                    champ_tag = idx.get("champion", "")
                    if champ_tag:
                        for ext in (".pth", ".pt"):
                            ckpt = os.path.join(CFG.AXON_STATE_DIR, subdir,
                                                f"{champ_tag}{ext}")
                            if os.path.exists(ckpt):
                                CFG.WARMSTART_CKPT = ckpt
                                break
                except Exception:
                    pass
                if CFG.WARMSTART_CKPT:
                    break
        if not CFG.EWC_PATH:
            ep = os.path.join(CFG.AXON_STATE_DIR, "AXON_EWC.pth")
            if os.path.exists(ep):
                CFG.EWC_PATH = ep

os.makedirs(CFG.WORK_DIR, exist_ok=True)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================================
#  DATA DISCOVERY
# ============================================================================
def _derive_source(filepath: str) -> str:
    try:
        parts = Path(filepath).relative_to(Path(CFG.INPUT_DIR)).parts
    except ValueError:
        return "unknown"
    if not parts:
        return "unknown"
    skip = {"datasets", "input", "kaggle"}
    if parts[0].lower() in skip and len(parts) >= 3:
        return parts[2].lower()
    return parts[0].lower()


def _label_for_directory(dirpath: str) -> Optional[int]:
    leaf_tokens = set(
        Path(dirpath).name.lower().replace("_", " ").replace("-", " ").split()
    )
    has_fake = bool(leaf_tokens & CFG.FAKE_TAGS)
    has_real = bool(leaf_tokens & CFG.REAL_TAGS)
    if has_fake and not has_real:
        return 1
    if has_real and not has_fake:
        return 0
    low = dirpath.lower()
    if any(hint in low for hint in CFG.AMBIGUOUS_HINTS):
        return None
    all_tokens = {
        t for part in Path(dirpath).parts
        for t in part.replace("_", " ").replace("-", " ").split()
    }
    has_fake = bool(all_tokens & CFG.FAKE_TAGS)
    has_real = bool(all_tokens & CFG.REAL_TAGS)
    if has_fake and not has_real:
        return 1
    if has_real and not has_fake:
        return 0
    for keyword, label in CFG.SOURCE_OVERRIDES.items():
        if keyword in low:
            return label
    return None


def discover_data(use_cache: bool = True) -> List[Tuple[str, int, str]]:
    if use_cache and os.path.exists(CFG.CACHE_JSON):
        try:
            items = [tuple(x) for x in json.load(open(CFG.CACHE_JSON))]
            if items and all(os.path.exists(p) for p, _, _ in items[:50]):
                log(f"Discovered (cached) {len(items):,} items")
                return items
        except Exception:
            pass

    items: List[Tuple[str, int, str]] = []
    seen: set = set()

    for dirpath, dirnames, filenames in os.walk(CFG.INPUT_DIR):
        low = dirpath.lower()
        if any(hint in low for hint in CFG.SKIP_HINTS):
            dirnames[:] = []
            continue

        label = _label_for_directory(dirpath)
        if label is None:
            continue

        source = _derive_source(dirpath)

        for fname in filenames:
            if not fname.lower().endswith(CFG.IMG_EXTENSIONS):
                continue
            fpath = os.path.join(dirpath, fname)
            if fpath in seen:
                continue
            seen.add(fpath)
            items.append((fpath, label, source))

    if items:
        with open(CFG.CACHE_JSON, "w") as f:
            json.dump(items, f)

    n_fake = sum(1 for _, l, _ in items if l == 1)
    n_real = len(items) - n_fake
    n_sources = len({s for *_, s in items})
    log(f"Discovered {len(items):,} items  (fake={n_fake:,}, real={n_real:,}, "
        f"sources={n_sources})")
    return items


# ============================================================================
#  LEAKAGE CHECK — perceptual hash (average hash, 8x8)
# ============================================================================
def _perceptual_hash(filepath: str, grid: int = CFG.LEAK_HASH_GRID) -> Optional[int]:
    try:
        img = Image.open(filepath).convert("L").resize(
            (grid, grid), Image.BILINEAR
        )
        arr = np.asarray(img, dtype=np.float32)
        bits = (arr > arr.mean()).flatten().astype(np.uint8)
        hash_val = 0
        for b in bits:
            hash_val = (hash_val << 1) | int(b)
        return hash_val
    except Exception:
        return None


def _hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def leakage_check(
    train_items: List[Tuple],
    val_items: List[Tuple],
    max_check: int = 4000,
) -> Dict[str, Any]:
    log("Stage 0: leakage check (train/val near-duplicate hashing)...")
    rng = random.Random(SEED)
    tr = train_items if len(train_items) <= max_check else rng.sample(train_items, max_check)
    va = val_items   if len(val_items)   <= max_check else rng.sample(val_items, max_check)

    tr_hashes = [h for h in (_perceptual_hash(p) for p, _, _ in tr) if h is not None]
    va_hashes = [h for h in (_perceptual_hash(p) for p, _, _ in va) if h is not None]

    if not tr_hashes or not va_hashes:
        log("  leakage check skipped (no hashes computed)")
        return {"checked_train": 0, "checked_val": 0,
                "leaks": 0, "leak_rate": 0.0, "verdict": "SKIPPED"}

    tr_set = set(tr_hashes)
    leaks = sum(
        1 for h in va_hashes
        if any(_hamming_distance(h, th) <= CFG.LEAK_HAMMING_THRESH for th in tr_set)
    )
    rate = leaks / max(1, len(va_hashes))
    if rate < 0.01:
        verdict = "OK"
    elif rate < 0.05:
        verdict = "SUSPICIOUS"
    else:
        verdict = "SEVERE LEAKAGE"

    log(f"  leakage: {leaks}/{len(va_hashes)} val images near-duplicate in train "
        f"(rate={rate:.3%}) -> {verdict}")

    if rate >= 0.05:
        log("  *** WARNING: high train/val overlap. AUC below is likely INFLATED. ***")

    return {
        "checked_train": len(tr_hashes), "checked_val": len(va_hashes),
        "leaks": leaks, "leak_rate": rate, "verdict": verdict,
    }


# ============================================================================
#  AUGMENTATION PIPELINE
# ============================================================================
class RandomDownUp:
    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        w, h = img.size
        scale = random.uniform(0.4, 0.9)
        small = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.BILINEAR,
        )
        return small.resize((w, h), Image.BILINEAR)


TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((CFG.IMG_SIZE + 32, CFG.IMG_SIZE + 32)),
    transforms.RandomCrop(CFG.IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
    transforms.RandomApply([transforms.GaussianBlur(3, (0.1, 2.0))], p=0.3),
    RandomDownUp(CFG.DOWNUP_PROB),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.12)),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


def _load_image(path: str) -> Image.Image:
    try:
        img = Image.open(path).convert("RGB")
        img.load()
        return img
    except Exception:
        return Image.new("RGB", (CFG.IMG_SIZE, CFG.IMG_SIZE), (128, 128, 128))


def _apply_jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def _apply_downscale(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    small = img.resize(
        (max(1, int(w * scale)), max(1, int(h * scale))),
        Image.BILINEAR,
    )
    return small.resize((w, h), Image.BILINEAR)


def _self_blended_image(img: Image.Image) -> Image.Image:
    try:
        w, h = img.size
        src = img.copy()
        if random.random() < 0.5:
            s = random.uniform(0.95, 1.05)
            src = src.resize(
                (max(1, int(w * s)), max(1, int(h * s)))
            ).resize((w, h))
        if random.random() < 0.5:
            src = src.filter(ImageFilter.GaussianBlur(random.uniform(0.4, 1.4)))
        if random.random() < 0.5:
            src = ImageEnhance.Brightness(src).enhance(random.uniform(0.9, 1.1))
        mask = Image.new("L", (w, h), 0)
        cx, cy = w // 2, int(h * 0.45)
        rx = int(w * random.uniform(0.28, 0.42))
        ry = int(h * random.uniform(0.30, 0.45))
        ImageDraw.Draw(mask).ellipse(
            [cx - rx, cy - ry, cx + rx, cy + ry], fill=255
        )
        mask = mask.filter(ImageFilter.GaussianBlur(random.uniform(5, 15)))
        return Image.composite(src, img, mask)
    except Exception:
        return img


# ============================================================================
#  DATASET
# ============================================================================
class DeepfakeDataset(Dataset):
    def __init__(
        self,
        items: List[Tuple[str, int, str]],
        mode: str = "train",
        eval_degrade: Optional[Tuple[str, float]] = None,
    ):
        self.paths = [it[0] for it in items]
        self.labels = [float(it[1]) for it in items]
        self.sources = [it[2] for it in items]
        self.mode = mode
        self.eval_degrade = eval_degrade

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        label = self.labels[idx]
        img = _load_image(path)

        is_sbi = False

        if self.mode == "train":
            if label == 0.0 and random.random() < CFG.SBI_PROB:
                img = _self_blended_image(img)
                label = 1.0
                is_sbi = True
            if random.random() < CFG.JPEG_PROB:
                img = _apply_jpeg(img, random.randint(30, 90))
            tensor = TRAIN_TRANSFORM(img)
        else:
            if self.eval_degrade is not None:
                kind, val = self.eval_degrade
                if kind == "jpeg":
                    img = _apply_jpeg(img, int(val))
                elif kind == "downscale":
                    img = _apply_downscale(img, val)
            tensor = EVAL_TRANSFORM(img)

        return tensor, torch.tensor(label, dtype=torch.float32), path, is_sbi


# ============================================================================
#  DATALOADER FACTORY
# ============================================================================
def _per_source_cap(
    items: List[Tuple],
    frac: float = CFG.PER_SRC_CAP_FRAC,
    pool_size: int = CFG.TRAIN_POOL,
) -> List[Tuple]:
    by_source: Dict[str, list] = defaultdict(list)
    for it in items:
        src = it[2] if len(it) > 2 else "unknown"
        by_source[src].append(it)

    cap = max(1, int(pool_size * frac))
    result = []
    for group in by_source.values():
        if len(group) > cap:
            result.extend(random.sample(group, cap))
        else:
            result.extend(group)
    random.shuffle(result)
    return result


def make_dataloader(
    items: List[Tuple],
    mode: str,
    batch_size: int,
    balanced: bool = False,
    steps_per_epoch: Optional[int] = None,
    eval_degrade: Optional[Tuple[str, float]] = None,
) -> DataLoader:
    dataset = DeepfakeDataset(items, mode, eval_degrade=eval_degrade)
    kw = dict(
        num_workers=CFG.NUM_WORKERS,
        pin_memory=AMP_OK,
        persistent_workers=(CFG.NUM_WORKERS > 0),
    )

    if balanced:
        labels = dataset.labels
        n_real = labels.count(0.0)
        n_fake = labels.count(1.0)
        if n_real == 0 or n_fake == 0:
            return DataLoader(
                dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kw
            )
        weights = torch.tensor([
            1.0 / n_real if l == 0.0 else 1.0 / n_fake
            for l in labels
        ])
        num_samples = min(
            weights.numel(),
            (steps_per_epoch or CFG.S1_STEPS_PER_EPOCH) * batch_size,
        )
        sampler = WeightedRandomSampler(weights, num_samples, replacement=True)
        return DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, drop_last=True, **kw
        )

    return DataLoader(
        dataset, batch_size=batch_size,
        shuffle=(mode == "train"), drop_last=(mode == "train"), **kw,
    )


# ============================================================================
#  MODEL ARCHITECTURE
# ============================================================================
class FrequencyBranch(nn.Module):
    """
    P-6: REAL frequency-domain branch. Computes the 2D FFT log-magnitude
    spectrum in fp32 (outside autocast — half-precision FFT is inaccurate),
    downsamples it, and runs a small conv net. AI generators leave
    high-frequency spectral peaks and grid artifacts visible here.
    """
    def __init__(self, hidden_dim: int = 64, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(hidden_dim, out_dim), nn.GELU(),
        )
        _m = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        _s = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("_mean", _m, persistent=False)
        self.register_buffer("_std", _s, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with autocast(enabled=False):
            xf = (x.float() * self._std + self._mean).clamp(0, 1)
            xf = xf - xf.mean(dim=(2, 3), keepdim=True)
            mag = torch.abs(torch.fft.fftshift(
                torch.fft.fft2(xf, norm="ortho"))).clamp_min(1e-6)
            spec = torch.log1p(mag)
            spec = F.interpolate(spec, size=(64, 64), mode="bilinear",
                                 align_corners=False)
        return self.net(spec)


def _spectral_norm_wrapper(module: nn.Module) -> nn.Module:
    if CFG.USE_SPECTRAL_NORM:
        return nn.utils.spectral_norm(module)
    return module


class PhantomDetector(nn.Module):
    """
    Multi-scale DINOv2 backbone with hyperspherical projection and
    spectral-normed classification head.

    P-2: multi-scale is now a properly registered module built in __init__,
    fed by timm forward_intermediates (single backbone pass, DataParallel-
    safe, no hooks, no lazy creation).
    """

    def __init__(self, dropout: float = CFG.HEAD_DROPOUT):
        super().__init__()
        self.backbone = timm.create_model(
            CFG.BACKBONE, pretrained=True, num_classes=0, img_size=CFG.IMG_SIZE,
        )
        backbone_dim = self.backbone.num_features

        for p in self.backbone.parameters():
            p.requires_grad = False
        for m in self.backbone.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad = True

        # ---- multi-scale (P-2): fixed indices + registered projection ----
        self.multi_scale = CFG.MULTI_SCALE and hasattr(
            self.backbone, "forward_intermediates")
        self._ms_indices: List[int] = []
        self.ms_proj: Optional[nn.Linear] = None
        if self.multi_scale and hasattr(self.backbone, "blocks"):
            n_blocks = len(self.backbone.blocks)
            k = min(CFG.MS_N_BLOCKS, n_blocks)
            step = max(1, n_blocks // k)
            self._ms_indices = list(range(step - 1, n_blocks, step))[:k]
            concat_dim = backbone_dim * (1 + len(self._ms_indices))
            self.ms_proj = nn.Linear(concat_dim, backbone_dim)
            with torch.no_grad():
                # Identity on the final pooled feature, zeros elsewhere:
                # starts equivalent to no multi-scale, learns to mix.
                self.ms_proj.weight.zero_()
                self.ms_proj.weight[:, :backbone_dim].copy_(
                    torch.eye(backbone_dim))
                self.ms_proj.bias.zero_()
        else:
            self.multi_scale = False

        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, CFG.PROBE_DIM),
            nn.GELU(),
            nn.LayerNorm(CFG.PROBE_DIM),
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            _spectral_norm_wrapper(nn.Linear(CFG.PROBE_DIM, CFG.PROBE_DIM // 2)),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            _spectral_norm_wrapper(nn.Linear(CFG.PROBE_DIM // 2, 1)),
        )

        self.freq_branch = FrequencyBranch() if CFG.FREQUENCY_BRANCH else None
        self.freq_head = nn.Linear(32, 1) if CFG.FREQUENCY_BRANCH else None

        try:
            self.backbone.set_grad_checkpointing()
        except Exception:
            pass

    def _resize_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == (CFG.IMG_SIZE, CFG.IMG_SIZE):
            return x
        return F.interpolate(
            x, size=(CFG.IMG_SIZE, CFG.IMG_SIZE), mode="bilinear",
            align_corners=False,
        )

    def _backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Single backbone pass. Returns (B, backbone_dim) fp32."""
        if self.multi_scale:
            final_toks, inters = self.backbone.forward_intermediates(
                x, indices=self._ms_indices, norm=True,
                output_fmt="NLC", intermediates_only=False,
            )
            pooled = self.backbone.forward_head(final_toks).float()
            ms_feats = [t.float().mean(dim=1) for t in inters]  # mean over tokens
            return self.ms_proj(torch.cat([pooled] + ms_feats, dim=-1))
        return self.backbone(x).float()

    def forward(
        self,
        x: torch.Tensor,
        return_emb: bool = False,
        return_consist: bool = False,
    ):
        x = self._resize_input(x)
        feat = self._backbone_features(x)

        emb = F.normalize(self.proj(feat), dim=-1)
        logit = self.head(emb).squeeze(-1)

        freq_logit = None
        if self.freq_branch is not None and self.freq_head is not None:
            freq_feat = self.freq_branch(x)
            freq_logit = self.freq_head(freq_feat).squeeze(-1)

        if return_consist and self.training:
            noise = torch.randn_like(emb) * CFG.EMB_NOISE_STD
            emb_v = F.normalize(emb + noise, dim=-1)
            logit_v = self.head(emb_v).squeeze(-1)
        else:
            logit_v = None

        if return_emb and return_consist:
            return logit, emb, logit_v, freq_logit
        if return_emb:
            return logit, emb, freq_logit
        if return_consist:
            return logit, logit_v, freq_logit
        return logit, freq_logit

    def set_stage(self, stage: int):
        for p in self.parameters():
            p.requires_grad = False
        for m in self.backbone.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True
        if self.freq_branch is not None:
            for p in self.freq_branch.parameters():
                p.requires_grad = True
            for p in self.freq_head.parameters():
                p.requires_grad = True
        if self.ms_proj is not None:
            for p in self.ms_proj.parameters():
                p.requires_grad = True
        if stage == 1:
            for p in self.proj.parameters():
                p.requires_grad = True

    def trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_params())


# ============================================================================
#  LOSSES
# ============================================================================
class FocalLoss(nn.Module):
    def __init__(self, smooth: float = CFG.FOCAL_SMOOTH):
        super().__init__()
        self.smooth = smooth

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0,
    ) -> torch.Tensor:
        t = targets * (1.0 - self.smooth) + 0.5 * self.smooth
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        p = torch.sigmoid(logits)
        pt = (p * t + (1.0 - p) * (1.0 - t)).clamp(1e-6, 1.0 - 1e-6)
        focal_weight = (1.0 - pt).pow(gamma)
        return (focal_weight * bce).mean()


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = CFG.SUPCON_TEMP):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        emb: torch.Tensor,
        labels: torch.Tensor,
        sbi_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        emb = F.normalize(emb.float(), dim=-1)
        n = emb.shape[0]
        if n < 2:
            return torch.zeros((), device=emb.device)

        if sbi_mask is not None and sbi_mask.any():
            valid = ~sbi_mask
            if valid.sum() < 2:
                return torch.zeros((), device=emb.device)
            emb = emb[valid]
            labels = labels[valid]
            n = emb.shape[0]

        lab = (labels > 0.5).long().view(-1, 1)
        sim = emb @ emb.t() / self.temperature
        self_mask = torch.eye(n, device=emb.device, dtype=torch.bool)
        pos = (lab == lab.t()) & (~self_mask)
        sim = sim.masked_fill(self_mask, -1e9)
        logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)

        cnt = pos.sum(1)
        loss_per_sample = -(logp * pos).sum(1) / cnt.clamp_min(1)
        valid_mask = cnt > 0
        if not valid_mask.any():
            return torch.zeros((), device=emb.device)
        return loss_per_sample[valid_mask].mean()


# ============================================================================
#  EMA
# ============================================================================
class EMA:
    def __init__(self, model: nn.Module, decay: float = CFG.EMA_DECAY):
        self.decay = decay
        self._backup: Dict[str, torch.Tensor] = {}
        self._refresh(model)

    def _unwrapped(self, model: nn.Module) -> nn.Module:
        return model.module if hasattr(model, "module") else model

    @torch.no_grad()
    def _refresh(self, model: nn.Module):
        m = self._unwrapped(model)
        self.shadow = {n: p.detach().clone() for n, p in m.named_parameters()
                       if p.requires_grad}
        self.buffers = {n: b.detach().clone() for n, b in m.named_buffers()}

    def refresh(self, model: nn.Module):
        self._backup = {}
        self._refresh(model)

    @torch.no_grad()
    def update(self, model: nn.Module):
        m = self._unwrapped(model)
        for name, param in m.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.detach(), alpha=1.0 - self.decay
                )
        for name, buf in m.named_buffers():
            if name in self.buffers:
                self.buffers[name].copy_(buf.detach())

    @torch.no_grad()
    def apply(self, model: nn.Module):
        m = self._unwrapped(model)
        self._backup = {
            n: p.detach().clone()
            for n, p in m.named_parameters() if p.requires_grad
        }
        for name, param in m.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.copy_(self.shadow[name].to(param.device, param.dtype))
        for name, buf in m.named_buffers():
            if name in self.buffers:
                buf.copy_(self.buffers[name].to(buf.device, buf.dtype))

    @torch.no_grad()
    def restore(self, model: nn.Module):
        m = self._unwrapped(model)
        for name, param in m.named_parameters():
            if param.requires_grad and name in self._backup:
                param.copy_(self._backup[name])
        self._backup = {}


# ============================================================================
#  AXON-STATE INTEGRATION
# ============================================================================
def load_replay_buffer(replay_path: str) -> List[Tuple[str, int, str]]:
    if not os.path.exists(replay_path):
        log(f"  replay buffer not found: {replay_path}")
        return []
    try:
        with open(replay_path) as f:
            raw = json.load(f)
        items = [(r[0], int(r[1]), r[2] if len(r) > 2 else "replay")
                 for r in raw if len(r) >= 2 and os.path.exists(r[0])]
        n_real = sum(1 for _, l, _ in items if l == 0)
        n_fake = len(items) - n_real
        log(f"  Loaded replay buffer: {len(items):,} items "
            f"(real={n_real:,}, fake={n_fake:,}) from {replay_path}")
        return items
    except Exception as e:
        log(f"  replay buffer load failed: {e}")
        return []


def find_axon_state(input_dir: str) -> Optional[str]:
    candidates = []
    for root, dirs, files in os.walk(input_dir):
        dirs[:] = [d for d in dirs if d.lower() not in {".cache", "__pycache__"}]
        if "AXON_LEDGER.json" in files and "AXON_REPLAY.json" in files:
            candidates.append(root)
    if candidates:
        candidates.sort(key=len)
        log(f"  Auto-detected axon-state at: {candidates[0]}")
        return candidates[0]
    return None


def load_champion_ckpt(ckpt_path: str, model: nn.Module, ema: "EMA") -> bool:
    if not os.path.exists(ckpt_path):
        return False
    try:
        blob = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    except Exception as e:
        log(f"  champion ckpt load failed ({e})")
        return False

    m = model.module if hasattr(model, "module") else model

    if isinstance(blob, dict) and "model" in blob:
        if load_checkpoint(ckpt_path, model, ema):
            log(f"  Loaded champion (PHANTOM format): {ckpt_path}")
            return True

    try:
        if isinstance(blob, dict) and "model_trainable" in blob:
            # AXON format: trainable params + buffers stored separately
            sd = dict(blob.get("model_trainable", {}))
            sd.update(blob.get("buffers", {}))
        elif isinstance(blob, dict) and "state_dict" in blob:
            sd = blob["state_dict"]
        elif isinstance(blob, dict) and "model_state_dict" in blob:
            sd = blob["model_state_dict"]
        elif isinstance(blob, dict):
            sample_keys = list(blob.keys())[:5]
            if any("." in k for k in sample_keys):
                sd = blob
            else:
                log("  champion ckpt unrecognized format")
                return False
        else:
            sd = blob

        cur_sd = m.state_dict()
        matched = {k: v for k, v in sd.items()
                   if k in cur_sd and tuple(cur_sd[k].shape) == tuple(v.shape)}
        if matched:
            m.load_state_dict(matched, strict=False)
            ema.refresh(model)
            log(f"  Loaded champion (legacy format): {ckpt_path} "
                f"({len(matched)}/{len(cur_sd)} keys matched)")
            return True
        log("  champion ckpt: no matching keys (architecture mismatch is "
            "expected for AXON multi-stream champions)")
        return False
    except Exception as e:
        log(f"  champion ckpt legacy load failed ({e})")
        return False


class EWCLoss(nn.Module):
    """
    P-5: parses BOTH the AXON format {"fisher": {...}, "optima": {...}}
    and a flat {param_name: fisher_tensor} dict. When AXON optima are
    present the penalty anchors at those optima (the point Fisher was
    computed at) rather than the current weights.
    """
    def __init__(self, lambda_ewc: float = 100.0):
        super().__init__()
        self.lambda_ewc = lambda_ewc
        self.fisher: Dict[str, torch.Tensor] = {}
        self.params: Dict[str, torch.Tensor] = {}

    def load_fisher(self, ewc_path: str, model: nn.Module):
        if not os.path.exists(ewc_path):
            log(f"  EWC file not found: {ewc_path}")
            return
        try:
            blob = torch.load(ewc_path, map_location=DEVICE, weights_only=False)
        except Exception as e:
            log(f"  EWC load failed: {e}")
            return

        m = model.module if hasattr(model, "module") else model
        cur_params = {n: p for n, p in m.named_parameters() if p.requires_grad}

        if isinstance(blob, dict) and "fisher" in blob:
            fisher_dict = blob.get("fisher", {})
            optima_dict = blob.get("optima", {})
        elif isinstance(blob, dict):
            fisher_dict = blob
            optima_dict = {}
        else:
            log("  EWC: unrecognized file format")
            return

        for name, fisher_val in fisher_dict.items():
            if not torch.is_tensor(fisher_val):
                continue
            if name in cur_params and fisher_val.shape == cur_params[name].shape:
                self.fisher[name] = fisher_val.to(DEVICE)
                anchor = optima_dict.get(name)
                if torch.is_tensor(anchor) and anchor.shape == fisher_val.shape:
                    self.params[name] = anchor.to(DEVICE)
                else:
                    self.params[name] = cur_params[name].detach().clone()

        if self.fisher:
            log(f"  EWC: loaded {len(self.fisher)} Fisher diagonals from "
                f"{ewc_path}")
        else:
            log(f"  EWC: no matching parameters found in {ewc_path}")

    def forward(self, model: nn.Module) -> torch.Tensor:
        if not self.fisher:
            return torch.zeros((), device=DEVICE)
        m = model.module if hasattr(model, "module") else model
        loss = torch.zeros((), device=DEVICE)
        for name, param in m.named_parameters():
            if name in self.fisher and param.requires_grad:
                loss = loss + (self.fisher[name]
                               * (param - self.params[name]).pow(2)).sum()
        return self.lambda_ewc * loss


# ============================================================================
#  CROSS-GENERATOR SPLIT (2-class guarantee, slice/val separation)
# ============================================================================
def split_cross_generator(
    items: List[Tuple[str, int, str]],
) -> Tuple[List[Tuple], List[Tuple], Dict[str, List[Tuple]], bool]:
    by_source: Dict[str, List[Tuple]] = defaultdict(list)
    for item in items:
        by_source[item[2]].append(item)

    all_sources = sorted(by_source)
    rng = random.Random(SEED)
    rng.shuffle(all_sources)

    def _source_class(src: str) -> int:
        labels = {l for _, l, _ in by_source[src]}
        if labels == {0}:
            return 0
        if labels == {1}:
            return 1
        return 2  # mixed

    both_class = [s for s in all_sources if _source_class(s) == 2]
    single_class = [s for s in all_sources if _source_class(s) != 2]

    k_both = max(0, int(len(both_class) * CFG.HOLDOUT_SRC_FRAC))
    val_sources = set(both_class[:k_both])

    real_singles = [s for s in single_class if _source_class(s) == 0]
    fake_singles = [s for s in single_class if _source_class(s) == 1]
    if real_singles:
        val_sources.add(real_singles[0])
    if fake_singles:
        val_sources.add(fake_singles[0])
    n_hold_single = max(2, int(len(single_class) * CFG.HOLDOUT_SRC_FRAC))
    for s in single_class:
        if len(val_sources) >= n_hold_single:
            break
        val_sources.add(s)

    train: List[Tuple] = []
    val: List[Tuple] = []
    for src, group in by_source.items():
        if src in val_sources:
            shuffled = list(group)
            rng.shuffle(shuffled)
            val.extend(shuffled[:CFG.VAL_CAP_PER_SRC])
        else:
            train.extend(group)

    val_labels = {l for _, l, _ in val}
    if len(val_labels) < 2 and train:
        needed = 0 if 1 in val_labels else 1
        for src in list(by_source):
            if src in val_sources:
                continue
            if all(l == needed for _, l, _ in by_source[src]):
                val.extend(by_source[src][:CFG.VAL_CAP_PER_SRC])
                train = [it for it in train if it[2] != src]
                val_sources.add(src)
                break

    slices: Dict[str, List[Tuple]] = {}
    slice_paths: set = set()
    for src in val_sources:
        src_items = [it for it in val if it[2] == src]
        src_labels = {l for _, l, _ in src_items}
        if len(src_labels) >= 2:
            slices[src] = src_items
            for it in src_items:
                slice_paths.add(it[0])

    val_main = [it for it in val if it[0] not in slice_paths]
    main_labels = {l for _, l, _ in val_main}
    if len(main_labels) < 2:
        val_main = val  # fallback: accept overlap rather than broken AUC

    val_labels = {l for _, l, _ in val_main}
    ok = len(val_labels) >= 2

    log(f"Cross-generator split: "
        f"train={len(train):,} ({len(by_source) - len(val_sources)} src) | "
        f"val_main={len(val_main):,} ({len(val_sources)} UNSEEN src) | "
        f"2-class slices={len(slices)} | val_2class={ok}")

    if not ok:
        log("  *** WARNING: could not build a 2-class validation pool. "
            "AUC will be undefined. ***")

    return train, val_main, slices, ok


# ============================================================================
#  HARD-NEGATIVE MINING (P-3: TRAIN pool only — never validation)
# ============================================================================
def mine_hard_negatives(
    model: nn.Module,
    ema: "EMA",
    train_items: List[Tuple],
    k: int = CFG.HARD_NEG_K,
) -> List[Tuple]:
    """
    One-shot mining from a sample of TRAINING items only. v3.0 mined from
    val_items and added them to the training pool — direct leakage of the
    held-out generators. Never do that.
    """
    if not train_items or k <= 0:
        return []

    mine_pool = (train_items
                 if len(train_items) <= CFG.HARD_NEG_MINE_POOL
                 else random.sample(train_items, CFG.HARD_NEG_MINE_POOL))

    ema.apply(model)
    model.eval()

    loader = make_dataloader(mine_pool, "eval", CFG.EVAL_BATCH_SIZE)
    all_probs: List[Tuple[float, int, str]] = []

    with torch.no_grad():
        for imgs, labels, paths, _ in loader:
            imgs = imgs.to(DEVICE)
            with autocast(DTYPE, AMP_OK):
                logits, _ = model(imgs)
                probs = torch.sigmoid(logits.float()).cpu()
            for p, l, path_str in zip(probs, labels, paths):
                all_probs.append((float(p), int(l), str(path_str)))

    del loader
    gc.collect()
    ema.restore(model)
    model.train()

    path_to_item = {it[0]: it for it in mine_pool}
    class_entries: Dict[int, List[Tuple[float, Tuple]]] = defaultdict(list)
    for prob, label, path_str in all_probs:
        item = path_to_item.get(path_str)
        if item is not None:
            # hardest = confidently WRONG first, then uncertain
            wrongness = prob if label == 0 else (1.0 - prob)
            class_entries[label].append((-wrongness, item))

    selected = []
    for label, entries in class_entries.items():
        entries.sort(key=lambda x: x[0])
        selected.extend([e[1] for e in entries[:k]])

    log(f"  Hard-negative mining (train-only pool of {len(mine_pool)}): "
        f"selected {len(selected)} samples")
    return selected


# ============================================================================
#  TTA
# ============================================================================
def _tta_views(imgs: torch.Tensor) -> List[torch.Tensor]:
    views = [imgs, torch.flip(imgs, dims=[3])]
    B, C, H, W = imgs.shape
    ch, cw = int(H * 0.9), int(W * 0.9)
    top, left = (H - ch) // 2, (W - cw) // 2
    center_crop = F.interpolate(
        imgs[:, :, top:top + ch, left:left + cw],
        size=(H, W), mode="bilinear", align_corners=False,
    )
    views.append(center_crop)
    views.append(torch.flip(center_crop, dims=[3]))
    return views


# ============================================================================
#  EVALUATION / METRICS
# ============================================================================
@torch.no_grad()
def predict(
    model: nn.Module,
    items: List[Tuple],
    ema: Optional["EMA"] = None,
    tta: bool = True,
    eval_degrade: Optional[Tuple[str, float]] = None,
) -> Dict[str, Tuple[float, int]]:
    if not items:
        return {}

    if ema is not None:
        ema.apply(model)
    model.eval()

    preds: Dict[str, Tuple[float, int]] = {}
    loader = make_dataloader(items, "eval", CFG.EVAL_BATCH_SIZE,
                             eval_degrade=eval_degrade)

    try:
        for imgs, labels, paths, _ in loader:
            imgs = imgs.to(DEVICE)
            with autocast(DTYPE, AMP_OK):
                views = _tta_views(imgs) if tta else [imgs]
                logit_sum = None
                for v in views:
                    logits, _ = model(v)
                    logit_sum = logits if logit_sum is None else logit_sum + logits
                avg_logit = logit_sum / len(views)
                probs = torch.sigmoid(avg_logit.float()).cpu().tolist()

            for path, prob, label in zip(paths, probs, labels.tolist()):
                preds[path] = (float(prob), int(label))
    finally:
        del loader
        gc.collect()
        if AMP_OK:
            torch.cuda.empty_cache()

    if ema is not None:
        ema.restore(model)
    model.train()
    return preds


def compute_auc(preds: Dict, items: List[Tuple]) -> float:
    probs, labels = [], []
    for it in items:
        r = preds.get(it[0])
        if r is not None:
            probs.append(r[0])
            labels.append(r[1])
    if len(set(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, probs)


def compute_accuracy(preds: Dict, items: List[Tuple]) -> float:
    probs, labels = [], []
    for it in items:
        r = preds.get(it[0])
        if r is not None:
            probs.append(1.0 if r[0] > 0.5 else 0.0)
            labels.append(r[1])
    if not labels:
        return float("nan")
    return accuracy_score(labels, probs)


def bootstrap_auc_ci(
    preds: Dict,
    items: List[Tuple],
    n_boot: int = CFG.BOOTSTRAP_N,
    seed: int = SEED,
) -> Tuple[float, float, float]:
    probs, labels = [], []
    for it in items:
        r = preds.get(it[0])
        if r is not None:
            probs.append(r[0])
            labels.append(r[1])
    probs = np.array(probs)
    labels = np.array(labels)
    if len(set(labels.tolist())) < 2 or len(labels) < 10:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.RandomState(seed)
    indices = np.arange(len(labels))
    aucs = []
    for _ in range(n_boot):
        s = rng.choice(indices, size=len(indices), replace=True)
        if len(set(labels[s].tolist())) < 2:
            continue
        aucs.append(roc_auc_score(labels[s], probs[s]))

    if not aucs:
        return float("nan"), float("nan"), float("nan")

    aucs = np.array(aucs)
    return (
        float(np.mean(aucs)),
        float(np.percentile(aucs, 2.5)),
        float(np.percentile(aucs, 97.5)),
    )


def evaluate(
    model: nn.Module,
    val_items: List[Tuple],
    slices: Dict[str, List[Tuple]],
    ema: Optional["EMA"] = None,
) -> Tuple[Dict[str, Any], Dict]:
    # Predict over val_main AND slice items in one pass
    union = list(val_items)
    seen = {it[0] for it in union}
    for group in slices.values():
        for it in group:
            if it[0] not in seen:
                union.append(it)
                seen.add(it[0])

    preds = predict(model, union, ema=ema, tta=CFG.TTA_ENABLED)
    auc = compute_auc(preds, val_items)
    acc = compute_accuracy(preds, val_items)

    slice_aucs = {}
    for src, group in slices.items():
        a = compute_auc(preds, group)
        if not math.isnan(a):
            slice_aucs[src] = a

    worst_slice = min(slice_aucs.values()) if slice_aucs else auc

    return {
        "auc": safe_float(auc),
        "acc": safe_float(acc),
        "worst_slice": safe_float(worst_slice, safe_float(auc)),
        "slices": slice_aucs,
    }, preds


# ============================================================================
#  CALIBRATION (Temperature Scaling)
# ============================================================================
def fit_temperature(preds: Dict, items: List[Tuple]) -> float:
    logits_list, labels_list = [], []
    for it in items:
        r = preds.get(it[0])
        if r is None:
            continue
        p = min(max(r[0], 1e-6), 1.0 - 1e-6)
        logits_list.append(math.log(p / (1.0 - p)))
        labels_list.append(float(r[1]))

    if len(set(labels_list)) < 2:
        return 1.0

    z = torch.tensor(logits_list, dtype=torch.float32)
    y = torch.tensor(labels_list, dtype=torch.float32)
    log_T = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        T = torch.exp(log_T).clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(z / T, y)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:
        return 1.0

    return float(torch.exp(log_T).clamp(0.05, 20.0).item())


def apply_temperature(preds: Dict, T: float) -> Dict[str, Tuple[float, int]]:
    out = {}
    for key, (p, l) in preds.items():
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        z = math.log(p / (1.0 - p)) / max(T, 1e-6)
        out[key] = (1.0 / (1.0 + math.exp(-z)), l)
    return out


def compute_ece(
    preds: Dict, items: List[Tuple], n_bins: int = CFG.ECE_BINS,
) -> float:
    probs, labels = [], []
    for it in items:
        r = preds.get(it[0])
        if r is not None:
            probs.append(r[0])
            labels.append(r[1])
    if not probs:
        return float("nan")
    probs = np.array(probs)
    labels = np.array(labels)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs > bins[i]) & (probs <= bins[i + 1])
        if mask.sum() == 0:
            continue
        confidence = probs[mask].mean()
        accuracy = (labels[mask] == (probs[mask] > 0.5)).mean()
        ece += (mask.sum() / len(probs)) * abs(confidence - accuracy)
    return float(ece)


# ============================================================================
#  CHECKPOINT SAVE / LOAD
# ============================================================================
def _persistable_params(m: nn.Module) -> Dict[str, torch.Tensor]:
    """
    P-1: persist every parameter the head COULD need at inference:
    all non-backbone params (proj, head, ms_proj, freq branch — regardless
    of the current stage's requires_grad flags) plus trainable backbone
    params (the LayerNorms). Frozen pretrained backbone weights are
    reproducible from timm and stay out to keep checkpoints small.
    """
    return {
        n: p.detach().cpu()
        for n, p in m.named_parameters()
        if p.requires_grad or not n.startswith("backbone.")
    }


def save_checkpoint(path: str, model: nn.Module, ema: "EMA",
                    report: Dict, extra: Optional[Dict] = None):
    m = model.module if hasattr(model, "module") else model
    blob = {
        "model": _persistable_params(m),
        "buffers": {n: b.detach().cpu() for n, b in m.named_buffers()},
        "ema_shadow": {k: v.detach().cpu() for k, v in ema.shadow.items()},
        "ema_buffers": {k: v.detach().cpu() for k, v in ema.buffers.items()},
        "report": report,
        "timestamp": datetime.now().isoformat(),
    }
    if extra:
        blob.update(extra)
    if CFG.AXON_STATE_DIR:
        blob["axon_state_dir"] = CFG.AXON_STATE_DIR
    if CFG.WARMSTART_CKPT:
        blob["warmstart_ckpt"] = CFG.WARMSTART_CKPT
    tmp = path + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, model: nn.Module, ema: "EMA") -> bool:
    if not os.path.exists(path):
        return False
    try:
        blob = torch.load(path, map_location=DEVICE, weights_only=False)
    except Exception:
        return False

    m = model.module if hasattr(model, "module") else model
    cur_sd = m.state_dict()

    param_sd = {k: v for k, v in blob.get("model", {}).items()
                if k in cur_sd and tuple(cur_sd[k].shape) == tuple(v.shape)}
    m.load_state_dict(param_sd, strict=False)

    with torch.no_grad():
        for name, buf in m.named_buffers():
            stored = blob.get("buffers", {}).get(name)
            if stored is not None and tuple(buf.shape) == tuple(stored.shape):
                buf.copy_(stored.to(buf.device, buf.dtype))

    if blob.get("ema_shadow"):
        params = {n: p.shape for n, p in m.named_parameters()}
        ema.shadow = {
            k: v.to(DEVICE)
            for k, v in blob["ema_shadow"].items()
            if k in params and tuple(params[k]) == tuple(v.shape)
        }
    if blob.get("ema_buffers"):
        ema.buffers = {
            k: v.to(DEVICE) for k, v in blob["ema_buffers"].items()
        }

    return True


def save_resume_state(
    path: str, stage: int, epoch: int, global_step: int,
    model: nn.Module, ema: "EMA", opt, sched, scaler, best_score: float,
):
    m = model.module if hasattr(model, "module") else model
    blob = {
        "stage": stage,
        "epoch": epoch,
        "global_step": global_step,
        "best_score": best_score,
        "model_full": m.state_dict(),
        "ema_shadow": {k: v.detach().cpu() for k, v in ema.shadow.items()},
        "ema_buffers": {k: v.detach().cpu() for k, v in ema.buffers.items()},
        "opt": opt.state_dict() if opt is not None else None,
        "sched": sched.state_dict() if sched is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "timestamp": datetime.now().isoformat(),
    }
    tmp = path + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)


def load_resume_state(
    path: str, model: nn.Module, ema: "EMA",
    opt=None, sched=None, scaler=None,
) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        blob = torch.load(path, map_location=DEVICE, weights_only=False)
    except Exception as e:
        log(f"  resume load failed ({e}); starting fresh")
        return None

    try:
        m = model.module if hasattr(model, "module") else model
        m.load_state_dict(blob["model_full"], strict=False)

        params = {n: p.shape for n, p in m.named_parameters()}
        ema.shadow = {
            k: v.to(DEVICE)
            for k, v in blob.get("ema_shadow", {}).items()
            if k in params and tuple(params[k]) == tuple(v.shape)
        }
        ema.buffers = {
            k: v.to(DEVICE) for k, v in blob.get("ema_buffers", {}).items()
        }

        if opt is not None and blob.get("opt"):
            try:
                opt.load_state_dict(blob["opt"])
            except Exception as e:
                log(f"  (opt state skipped: {e})")
        if sched is not None and blob.get("sched"):
            try:
                sched.load_state_dict(blob["sched"])
            except Exception as e:
                log(f"  (sched state skipped: {e})")
        if scaler is not None and blob.get("scaler"):
            try:
                scaler.load_state_dict(blob["scaler"])
            except Exception as e:
                log(f"  (scaler state skipped: {e})")

        log(f"  RESUMED: stage={blob['stage']} epoch={blob['epoch']} "
            f"step={blob['global_step']} best={blob.get('best_score', -1):.4f}")
        return {
            "stage": blob["stage"], "epoch": blob["epoch"],
            "global_step": blob["global_step"],
            "best_score": blob.get("best_score", -1.0),
        }
    except Exception as e:
        log(f"  resume state incompatible ({e}); starting fresh")
        return None


# ============================================================================
#  TRAIN EPOCH (P-4: OOM retry uses an explicit, replaceable iterator)
# ============================================================================
def train_epoch(
    model: nn.Module,
    ema: "EMA",
    opt,
    sched,
    scaler,
    focal_loss: FocalLoss,
    supcon_loss: SupervisedContrastiveLoss,
    pool: List[Tuple],
    batch_size: int,
    max_steps: int,
    gamma: float,
    t_deadline: float,
    stage: int = 1,
    epoch: int = 0,
    start_step: int = 0,
    best_score: float = -1.0,
    ewc_loss: Optional[EWCLoss] = None,
) -> float:
    dl = make_dataloader(pool, "train", batch_size, balanced=True,
                         steps_per_epoch=max_steps)
    data_iter = iter(dl)
    model.train()

    step = 0
    oom_retries = 0
    current_batch = batch_size
    running_loss = 0.0
    loss_count = 0
    l_ewc = torch.zeros((), device=DEVICE)

    while step < max_steps and time.time() <= t_deadline:
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        # Fast-forward on resume (data only; sched state comes from resume blob)
        if step < start_step:
            step += 1
            continue

        imgs, labels, paths, sbi_flags = batch
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        sbi_mask = sbi_flags.to(DEVICE)

        try:
            with autocast(DTYPE, AMP_OK):
                logit, emb, logit_v, freq_logit = model(
                    imgs, return_emb=True, return_consist=True
                )

                l_focal = focal_loss(logit, labels, gamma)
                l_supcon = supcon_loss(emb, labels, sbi_mask=sbi_mask.bool())
                l_consist = F.mse_loss(torch.sigmoid(logit_v),
                                       torch.sigmoid(logit.detach()))

                total_loss = (l_focal
                              + CFG.SUPCON_LAMBDA * l_supcon
                              + CFG.CONSIST_LAMBDA * l_consist)

                if freq_logit is not None:
                    l_freq = F.binary_cross_entropy_with_logits(
                        freq_logit, labels
                    )
                    total_loss = total_loss + CFG.FREQ_LAMBDA * l_freq

                if ewc_loss is not None:
                    l_ewc = ewc_loss(model)
                    total_loss = total_loss + l_ewc

                if not torch.isfinite(total_loss):
                    step += 1
                    continue

            opt.zero_grad(set_to_none=True)

            if USE_SCALER:
                scaler.scale(total_loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    (model.module if hasattr(model, "module") else model)
                    .trainable_params(), CFG.GRAD_CLIP
                )
                scaler.step(opt)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (model.module if hasattr(model, "module") else model)
                    .trainable_params(), CFG.GRAD_CLIP
                )
                opt.step()

            if sched is not None:
                sched.step()

            ema.update(model)
            running_loss += float(total_loss.item())
            loss_count += 1

        except RuntimeError as e:
            if "out of memory" in str(e).lower() and oom_retries < 2:
                torch.cuda.empty_cache()
                gc.collect()
                current_batch = max(4, current_batch // 2)
                oom_retries += 1
                log(f"  OOM -> halving batch to {current_batch}, "
                    f"rebuilding loader")
                # P-4: replace BOTH the loader and the live iterator
                del dl, data_iter
                gc.collect()
                dl = make_dataloader(pool, "train", current_batch,
                                     balanced=True, steps_per_epoch=max_steps)
                data_iter = iter(dl)
                continue
            raise

        if step % CFG.LOG_EVERY == 0 and loss_count > 0:
            ewc_str = (f"  ewc={safe_float(l_ewc.item()):.4f}"
                       if ewc_loss is not None else "")
            log(f"    step {step}/{max_steps}  "
                f"loss={safe_float(total_loss.item()):.4f}  "
                f"focal={safe_float(l_focal.item()):.4f}  "
                f"supcon={safe_float(l_supcon.item()):.4f}  "
                f"consist={safe_float(l_consist.item()):.4f}{ewc_str}")

        if step > 0 and step % CFG.RESUME_EVERY_STEPS == 0:
            save_resume_state(
                CFG.CKPT_RESUME, stage, epoch, step,
                model, ema, opt, sched, scaler, best_score,
            )

        step += 1

    del dl, data_iter
    gc.collect()
    if AMP_OK:
        torch.cuda.empty_cache()

    return running_loss / max(1, loss_count)


# ============================================================================
#  INFERENCE-ONLY MODE
# ============================================================================
@torch.no_grad()
def infer_single(model: nn.Module, image_path: str, checkpoint_path: str,
                 tta: bool = True) -> Dict[str, Any]:
    ema = EMA(model)
    if not load_checkpoint(checkpoint_path, model, ema):
        raise RuntimeError(f"Cannot load checkpoint: {checkpoint_path}")

    ema.apply(model)
    model.eval()

    img = _load_image(image_path)
    tensor = EVAL_TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with autocast(DTYPE, AMP_OK):
        if tta:
            views = _tta_views(tensor)
            logit_sum = None
            for v in views:
                l, _ = model(v)
                logit_sum = l if logit_sum is None else logit_sum + l
            logit = logit_sum / len(views)
        else:
            logit, _ = model(tensor)

    prob = float(torch.sigmoid(logit.float()).item())
    label = "FAKE" if prob > 0.5 else "REAL"
    confidence = max(prob, 1.0 - prob)

    ema.restore(model)

    return {
        "path": image_path,
        "probability_fake": round(prob, 6),
        "label": label,
        "confidence": round(confidence, 6),
        "tta": tta,
        "backbone": CFG.BACKBONE,
    }


# ============================================================================
#  MAIN PIPELINE
# ============================================================================
def run(smoke: bool = False):
    t0 = time.time()
    t_deadline = t0 + CFG.WALL_CLOCK_H * 3600

    if smoke:
        CFG.S1_EPOCHS = 1
        CFG.S1_STEPS_PER_EPOCH = 4
        CFG.S2_EPOCHS = 1
        CFG.S2_STEPS_PER_EPOCH = 3
        CFG.BATCH_SIZE = 4
        CFG.EVAL_BATCH_SIZE = 4
        CFG.NUM_WORKERS = 0
        CFG.TRAIN_POOL = 200
        CFG.VAL_CAP_PER_SRC = 40
        CFG.BOOTSTRAP_N = 50
        CFG.RESUME_EVERY_STEPS = 2
        CFG.HARD_NEG_MINE_POOL = 100
        t_deadline = t0 + 600

    log("=" * 64)
    log("  PHANTOM v3.1 — Deepfake Detector (axon-state compatible)")
    log("  generalize -> guarded-adapt -> honest evaluation")
    log("=" * 64)
    log(f"  device={DEVICE}  amp={AMP_OK}  dtype={DTYPE}  bf16={USE_BF16}  "
        f"smoke={smoke}")
    log(f"  backbone={CFG.BACKBONE}  multi_scale={CFG.MULTI_SCALE}  "
        f"freq_branch={CFG.FREQUENCY_BRANCH}")

    # ---- Stage 0: Axon-state integration ----
    _resolve_axon_paths()
    if CFG.AXON_STATE_DIR:
        log(f"  axon-state dir: {CFG.AXON_STATE_DIR}")
        log(f"  replay:  {CFG.REPLAY_PATH or '(none)'}")
        log(f"  warmstart: {CFG.WARMSTART_CKPT or '(none)'}")
        log(f"  ewc: {CFG.EWC_PATH or '(none)'} (lambda={CFG.EWC_LAMBDA})")

    # ---- Stage 0: Data discovery ----
    items = discover_data(use_cache=not smoke)

    replay_items = []
    if CFG.REPLAY_PATH:
        replay_items = load_replay_buffer(CFG.REPLAY_PATH)

    min_items = 20 if smoke else 100
    if len(items) < min_items and not replay_items:
        log(f"Not enough labeled data ({len(items)} < {min_items}). Aborting.")
        return

    # ---- Stage 0: Cross-generator split + leakage check ----
    train_items, val_items, slices, val_ok = split_cross_generator(items)

    if replay_items:
        # Replay goes to TRAIN only — and never items that landed in val
        val_paths = {it[0] for it in val_items}
        for group in slices.values():
            val_paths.update(it[0] for it in group)
        replay_items = [it for it in replay_items if it[0] not in val_paths]
        train_items = train_items + replay_items
        log(f"  Merged replay -> train pool now {len(train_items):,} items")
    if not val_ok:
        log("FATAL: validation pool is single-class; cannot measure AUC.")
        return

    leak_report = leakage_check(train_items, val_items)

    # ---- Model ----
    model = PhantomDetector().to(DEVICE)
    dp = nn.DataParallel(model) if (AMP_OK and torch.cuda.device_count() > 1) else model
    focal_loss = FocalLoss()
    supcon_loss = SupervisedContrastiveLoss(CFG.SUPCON_TEMP)

    ema_pre = EMA(model)
    if CFG.WARMSTART_CKPT:
        if load_champion_ckpt(CFG.WARMSTART_CKPT, model, ema_pre):
            log("  *** CHAMPION WARMSTART ACTIVE ***")
        else:
            log(f"  Champion warmstart failed: {CFG.WARMSTART_CKPT}")
            CFG.WARMSTART_CKPT = ""

    ewc = None
    if CFG.EWC_PATH and CFG.EWC_LAMBDA > 0:
        ewc = EWCLoss(lambda_ewc=CFG.EWC_LAMBDA)
        ewc.load_fisher(CFG.EWC_PATH, model)
        if not ewc.fisher:
            ewc = None
            log("  EWC disabled (no matching parameters)")
    else:
        log(f"  EWC: disabled (path={bool(CFG.EWC_PATH)}, "
            f"lambda={CFG.EWC_LAMBDA})")

    # ================================================================
    #  STAGE 1: GENERALIZE
    # ================================================================
    log("\n--- STAGE 1: GENERALIZE (LN + proj + head) ---")
    model.set_stage(1)
    log(f"  trainable params: {model.n_trainable():,}")

    if CFG.WARMSTART_CKPT:
        ema = ema_pre
        ema.refresh(model)   # re-key shadow to stage-1 trainable set
    else:
        ema = EMA(model)
    opt = torch.optim.AdamW(
        model.trainable_params(), lr=CFG.S1_LR, weight_decay=CFG.WEIGHT_DECAY,
    )
    total_steps_s1 = max(1, CFG.S1_EPOCHS * CFG.S1_STEPS_PER_EPOCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps_s1)
    scaler = GradScaler(USE_SCALER)

    resume = load_resume_state(CFG.CKPT_RESUME, model, ema, opt, sched, scaler)
    start_stage = resume["stage"] if resume else 1
    start_epoch = resume["epoch"] if resume else 0
    start_step  = resume["global_step"] if resume else 0
    best_score  = resume["best_score"] if resume else -1.0

    for epoch in range(CFG.S1_EPOCHS):
        if start_stage > 1:
            break
        if epoch < start_epoch:
            continue
        if time.time() > t_deadline:
            log("  wall-clock deadline reached in stage 1")
            break

        pool = _per_source_cap(train_items)
        if len(pool) > CFG.TRAIN_POOL:
            pool = random.sample(pool, CFG.TRAIN_POOL)

        # P-3: mine from TRAIN items only — never validation
        if (CFG.HARD_NEG_MINING
                and epoch > 0
                and epoch % CFG.HARD_NEG_EVERY == 0):
            hard_negs = mine_hard_negatives(dp, ema, train_items, CFG.HARD_NEG_K)
            if hard_negs:
                pool = pool + hard_negs
                log(f"  added {len(hard_negs)} hard negatives to pool")

        s_step = start_step if (epoch == start_epoch and resume) else 0

        avg_loss = train_epoch(
            dp, ema, opt, sched, scaler, focal_loss, supcon_loss,
            pool, CFG.BATCH_SIZE, CFG.S1_STEPS_PER_EPOCH, CFG.FOCAL_GAMMA,
            t_deadline, stage=1, epoch=epoch, start_step=s_step,
            best_score=best_score, ewc_loss=ewc,
        )

        rep, _ = evaluate(model, val_items, slices, ema)
        score = 0.5 * rep["auc"] + 0.5 * rep["worst_slice"]
        ewc_tag = " +EWC" if ewc is not None else ""
        log(f"  [S1 epoch {epoch}] loss={avg_loss:.4f}  "
            f"cross-gen AUC={rep['auc']:.4f}  acc={rep['acc']:.4f}  "
            f"worst_slice={rep['worst_slice']:.4f}  score={score:.4f}{ewc_tag}")

        if score > best_score:
            best_score = score
            save_checkpoint(CFG.CKPT_STAGE1, model, ema, rep)
            log(f"    saved stage-1 best (score={best_score:.4f})")

        save_resume_state(
            CFG.CKPT_RESUME, 1, epoch + 1, 0,
            model, ema, opt, sched, scaler, best_score,
        )

    load_checkpoint(CFG.CKPT_STAGE1, model, ema)

    # ================================================================
    #  STAGE 2: ADAPT (guarded)
    # ================================================================
    if start_stage >= 3:
        log("\n--- STAGE 2 already complete (resumed) -> skipping to eval ---")
        kept_best = best_score
    else:
        log("\n--- STAGE 2: ADAPT (LN + head, low LR, guarded) ---")
        model.set_stage(2)
        log(f"  trainable params: {model.n_trainable():,}")

        ema.refresh(model)
        opt = torch.optim.AdamW(
            model.trainable_params(), lr=CFG.S2_LR,
            weight_decay=CFG.WEIGHT_DECAY,
        )
        total_steps_s2 = max(1, CFG.S2_EPOCHS * CFG.S2_STEPS_PER_EPOCH)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps_s2)
        scaler = GradScaler(USE_SCALER)

        base_rep, _ = evaluate(model, val_items, slices, ema)
        base_score = 0.5 * base_rep["auc"] + 0.5 * base_rep["worst_slice"]
        log(f"  stage-2 baseline score={base_score:.4f} "
            f"(AUC={base_rep['auc']:.4f})")
        # P-1: this save (and all later ones) now includes proj even though
        # stage 2 froze it — _persistable_params keeps all non-backbone params.
        save_checkpoint(CFG.CKPT_FINAL, model, ema, base_rep)
        kept_best = base_score

        s2_start_epoch = start_epoch if (resume and start_stage == 2) else 0
        s2_start_step  = start_step  if (resume and start_stage == 2) else 0

        for epoch in range(CFG.S2_EPOCHS):
            if epoch < s2_start_epoch:
                continue
            if time.time() > t_deadline:
                log("  wall-clock deadline reached in stage 2")
                break

            pool = _per_source_cap(train_items)
            if len(pool) > CFG.TRAIN_POOL:
                pool = random.sample(pool, CFG.TRAIN_POOL)

            s_step = (s2_start_step
                      if (epoch == s2_start_epoch and resume and start_stage == 2)
                      else 0)

            avg_loss = train_epoch(
                dp, ema, opt, sched, scaler, focal_loss, supcon_loss,
                pool, CFG.BATCH_SIZE, CFG.S2_STEPS_PER_EPOCH, CFG.FOCAL_GAMMA,
                t_deadline, stage=2, epoch=epoch, start_step=s_step,
                best_score=kept_best, ewc_loss=ewc,
            )

            rep, _ = evaluate(model, val_items, slices, ema)
            score = 0.5 * rep["auc"] + 0.5 * rep["worst_slice"]
            ewc_tag = " +EWC" if ewc is not None else ""
            log(f"  [S2 epoch {epoch}] loss={avg_loss:.4f}  "
                f"cross-gen AUC={rep['auc']:.4f}  "
                f"worst_slice={rep['worst_slice']:.4f}  "
                f"score={score:.4f}{ewc_tag}")

            if score > kept_best:
                kept_best = score
                save_checkpoint(CFG.CKPT_FINAL, model, ema, rep)
                log(f"    KEPT stage-2 epoch (score={kept_best:.4f})")
            else:
                log(f"    rejected (score {score:.4f} <= best {kept_best:.4f})")

            save_resume_state(
                CFG.CKPT_RESUME, 2, epoch + 1, 0,
                model, ema, opt, sched, scaler, kept_best,
            )

        save_resume_state(
            CFG.CKPT_RESUME, 3, 0, 0,
            model, ema, opt, sched, scaler, kept_best,
        )

    # ================================================================
    #  STAGE 3: HONEST EVALUATION
    # ================================================================
    log("\n--- STAGE 3: HONEST EVALUATION (final checkpoint) ---")

    final_model = PhantomDetector().to(DEVICE)
    final_ema = EMA(final_model)
    if not load_checkpoint(CFG.CKPT_FINAL, final_model, final_ema):
        log("  (no CKPT_FINAL — falling back to stage-1 best)")
        load_checkpoint(CFG.CKPT_STAGE1, final_model, final_ema)

    preds_tta = predict(final_model, val_items, ema=final_ema, tta=True)
    preds_base = predict(final_model, val_items, ema=final_ema, tta=False)

    auc_base = compute_auc(preds_base, val_items)
    auc_tta = compute_auc(preds_tta, val_items)
    acc = compute_accuracy(preds_base, val_items)
    mean_b, lo_b, hi_b = bootstrap_auc_ci(preds_base, val_items)

    T = fit_temperature(preds_base, val_items)
    preds_cal = apply_temperature(preds_base, T)
    ece_before = compute_ece(preds_base, val_items)
    ece_after = compute_ece(preds_cal, val_items)

    conf_sorted = sorted(
        ((abs(preds_base.get(it[0], (0.5, 0))[0] - 0.5), it[0])
         for it in val_items),
        reverse=True,
    )
    keep_n = int(len(conf_sorted) * CFG.ABSTAIN_TARGET_COVERAGE)
    keep_paths = {c[1] for c in conf_sorted[:keep_n]}
    abstain_items = [it for it in val_items if it[0] in keep_paths]
    auc_abstain = compute_auc(preds_base, abstain_items)

    slice_preds = predict(
        final_model,
        [it for group in slices.values() for it in group],
        ema=final_ema, tta=False,
    ) if slices else {}
    per_generator = {}
    for src, group in slices.items():
        per_generator[src] = {
            "auc_base": safe_float(compute_auc(slice_preds, group), float("nan")),
            "n": len(group),
        }

    robustness = {"jpeg": {}, "downscale": {}}
    jpeg_qualities = [90, 50] if smoke else [95, 90, 70, 50, 30]
    for q in jpeg_qualities:
        p = predict(final_model, val_items, ema=final_ema, tta=False,
                    eval_degrade=("jpeg", q))
        robustness["jpeg"][str(q)] = safe_float(
            compute_auc(p, val_items), float("nan"))
        log(f"  robustness JPEG q={q}: AUC={robustness['jpeg'][str(q)]:.4f}")

    downscale_factors = [0.5] if smoke else [0.75, 0.5, 0.25]
    for sc in downscale_factors:
        p = predict(final_model, val_items, ema=final_ema, tta=False,
                    eval_degrade=("downscale", sc))
        robustness["downscale"][str(sc)] = safe_float(
            compute_auc(p, val_items), float("nan"))
        log(f"  robustness downscale x{sc}: AUC="
            f"{robustness['downscale'][str(sc)]:.4f}")

    report = {
        "model": "PHANTOM v3.1",
        "backbone": CFG.BACKBONE,
        "axon_state": CFG.AXON_STATE_DIR or None,
        "warmstart": CFG.WARMSTART_CKPT or None,
        "ewc_lambda": CFG.EWC_LAMBDA,
        "replay_items": len(replay_items) if replay_items else 0,
        "multi_scale": CFG.MULTI_SCALE,
        "frequency_branch": CFG.FREQUENCY_BRANCH,
        "leakage": leak_report,
        "auc_base_no_tta": safe_float(auc_base, float("nan")),
        "auc_with_tta": safe_float(auc_tta, float("nan")),
        "auc_bootstrap_mean": safe_float(mean_b, float("nan")),
        "auc_95ci": [safe_float(lo_b, float("nan")),
                     safe_float(hi_b, float("nan"))],
        "accuracy": safe_float(acc, float("nan")),
        "temperature": T,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "abstain_coverage": CFG.ABSTAIN_TARGET_COVERAGE,
        "auc_at_coverage": safe_float(auc_abstain, float("nan")),
        "per_generator": per_generator,
        "robustness": robustness,
        "n_train": len(train_items),
        "n_val": len(val_items),
        "n_val_sources": len(set(it[2] for it in val_items)),
        "trainable_params_stage1": model.n_trainable(),
        "elapsed_hours": (time.time() - t0) / 3600,
    }

    with open(CFG.REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    log("\n" + "=" * 64)
    log("  PHANTOM v3.1 — HONEST REPORT")
    log("=" * 64)
    log(f"  Backbone:             {CFG.BACKBONE}")
    if CFG.AXON_STATE_DIR:
        log(f"  Axon-state:           {CFG.AXON_STATE_DIR}")
    if CFG.WARMSTART_CKPT:
        log(f"  Warmstart:            {os.path.basename(CFG.WARMSTART_CKPT)}")
    if ewc is not None:
        log(f"  EWC lambda:           {CFG.EWC_LAMBDA}")
    if replay_items:
        log(f"  Replay items:         {len(replay_items):,}")
    log(f"  Multi-scale:          {CFG.MULTI_SCALE}")
    log(f"  Frequency branch:     {CFG.FREQUENCY_BRANCH}")
    log(f"  Base AUC (no TTA):    {report['auc_base_no_tta']:.4f}")
    log(f"  TTA  AUC:             {report['auc_with_tta']:.4f}")
    log(f"  Bootstrap 95% CI:     [{report['auc_95ci'][0]:.4f}, "
        f"{report['auc_95ci'][1]:.4f}]")
    log(f"  Accuracy:             {report['accuracy']:.4f}")
    log(f"  ECE before/after cal: {ece_before:.4f} -> {ece_after:.4f}  "
        f"(T={T:.3f})")
    log(f"  AUC @ {int(CFG.ABSTAIN_TARGET_COVERAGE * 100)}% coverage: "
        f"{report['auc_at_coverage']:.4f}")
    log(f"  Leakage:              {leak_report.get('leak_rate', 0):.3%} "
        f"({leak_report.get('verdict', '?')})")
    log("  Per-generator AUC (base):")
    for src, data in sorted(per_generator.items()):
        log(f"    {src:<28} AUC={data['auc_base']:.4f}  (n={data['n']})")
    log("  Robustness (JPEG):")
    for q, auc_val in robustness["jpeg"].items():
        log(f"    q={q:<4} AUC={auc_val:.4f}")
    log("  Robustness (downscale):")
    for sc, auc_val in robustness["downscale"].items():
        log(f"    x{sc:<4} AUC={auc_val:.4f}")

    if (report["auc_base_no_tta"] > 0.999
            and leak_report.get("leak_rate", 0) < 0.01):
        log("  *** NOTE: AUC>0.999 with low leakage on UNSEEN generators is "
            "unusual. ***")
        log("  *** Verify your holdout sources are truly disjoint "
            "generators. ***")

    log(f"\n  Full report -> {CFG.REPORT_JSON}")
    log(f"  Elapsed: {report['elapsed_hours']:.2f}h")
    log("=" * 64)


# ============================================================================
#  CLI ENTRY POINT
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="PHANTOM v3.1 — Deepfake Detector (axon-state compatible)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--smoke-test", action="store_true",
                        help="Tiny fast run that exercises all code paths")
    parser.add_argument("--input", type=str, default=None,
                        help="Override input data directory")
    parser.add_argument("--work", type=str, default=None,
                        help="Override working/output directory")
    parser.add_argument("--inference", type=str, default=None,
                        help="Run inference on a single image (requires --ckpt)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint for inference mode")
    parser.add_argument("--no-tta", action="store_true",
                        help="Disable test-time augmentation")
    parser.add_argument("--axon-state", type=str, default=None,
                        help="Path to axon-state directory")
    parser.add_argument("--replay", type=str, default=None,
                        help="Path to AXON_REPLAY.json")
    parser.add_argument("--warmstart", type=str, default=None,
                        help="Path to champion .pth for weight warmstart")
    parser.add_argument("--ewc", type=str, default=None,
                        help="Path to AXON_EWC.pth")
    parser.add_argument("--ewc-lambda", type=float, default=0.0,
                        help="EWC penalty weight (0=disabled)")
    args = parser.parse_args()

    if args.input:
        CFG.INPUT_DIR = args.input
    if args.work:
        CFG.WORK_DIR = args.work
        CFG.CKPT_STAGE1 = f"{CFG.WORK_DIR}/phantom_stage1.pth"
        CFG.CKPT_FINAL  = f"{CFG.WORK_DIR}/phantom_final.pth"
        CFG.CKPT_RESUME = f"{CFG.WORK_DIR}/phantom_resume.pth"
        CFG.CACHE_JSON  = f"{CFG.WORK_DIR}/phantom_items.json"
        CFG.REPORT_JSON = f"{CFG.WORK_DIR}/phantom_report.json"
        os.makedirs(CFG.WORK_DIR, exist_ok=True)

    if args.axon_state:
        CFG.AXON_STATE_DIR = args.axon_state
    elif not CFG.AXON_STATE_DIR:
        detected = find_axon_state(CFG.INPUT_DIR)
        if detected:
            CFG.AXON_STATE_DIR = detected
    if args.replay:
        CFG.REPLAY_PATH = args.replay
    if args.warmstart:
        CFG.WARMSTART_CKPT = args.warmstart
    if args.ewc:
        CFG.EWC_PATH = args.ewc
    if args.ewc_lambda > 0:
        CFG.EWC_LAMBDA = args.ewc_lambda

    if args.inference:
        if not args.ckpt:
            parser.error("--inference requires --ckpt")
        model = PhantomDetector().to(DEVICE)
        result = infer_single(
            model, args.inference, args.ckpt, tta=not args.no_tta
        )
        print(json.dumps(result, indent=2))
        return

    try:
        run(smoke=args.smoke_test)
    except Exception:
        log("FATAL:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
