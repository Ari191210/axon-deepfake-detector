"""
AXON SOTA v10.0 -- v9.0 with the confirmed-by-execution faults closed.

Every fix below was reproduced in an isolated CPU harness against real
timm 1.0.28 / torch 2.13 BEFORE being written, not read-and-guessed.

CHANGELOG v9.0 -> v10.0

  [FIX-A / BLOCKING] DINO_FORCE_224 crashed the run in round 0.
    v9 set size_dino=224 but still BUILT bb_dino at its native 518px.
    timm's PatchEmbed has strict_img_size=True, so the very first
    DetectorModel() -> _selftest() -> forward(zeros(2,3,224,224)) raised
      AssertionError: Input height (224) doesn't match model (518).
    The model has to be CONSTRUCTED at 224 so timm resamples pos_embed on
    load.  ->  create_model(..., img_size=CFG.IMG_SIZE)

  [FIX-B / CRITICAL, SILENT] SupConLoss treated EVERY pair as a positive.
    `hard` was 1-D, so `hard.t()` is a no-op and `hard == hard.t()` is an
    elementwise all-True vector, which broadcast against ~eye into an
    all-ones off-diagonal mask.  ->  hard = (labels > 0.5).long().view(-1, 1)

  [FIX-C / HIGH, SILENT] SOURCE_OVERRIDES poisoned labels.
    The override loop substring-matched the FULL path and short-circuited
    BEFORE the leaf-folder check, so any dataset named after the corpus it
    was generated from had every image relabeled.  ->  overrides are now the
    LAST resort, only consulted when neither the leaf name nor the path
    tokens give a signal.

  [FIX-D] EMA could die permanently after an arch change.
    ->  `if not restored or not ema.shadow: ema.refresh(model)`

  [FIX-E] MIL ran EVA-02 twice per forward.
    ->  one forward_features pass; pooled feat and MIL tokens both from it.

  [FIX-F] DEDUP_REALS_XSOURCE was a guaranteed no-op. Replaced with a real
    perceptual 16x16 grayscale average-hash, gated behind
    CFG.XSOURCE_AHASH. Default: OFF.

  [FIX-G] EfficientNetV2-M's native size is 384 -> CFG.CNN_FORCE_224.

  [FIX-H] Duplicate self._selftest() in __init__ removed.

  [FIX-I] FALLBACK_CACHE_VERS staging fixed.

  [FIX-J] param_groups() membership uses id()-keyed set.

  [FIX-K] _check_unfreeze_transition() honours read_only.

  [CACHE / STATE] CACHE_VER "_v9" -> "_v10", FALLBACK_CACHE_VERS = ().

SMOKE TEST: WALL_H=0.15, STEPS_PER_ROUND=20 before the 8.5h run.
"""
import os, sys, gc, json, time, math, random, hashlib, copy, traceback, io
import glob, shutil, subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFile, ImageFilter, ImageDraw, ImageEnhance
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded")
except Exception:
    print("No HF_TOKEN secret -- backbone downloads slower but OK")

try:
    import psutil
    _PROC = psutil.Process(os.getpid())
    def ram_gb(): return _PROC.memory_info().rss / 1e9
except Exception:
    def ram_gb(): return -1.0

try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
    def autocast(device_type="cuda", dtype=None, enabled=True):
        return _autocast(device_type, dtype=dtype, enabled=enabled)
    def GradScaler(device="cuda", enabled=True):
        try: return _GradScaler(device, enabled=enabled)
        except TypeError: return _GradScaler(enabled=enabled)
except ImportError:
    from torch.cuda.amp import autocast as _cuda_autocast, GradScaler as _GradScaler
    def autocast(device_type="cuda", dtype=None, enabled=True):
        return _cuda_autocast(dtype=dtype, enabled=enabled)
    def GradScaler(device="cuda", enabled=True):
        return _GradScaler(enabled=enabled)
from torchvision import transforms

try:
    import timm
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm>=1.0.3"])
    import timm

from sklearn.metrics import roc_auc_score, accuracy_score

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_OK = torch.cuda.is_available()

def _bf16_really_supported():
    if not AMP_OK: return False
    try:
        major, _ = torch.cuda.get_device_capability(0)
        return major >= 8 and torch.cuda.is_bf16_supported()
    except Exception:
        return False

USE_BF16 = _bf16_really_supported()
DTYPE  = torch.bfloat16 if USE_BF16 else torch.float16
USE_SCALER = AMP_OK and not USE_BF16

def log(msg):
    r = ram_gb()
    tag = f" | RAM {r:.1f}GB" if r >= 0 else ""
    print(f"[{datetime.now().strftime('%H:%M:%S')}{tag}] {msg}", flush=True)

def free_disk_gb(path=None):
    try:
        return shutil.disk_usage(path or "/kaggle/working").free / 1e9
    except Exception:
        return -1.0

# CONFIG
class CFG:
    INPUT  = "/kaggle/input"
    WORK   = "/kaggle/working"
    REGISTRY    = f"{WORK}/registry"
    JUDGE_CACHE = f"{WORK}/judge_split.json"
    LEDGER      = f"{WORK}/AXON_LEDGER.json"
    REPLAY      = f"{WORK}/AXON_REPLAY.json"
    EWC_PATH    = f"{WORK}/AXON_EWC.pth"
    ITEMS_CACHE = f"{WORK}/AXON_ITEMS.json"
    WORKER_DIR  = f"{WORK}/gpu_workers"

    CACHE_VER = "_v10"           # [FIX-C] bumped: the LABELS in every older
                                 # cache may be poisoned. One clean rescan.
    FALLBACK_CACHE_VERS = ()     # [FIX-C] deliberately empty.

    # [FIX-C] EWC / replay are keyed to the label logic, not just the file
    # list. Bump this string whenever discover()'s labelling changes.
    LABEL_LOGIC_VER   = "v10-overrides-are-last-resort"
    LABEL_VER_MARKER  = f"{WORK}/AXON_LABEL_LOGIC.txt"
    UNFREEZE_MARKER   = f"{WORK}/AXON_UNFREEZE_MODE.txt"

    MIN_BATCHES = 10
    MAX_SKIPS   = 5

    SOURCE_WRAPPER = {"datasets", "input", "kaggle"}

    # [FIX-C] LAST RESORT ONLY -- consulted when neither the leaf folder name
    # nor the path tokens give any signal.
    SOURCE_OVERRIDES = {
        "celeba": 0, "ffhq": 0, "flickr": 0,
        "ai-elites": 1, "ai_elites": 1, "gravex": 1,
        "chatgpt": 1, "gemini": 1,
    }
    AMBIGUOUS_ROOT_HINTS = ["140k", "real and fake", "real-and-fake",
                            "real_and_fake", "real vs fake", "real-vs-fake",
                            "fake-vs-real", "fake vs real"]
    PRIOR_BRAIN_HINTS = ["project33", "brain", "axon"]

    CSV_DATASETS = [
        {"src": "ddd2026",
         "csv_glob": "*deepfake-detection-dataset-2026*/**/FINAL_DATASET.csv",
         "url_col": "image_url", "label_text_col": "label",
         "label_num_col": "label_numeric", "split_col": None,
         "cache_dir": f"{WORK}/csv_img_cache/ddd2026",
         "max_download": 6600, "dl_timeout": 8},
    ]
    CSV_DL_WORKERS = 16

    DATASET_SRC_HINTS = {
        "deepdetect-2025": "deepdetect2025", "ddata": "deepdetect2025",
        "deepfake-detection-dataset-2026": "ddd2026",
    }
    HELDOUT_SPLIT_TOKENS = {"test", "testing", "val", "valid", "validation"}

    REAL_TAGS = {"real","genuine","authentic","original","ffhq","celeba",
                 "pristine","unaltered","flickr","faces","youtube"}
    FAKE_TAGS = {"fake","synthetic","deepfake","generated","ai","gan",
                 "diffusion","sdxl","midjourney","stylegan","dalle",
                 "ai-elites","gravex","cifake","synthesis","fakeavceleb"}
    IMG_EXT = (".jpg",".jpeg",".png",".webp",".bmp")

    BACKBONE_TF   = "eva02_base_patch14_224.mim_in22k"
    BACKBONE_CNN  = "tf_efficientnetv2_m.in21k_ft_in1k"
    CLIP_BACKBONE = "vit_base_patch16_clip_224.openai"
    DINO_BACKBONE = "vit_base_patch14_dinov2.lvd142m"

    # [FIX-A] DINOv2's pretrained_cfg is 518x518 with strict_img_size=True.
    # The model must be CONSTRUCTED at 224 (timm then resamples pos_embed).
    DINO_FORCE_224 = True
    # [FIX-G] EfficientNetV2-M's native size is 384. Fully convolutional,
    # accepts 224 fine (same 1280-dim output).
    CNN_FORCE_224  = True

    USE_DUAL = True
    USE_FREQ = True
    UNFREEZE_LN = True
    UNFREEZE_BACKBONE_MODE    = "none"   # "none" | "last_block" | "ln_only" | "bias_only"
    UNFREEZE_BACKBONES        = ("tf","cnn","dino","clip")
    UNFREEZE_BACKBONE_LR_MULT = 1/30.0

    IMG_SIZE = 224
    PROBE_DIM = 256
    BATCH = 4
    EVAL_BATCH     = 32

    VAL_FRAC = 0.15
    VAL_CAP        = 12000
    SLICE_VAL_CAP  = 4000

    NUM_WORKERS   = 0
    PIN_MEMORY    = False
    TRAIN_POOL    = 60000
    PER_SOURCE_CAP_FRAC = 0.25
    RAM_LIMIT_GB       = 11.0
    RAM_LIMIT_GB_DUAL  = 6.0
    WATCH_EVERY   = 50

    # [FIX-F] real perceptual 16x16 grayscale average-hash. OFF by default.
    DEDUP_REALS_XSOURCE = False
    XSOURCE_AHASH_SIZE  = 16

    TTA = True
    TTA_MULTISCALE = True
    SBI_P = 0.25
    GATE_ENTROPY_LAM = 0.02
    AUX_LAM = 0.3
    WORST_SLICE_WEIGHT = 0.5

    PROMOTE_MARGIN = 0.0005
    REGRESSION_TOL = 0.01
    DRIFT_DROP = 0.05

    STEPS_PER_ROUND = 600
    ROUND_MAX_S = 2400
    REPLAY_CAP = 4000
    WALL_H = 8.5
    HEADROOM_S = 600
    SEED = 42

    MIN_FREE_GB   = 3.0
    ROLLBACK_KEEP = 3

    USE_CUTMIX  = True
    CUTMIX_P    = 0.25
    CUTMIX_BETA = 1.0
    DOWNUP_P    = 0.3

    USE_DCT   = True
    DCT_BLOCK = 8

    USE_AE    = True
    AE_LATENT = 64
    AE_LAM    = 0.1

    USE_CLIP  = True
    USE_DINO  = True

    USE_CONTRASTIVE  = True
    CONTRASTIVE_LAM  = 0.1
    CONTRASTIVE_TEMP = 0.1

    USE_MIL       = True
    MIL_TOPK_FRAC = 0.1
    MIL_WEIGHT    = 0.3

    USE_ADV       = False
    ADV_MODE      = "fgsm"
    ADV_EPS       = 2/255
    ADV_STEPS     = 3
    ADV_FRAC      = 0.3
    ADV_START_AUC = 0.97

    EVAL_JPEG_PURIFY_Q = 75
    USE_EVAL_PURIFY    = False

    USE_DUAL_GPU        = True
    WORKER_TIMEOUT_SLOP = 300

os.makedirs(CFG.WORK, exist_ok=True)
os.makedirs(CFG.REGISTRY, exist_ok=True)
os.makedirs(CFG.WORKER_DIR, exist_ok=True)

# DATA DISCOVERY
def _file_sig(path):
    """Kaggle-safe dedup key: os.stat() ONLY (size + mtime_ns). NO read."""
    try:
        st = os.stat(path)
        return f"{st.st_size}_{st.st_mtime_ns}"
    except Exception:
        return None

def _ahash(path, n=None):
    """[FIX-F] REAL perceptual key: NxN grayscale average-hash."""
    n = n or CFG.XSOURCE_AHASH_SIZE
    try:
        with Image.open(path) as im:
            g = im.convert("L").resize((n, n), Image.BILINEAR)
        a = np.asarray(g, dtype=np.float32)
        bits = (a > a.mean()).astype(np.uint8).flatten()
        return np.packbits(bits).tobytes()
    except Exception:
        return None

def _derive_source(dp):
    """Structural Kaggle-mount parsing (no hardcoded username)."""
    try:
        rp = Path(dp).relative_to(Path(CFG.INPUT)).parts
    except ValueError:
        return "unknown"
    if not rp:
        return "unknown"
    if rp[0].lower() == "datasets" and len(rp) >= 3:
        src = rp[2]
    else:
        src = rp[0]
    src = src.lower()
    low = dp.lower()
    for hint, name in CFG.DATASET_SRC_HINTS.items():
        if hint in low: return name
    return src

def _label_for_dir(dp):
    """[FIX-C] Label a directory. Precedence, most specific first:
         1. leaf folder name   ('.../fake' -> 1, '.../real' -> 0)
         2. ambiguity guard    (unlabelable root -> skip)
         3. path tokens        ('.../ffhq-dataset/images' -> 0)
         4. SOURCE_OVERRIDES   (LAST resort only)
    Returns None to mean 'skip this directory'."""
    low = dp.lower()
    leaf_tok = set(Path(dp).name.lower().replace("_", " ").replace("-", " ").split())
    lf, lr = bool(leaf_tok & CFG.FAKE_TAGS), bool(leaf_tok & CFG.REAL_TAGS)
    if lf and not lr: return 1
    if lr and not lf: return 0

    amb = any(h in low for h in CFG.AMBIGUOUS_ROOT_HINTS)
    phf = any(t in CFG.FAKE_TAGS for part in Path(dp).parts
              for t in part.replace("_", " ").replace("-", " ").split())
    phr = any(t in CFG.REAL_TAGS for part in Path(dp).parts
              for t in part.replace("_", " ").replace("-", " ").split())
    if amb or (phf and phr): return None
    if phf and not phr: return 1
    if phr and not phf: return 0

    for key, lab in CFG.SOURCE_OVERRIDES.items():
        if key in low: return lab
    return None

import csv as _csv
import urllib.request as _urlreq
from concurrent.futures import ThreadPoolExecutor, as_completed

def _csv_label(row, spec):
    tc = spec.get("label_text_col")
    if tc and tc in row and row[tc].strip():
        v = row[tc].strip().lower()
        if v in ("fake","ai","ai generated","synthetic","generated"): return 1
        if v in ("real","authentic","genuine","pristine"): return 0
    nc = spec.get("label_num_col")
    if nc and nc in row and str(row.get(nc,"")).strip() not in ("","None"):
        try: return 0 if int(float(row[nc])) == 1 else 1
        except ValueError: pass
    return None

def _csv_split(row, spec):
    sc = spec.get("split_col"); cand = []
    if sc and sc in row: cand.append(row[sc])
    for k in row:
        if "split" in k.lower() or k.lower() in ("set","subset","fold","partition"):
            cand.append(row[k])
    for c in cand:
        cl = str(c).strip().lower()
        if cl in CFG.HELDOUT_SPLIT_TOKENS: return "test"
        if cl in ("train","training"): return "train"
    return "train"

def _fetch_one(args):
    url, dst, timeout = args
    if os.path.exists(dst) and os.path.getsize(dst) > 512: return dst
    try:
        req = _urlreq.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with _urlreq.urlopen(req, timeout=timeout) as r:
            ctype = (r.headers.get("Content-Type") or "").lower()
            data = r.read()
        if len(data) < 512 or "image" not in ctype: return None
        Image.open(io.BytesIO(data)).verify()
        with open(dst, "wb") as f: f.write(data)
        return dst
    except Exception:
        return None

def discover_csv(spec):
    os.makedirs(spec["cache_dir"], exist_ok=True)
    pattern = spec.get("csv_glob", "**/*.csv"); filename = os.path.basename(pattern)
    slug = [t for t in pattern.lower().replace("*"," ").replace("/"," ").split()
            if t and not t.endswith(".csv")]
    base = CFG.INPUT.count(os.sep)
    cands = []
    for dp, dirs, files in os.walk(CFG.INPUT):
        if CFG.WORK in dp:
            dirs[:] = []; continue
        if dp.count(os.sep) - base > 6:
            dirs[:] = []; continue
        if filename in files:
            cands.append(os.path.join(dp, filename))
    hits = [h for h in cands if all(s in h.lower() for s in slug)]
    hits = sorted(set(hits))
    if not hits:
        log(f"  CSV[{spec['src']}]: no CSV matching '{pattern}' -- skipped"); return []
    rows = []
    with open(hits[0], newline="", encoding="utf-8", errors="ignore") as f:
        for row in _csv.DictReader(f):
            url = (row.get(spec["url_col"]) or "").strip()
            lab = _csv_label(row, spec)
            if not url or lab is None: continue
            ext = os.path.splitext(url.split("?")[0])[1].lower()
            if ext not in CFG.IMG_EXT: ext = ".jpg"
            fid = hashlib.md5(url.encode()).hexdigest()[:16]
            dst = os.path.join(spec["cache_dir"], f"{fid}{ext}")
            rows.append((url, dst, lab, _csv_split(row, spec)))
            if len(rows) >= spec["max_download"]: break
    need = [(u,d,spec["dl_timeout"]) for (u,d,_,_) in rows
            if not (os.path.exists(d) and os.path.getsize(d) > 512)]
    if need:
        log(f"  CSV[{spec['src']}]: {len(rows):,} rows -- downloading {len(need):,} missing")
        with ThreadPoolExecutor(max_workers=CFG.CSV_DL_WORKERS) as ex:
            futs = {ex.submit(_fetch_one, t): t[1] for t in need}
            for fut in as_completed(futs): fut.result()
    else:
        log(f"  CSV[{spec['src']}]: all {len(rows):,} images already cached")
    items, ok = [], 0
    for (url, dst, lab, split) in rows:
        if os.path.exists(dst) and os.path.getsize(dst) > 512:
            items.append((dst, lab, spec["src"], split)); ok += 1
    log(f"  CSV[{spec['src']}]: usable {ok:,}/{len(rows):,} images")
    return items

def discover(root=CFG.INPUT, dedup=True, write_cache=True):
    _CACHE = CFG.ITEMS_CACHE.replace(".json", f"{CFG.CACHE_VER}.json")
    if not os.path.exists(_CACHE):
        for old_ver in CFG.FALLBACK_CACHE_VERS:      # empty in v10 by design
            old = CFG.ITEMS_CACHE.replace(".json", f"{old_ver}.json")
            if os.path.exists(old):
                try:
                    cached = [tuple(x) for x in json.load(open(old))]
                    if cached and all(os.path.exists(p) for p, _, _ in cached[:50]):
                        log(f"Discovered (prior cache {old_ver}) {len(cached):,} -- reusing")
                        if write_cache:
                            json.dump(cached, open(_CACHE, "w"))
                            old_split = CFG.ITEMS_CACHE.replace(".json", f"{old_ver}_split.json")
                            if os.path.exists(old_split):
                                try:
                                    shutil.copy(old_split,
                                                _CACHE.replace(".json","_split.json"))
                                except Exception:
                                    pass
                        return cached
                except Exception:
                    pass
    if os.path.exists(_CACHE):
        try:
            cached = [tuple(x) for x in json.load(open(_CACHE))]
            if cached and all(os.path.exists(p) for p, _, _ in cached[:50]):
                nf = sum(1 for _, l, _ in cached if l == 1)
                by = defaultdict(lambda:[0,0])
                for _, l, s in cached: by[s][l] += 1
                log(f"Discovered (cached) {len(cached):,}  fake={nf:,} real={len(cached)-nf:,}  "
                    f"sources={len(by)}")
                for s,(nr,nff) in sorted(by.items()):
                    log(f"    {s[:40]:40s} real={nr:,} fake={nff:,}")
                return cached
            log("  Discovery cache stale -- rebuilding.")
        except Exception:
            log("  Discovery cache unreadable -- rebuilding.")

    items, seen_path, seen_sig = [], set(), set()
    dup = 0
    for dp, dirs, files in os.walk(root):
        if CFG.WORK in dp:
            dirs[:] = []; continue
        low = dp.lower()
        if any(h in low for h in CFG.PRIOR_BRAIN_HINTS):
            dirs[:] = []; continue

        label = _label_for_dir(dp)          # [FIX-C]
        if label is None: continue

        src = _derive_source(dp)
        parts_tok = {p.lower() for part in Path(dp).parts
                     for p in part.replace("_"," ").replace("-"," ").split()}
        split = "test" if (parts_tok & CFG.HELDOUT_SPLIT_TOKENS) else "train"

        for f in files:
            if not f.lower().endswith(CFG.IMG_EXT): continue
            fp = os.path.join(dp, f)
            if fp in seen_path: continue
            seen_path.add(fp)
            if dedup:
                sig = _file_sig(fp)
                if sig is not None:
                    if sig in seen_sig: dup += 1; continue
                    seen_sig.add(sig)
            items.append((fp, label, src, split))

    for spec in CFG.CSV_DATASETS:
        for (fp, lab, src, split) in discover_csv(spec):
            if fp in seen_path: continue
            seen_path.add(fp)
            if dedup:
                sig = _file_sig(fp)
                if sig is not None:
                    if sig in seen_sig: dup += 1; continue
                    seen_sig.add(sig)
            items.append((fp, lab, src, split))

    if CFG.DEDUP_REALS_XSOURCE:
        # [FIX-F] now keyed on a real perceptual hash.
        log(f"  cross-source real dedup: hashing reals (decodes each image once)...")
        seen_real = {}; kept = []; xdup = 0
        for it in items:
            fp, lab, src, split = it
            if lab == 0:
                h = _ahash(fp)
                if h is not None:
                    if h in seen_real: xdup += 1; continue
                    seen_real[h] = src
            kept.append(it)
        items = kept
        log(f"  cross-source real dedup: dropped {xdup:,} duplicate real face(s)")
        del seen_real; gc.collect()

    if write_cache:
        json.dump(items, open(_CACHE.replace(".json","_split.json"), "w"))
    items3 = [(fp, lab, src) for (fp, lab, src, _s) in items]
    del seen_path, seen_sig; gc.collect()

    nf = sum(1 for _, l, _ in items3 if l == 1)
    by = defaultdict(lambda:[0,0])
    for _, l, s in items3: by[s][l] += 1
    log(f"Discovered {len(items3):,}  fake={nf:,} real={len(items3)-nf:,}  "
        f"sources={len(by)} deduped={dup:,}")
    for s,(nr,nff) in sorted(by.items()):
        log(f"    {s[:40]:40s} real={nr:,} fake={nff:,}")
    if write_cache:
        json.dump(items3, open(_CACHE, "w"))
    return items3

_MEAN = [0.485, 0.456, 0.406]; _STD = [0.229, 0.224, 0.225]

class RandomDownUpscale:
    def __init__(self, min_scale=0.4, max_scale=0.9, p=0.3):
        self.min_scale, self.max_scale, self.p = min_scale, max_scale, p
    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        s = random.uniform(self.min_scale, self.max_scale)
        small = img.resize((max(1,int(w*s)), max(1,int(h*s))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)

TRAIN_TF = transforms.Compose([
    transforms.Resize((CFG.IMG_SIZE + 32, CFG.IMG_SIZE + 32)),
    transforms.RandomCrop(CFG.IMG_SIZE),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.2, 0.2, 0.15, 0.03),
    transforms.RandomApply([transforms.GaussianBlur(3, (0.1, 2.0))], p=0.3),
    transforms.RandomApply([transforms.RandomResizedCrop(
        CFG.IMG_SIZE, scale=(0.7, 1.0))], p=0.2),
    RandomDownUpscale(p=CFG.DOWNUP_P),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.12)),
])
EVAL_TF = transforms.Compose([
    transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

def _load(path):
    try:
        im = Image.open(path).convert("RGB"); im.load(); return im
    except Exception:
        return Image.new("RGB", (CFG.IMG_SIZE, CFG.IMG_SIZE), (128, 128, 128))

def _jpeg_recompress(img, quality):
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=quality)
    buf.seek(0); return Image.open(buf).copy()

def _sbi(img: Image.Image) -> Image.Image:
    try:
        w, h = img.size
        src = img.copy()
        if random.random() < 0.5:
            s = random.uniform(0.95, 1.05)
            src = src.resize((int(w*s), int(h*s))).resize((w, h))
        if random.random() < 0.5:
            src = src.filter(ImageFilter.GaussianBlur(random.uniform(0.4, 1.4)))
        if random.random() < 0.5:
            src = ImageEnhance.Brightness(src).enhance(random.uniform(0.9, 1.1))
        mask = Image.new("L", (w, h), 0)
        cx, cy = w//2, int(h*0.45)
        rx, ry = int(w*random.uniform(0.28,0.42)), int(h*random.uniform(0.30,0.45))
        ImageDraw.Draw(mask).ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(random.uniform(5, 15)))
        return Image.composite(src, img, mask)
    except Exception:
        return img

class FakeDS(Dataset):
    def __init__(self, items, mode="train", jpeg_p=0.3, sbi_p=CFG.SBI_P):
        self.paths  = [it[0] for it in items]
        self.labels = [float(it[1]) for it in items]
        self.mode, self.jpeg_p, self.sbi_p = mode, jpeg_p, sbi_p
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        path, label = self.paths[i], self.labels[i]
        img = _load(path)
        if self.mode == "train" and label == 0.0 and random.random() < self.sbi_p:
            img = _sbi(img); label = 1.0
        if self.mode == "train" and random.random() < self.jpeg_p:
            img = _jpeg_recompress(img, random.randint(40, 95))
        if self.mode == "eval" and CFG.USE_EVAL_PURIFY and CFG.EVAL_JPEG_PURIFY_Q is not None:
            img = _jpeg_recompress(img, CFG.EVAL_JPEG_PURIFY_Q)
        t = (TRAIN_TF if self.mode == "train" else EVAL_TF)(img)
        return t, torch.tensor(label), path

def _per_source_cap(items, cap_frac=CFG.PER_SOURCE_CAP_FRAC, pool_size=CFG.TRAIN_POOL):
    by = defaultdict(list)
    for it in items: by[it[2] if len(it) > 2 else "?"].append(it)
    cap = max(1, int(pool_size * cap_frac))
    out = []
    for src, group in by.items():
        if len(group) > cap:
            out.extend(random.sample(group, cap))
        else:
            out.extend(group)
    random.shuffle(out)
    return out

def make_loader(items, mode, jpeg_p=0.3, batch=CFG.BATCH, balanced=False, sbi_p=CFG.SBI_P):
    ds = FakeDS(items, mode, jpeg_p, sbi_p)
    kw = dict(num_workers=CFG.NUM_WORKERS, pin_memory=CFG.PIN_MEMORY)
    if balanced:
        labs = ds.labels
        nr = labs.count(0.0); nf = labs.count(1.0)
        if nr == 0 or nf == 0:
            return DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True, **kw)
        w = torch.tensor([1.0/nr if l == 0.0 else 1.0/nf for l in labs])
        n_draw = min(w.numel(), CFG.STEPS_PER_ROUND * batch)
        sampler = WeightedRandomSampler(w, n_draw, replacement=True)
        return DataLoader(ds, batch_size=batch, sampler=sampler, drop_last=True, **kw)
    return DataLoader(ds, batch_size=batch, shuffle=False, **kw)

def _cutmix_batch(imgs, lbs, beta=1.0):
    lam = float(np.random.beta(beta, beta))
    B, C, H, W = imgs.shape
    perm = torch.randperm(B, device=imgs.device)
    rh, rw = int(H * math.sqrt(1-lam)), int(W * math.sqrt(1-lam))
    cy, cx = random.randint(0, H), random.randint(0, W)
    y1, y2 = max(0, cy-rh//2), min(H, cy+rh//2)
    x1, x2 = max(0, cx-rw//2), min(W, cx+rw//2)
    imgs_mixed = imgs.clone()
    imgs_mixed[:, :, y1:y2, x1:x2] = imgs[perm][:, :, y1:y2, x1:x2]
    lam_adj = 1.0 - ((y2-y1)*(x2-x1) / float(H*W))
    lbs_mixed = lam_adj * lbs + (1-lam_adj) * lbs[perm]
    return imgs_mixed, lbs_mixed, perm

class FrequencyStream(nn.Module):
    _M = torch.tensor(_MEAN).view(1,3,1,1); _S = torch.tensor(_STD).view(1,3,1,1)
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.GroupNorm(8,32), nn.GELU(),
            nn.Conv2d(32,64,3,2,1), nn.GroupNorm(8,64), nn.GELU(),
            nn.Conv2d(64,64,3,padding=1,groups=64), nn.GELU(),
            nn.Conv2d(64,128,1), nn.GroupNorm(8,128), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.proj = nn.Sequential(nn.Linear(128,out_dim), nn.LayerNorm(out_dim), nn.GELU())
    def forward(self, x):
        with autocast("cuda", enabled=False):
            m = self._M.to(x.device, torch.float32); s = self._S.to(x.device, torch.float32)
            xp = (x.float()*s+m).clamp(0,1); xp = xp - xp.mean(dim=(2,3), keepdim=True)
            mag = torch.abs(torch.fft.fftshift(
                torch.fft.fft2(xp, norm="ortho"))).clamp_min(1e-6)
            mag = torch.log1p(mag)
        return self.proj(self.net(mag))

class BlockDCTStream(nn.Module):
    def __init__(self, out_dim=128, block=8):
        super().__init__()
        self.block = block
        basis = self._dct_basis(block)
        self.register_buffer("basis", basis, persistent=False)
        in_feat = 3 * block * block
        self.net = nn.Sequential(
            nn.Conv2d(in_feat, 128, 1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1, groups=128), nn.GELU(),
            nn.Conv2d(128, 128, 1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.proj = nn.Sequential(nn.Linear(128, out_dim), nn.LayerNorm(out_dim), nn.GELU())

    @staticmethod
    def _dct_basis(n):
        b = torch.zeros(n, n)
        for u in range(n):
            cu = math.sqrt(1.0/n) if u == 0 else math.sqrt(2.0/n)
            for x in range(n):
                b[u, x] = cu * math.cos(math.pi * (2*x+1) * u / (2*n))
        return b

    def forward(self, x):
        with autocast("cuda", enabled=False):
            xf = x.float()
            b = self.block
            B, C, H, W = xf.shape
            ph, pw = (-H) % b, (-W) % b
            if ph or pw:
                xf = F.pad(xf, (0, pw, 0, ph))
                H, W = xf.shape[-2:]
            patches = xf.unfold(2, b, b).unfold(3, b, b)
            nH, nW = patches.shape[2], patches.shape[3]
            basis = self.basis.to(xf.device, xf.dtype)
            d = torch.einsum('uk,bcnmkl->bcnmul', basis, patches)
            d = torch.einsum('bcnmul,lv->bcnmuv', d, basis.t())
            d = d.permute(0, 1, 4, 5, 2, 3).contiguous()
            d = d.reshape(B, C*b*b, nH, nW)
            d = torch.log1p(d.abs())
            return self.proj(self.net(d))

class ReconResidualStream(nn.Module):
    def __init__(self, out_dim=128, latent=CFG.AE_LATENT):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3,32,4,2,1), nn.GroupNorm(8,32), nn.GELU(),
            nn.Conv2d(32,64,4,2,1), nn.GroupNorm(8,64), nn.GELU(),
            nn.Conv2d(64,latent,4,2,1), nn.GroupNorm(min(8,latent),latent), nn.GELU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent,64,4,2,1), nn.GroupNorm(8,64), nn.GELU(),
            nn.ConvTranspose2d(64,32,4,2,1), nn.GroupNorm(8,32), nn.GELU(),
            nn.ConvTranspose2d(32,3,4,2,1), nn.Tanh(),
        )
        self.res_net = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.GroupNorm(8,32), nn.GELU(),
            nn.Conv2d(32,64,3,2,1), nn.GroupNorm(8,64), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.proj = nn.Sequential(nn.Linear(64,out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def _reconstruct(self, x):
        z = self.enc(x)
        r = self.dec(z)
        if r.shape[-2:] != x.shape[-2:]:
            r = F.interpolate(r, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return r

    def forward(self, x):
        with torch.no_grad():
            recon = self._reconstruct(x.float())
            residual = (x.float() - recon)
        feat = self.res_net(residual)
        return self.proj(feat)

    def ae_loss(self, x, is_real_mask):
        if is_real_mask.sum() == 0:
            return torch.zeros((), device=x.device)
        xr = x.float()[is_real_mask]
        recon = self._reconstruct(xr)
        return F.mse_loss(recon, xr)

class DetectorModel(nn.Module):
    def __init__(self, dropout=0.3, probe_dim=CFG.PROBE_DIM,
                 use_freq=CFG.USE_FREQ, use_dual=CFG.USE_DUAL, unfreeze_ln=CFG.UNFREEZE_LN,
                 use_dct=CFG.USE_DCT, use_ae=CFG.USE_AE, use_clip=CFG.USE_CLIP,
                 use_mil=CFG.USE_MIL, use_dino=CFG.USE_DINO):
        super().__init__()
        self.arch = dict(dropout=dropout, probe_dim=probe_dim, use_freq=use_freq,
                          use_dual=use_dual, unfreeze_ln=unfreeze_ln, use_dct=use_dct,
                          use_ae=use_ae, use_clip=use_clip, use_mil=use_mil, use_dino=use_dino)
        self.use_freq, self.use_dual = use_freq, use_dual
        self.use_dct, self.use_ae, self.use_clip, self.use_mil = use_dct, use_ae, use_clip, use_mil
        self.use_dino = use_dino

        # EVA-02 is genuinely global_pool='avg'-native at pretraining,
        # so this explicit override is correct/redundant here.
        self.bb_tf = timm.create_model(CFG.BACKBONE_TF, pretrained=True,
                                        num_classes=0, global_pool="avg")
        d_tf = self.bb_tf.num_features
        self.size_tf = self._req_size(self.bb_tf)
        for p in self.bb_tf.parameters(): p.requires_grad = False
        if unfreeze_ln:
            for mod in self.bb_tf.modules():
                if isinstance(mod, nn.LayerNorm):
                    for p in mod.parameters(): p.requires_grad = True

        streams = [("tf", d_tf)]

        if use_dual:
            # CNN -- no CLS-token concept, global_pool='avg' is the native choice.
            self.bb_cnn = timm.create_model(CFG.BACKBONE_CNN, pretrained=True,
                                             num_classes=0, global_pool="avg")
            # [FIX-G] run EfficientNetV2-M at 224 to avoid upsampling every batch.
            self.size_cnn = CFG.IMG_SIZE if CFG.CNN_FORCE_224 else self._req_size(self.bb_cnn)
            for p in self.bb_cnn.parameters(): p.requires_grad = False
            streams.append(("cnn", self.bb_cnn.num_features))

        if use_dino:
            # [FIX-A] DINOv2 must be CONSTRUCTED at 224; timm resamples pos_embed.
            # global_pool deliberately NOT overridden (CLS-token-native, no fc_norm).
            dino_kw = dict(pretrained=True, num_classes=0)
            if CFG.DINO_FORCE_224:
                dino_kw["img_size"] = CFG.IMG_SIZE
            self.bb_dino = timm.create_model(CFG.DINO_BACKBONE, **dino_kw)
            self.size_dino = self._req_size(self.bb_dino)
            if CFG.DINO_FORCE_224:
                self.size_dino = CFG.IMG_SIZE
            for p in self.bb_dino.parameters(): p.requires_grad = False
            streams.append(("dino", self.bb_dino.num_features))

        if use_clip:
            # Same CLS-token rationale as bb_dino: do not override global_pool.
            self.bb_clip = timm.create_model(CFG.CLIP_BACKBONE, pretrained=True,
                                             num_classes=0)
            self.size_clip = self._req_size(self.bb_clip)
            for p in self.bb_clip.parameters(): p.requires_grad = False
            streams.append(("clip", self.bb_clip.num_features))

        if use_freq:
            self.freq = FrequencyStream(128); streams.append(("frq", 128))

        if use_dct:
            self.dct = BlockDCTStream(128, block=CFG.DCT_BLOCK); streams.append(("dct", 128))

        if use_ae:
            self.ae = ReconResidualStream(128); streams.append(("ae", 128))

        self.projs = nn.ModuleDict({n: nn.Linear(d, probe_dim) for n, d in streams})
        self.gate = nn.Sequential(nn.Linear(probe_dim*len(streams), len(streams)),
                                  nn.Softmax(-1))
        self.head = nn.Sequential(nn.LayerNorm(probe_dim),
                    nn.Linear(probe_dim,128), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(128,1))
        self.aux = nn.ModuleDict({n: nn.Linear(probe_dim, 1) for n, _ in streams})
        self._names = [s[0] for s in streams]

        if use_mil:
            self.mil_head = nn.Linear(d_tf, 1)

        self._apply_backbone_unfreeze()
        self._selftest()            # [FIX-H] was called twice

    @staticmethod
    def _req_size(model):
        cfg = getattr(model, "pretrained_cfg", {}) or {}
        return int(cfg.get("input_size", (3, CFG.IMG_SIZE, CFG.IMG_SIZE))[-1])

    def _resize(self, x, size):
        return x if x.shape[-1] == size else F.interpolate(
            x, size=(size,size), mode="bilinear", align_corners=False)

    def _tf_forward(self, x):
        """[FIX-E] ONE EVA-02 pass -> (pooled_feature, token_map)."""
        toks = self.bb_tf.forward_features(self._resize(x, self.size_tf))
        pooled = self.bb_tf.forward_head(toks)
        return pooled.float(), toks

    def _feat(self, n, x):
        if n == "tf":   return self._tf_forward(x)[0]
        if n == "cnn":  return self.bb_cnn(self._resize(x, self.size_cnn)).float()
        if n == "dino": return self.bb_dino(self._resize(x, self.size_dino)).float()
        if n == "clip": return self.bb_clip(self._resize(x, self.size_clip)).float()
        if n == "frq":  return self.freq(x)
        if n == "dct":  return self.dct(x)
        if n == "ae":   return self.ae(x)
        raise ValueError(f"unknown stream '{n}'")

    def _mil_from_tokens(self, toks):
        """[FIX-E] Consumes the token map from the single _tf_forward pass."""
        if not self.use_mil or toks is None:
            return None
        try:
            feats = toks
            if feats.dim() == 4:                       # CNN-style (B,D,H,W)
                B, D, H, W = feats.shape
                feats = feats.flatten(2).transpose(1, 2)
            if feats.dim() != 3:
                return None
            n = feats.shape[1]
            root = int(round(math.sqrt(n)))
            if root*root == n - 1:                     # strip CLS token
                feats = feats[:, 1:, :]
            elif root*root != n:
                return None
            scores = self.mil_head(feats.float()).squeeze(-1)
            k = max(1, int(scores.shape[1] * CFG.MIL_TOPK_FRAC))
            topk = torch.topk(scores, k, dim=1).values
            return topk.mean(dim=1)
        except Exception:
            return None

    def forward(self, x, return_aux=False):
        # [FIX-E] single EVA-02 pass, reused for both the 'tf' embedding and MIL.
        tf_pooled, tf_toks = self._tf_forward(x)
        embs = []
        for n in self._names:
            f = tf_pooled if n == "tf" else self._feat(n, x)
            embs.append(F.normalize(self.projs[n](f), dim=-1))
        cat = torch.cat(embs, dim=-1)
        w = self.gate(cat)
        fused = sum(w[:, i:i+1] * embs[i] for i in range(len(embs)))
        logit = self.head(fused).squeeze(-1)
        mil_logit = self._mil_from_tokens(tf_toks) if self.use_mil else None
        if mil_logit is not None:
            combined = (1-CFG.MIL_WEIGHT)*logit + CFG.MIL_WEIGHT*mil_logit
        else:
            combined = logit
        if not return_aux:
            return combined
        # MIL already gets a training signal via `combined`; deliberately not
        # added into aux_logits (that would dilute the per-stream signal
        # and double-count MIL's supervision).
        aux_logits = {n: self.aux[n](embs[i]).squeeze(-1) for i, n in enumerate(self._names)}
        return {"logit": combined, "gate": w, "aux": aux_logits, "fused": fused,
                "mil_logit": mil_logit}

    @torch.no_grad()
    def _selftest(self):
        was = self.training; self.eval()
        try: _ = self.forward(torch.zeros(2,3,CFG.IMG_SIZE,CFG.IMG_SIZE))
        finally: self.train(was)

    def trainable_params(self): return [p for p in self.parameters() if p.requires_grad]
    def n_trainable(self): return sum(p.numel() for p in self.trainable_params())

    def _apply_backbone_unfreeze(self):
        """Idempotent post-construction mutation. Default mode 'none' leaves
        _unfrozen_backbone_ids empty, making param_groups() byte-identical to a
        single flat AdamW group."""
        # [FIX-J] id()-keyed, not a set of Tensors.
        self._unfrozen_backbone_ids = set()
        mode = CFG.UNFREEZE_BACKBONE_MODE
        if mode == "none":
            return
        name_to_bb = {"tf": getattr(self,"bb_tf",None), "cnn": getattr(self,"bb_cnn",None),
                      "dino": getattr(self,"bb_dino",None), "clip": getattr(self,"bb_clip",None)}
        for tag in CFG.UNFREEZE_BACKBONES:
            bb = name_to_bb.get(tag)
            if bb is None: continue
            if mode == "last_block" and hasattr(bb, "blocks"):
                target_modules = [bb.blocks[-1]]
            elif mode == "ln_only":
                target_modules = [m for m in bb.modules() if isinstance(m, nn.LayerNorm)]
            elif mode == "bias_only":
                target_modules = [bb]
            else:
                target_modules = []
            for mod in target_modules:
                for pname, p in mod.named_parameters():
                    if mode == "bias_only" and not pname.endswith("bias"):
                        continue
                    p.requires_grad = True
                    self._unfrozen_backbone_ids.add(id(p))

    def param_groups(self, base_lr, backbone_lr_mult=1.0):
        unfrozen = self._unfrozen_backbone_ids
        if not unfrozen:
            return [{"params": self.trainable_params(), "lr": base_lr}]
        head_params, backbone_params = [], []
        for p in self.parameters():
            if not p.requires_grad: continue
            (backbone_params if id(p) in unfrozen else head_params).append(p)
        groups = []
        if head_params:     groups.append({"params": head_params, "lr": base_lr})
        if backbone_params: groups.append({"params": backbone_params, "lr": base_lr*backbone_lr_mult})
        return groups

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow  = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        self.buffers = {n: b.detach().clone() for n,b in model.named_buffers()}
        self._bak = {}
    @torch.no_grad()
    def refresh(self, model):
        self.shadow  = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        self.buffers = {n: b.detach().clone() for n,b in model.named_buffers()}
    @torch.no_grad()
    def update(self, model):
        for n,p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1-self.decay)
        for n,b in model.named_buffers():
            if n in self.buffers: self.buffers[n].copy_(b.detach())
    @torch.no_grad()
    def apply(self, model):
        self._bak = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        for n,p in model.named_parameters():
            if p.requires_grad and n in self.shadow and self.shadow[n].shape == p.shape:
                p.copy_(self.shadow[n].to(p.device, p.dtype))
    @torch.no_grad()
    def restore(self, model):
        for n,p in model.named_parameters():
            if p.requires_grad and n in self._bak: p.copy_(self._bak[n])
        self._bak = {}

class FocalLoss(nn.Module):
    def __init__(self, smooth=0.05):
        super().__init__(); self.smooth = smooth
    def forward(self, logits, targets, gamma=2.0):
        t = targets*(1-self.smooth) + 0.5*self.smooth
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        p = torch.sigmoid(logits)
        pt = (p*targets + (1-p)*(1-targets)).clamp(1e-6, 1-1e-6)
        return ((1-pt).pow(gamma) * bce).mean()

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.t = temperature
    def forward(self, emb, labels):
        emb = F.normalize(emb.float(), dim=-1)
        n = emb.shape[0]
        if n < 2:
            return torch.zeros((), device=emb.device)
        # [FIX-B] view(-1, 1), NOT view(-1). A 1-D `hard` makes hard.t() a
        # no-op and treats EVERY pair (incl. real<->fake) as a positive.
        hard = (labels > 0.5).long().view(-1, 1)
        sim = emb @ emb.t() / self.t
        mask_self = torch.eye(n, device=emb.device, dtype=torch.bool)
        mask_pos = (hard == hard.t()) & (~mask_self)     # (n,1) == (1,n) -> (n,n)
        sim = sim.masked_fill(mask_self, -1e9)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        pos_count = mask_pos.sum(1).clamp_min(1)
        loss = -(log_prob * mask_pos).sum(1) / pos_count
        valid = mask_pos.sum(1) > 0
        if valid.sum() == 0:
            return torch.zeros((), device=emb.device)
        return loss[valid].mean()

def _fgsm_perturb(model, imgs, lbs, eps, loss_fn, gamma):
    """fp32 for the adversarial input-gradient (autocast disabled)."""
    imgs_adv = imgs.clone().detach().requires_grad_(True)
    with autocast("cuda", enabled=False):
        logit = model(imgs_adv.float())
        loss = loss_fn(logit, lbs.float(), gamma)
    grad = torch.autograd.grad(loss, imgs_adv, retain_graph=False)[0]
    return (imgs_adv + eps * grad.sign()).detach()

def _pgd_perturb(model, imgs, lbs, eps, steps, loss_fn, gamma):
    imgs_adv = imgs.clone().detach()
    alpha = (eps / max(1, steps)) * 2.0
    for _ in range(steps):
        imgs_adv.requires_grad_(True)
        with autocast("cuda", enabled=False):
            logit = model(imgs_adv.float())
            loss = loss_fn(logit, lbs.float(), gamma)
        grad = torch.autograd.grad(loss, imgs_adv)[0]
        imgs_adv = (imgs_adv + alpha * grad.sign()).detach()
        delta = torch.clamp(imgs_adv - imgs, -eps, eps)
        imgs_adv = (imgs + delta).detach()
    return imgs_adv

def _tta_views(imgs):
    views = [imgs, torch.flip(imgs, dims=[3])]
    if CFG.TTA_MULTISCALE:
        B, C, H, W = imgs.shape
        ch, cw = int(H*0.9), int(W*0.9)
        top, left = (H-ch)//2, (W-cw)//2
        cropped = imgs[:, :, top:top+ch, left:left+cw]
        zoomed = F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False)
        views.append(zoomed)
        views.append(torch.flip(zoomed, dims=[3]))
    return views

class EWC:
    def __init__(self, path=CFG.EWC_PATH, read_only=False):
        self.path, self.fisher, self.optima = path, {}, {}
        self.read_only = read_only
        if os.path.exists(path):
            try:
                b = torch.load(path, map_location="cpu", weights_only=True)
                self.fisher, self.optima = b.get("fisher", {}), b.get("optima", {})
                log(f"EWC: loaded Fisher for {len(self.fisher)} tensors"
                    f"{' (read-only)' if read_only else ''}")
            except Exception as e:
                log(f"EWC load skipped: {e}")

    def invalidate(self, reason=""):
        """Explicit reset only. Never deletes the on-disk file; consolidate()
        replaces it wholesale on the next promotion."""
        self.fisher, self.optima = {}, {}
        log(f"  EWC: in-memory state invalidated"
            f"{' -- ' + reason if reason else ''} (file kept intact)")

    def penalty(self, model, lam=200.0):
        if not self.fisher:
            return torch.zeros((), device=DEVICE, dtype=torch.float32)
        loss = torch.zeros((), device=DEVICE, dtype=torch.float32)
        for n,p in model.named_parameters():
            if not (p.requires_grad and n in self.fisher): continue
            f = self.fisher[n]
            if f.shape != p.shape: continue   # shape-safe per-tensor
            opt = self.optima[n]
            loss = loss + (f.to(p.device,torch.float32) *
                    (p.float()-opt.to(p.device,torch.float32)).pow(2)).sum()
        return lam * loss

    def consolidate(self, model, items, n=256):
        if not items: return
        if self.read_only:
            log("  EWC: consolidate skipped (read-only mode)"); return
        model.eval()
        fisher = {nm: torch.zeros_like(p, dtype=torch.float32)
                  for nm,p in model.named_parameters() if p.requires_grad}
        dl = make_loader(random.sample(items, min(n, len(items))), "eval", batch=8)
        cnt = 0
        for imgs, lbs, _ in dl:
            imgs, lbs = imgs.to(DEVICE), lbs.to(DEVICE)
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                with autocast("cuda", dtype=DTYPE, enabled=AMP_OK):
                    loss = F.binary_cross_entropy_with_logits(model(imgs), lbs.float())
                loss.backward()
            for nm,p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[nm] += p.grad.detach().float().pow(2)
            cnt += imgs.size(0)
        if cnt:
            for k in fisher: fisher[k] /= cnt
        self.fisher = {k: v.cpu() for k,v in fisher.items()}
        self.optima = {nm: p.detach().float().cpu()
                        for nm,p in model.named_parameters() if p.requires_grad}
        tmp = self.path + ".tmp"
        try:
            torch.save({"fisher": self.fisher, "optima": self.optima}, tmp)
            os.replace(tmp, self.path)
        except Exception as e:
            log(f"  EWC consolidate save failed: {e}")
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except Exception: pass
        model.zero_grad(set_to_none=True); del dl; gc.collect()
        model.train(); log("EWC: consolidated snapshot saved.")

class FrozenJudge:
    def __init__(self, all_items, write_cache=True):
        cache = CFG.JUDGE_CACHE.replace(".json", f"{CFG.CACHE_VER}.json")
        if os.path.exists(cache):
            try:
                d = json.load(open(cache))
                self.val    = [tuple(x) for x in d["val"]]
                self.train  = [tuple(x) for x in d["train"]]
                self.slices = {k: [tuple(x) for x in v] for k,v in d["slices"].items()}
                live = {it[2] for it in all_items}
                cached = {s for _,_,s in self.val} | {s for _,_,s in self.train}
                if live - cached: raise ValueError("stale")
                if len(self.val) > CFG.VAL_CAP: raise ValueError("oversized")
                log(f"Judge: loaded FROZEN split val={len(self.val):,} "
                    f"train={len(self.train):,} slices={len(self.slices)}")
                self._validate(); return
            except ValueError: pass
            except Exception: log("  Judge cache unreadable -- rebuilding.")

        split_path = CFG.ITEMS_CACHE.replace(".json", f"{CFG.CACHE_VER}_split.json")
        split_map = {}
        if os.path.exists(split_path):
            try:
                for fp,lab,src,sp in json.load(open(split_path)): split_map[fp] = sp
            except Exception: split_map = {}

        rng = random.Random(CFG.SEED)
        by_src = {}
        for it in all_items: by_src.setdefault(it[2], []).append(it)
        val, train, slices = [], [], {}
        for src, group in by_src.items():
            held = [g for g in group if split_map.get(g[0]) == "test"]
            pool = [g for g in group if split_map.get(g[0]) != "test"]
            if held:
                if len(held) > CFG.SLICE_VAL_CAP:
                    rng.shuffle(held); train.extend(held[CFG.SLICE_VAL_CAP:])
                    held = held[:CFG.SLICE_VAL_CAP]
                val.extend(held); train.extend(pool); slice_items = held
            else:
                rng.shuffle(group)
                k = min(max(1, int(len(group)*CFG.VAL_FRAC)), CFG.SLICE_VAL_CAP)
                val.extend(group[:k]); train.extend(group[k:]); slice_items = group[:k]
            if len({g[1] for g in slice_items}) >= 2:
                slices[src] = slice_items
        if len(val) > CFG.VAL_CAP:
            rng.shuffle(val); overflow = val[CFG.VAL_CAP:]; val = val[:CFG.VAL_CAP]
            train.extend(overflow); keep = {it[0] for it in val}
            slices = {s:[it for it in its if it[0] in keep] for s,its in slices.items()}
            slices = {s:v for s,v in slices.items() if len({it[1] for it in v})>=2}
            log(f"  Judge: val capped at {CFG.VAL_CAP:,}")
        self.val, self.train, self.slices = val, train, slices
        if write_cache:
            json.dump({"val":val,"train":train,"slices":slices}, open(cache,"w"))
        log(f"Judge: created FROZEN split val={len(val):,} train={len(train):,} "
            f"slices={len(slices)}")
        self._validate()

    def label_map(self):
        """[FIX-C] path -> label over everything the judge knows about."""
        m = {}
        for it in self.train: m[it[0]] = int(it[1])
        for it in self.val:   m[it[0]] = int(it[1])
        return m

    def _validate(self):
        self.val_ok = len({l for _,l,_ in self.val}) >= 2
        if not self.val_ok: log("  Judge val single-class -- scoring deferred.")

    @torch.no_grad()
    def _predict(self, model, items, ema=None):
        if not items: return {}
        if ema is not None: ema.apply(model)
        model.eval(); out = {}
        dl = make_loader(items, "eval", batch=CFG.EVAL_BATCH)
        try:
            for imgs, lbs, paths in dl:
                imgs = imgs.to(DEVICE)
                with autocast("cuda", dtype=DTYPE, enabled=AMP_OK):
                    views = _tta_views(imgs) if CFG.TTA else [imgs]
                    logit_sum = None
                    for v in views:
                        lg = model(v)
                        logit_sum = lg if logit_sum is None else logit_sum + lg
                    logit = logit_sum / len(views)
                probs = torch.sigmoid(logit).float().cpu().tolist()
                for pa,pr,l in zip(paths, probs, lbs.tolist()): out[pa] = (pr, l)
        finally:
            del dl; gc.collect()
        if ema is not None: ema.restore(model)
        model.train(); return out

    @staticmethod
    def _auc_acc(pred, items):
        ps, ls = [], []
        for it in items:
            r = pred.get(it[0])
            if r is None: continue
            ps.append(r[0]); ls.append(r[1])
        if len(set(ls)) < 2: return float("nan"), float("nan")
        return (roc_auc_score(ls, ps),
                accuracy_score(ls, [1 if p>0.5 else 0 for p in ps]))

    def evaluate(self, model, ema=None):
        union = list(self.val); seen = {it[0] for it in union}
        for its in self.slices.values():
            for it in its:
                if it[0] not in seen: union.append(it); seen.add(it[0])
        pred = self._predict(model, union, ema)
        auc, acc = self._auc_acc(pred, self.val)
        slice_auc = {}
        for src, its in self.slices.items():
            a,_ = self._auc_acc(pred, its)
            if not math.isnan(a): slice_auc[src] = a
        del pred; gc.collect()
        worst = min(slice_auc.values()) if slice_auc else auc
        return {"auc": auc, "acc": acc, "slices": slice_auc, "worst_slice": worst}

def _trainable_state_dict(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}

def _all_buffers(model):
    return {n: b.detach().cpu() for n, b in model.named_buffers()}

def _extract_state(blob):
    if not isinstance(blob, dict):
        return None
    if "model_trainable" in blob:
        merged = dict(blob.get("model_trainable", {}))
        merged.update(blob.get("buffers", {}))
        return merged
    if "model" in blob:
        return blob["model"]
    return blob

def _shape_safe_load(model, state_dict):
    if not state_dict:
        return 0, 0
    cur = model.state_dict()
    compat = {k: v for k, v in state_dict.items() if k in cur and tuple(cur[k].shape) == tuple(v.shape)}
    skipped = len(state_dict) - len(compat)
    model.load_state_dict(compat, strict=False)
    return len(compat), skipped

class Registry:
    def __init__(self):
        self.ip = f"{CFG.REGISTRY}/index.json"
        self.index = (json.load(open(self.ip)) if os.path.exists(self.ip)
                      else {"champion": None, "lineage": [], "meta": {}})
        self.index.setdefault("champion", None)
        self.index.setdefault("lineage", [])
        self.index.setdefault("meta", {})

    def _persist(self):
        tmp = self.ip + ".tmp"
        json.dump(self.index, open(tmp, "w"), indent=2)
        os.replace(tmp, self.ip)

    def rebaseline_champion(self, judge):
        c = self.index.get("champion")
        if not c: return
        fp = f"{CFG.REGISTRY}/{c}.pth"
        if not os.path.exists(fp): log("  Re-baseline skipped: .pth missing."); return
        sig = hashlib.md5(json.dumps({"v":len(judge.val),
                "s":{k:len(v) for k,v in sorted(judge.slices.items())}},
                sort_keys=True).encode()).hexdigest()[:12]
        meta = self.index.setdefault("meta", {}).setdefault(c, {})
        if meta.get("judge_sig") == sig:
            log(f"  Champion '{c}' already baselined -- skip."); return
        log(f"  Re-baselining champion '{c}' on the {CFG.CACHE_VER} judge...")
        b = torch.load(fp, map_location="cpu", weights_only=False)
        arch = b.get("arch") or {}
        model = (DetectorModel(**arch) if arch else DetectorModel()).float().to(DEVICE)
        kept, skipped = _shape_safe_load(model, _extract_state(b))
        if skipped:
            log(f"  re-baseline: {skipped} tensor(s) dropped, {kept} kept")
        ema = EMA(model)
        if b.get("ema_shadow"):
            cur = {n:p.shape for n,p in model.named_parameters()}
            ema.shadow  = {k:v.to(DEVICE) for k,v in b["ema_shadow"].items()
                           if k in cur and tuple(cur[k])==tuple(v.shape)}
            ema.buffers = {k:v.to(DEVICE) for k,v in b.get("ema_buffers", {}).items()}
            new_report = judge.evaluate(model, ema) if ema.shadow else judge.evaluate(model)
        else:
            new_report = judge.evaluate(model)
        old_auc = (b.get("report") or {}).get("auc")
        free = free_disk_gb()
        saved_ok = False
        if free < CFG.MIN_FREE_GB:
            log(f"  SKIP re-baseline SAVE: only {free:.2f}GB free")
        else:
            tmp = fp + ".tmp"
            try:
                torch.save({"model_trainable": _trainable_state_dict(model),
                            "buffers": _all_buffers(model),
                            "arch": model.arch,
                            "ema_shadow": ema.shadow, "ema_buffers": ema.buffers,
                            "report": new_report, "parent": b.get("parent"),
                            "ts": datetime.now().isoformat()}, tmp)
                os.replace(tmp, fp)
            except Exception as e:
                log(f"  re-baseline SAVE FAILED: {e}")
                if os.path.exists(tmp):
                    try: os.remove(tmp)
                    except Exception: pass
                del model, ema; gc.collect(); torch.cuda.empty_cache()
                return
            meta["report"] = new_report; meta["arch"] = model.arch; meta["judge_sig"] = sig
            self._persist(); saved_ok = True
        oa = f"{old_auc:.4f}" if isinstance(old_auc,(int,float)) and not math.isnan(old_auc) else "n/a"
        na = f"{new_report['auc']:.4f}" if not math.isnan(new_report['auc']) else "n/a"
        if saved_ok:
            log(f"  Champion '{c}' re-baselined: AUC {oa}->{na}  "
                f"worst_slice={new_report.get('worst_slice', float('nan')):.4f}")
        else:
            log(f"  Champion '{c}' evaluated (AUC {oa}->{na}) but not saved")
        if (isinstance(old_auc,(int,float)) and not math.isnan(old_auc)
                and not math.isnan(new_report["auc"])
                and new_report["auc"] < old_auc - 0.05):
            log(f"  NOTE: the champion's AUC dropped sharply on the corrected judge -- "
                f"EXPECTED if it was fitted under the old (poisoned) label logic -- "
                f"this is the first honest number. If it stays low, delete registry/ "
                f"and reseed from scratch.")
        del model, ema; gc.collect(); torch.cuda.empty_cache()

    def save_challenger(self, model, ema, report, parent, tag):
        free = free_disk_gb()
        if free < CFG.MIN_FREE_GB:
            log(f"  SKIP SAVE '{tag}': only {free:.2f}GB free")
            self.index["meta"][tag] = {"report": report, "arch": model.arch, "unsaved": True}
            self._persist(); return False
        fp = f"{CFG.REGISTRY}/{tag}.pth"
        tmp = fp + ".tmp"
        try:
            torch.save({"model_trainable": _trainable_state_dict(model),
                        "buffers": _all_buffers(model),
                        "arch": model.arch,
                        "ema_shadow": ema.shadow, "ema_buffers": ema.buffers,
                        "report": report, "parent": parent, "tag": tag,
                        "ts": datetime.now().isoformat()}, tmp)
            os.replace(tmp, fp)
        except Exception as e:
            log(f"  SAVE FAILED '{tag}': {e}")
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except Exception: pass
            self.index["meta"][tag] = {"report": report, "arch": model.arch, "unsaved": True}
            self._persist(); return False
        self.index["meta"][tag] = {"report": report, "arch": model.arch}
        self._persist(); return True

    def promote(self, tag, report):
        self.index["champion"] = tag
        self.index["lineage"].append({"tag":tag,"auc":report["auc"],
            "worst_slice":report.get("worst_slice"), "ts":datetime.now().isoformat()})
        self._persist()
        log(f"  PROMOTED '{tag}' AUC={report['auc']:.4f} "
            f"worst={report.get('worst_slice',float('nan')):.4f}")
        self._prune_old_champions()

    def _prune_old_champions(self):
        lineage = self.index.get("lineage", [])
        keep_tags = {e["tag"] for e in lineage[-CFG.ROLLBACK_KEEP:]}
        champ = self.index.get("champion")
        if champ: keep_tags.add(champ)
        pruned = 0
        for f in glob.glob(f"{CFG.REGISTRY}/*.pth"):
            tag = os.path.splitext(os.path.basename(f))[0]
            if tag not in keep_tags:
                try: os.remove(f); pruned += 1
                except Exception: pass
        if pruned:
            log(f"  pruned {pruned} old checkpoint(s), kept {sorted(keep_tags)}")

    def rollback(self):
        """Walks back until it finds a tag whose checkpoint actually exists."""
        lineage = self.index["lineage"]
        while len(lineage) >= 2:
            lineage.pop()
            prev = lineage[-1]["tag"]
            if os.path.exists(f"{CFG.REGISTRY}/{prev}.pth"):
                self.index["champion"] = prev; self._persist()
                log(f"  ROLLED BACK -> '{prev}'")
                return prev
            log(f"  rollback target '{prev}' has no checkpoint "
                f"-- trying further back")
        log("  no prior champion with an available checkpoint to roll "
            "back further; keeping current (degraded) champion")
        return self.index["champion"]

    def _meta_or_load(self, key):
        c = self.index.get("champion")
        if not c: return None
        cached = self.index.get("meta", {}).get(c)
        if cached and key in cached: return cached[key]
        fp = f"{CFG.REGISTRY}/{c}.pth"
        if not os.path.exists(fp): return None
        b = torch.load(fp, map_location="cpu", weights_only=False)
        return b.get(key)

    def champion_arch(self):   return self._meta_or_load("arch")
    def champion_report(self): return self._meta_or_load("report")

    def load_into(self, model, ema, tag=None):
        tag = tag or self.index.get("champion")
        if not tag:
            return False
        fp = f"{CFG.REGISTRY}/{tag}.pth"
        if not os.path.exists(fp):
            log(f"  CRITICAL: champion '{tag}' is set in index.json but {tag}.pth is "
                f"MISSING from registry/ -- proceeding from a fresh init; champion "
                f"+ EWC relevance silently lost. Check pruning/restore logs above.")
            return False
        b = torch.load(fp, map_location=DEVICE, weights_only=False)
        kept, skipped = _shape_safe_load(model, _extract_state(b))
        if skipped:
            log(f"  arch drift on load: {skipped} dropped, {kept} kept")
        if b.get("ema_shadow"):
            cur = {n:p.shape for n,p in model.named_parameters()}
            ema.shadow = {k:v.to(DEVICE) for k,v in b["ema_shadow"].items()
                          if k in cur and tuple(cur[k])==tuple(v.shape)}
            ema.buffers = {k:v.to(DEVICE) for k,v in b.get("ema_buffers", {}).items()}
            kept_ema, total_ema = len(ema.shadow), len(b["ema_shadow"])
            ra = b.get("report", {}).get("auc", float("nan"))
            ra = f"{ra:.4f}" if isinstance(ra,(int,float)) and not math.isnan(ra) else "n/a"
            log(f"  Loaded champion '{tag}' + EMA ({kept_ema}/{total_ema} ok, AUC={ra})")
            return kept_ema > 0 or kept > 0
        log(f"  Loaded champion '{tag}' (no EMA)"); return kept > 0

def _restore_registry_selective(state_src):
    os.makedirs(CFG.REGISTRY, exist_ok=True)
    src_reg = os.path.join(state_src, "registry")
    idx_path = os.path.join(src_reg, "index.json")
    if not os.path.exists(idx_path):
        log("  no index.json in prior registry -- skipping restore."); return
    shutil.copy(idx_path, os.path.join(CFG.REGISTRY, "index.json"))
    idx = json.load(open(idx_path))
    keep_tags = {idx.get("champion")}
    keep_tags |= {e["tag"] for e in idx.get("lineage", [])[-CFG.ROLLBACK_KEEP:]}
    keep_tags.discard(None)
    copied = 0
    champ = idx.get("champion")
    champ_copied = champ is None
    for tag in keep_tags:
        fp = os.path.join(src_reg, f"{tag}.pth")
        if os.path.exists(fp):
            shutil.copy(fp, os.path.join(CFG.REGISTRY, f"{tag}.pth")); copied += 1
            if tag == champ: champ_copied = True
    log(f"  Registry restored selectively: {copied} checkpoint(s)")
    if not champ_copied:
        log(f"  CRITICAL: prior state's champion '{champ}' has no checkpoint in restored "
            f"registry -- restore is incomplete. The next round will train fresh.")

def _bounded_pth_search(hints, max_depth=4):
    base = CFG.INPUT.count(os.sep)
    found = []
    for dp, dirs, files in os.walk(CFG.INPUT):
        if CFG.WORK in dp:
            dirs[:] = []; continue
        if dp.count(os.sep) - base > max_depth:
            dirs[:] = []; continue
        for f in files:
            if f.lower().endswith((".pth", ".pt")):
                fp = os.path.join(dp, f)
                if (any(h in fp.lower() for h in hints)
                        and "ewc" not in f.lower() and "replay" not in f.lower()):
                    found.append(Path(fp))
    return found

def import_prior_brain(model, judge):
    cands = _bounded_pth_search(CFG.PRIOR_BRAIN_HINTS, max_depth=4)
    if not cands:
        log("Brain import: none found."); return False
    base = judge.evaluate(model)["auc"]; base = 0.5 if math.isnan(base) else base
    best = base
    for pf in sorted(cands, key=lambda p: p.stat().st_size, reverse=True):
        try: raw = torch.load(str(pf), map_location="cpu", weights_only=False)
        except Exception as e: log(f"  {pf.name}: {e} -- SKIP"); continue
        sd = _extract_state(raw)
        if not isinstance(sd, dict) or not sd:
            log(f"  {pf.name}: no usable state_dict -- SKIP"); continue
        compat = {k.replace("module.","").replace("model.",""): v for k,v in sd.items()}
        backup = copy.deepcopy(model.state_dict())
        kept, skipped = _shape_safe_load(model, compat)
        if kept == 0:
            log(f"  {pf.name}: 0 compatible -- SKIP"); continue
        score = judge.evaluate(model)["auc"]
        if not math.isnan(score) and score >= best:
            log(f"  {pf.name}: +{kept} tensors AUC {best:.4f}->{score:.4f} -- KEPT"); best = score
        else:
            model.load_state_dict(backup, strict=False)
            log(f"  {pf.name}: AUC {score:.4f} < {best:.4f} -- REVERTED")
        del backup; gc.collect()
    return best > base

def _marker_transition(path, cur, ewc, what, read_only=False):
    """[FIX-C/FIX-K] Generic once-per-transition guard. Invalidates in-memory
    EWC state exactly once when an axis of the setup changes. Read-only workers
    never WRITE the marker."""
    prev = None
    if os.path.exists(path):
        try: prev = open(path).read().strip()
        except Exception: prev = None
    if prev == cur:
        return
    if prev is not None and ewc is not None:
        ewc.invalidate(reason=f"{what} changed '{prev}' -> '{cur}'")
    if read_only:
        return
    tmp = path + ".tmp"
    with open(tmp, "w") as f: f.write(cur)
    os.replace(tmp, path)

class EvolutionController:
    SEARCH = {
        "lr":       [3e-5, 5e-5, 1e-4, 2e-4],
        "dropout":  [0.2, 0.25, 0.3],
        "gamma":    [2.0, 2.5, 3.0],
        "jpeg_p":   [0.3, 0.5, 0.6],
        "ewc_lam":  [50.0, 100.0, 200.0],
        "sbi_p":    [0.0, 0.15, 0.25, 0.35],
    }

    def __init__(self, judge, registry, read_only=False):
        self.judge, self.reg = judge, registry
        self.ewc = EWC(read_only=read_only)
        _marker_transition(CFG.UNFREEZE_MARKER, CFG.UNFREEZE_BACKBONE_MODE,
                           self.ewc, "unfreeze mode", read_only)
        _marker_transition(CFG.LABEL_VER_MARKER, CFG.LABEL_LOGIC_VER,
                           self.ewc, "label logic", read_only)
        self.loss_fn = FocalLoss()
        self.replay = self._load_replay()

    def _load_replay(self):
        """[FIX-C] Replay stores its own labels, which may have been written
        under the OLD (poisoned) label logic. Re-validate against the freshly
        discovered label map and drop disagreements."""
        if not os.path.exists(CFG.REPLAY): return []
        try: raw = [tuple(x) for x in json.load(open(CFG.REPLAY))]
        except Exception: return []
        keep = [r for r in raw if os.path.exists(r[0])]
        missing = len(raw) - len(keep)

        lm = self.judge.label_map()
        revalidated, relabel_drop, unknown_drop = [], 0, 0
        for r in keep:
            fp, lab = r[0], int(r[1])
            truth = lm.get(fp)
            if truth is None:
                unknown_drop += 1     # no longer discovered at all -> drop
                continue
            if truth != lab:
                relabel_drop += 1     # label changed under the corrected logic
                continue
            revalidated.append(r)
        if missing or relabel_drop or unknown_drop:
            log(f"  Replay: {len(raw):,} -> {len(revalidated):,} "
                f"(missing={missing:,} relabeled={relabel_drop:,} "
                f"no-longer-discovered={unknown_drop:,})")
        return revalidated

    def _save_replay(self): json.dump(self.replay[-CFG.REPLAY_CAP:], open(CFG.REPLAY,"w"))

    def _cfg(self, rng): return {k: rng.choice(v) for k,v in self.SEARCH.items()}

    def _build_challenger(self, cfg):
        """Never touches EWC. penalty() already shape-guards per tensor, and
        consolidate() rebuilds Fisher from scratch scoped to whatever actually
        gets promoted."""
        stored = self.reg.champion_arch()
        cfg_arch = dict(dropout=cfg["dropout"], probe_dim=CFG.PROBE_DIM,
                        use_freq=CFG.USE_FREQ, use_dual=CFG.USE_DUAL, unfreeze_ln=CFG.UNFREEZE_LN,
                        use_dct=CFG.USE_DCT, use_ae=CFG.USE_AE, use_clip=CFG.USE_CLIP,
                        use_mil=CFG.USE_MIL, use_dino=CFG.USE_DINO)
        if stored:
            STRUCT = ("probe_dim","use_freq","use_dual","unfreeze_ln",
                      "use_dct","use_ae","use_clip","use_mil","use_dino")
            changed = [k for k in STRUCT if stored.get(k) != cfg_arch.get(k)]
            if changed:
                log(f"  ARCH CHANGE (exploring only, not yet committed): "
                    f"{{ {', '.join(changed)} }}")
                return DetectorModel(**cfg_arch).float().to(DEVICE)
            arch = dict(stored); arch["dropout"] = cfg["dropout"]
            return DetectorModel(**arch).float().to(DEVICE)
        return DetectorModel(dropout=cfg["dropout"]).float().to(DEVICE)

    def _train(self, model, ema, cfg, items, t_dead, champion_auc=None, ram_limit=None):
        ram_limit = ram_limit if ram_limit is not None else CFG.RAM_LIMIT_GB
        pool = items + self.replay
        pool = _per_source_cap(pool)
        if len(pool) > CFG.TRAIN_POOL:
            pool = random.sample(pool, CFG.TRAIN_POOL)
        log(f"  train pool: {len(pool):,}")
        opt = torch.optim.AdamW(model.param_groups(cfg["lr"], CFG.UNFREEZE_BACKBONE_LR_MULT),
                                 weight_decay=1e-4)
        scaler = GradScaler("cuda", enabled=USE_SCALER)
        dl = make_loader(pool, "train", jpeg_p=cfg["jpeg_p"], sbi_p=cfg["sbi_p"])
        del pool; gc.collect()
        try: n_avail = len(dl)
        except TypeError: n_avail = CFG.STEPS_PER_ROUND
        if n_avail < CFG.MIN_BATCHES:
            log(f"  only {n_avail} batches -- skip"); del opt,scaler,dl; gc.collect(); return None
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, CFG.STEPS_PER_ROUND))
        model.train(); step = 0; new_unc = []
        t_round = time.time() + CFG.ROUND_MAX_S; aborted = False

        supcon = SupConLoss(CFG.CONTRASTIVE_TEMP) if CFG.USE_CONTRASTIVE else None
        adv_active = CFG.USE_ADV and (champion_auc is not None and champion_auc >= CFG.ADV_START_AUC)
        if CFG.USE_ADV and not adv_active:
            log(f"  adversarial gated off (champion AUC {champion_auc} < {CFG.ADV_START_AUC})")

        for imgs, lbs, paths in dl:
            if step >= CFG.STEPS_PER_ROUND or time.time() > min(t_dead, t_round): break
            if step % CFG.WATCH_EVERY == 0:
                r = ram_gb()
                if r >= 0 and r > ram_limit:
                    log(f"    RAM {r:.1f}GB -- abort"); aborted = True; break
                free = free_disk_gb()
                if free >= 0 and free < CFG.MIN_FREE_GB:
                    log(f"    Disk {free:.2f}GB -- abort"); aborted = True; break
            imgs, lbs = imgs.to(DEVICE), lbs.to(DEVICE)
            did_cutmix = False
            if CFG.USE_CUTMIX and random.random() < CFG.CUTMIX_P:
                imgs, lbs, _ = _cutmix_batch(imgs, lbs, beta=CFG.CUTMIX_BETA)
                did_cutmix = True
            if adv_active and CFG.ADV_FRAC > 0 and not did_cutmix:
                n_adv = max(1, int(imgs.shape[0]*CFG.ADV_FRAC))
                idx = torch.randperm(imgs.shape[0], device=imgs.device)[:n_adv]
                with torch.enable_grad():
                    if CFG.ADV_MODE == "pgd":
                        adv_imgs = _pgd_perturb(model, imgs[idx], lbs[idx], CFG.ADV_EPS,
                                                 CFG.ADV_STEPS, self.loss_fn, cfg["gamma"])
                    else:
                        adv_imgs = _fgsm_perturb(model, imgs[idx], lbs[idx], CFG.ADV_EPS,
                                                  self.loss_fn, cfg["gamma"])
                imgs = imgs.clone(); imgs[idx] = adv_imgs.to(imgs.dtype)
            with autocast("cuda", dtype=DTYPE, enabled=AMP_OK):
                out = model(imgs, return_aux=True)
                logits, gate_w, aux, fused = out["logit"], out["gate"], out["aux"], out["fused"]
                loss = self.loss_fn(logits, lbs, cfg["gamma"])
                aux_loss = sum(F.binary_cross_entropy_with_logits(a, lbs.float()) for a in aux.values()) \
                           / max(len(aux), 1)
                ent = -(gate_w.clamp_min(1e-8) * gate_w.clamp_min(1e-8).log()).sum(-1).mean()
                loss = loss + CFG.AUX_LAM*aux_loss - CFG.GATE_ENTROPY_LAM*ent
                loss = loss + self.ewc.penalty(model, cfg["ewc_lam"])
                if supcon is not None:
                    loss = loss + CFG.CONTRASTIVE_LAM * supcon(fused, lbs)
                if CFG.USE_AE and hasattr(model, "ae") and not did_cutmix:
                    is_real = (lbs < 0.5)
                    loss = loss + CFG.AE_LAM * model.ae.ae_loss(imgs, is_real)
            opt.zero_grad(set_to_none=True)
            if USE_SCALER:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.trainable_params(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.trainable_params(), 1.0); opt.step()
            sched.step(); ema.update(model)
            if not did_cutmix:
                with torch.no_grad(): pr = torch.sigmoid(logits.detach()).cpu().tolist()
                for p,l,pa in zip(pr, lbs.cpu().tolist(), paths):
                    if 0.35 < p < 0.65: new_unc.append((pa, int(l), "uncertain"))
            step += 1
        del opt, scaler, dl; gc.collect()
        if aborted and step < CFG.MIN_BATCHES: return None
        return new_unc

    def _execute_round(self, items, rid, t_dead, ram_limit=None, tag_suffix=""):
        rng = random.Random(CFG.SEED + rid); cfg = self._cfg(rng)
        tag = f"ch_{rid}_{hashlib.md5(json.dumps(cfg).encode()).hexdigest()[:6]}"
        log(f"\nRound {rid}{tag_suffix}: '{tag}'  cfg={cfg}")
        model = self._build_challenger(cfg); ema = EMA(model)
        restored = self.reg.load_into(model, ema)
        # [FIX-D] load_into() can return True (weights kept) while ema.shadow
        # filtered down to {} on an arch change. Second guard reseeds EMA.
        if not restored or not ema.shadow:
            if restored and not ema.shadow:
                log("  EMA shadow empty after load (arch drift) -- refreshing")
            ema.refresh(model)
        prev_report = self.reg.champion_report() or {}
        champion_auc = prev_report.get("auc")
        new_unc = self._train(model, ema, cfg, items, t_dead,
                               champion_auc=champion_auc, ram_limit=ram_limit)
        if new_unc is None:
            del model, ema; gc.collect(); torch.cuda.empty_cache(); return None
        report = self.judge.evaluate(model, ema)
        log(f"  Judge AUC={report['auc']:.4f} acc={report['acc']:.4f} "
            f"worst_slice={report['worst_slice']:.4f} "
            f"slices={ {k:round(v,3) for k,v in list(report['slices'].items())[:5]} }")
        trainable = _trainable_state_dict(model)
        buffers = _all_buffers(model)
        result = {"tag": tag, "rid": rid, "cfg": cfg, "report": report,
                  "trainable": trainable, "buffers": buffers, "arch": model.arch,
                  "ema_shadow": ema.shadow, "ema_buffers": ema.buffers, "new_unc": new_unc}
        del model, ema; gc.collect(); torch.cuda.empty_cache()
        return result

    def run_round(self, items, rid, t_dead):
        result = self._execute_round(items, rid, t_dead)
        if result is None:
            return None
        tag, cfg, report = result["tag"], result["cfg"], result["report"]
        decision = self._commit_result(result)
        ledger_add({"round":rid,"tag":tag,"auc":report["auc"],
                    "worst_slice":report["worst_slice"],"decision":decision,"cfg":cfg})
        return report

    def _commit_result(self, result):
        tag, report = result["tag"], result["report"]
        decision = self._gate(report)
        if decision == "promote":
            model = DetectorModel(**result["arch"]).float().to(DEVICE)
            _shape_safe_load(model, result["trainable"])
            with torch.no_grad():
                for n, b in model.named_buffers():
                    if n in result["buffers"]:
                        b.copy_(result["buffers"][n].to(b.device, b.dtype))
            ema = EMA(model)
            ema.shadow  = {k: v.to(DEVICE) for k, v in result["ema_shadow"].items()}
            ema.buffers = {k: v.to(DEVICE) for k, v in result["ema_buffers"].items()}
            saved = self.reg.save_challenger(model, ema, report,
                                              self.reg.index["champion"], tag)
            if saved:
                self.reg.promote(tag, report)
                self.replay = (self.replay + result["new_unc"])[-CFG.REPLAY_CAP:]
                self._save_replay()
                self.ewc.consolidate(model, self.judge.train)
            else:
                log(f"  '{tag}' promotion skipped -- save failed, reject")
                decision = "reject"
            del model, ema; gc.collect(); torch.cuda.empty_cache()
        elif decision == "rollback":
            self.reg.rollback()
        else:
            log(f"  '{tag}' rejected (weights discarded)")
        return decision

    @staticmethod
    def _score(report):
        a = report["auc"]; w = report.get("worst_slice", a)
        if math.isnan(a): return float("nan")
        if w is None or math.isnan(w): w = a
        return (1-CFG.WORST_SLICE_WEIGHT)*a + CFG.WORST_SLICE_WEIGHT*w

    def _gate(self, report):
        s_new = self._score(report)
        if math.isnan(s_new): return "reject"
        prev = self.reg.champion_report()
        if prev is None or math.isnan(prev.get("auc", float("nan"))):
            return "promote"
        s_old = self._score(prev)
        current_auc = prev["auc"]
        dynamic_margin = max(0.0001, CFG.PROMOTE_MARGIN * ((1.0-current_auc)*100))
        beats = s_new >= s_old + dynamic_margin
        regressed = [(sname, round(prev["slices"][sname]-a,3))
                     for sname,a in report["slices"].items()
                     if sname in prev.get("slices",{})
                     and a < prev["slices"][sname] - CFG.REGRESSION_TOL]
        if beats and not regressed: return "promote"
        if regressed: log(f"  regression: {regressed}")
        if report["auc"] < current_auc - CFG.DRIFT_DROP: return "rollback"
        return "reject"

def ledger_add(row):
    try:
        d = (json.load(open(CFG.LEDGER)) if os.path.exists(CFG.LEDGER) else {"rounds": []})
        d["rounds"].append({**row, "ts":datetime.now().isoformat()})
        json.dump(d, open(CFG.LEDGER,"w"), indent=2)
    except Exception: pass

def _self_script_path():
    try:
        p = os.path.abspath(__file__)
        if os.path.exists(p) and p.endswith(".py"):
            return p
    except NameError:
        pass
    return None

def _worker_main(args_path, result_path):
    try:
        args = json.load(open(args_path))
        rid = args["rid"]; t_dead = args["t_dead"]
        tag_suffix = args.get("tag_suffix", "")
        ram_limit = args.get("ram_limit", CFG.RAM_LIMIT_GB_DUAL)
        log(f"[worker] starting rid={rid} suffix='{tag_suffix}'")
        items = discover(write_cache=False)
        judge = FrozenJudge(items, write_cache=False)
        registry = Registry()
        coach = EvolutionController(judge, registry, read_only=True)
        result = coach._execute_round(judge.train, rid, t_dead,
                                       ram_limit=ram_limit, tag_suffix=tag_suffix)
        if result is None:
            torch.save({"skipped": True}, result_path); log("[worker] skipped"); return
        torch.save(result, result_path)
        log(f"[worker] done: '{result['tag']}' AUC={result['report']['auc']:.4f}")
    except Exception:
        err = traceback.format_exc()
        log(f"[worker] FATAL:\n{err}")
        try:
            with open(result_path + ".failed", "w") as f: f.write(err)
        except Exception: pass
        sys.exit(1)

def _launch_gpu_worker(script_path, physical_gpu_id, rid, t_dead, tag_suffix, ram_limit):
    os.makedirs(CFG.WORKER_DIR, exist_ok=True)
    stem = f"{rid}{tag_suffix or '_a'}"
    args_path   = f"{CFG.WORKER_DIR}/args_{stem}.json"
    result_path = f"{CFG.WORKER_DIR}/result_{stem}.pth"
    log_path    = f"{CFG.WORKER_DIR}/log_{stem}.txt"
    for p in (args_path, result_path, result_path + ".failed"):
        if os.path.exists(p):
            try: os.remove(p)
            except Exception: pass
    json.dump({"rid": rid, "t_dead": t_dead, "tag_suffix": tag_suffix,
               "ram_limit": ram_limit}, open(args_path, "w"))
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
    logf = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, script_path, "--gpu-worker", args_path, "--result", result_path],
        env=env, stdout=logf, stderr=subprocess.STDOUT)
    return proc, result_path, log_path, logf

def run_double_round(coach, rid, t_dead, gpu_ids, script_path, physical_gpu_ids=None):
    if physical_gpu_ids is None:
        physical_gpu_ids = [str(g) for g in gpu_ids]
    per_worker_ram = CFG.RAM_LIMIT_GB_DUAL
    worker_deadline = min(t_dead, time.time() + CFG.ROUND_MAX_S)
    procs = []
    for i, gpu_id in enumerate(gpu_ids):
        r = rid + i
        suffix = f"_g{gpu_id}"
        proc, result_path, log_path, logf = _launch_gpu_worker(
            script_path, physical_gpu_ids[i], r, worker_deadline, suffix, per_worker_ram)
        procs.append((proc, result_path, log_path, logf, r, gpu_id))
        log(f"  launched worker GPU {gpu_id} round {r} -> {log_path}")
    results = []
    kill_at = worker_deadline + CFG.WORKER_TIMEOUT_SLOP
    for proc, result_path, log_path, logf, r, gpu_id in procs:
        remaining = max(1, kill_at - time.time())
        try: proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            log(f"  worker GPU {gpu_id} round {r} timeout -- kill")
            proc.kill()
            try: proc.wait(timeout=30)
            except Exception: pass
        try: logf.close()
        except Exception: pass
        if proc.returncode != 0:
            log(f"  worker GPU {gpu_id} round {r} exited {proc.returncode} "
                f"-- see {log_path}")
            continue
        if not os.path.exists(result_path):
            log(f"  worker GPU {gpu_id} round {r} no result"); continue
        try: result = torch.load(result_path, map_location="cpu", weights_only=False)
        except Exception as e:
            log(f"  worker result unreadable: {e}"); continue
        finally:
            try: os.remove(result_path)
            except Exception: pass
        if result.get("skipped"): continue
        results.append(result)
    reports = []
    for result in results:
        decision = coach._commit_result(result)
        ledger_add({"round": result["rid"], "tag": result["tag"], "auc": result["report"]["auc"],
                    "worst_slice": result["report"]["worst_slice"],
                    "decision": decision, "cfg": result["cfg"]})
        reports.append(result["report"])
    return reports

def main():
    t0 = time.time(); t_dead = t0 + CFG.WALL_H*3600
    log("=" * 66)
    log("  AXON SOTA v10.0 -- DINOv2 crash fixed, SupCon fixed, labels fixed")
    log("=" * 66)
    if AMP_OK:
        log(f"  GPU: {torch.cuda.get_device_name(0)}  dtype={DTYPE}  bf16={USE_BF16}  "
            f"visible_gpus={torch.cuda.device_count()}")

    hits = sorted(set(glob.glob(f"{CFG.INPUT}/**/registry/index.json", recursive=True)))
    if hits:
        def _rank(idx):
            try:
                d = json.load(open(idx)); lin = d.get("lineage", [])
                return (len(lin), max((r.get("auc",0) or 0) for r in lin) if lin else 0.0,
                        os.path.getmtime(idx))
            except Exception: return (-1,-1.0,0)
        ranked = sorted(hits, key=_rank, reverse=True); chosen = ranked[0]
        if len(ranked) > 1:
            log(f"{len(ranked)} AXON states -- choosing most evolved:")
            for h in ranked:
                r = _rank(h); log(f"     rounds={r[0]} best_auc={r[1]:.4f}  {h}")
        STATE_SRC = os.path.dirname(os.path.dirname(chosen))
        log(f"Prior AXON state at {STATE_SRC}")

        staged = ["AXON_LEDGER.json", "AXON_EWC.pth", "AXON_REPLAY.json",
                  "AXON_UNFREEZE_MODE.txt", "AXON_LABEL_LOGIC.txt",
                  f"AXON_ITEMS{CFG.CACHE_VER}.json",
                  f"AXON_ITEMS{CFG.CACHE_VER}_split.json",
                  f"judge_split{CFG.CACHE_VER}.json"]
        # [FIX-I] also stage prior-version caches so discover()'s fallback can fire.
        for v in CFG.FALLBACK_CACHE_VERS:
            staged += [f"AXON_ITEMS{v}.json", f"AXON_ITEMS{v}_split.json",
                       f"judge_split{v}.json"]
        for name in staged:
            src = os.path.join(STATE_SRC, name)
            if os.path.exists(src): shutil.copy(src, os.path.join(CFG.WORK, name))

        _restore_registry_selective(STATE_SRC)
        sc = os.path.join(STATE_SRC,"csv_img_cache")
        if os.path.isdir(sc):
            shutil.copytree(sc, f"{CFG.WORK}/csv_img_cache", dirs_exist_ok=True)
        log("Restored champion + ledger + EWC + caches")
        log(f"  CACHE_VER is now {CFG.CACHE_VER} and FALLBACK is empty: the label "
            f"logic changed, so EVERY older cache is discarded -- one clean "
            f"discovery rescan. Champion weights are preserved but were fitted "
            f"under the old labels -- watch the re-baseline AUC.")
    else:
        log("No prior state -- fresh run.")

    if os.environ.get("AXON_RESUME","0") == "1" and not os.path.exists(f"{CFG.REGISTRY}/index.json"):
        raise RuntimeError("AXON_RESUME=1 but no champion restored.")

    items = discover(); gc.collect(); torch.cuda.empty_cache()
    if len(items) < 100:
        log("Not enough labeled data."); return

    judge = FrozenJudge(items); registry = Registry()
    registry._prune_old_champions()
    if registry.index.get("champion") is not None:
        registry.rebaseline_champion(judge)
    coach = EvolutionController(judge, registry)

    if registry.index["champion"] is None:
        log("\nSeeding baseline...")
        seed = DetectorModel().float().to(DEVICE)
        log(f"  trainable params: {seed.n_trainable():,}")
        import_prior_brain(seed, judge)
        report = judge.evaluate(seed)
        if judge.val_ok and not math.isnan(report["auc"]):
            se = EMA(seed); se.refresh(seed)
            registry.save_challenger(seed, se, report, None, "seed_brain")
            registry.promote("seed_brain", report)
        else:
            log("  Baseline deferred.")
        del seed; gc.collect(); torch.cuda.empty_cache()
    else:
        cr = registry.champion_report() or {}; cv = cr.get("auc")
        cstr = f"{cv:.4f}" if isinstance(cv,(int,float)) and not math.isnan(cv) else "n/a"
        log(f"\nResuming champion '{registry.index['champion']}' AUC={cstr} "
            f"worst_slice={cr.get('worst_slice','n/a')}")

    num_gpus = torch.cuda.device_count() if AMP_OK else 0
    script_path = _self_script_path()
    dual_active = (num_gpus >= 2) and CFG.USE_DUAL_GPU and (script_path is not None)
    if CFG.USE_DUAL_GPU and num_gpus >= 2 and script_path is None:
        log("  Dual-GPU requested but no self-relaunch target -- single-GPU.")
    _cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    _physical_gpu_ids = ([x.strip() for x in _cvd.split(",") if x.strip()]
                         if _cvd else [str(i) for i in range(num_gpus)])
    gpu_ids = list(range(min(2, num_gpus))) if dual_active else []
    if dual_active and len(_physical_gpu_ids) < len(gpu_ids):
        log(f"  GPU id mapping short -- disabling dual-GPU."); dual_active = False; gpu_ids = []
    if dual_active:
        log(f"  Dual-GPU ACTIVE -- GPUs {gpu_ids}")

    ledger = json.load(open(CFG.LEDGER)) if os.path.exists(CFG.LEDGER) else {"rounds": []}
    rid = (ledger["rounds"][-1]["round"] + 1) if ledger["rounds"] else 0
    log(f"  Starting evolution at round {rid}")
    skips = 0
    while time.time() < t_dead - CFG.HEADROOM_S:
        free = free_disk_gb()
        if free >= 0 and free < CFG.MIN_FREE_GB:
            log(f"  Disk low ({free:.2f}GB) -- stop early."); break
        if dual_active:
            reports = run_double_round(coach, rid, t_dead, gpu_ids, script_path,
                                        physical_gpu_ids=[_physical_gpu_ids[g] for g in gpu_ids])
            rid += len(gpu_ids)
            if not reports:
                skips += 1
                if skips >= CFG.MAX_SKIPS: log(f"  {skips} skips -- stop."); break
                time.sleep(2); continue
            skips = 0
        else:
            result = coach.run_round(judge.train, rid, t_dead)
            if result is None:
                skips += 1
                if skips >= CFG.MAX_SKIPS: log(f"  {skips} skips -- stop."); break
                time.sleep(2); continue
            skips = 0
            rid += 1

    champ = registry.index.get("champion"); rep = registry.champion_report() or {}
    av = rep.get("auc")
    auc_str = f"{av:.4f}" if isinstance(av,(int,float)) and not math.isnan(av) else "n/a"
    log("\n" + "=" * 66)
    log(f"  RUN COMPLETE -- champion '{champ}' AUC={auc_str} "
        f"worst_slice={rep.get('worst_slice','n/a')}")
    log(f"  Lineage rounds: {len(registry.index.get('lineage',[]))}")
    log(f"  Elapsed: {(time.time()-t0)/3600:.2f}h")
    log("  Save a NEW VERSION of 'axon-state' from this Output.")
    log("=" * 66)

if __name__ == "__main__":
    if "--gpu-worker" in sys.argv:
        _i = sys.argv.index("--gpu-worker")
        _args_path = sys.argv[_i + 1]
        _rj = sys.argv.index("--result")
        _result_path = sys.argv[_rj + 1]
        _worker_main(_args_path, _result_path)
    else:
        try: main()
        except Exception:
            log("FATAL:"); traceback.print_exc()
