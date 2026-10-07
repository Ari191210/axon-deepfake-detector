# Changelog

No git history exists for this project — both scripts carry their own version history inline in their docstrings/comments. This changelog is reconstructed from that inline documentation.

## PHANTOM v3.1 (`src/phantom_v3.py`)
Patch notes, fixes on top of v3.0 (each was a run-breaking or eval-corrupting bug):

- **[P-1] Checkpoint bug (critical):** Stage-2 checkpoints only saved parameters with `requires_grad=True`. Stage 2 freezes `proj`, so the final checkpoint was saved *without* the projection weights — honest evaluation and all inference then ran with a random projection. Checkpoints now persist every non-backbone parameter plus trainable backbone parameters.
- **[P-2] Multi-scale bug (critical):** the multi-scale projection layer was lazily created inside `forward()`, after the optimizer/EMA were built — so it was never trained, never in the EMA, recreated with fresh random weights on every DataParallel replica each step, and missing at checkpoint load. Rewritten to use a properly registered module built in `__init__`, fed by `timm`'s `forward_intermediates` in a single DataParallel-safe backbone pass.
- **[P-3] Data leakage (critical):** hard-negative mining mined from validation items and added them to the training pool, training directly on the held-out generators and corrupting the cross-generator AUC. Mining now draws only from a train-only sample.
- **[P-4] OOM retry bug:** on out-of-memory, the retry rebuilt the dataloader but the training loop kept iterating the *old* iterator at the old batch size, so the retry never actually took effect. Rewritten with an explicit iterator that is actually replaced on OOM.
- **[P-5] EWC loader bug:** AXON's EWC file stores `{"fisher": {...}, "optima": {...}}`; the v3.0 loader iterated the top-level dict directly and crashed. The loader now parses both AXON's format and a flat format, and anchors the penalty at AXON's stored optima when present.
- **[P-6] Frequency branch was a no-op:** the "DCT" branch was just an 8x8 average-pool — a 64-pixel thumbnail with no frequency content at all. Replaced with a real FFT log-magnitude branch, computed in fp32 outside autocast (half-precision FFT is inaccurate).
- **[P-7] AXON path mismatch:** champion auto-detection looked only in a `champions/` folder; AXON's registry actually lives in `registry/`. Both are checked now.

Everything else (two-stage training, cross-generator split with a 2-class guarantee, leakage hashing, bootstrap CI, calibration, robustness curves, abstention, resume) is unchanged from v3.0.

## AXON v10 (`src/axon_sota_v10.py`)
Fixes on top of v9.0, each reproduced in an isolated CPU harness before being written:

- **[FIX-A, blocking]** `DINO_FORCE_224` crashed the run in round 0 — the DINOv2 backbone was built at its native 518px despite the config saying 224px, and timm's strict image-size check raised an assertion on the very first forward pass. Fixed by constructing the model at 224px so timm resamples the position embedding on load.
- **[FIX-B, critical/silent]** `SupConLoss` treated every pair in a batch as a positive pair, because a label tensor was accidentally 1-D, making an equality comparison a no-op that produced an all-true mask. Fixed by reshaping labels before the comparison.
- **[FIX-C, high/silent]** `SOURCE_OVERRIDES` (a last-resort dataset-labeling rule) was substring-matching full paths and firing *before* the proper leaf-folder label check, mislabeling any dataset whose name happened to contain an override keyword. Overrides are now consulted only as an actual last resort.
- **[FIX-D]** the EMA (exponential moving average of weights) could die permanently after an architecture change; fixed to refresh when the shadow state is missing or empty.
- **[FIX-E]** the EVA-02 backbone was being run twice per forward pass (once for pooled features, once for MIL tokens); reduced to one pass that supplies both.
- **[FIX-F]** a cross-source deduplication step (`DEDUP_REALS_XSOURCE`) was a guaranteed no-op; replaced with a real perceptual 16x16 grayscale average-hash (off by default).
- **[FIX-G]** forced EfficientNetV2-M to run at 224px (its native input size is 384px).
- **[FIX-H]** removed a duplicated self-test call in `__init__`.
- **[FIX-I]** fixed a cache-versioning/staging bug for fallback cache versions.
- **[FIX-J]** parameter-group membership now uses an `id()`-keyed set.
- **[FIX-K]** an unfreeze-transition check now honors the `read_only` flag.
