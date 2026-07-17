# PGMM Replication — Design Spec

**Paper:** Pose-Guided Fine-Grained Sign Language Video Generation (ECCV 2024)
**arXiv:** 2409.16709 · **Authors:** Shi, Hu, Shang, Feng, Liu, Feng
**Date:** 2026-07-17
**Status:** Approved design, pending implementation plan

---

## 1. Goal & Success Criteria

**Goal:** Replication — verify the paper's quantitative claims by independently reproducing its reported numbers.

**Primary success criterion:** Reproduce the LSA64 rows of Table 1 within tolerance, for both the TPSMM baseline and PGMM, using our own implementation.

| Method | L1↓ | SSIM↑ | LPIPS↓ | FVD↓ | TCD↓ |
|---|---|---|---|---|---|
| MRAA (cited, not retrained) | 0.01196 | 0.9324 | 0.01996 | 184.668 | 0.150 |
| TPSMM (retrained — our control) | 0.01342 | 0.9208 | 0.02261 | 182.785 | 0.130 |
| PGMM (retrained — the claim) | 0.01144 | 0.9395 | 0.01638 | 154.726 | 0.125 |

**Tolerance:** ±5% relative on L1 and LPIPS; ±0.005 absolute on SSIM. **FVD is exempt from the numeric tolerance** — it is notoriously implementation-sensitive (I3D checkpoint, resize, clip length each shift it), so we hold ourselves only to reproducing the *ordering and relative gap*, and we state that limitation explicitly in results rather than pretending to an absolute match.

**The real claim under test:** PGMM > TPSMM. Everything else is context.

**Secondary criterion:** Reproduce the ordering of Table 4 (ablations).

**A replication that shows the claim does NOT hold is a successful replication.** We are testing, not confirming. No result is massaged toward the paper's numbers.

## 2. Scope

**In scope (now):** LSA64. PGMM + TPSMM baseline + 5 ablations. Metrics: L1, SSIM, LPIPS, FVD, TCD.

**Deferred to a future spec cycle:** PHOENIX14T → unlocks WER. Access is requested on day 1 because latency is weeks, but the dataset's arrival does **not** expand this spec — it triggers its own brainstorm → spec → plan cycle (different resolution, different eval, a recognizer dependency). This spec stays scoped to a single implementation plan.

**Out of scope (YAGNI):**
- Retraining MRAA and DynaST. They are independently published methods; PGMM is built on TPSMM, so TPSMM is the only control that isolates the contribution. Their numbers are cited. This halves compute.
- WLASL-2000. Distributed as YouTube links; link rot makes the paper's 21,083-video split unreproducible. Any number we got would be incomparable.
- CSL-Daily. Access is requested on day 1 (asking is free), but approval is uncertain and often slow; the project plans as though it will never arrive. If it does, it gets its own spec cycle like PHOENIX14T.
- Any novel extension to the method. This is replication.

## 3. Constraints

**Compute: Kaggle, 2×T4.**
- 30 GPU-hours/week quota. Session cap ~9–12h (sources conflict; design assumes 9h).
- **Quota bills session wall-clock, not GPU-hours** → using both T4s in one session doubles work at no quota cost. This drives the DataParallel decision.
- T4 = 16GB, Turing. Has fp16 tensor cores. **No bf16. No flash-attn** (needs Ampere+); memory-efficient SDPA backend works.
- ~4 CPU cores → data loading is a real bottleneck; drives the frame-pre-extraction decision.
- `/kaggle/working` ~20GB, wiped between sessions → checkpoint chaining via Kaggle Dataset versions is mandatory, not optional.

**Paper's own setup for reference:** 1× RTX 3090, 128×128, 100 epochs, Adam lr 2e-4 ÷10 at epochs 60 and 90, 5 source/driving pairs sampled per video.

**Upstream availability:**
- Official PGMM repo (`shitongkai/PGMM`): **empty placeholder**, README says "Under Construction", 3 commits, no code/weights. Full reimplementation required.
- TPSMM (`yoyo-nb/Thin-Plate-Spline-Motion-Model`): **full training + inference code, MIT**. This is the foundation.
- HRNet-W32 COCO-WholeBody: pretrained weights via MMPose.
- LSA64: freely downloadable (~1.9GB), non-commercial, attribution required.

## 4. Approach

**Fork TPSMM; add PGMM alongside it.**

Rationale: the baseline and PGMM then share the data pipeline, `L_r`, and the eval harness *exactly*. The only difference between the R0 and R1 runs is the paper's contribution — so the measured delta is the claim, uncontaminated. Ablations reduce to config flags.

Rejected: clean-room rewrite. If our numbers diverged we could not tell whether PGMM is wrong or our TPSMM is wrong — which destroys the point of a replication.

**Discipline:** `third_party/tpsmm/` is a pinned fork and its **numerics are not touched**. New code lives in `pgmm/`. Any unavoidable upstream edit gets its own commit and an entry in the decision log.

## 5. Architecture

```
paper_pgmm/
├── third_party/tpsmm/        # pinned fork (MIT), numerics frozen
├── pgmm/
│   ├── modules/
│   │   ├── cmm.py            # Coarse Motion Module
│   │   ├── pfm.py            # Pose Fusion Module
│   │   ├── posenet.py        # heatmaps -> fine-grained pose features
│   │   ├── pose_estimator.py # frozen HRNet-W32 wrapper, 36 heatmaps
│   │   └── generator.py      # (CMM, PFM, Conv, Upsample) x N + sigmoid
│   ├── losses/
│   │   ├── pose_distance.py  # L_pd (soft-argmax)
│   │   └── align.py          # L_align
│   ├── metrics/
│   │   ├── tcd.py            # the paper's new metric
│   │   ├── fvd.py            # I3D-based
│   │   └── basic.py          # L1, SSIM, LPIPS
│   ├── data/lsa64.py
│   └── config/
│       ├── lsa64-tpsmm.yaml      # R0 baseline
│       ├── lsa64-pgmm.yaml       # R1 full
│       └── ablations/            # R2..R5
├── kaggle/
│   ├── train.ipynb
│   └── chain.py              # checkpoint push/pull across sessions
├── docs/
│   ├── decisions.md          # every paper ambiguity, our choice, evidence
│   └── results.md            # our numbers vs paper's, incl. failures
└── tests/
```

### Module specs (from the paper)

**Decomposition.** PGMM splits a sign-language image into appearance `X_A`, coarse motion `X_M`, fine detail `X_D`. Mapping: `Î_d = PGMM(I_s, I_d)`.

**CMM** — transfers coarse structural motion via flow warping without altering appearance:
```
X_out = [X_in, (1 − M(X_m)) ⊙ T(F(X_m), X_in)]
```
`F` = Conv-Flow (motion features → optical flow), `M` = Conv-Mask (→ occlusion map), `T` = warp (`grid_sample`). **Output channels double (2·C_in)**; the following Conv Module must halve them back. TPSMM's `DenseMotionNetwork` already yields flow + multi-resolution occlusion maps — reuse, don't reinvent.

**PFM** — cross-attention fusing pose (Query) with RGB (Key/Value):
```
X_out = X_in + softmax(Q(X_p)·K(X_in)ᵀ / √C_in)·V(X_in)
```
Each of Q/K/V = 3×3 conv + InstanceNorm + ReLU.

**PoseNet** — interpolation + 3 upsampling blocks (3×3 transposed conv stride 2 + IN + ReLU). Resizes heatmaps to the largest image-feature scale, then progressively extracts fine-grained pose features.

**Pose estimator** — HRNet-W32 pretrained on COCO-WholeBody, **frozen**, emitting 36 heatmaps.

**Generator** — "multiple sets of a Coarse Motion Module, a Pose Fusion Module, a Convolution Module, and an Up-sampling module, and a sigmoid at the end." Conv Module = 2 ResBlocks. Upsample = interpolate + 3×3 conv + IN + ReLU.

**Motion backbone (inherited)** — TPSMM: ResNet-18, 100 keypoints in 20 TPS groups (K=20).

### Losses

```
L = L_r + L_pd + L_align
L_pd    = (1/K) Σᵢ |kᵢ(P_e(Î_d)) − kᵢ(H)|         , K = 36
L_align = |Y_s − Y_d|
          Y_sᵢ = Gᵢ(X_s, X_m, X_p)   (source, with motion)
          Y_dᵢ = Gᵢ(X_d, ∅, X_p)     (driving, no CMM)
```
`L_r` = TPSMM's existing perceptual + equivariance + background + warp losses, weights inherited verbatim.

**Cost note:** `L_align` requires a **second generator forward pass** per step. Budget for it.

### TCD (the paper's new metric)

```
D   = |Î_t − ½(I_{t−1} + I_{t+1})|
TCD = (1/(N−2)) Σ_t (1/(H·W)) Σ_h Σ_w [D(h,w) > T]        , T = 0.5
```
Fraction of pixels whose deviation from the temporal midpoint exceeds T, averaged over frames. Note TCD is *not* a pure reconstruction error — fast-moving hands deviate from the temporal midpoint even under perfect reconstruction, which is why values sit around 0.13 rather than near zero.

## 6. Paper Ambiguities → Decisions

These are the project's real risk. Each gets an entry in `docs/decisions.md` with the evidence behind the choice. **This section is the reason the project could fail; it is not boilerplate.**

| # | Ambiguity | Decision |
|---|---|---|
| **D1** | **Which 36 of COCO-WholeBody's 133 keypoints?** Paper says only "We selected 36 heatmaps of pose points." Not derivable: hands alone are 42 (21+21), upper body 11, so 36 excludes even a full pair of hands. | **Unresolved — the single biggest unknown.** Ask the authors (issue/email). Meanwhile: define 2–3 candidate sets, pick empirically with a short run at M3, document the choice and its sensitivity. If no set reproduces the numbers, that finding is itself a legitimate replication result. |
| **D2** | **`L_pd` as written has no gradient.** `kᵢ` takes the argmax coordinate of a heatmap; argmax is not differentiable. | Use **soft-argmax** (spatial softmax / integral pose regression). This is the only reading under which the loss can train. Record as a deviation. |
| **D3** | **TCD's `T=0.5` applies to what range?** Value range never stated; per-channel vs. aggregated never stated. | **Calibrate against the paper's own number:** TPSMM must yield TCD = 0.130 on LSA64. Enumerate candidate readings ([0,1] mean-channel, [0,1] summed-channel, [0,255], …) and adopt the one reproducing 0.130 at M1. The paper's baseline row becomes the metric's calibration fixture. |
| **D4** | Generator block count unspecified ("multiple sets"). | Follow TPSMM's decoder depth (`num_down_blocks=3`; 128→64→32→16 and back). Deviation only if M3 misses. |
| **D5** | **PFM scale placement — possible hard blocker.** Global attention is O((H·W)²). At batch 16 fp32 the attention matrix alone is ~67MB @32², ~1GB @64², **~17GB @128²** — exceeding the T4's 16GB. If PFM truly sits at every decoder scale, it cannot run as literally described. | Use memory-efficient `F.scaled_dot_product_attention` (Turing-compatible backend). If still infeasible, restrict PFM to lower scales and **ablate the restriction**. Resolve at M2, before quota is spent. |
| **D6** | Weights of `L_pd`, `L_align` unspecified. `L = L_r + L_pd + L_align` implies 1.0, which is suspicious given TPSMM's perceptual weight is 10. | Start at 1.0. Log per-term magnitudes at M2; if a term is drowned out or dominant, tune and document. |
| **D7** | `L_r` sub-loss weights unspecified. | Moot — inherited verbatim from the TPSMM fork. |
| **D8** | FVD is implementation-sensitive (I3D checkpoint, resize, clip length all shift it). | Pin one widely-used implementation + I3D checkpoint, record the choice, and report FVD as relative rather than absolute. |

## 7. Data Pipeline

**LSA64:** 3,200 videos, 10 signers, 64 signs. Split 2,800 train / 400 test.

**Pre-extract frames to 128×128** (crop-and-resize per the paper) and ship as a versioned Kaggle Dataset. Reason: ~4 CPU cores cannot decode video fast enough to keep 2 T4s fed; on-the-fly decoding would make us CPU-bound. Preprocessing is a one-time cost.

Training samples 5 source/driving pairs per video per epoch, per the paper.

**Eval:** TPSMM-style reconstruction protocol on the 400-video test set.

## 8. Kaggle Training Harness

**Checkpoint chaining** (mandatory — a 100-epoch run cannot fit one session):
1. Checkpoint every N steps to `/kaggle/working`.
2. At session end, push as a new version of a Kaggle Dataset via the API.
3. Next session attaches the latest version and resumes.

Resume must restore model + optimizer + LR schedule + epoch/step + RNG state. **A resume that silently restarts the LR schedule would quietly corrupt every downstream number** — so resume correctness is verified at M0 by checking that a resumed run's loss curve matches an uninterrupted one.

**Both GPUs:** DataParallel (TPSMM already supports `device_ids`). DDP is faster but hostile inside notebooks; the quota is billed per session, so DataParallel already captures the main win.

**Precision:** fp32 first. AMP fp16 is risky around `grid_sample`/flow warping. Revisit only after M1, and only with a numerics check.

## 9. Experiment Matrix (LSA64)

| Run | Config | Purpose | Target |
|---|---|---|---|
| R0 | TPSMM baseline | control + TCD calibration | L1 .01342, SSIM .9208, LPIPS .02261, TCD .130 |
| R1 | PGMM full | **the claim** | L1 .01144, SSIM .9395, LPIPS .01638, TCD .125 |
| R2 | + pose (direct feature addition) | ablation (Table 4) | ordering |
| R3 | + PFM | ablation | ordering |
| R4 | + PFM, L_pd | ablation | ordering |
| R5 | + PFM, L_align | ablation | ordering |

Note Table 4 has no CMM-only row — CMM largely restructures warping TPSMM already does, so the baseline implicitly covers it.

## 10. Milestones

| | Milestone | Exit criterion |
|---|---|---|
| M0 | Harness | LSA64 preprocessed + published; TPSMM trains and **resumes across 2 sessions with a loss curve matching an uninterrupted run**. Real step-time measured → timeline re-estimated. |
| **M1** | **Baseline gate** | **TPSMM reproduces its own Table 1 row within tolerance, and TCD calibrates to 0.130 (D3 resolved).** |
| M2 | Modules | CMM/PFM/PoseNet/losses implemented; shape + gradient unit tests pass; overfits a single video; D5 resolved; loss magnitudes logged (D6). |
| M3 | The claim | R1 complete, compared against Table 1. D1 resolved empirically. |
| M4 | Ablations | R2–R5 complete, compared against Table 4. |
| M5 | Expansion | PHOENIX14T (gated on access) → WER via CorrNet. |

**M1 is a hard gate.** If TPSMM does not reproduce its published row, we stop and investigate rather than proceeding to PGMM. Rationale: PGMM's numbers are only meaningful *relative to* a trustworthy baseline. A PGMM number sitting on an unvalidated baseline measures nothing, and would waste weeks of quota producing an uninterpretable result.

## 11. Timeline

Each run ≈ 10–20 GPU-hours (estimate; L_align's second forward pass and the frozen HRNet forwards inflate PGMM runs above baseline). Six runs ≈ 60–120 hours. At 30h/week that is **2–4 weeks of pure training**, plus debugging — which historically exceeds training.

**Realistic: 8–12 weeks.**

This rests on an unmeasured step-time estimate. **M0 measures it and the timeline is re-derived there.** Treat the above as an order-of-magnitude claim, not a commitment.

## 12. Day-1 Parallel Actions

Long-latency items, started immediately because they block nothing else:
1. Submit PHOENIX14T access request.
2. Submit CSL-Daily access request (may never be granted; costs nothing to ask).
3. Download LSA64.
4. Open an issue on `shitongkai/PGMM` asking about code release **and the D1 keypoint selection**. If the authors release code mid-project, the plan changes substantially — better to learn that in week 1 than week 8.

## 13. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **D1 (36 keypoints) never resolved** | PGMM may be unreproducible as specified | Empirical search over candidate sets; ask authors; a negative result is publishable-grade information and gets reported as such |
| **D5: PFM doesn't fit in 16GB** | Blocks PGMM entirely | Memory-efficient SDPA; fall back to restricted scales + ablate. Resolved at M2, before heavy spend |
| M1 fails — TPSMM won't reproduce | Blocks everything downstream | The gate exists precisely to catch this early rather than after PGMM runs |
| Kaggle quota/session limits tighten | Timeline slips | Checkpoint chaining makes the work interruption-tolerant by construction |
| FVD mismatch vs. paper | Weakens one metric | Pin implementation; report relative gap; do not chase the absolute number |
| Authors release code mid-project | Plan invalidated (favorably) | Ask on day 1; re-plan promptly if so |

## 14. Reporting

`docs/results.md` records our numbers against the paper's, **including runs that fail to reproduce**. Every deviation from the paper is traceable to a numbered decision in `docs/decisions.md`. The output of this project is an honest replication report, not a matching table.
