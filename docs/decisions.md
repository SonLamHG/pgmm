# Decision Log

Every deviation from the paper, and every choice the paper left open, is
recorded here. A result that cannot be traced to a decision is not reportable.

| # | Ambiguity | Decision | Status |
|---|---|---|---|
| D1 | Which 36 of COCO-WholeBody's 133 keypoints | Unresolved; ask authors, resolve empirically at M3 | open |
| D2 | `L_pd` argmax is non-differentiable | Use soft-argmax | decided |
| D3 | TCD value range / channel reduction for `T=0.5` | Calibrate against TPSMM=0.130 at M1 | open |
| D4 | Generator block count | Follow TPSMM `num_down_blocks=3` | decided |
| D5 | PFM attention may exceed 16GB | Memory-efficient SDPA; restrict scales if needed | open |
| D6 | `L_pd`/`L_align` weights | Start 1.0, log magnitudes at M2 | decided |
| D7 | `L_r` sub-loss weights | Inherited verbatim: perceptual [10]x5, equivariance 10, warp 10, bg 10 | resolved |
| D17 | TPSMM silently 80/20-splits when `train/` is absent, ignoring D9 | Materialise the split on disk (`pgmm/data/layout.py`) | resolved |
| D18 | Paper's "five pairs per video" vs TPSMM's `num_repeats` 50–150 | `num_repeats: 5` — the semantics map exactly | resolved |
| D19 | LSA64 signers wear fluorescent gloves; TPSMM defaults apply colour jitter | Keep jitter (D7 inheritance); revisit if M1 misses | decided |
| D20 | How does the prepared data reach the training kernel? | `kernel_sources` mounts the preprocess kernel's auto-zipped output; extract in place. **Zero transfer.** | resolved |
| D21 | TPSMM's own checkpoint restores model/optimizer/LR but **not RNG** | Accept: resumes are correct but not bit-reproducible. M0's exit criterion revised. | decided |
| D8 | FVD implementation sensitivity | Pin one I3D; report relative only | open |
| D9 | LSA64 2800/400 split undefined | Seeded random default; signer-held-out alternative | open |
| D10 | LPIPS backbone (alex vs vgg) undefined | `lpips` pkg, `net='alex'` | decided |
| D11 | SSIM implementation undefined | `skimage`, `data_range=1.0`, `channel_axis=-1` | decided |
| D12 | Real step time / timeline unmeasured | Measure at M0, re-derive the estimate | open |
| D13 | LSA64 filename convention unverified | `NNN_NNN_NNN.mp4` = sign_signer_repetition — **confirmed** against the real archive | resolved |
| D14 | Paper does not say whether LSA64 raw or cut is used | Use **cut** (`justinvo277/lsa64-dataset`); retry raw if M1 misses | decided |
| D15 | Kaggle legacy API key shadows OAuth and cannot push kernels | Disable `~/.kaggle/kaggle.json` so the CLI resolves to OAuth | decided |
| D16 | A kernel has no Kaggle credentials, so it cannot push its own checkpoint | **Local orchestration** — the kernel never pushes; the workstation does | resolved |

## D1 — which 36 keypoints

The paper says only: "We selected 36 heatmaps of pose points." It never says
which 36 of COCO-WholeBody's 133. The count is not derivable by any obvious
reading:

| Group | Count |
|---|---|
| body | 17 |
| feet | 6 |
| face | 68 |
| left hand | 21 |
| right hand | 21 |

Two hands alone are 42, so 36 excludes even a full pair of hands — which is
surprising for sign language, where the hands carry most of the signal. Upper
body (11) + hands (42) = 53. No natural subset lands on 36.

**Status: the single biggest unknown in the project.** Resolution: ask the
authors; meanwhile define candidate sets and pick empirically at M3. If no
candidate reproduces the paper's numbers, that is itself a legitimate
replication finding and gets reported as one.

## D3 — TCD threshold semantics

`T = 0.5` is given, but neither the image value range nor the channel reduction
is. `[0,1]` vs `[0,255]` changes the answer by orders of magnitude; mean vs max
vs sum over channels changes it too.

**Resolution by calibration:** the paper publishes TPSMM's TCD on LSA64 =
0.130. We score our reproduced baseline under every candidate reading and adopt
the one that lands on the published value. This calibrates the *metric's
definition* against the paper. It does not tune the *model* — that distinction
is load-bearing and must not blur.

Candidates implemented in `pgmm/metrics/tcd.py`: `UNIT_MEAN`, `UNIT_MAX`,
`UNIT_SUM`, `BYTE_MEAN`.

## D9 — LSA64 split

The paper reports 2800 train / 400 test but never defines the partition.
LSA64 is 10 signers x 64 signs x 5 repetitions = 3200. A 2800/400 split is
87.5%/12.5%, which factors cleanly against neither 10 signers (would give
2560/640) nor 5 repetitions (also 2560/640). So the split is almost certainly a
plain random 87.5/12.5, not a structured hold-out.

**Decision:** default `split="random"`, seed 0. Keep `split="signer"`
selectable to test sensitivity.

**If M1 misses its target, re-examine this first.** A random split lets the same
signer appear in both train and test, which is *easier* than a held-out split
and would inflate all our numbers relative to the paper. If we come out better
than the paper, suspect this before anything else.

## D13 — LSA64 filename convention (RESOLVED)

`pgmm/data/lsa64_prepare.py` parses `NNN_NNN_NNN.mp4` as
`<sign>_<signer>_<repetition>`. Originally assumed from LSA64's documentation
with nobody having opened the archive. **Confirmed** on 2026-07-17 against the
real files, via `kaggle datasets files` — no download needed:

```
lsa64_dataset/LSA64/001/001_001_001.mp4
lsa64_dataset/LSA64/001/001_001_002.mp4
...
lsa64_dataset/LSA64/001/001_001_005.mp4
lsa64_dataset/LSA64/001/001_002_001.mp4
```

Two independent confirmations in that listing:

1. The **third** field cycles 1→5 and then the second increments — matching the
   5 repetitions per (sign, signer). So field 3 = repetition.
2. `justinvo277` groups files into directories named by the **first** field
   (`001/`), and LSA64 has 64 signs. So field 1 = sign, leaving field 2 =
   signer (10 values).

This matters because D9's split reads the signer field out of the filename; a
wrong convention would have produced a split that was not the one we documented,
silently.

**Fully confirmed** on 2026-07-17 by enumerating every file in
`justinvo277/lsa64-dataset` through the Kaggle API (still no download):

```
mp4 files    : 3200      <- exactly the paper's count
unparseable  : 0
signs        : 1..64   | distinct 64
signers      : 1..10   | distinct 10
repetitions  : 1..5    | distinct 5
```

64 x 10 x 5 = 3200 closes it. The convention, the clip count, and every field
range now match both the paper and `pgmm/data/lsa64_prepare.py`. D13 is settled.

## D14 — LSA64 raw vs cut

The paper says "LSA64" and gives 3200 videos, but never says which distribution.
Both exist and differ: raw (1.9GB) keeps the idle frames before and after each
sign; cut (1.5GB) is temporally segmented to the sign itself.

**Decision: cut**, via `justinvo277/lsa64-dataset` (1,505,037,238 bytes, matching
the official cut size).

Reasoning: idle frames are nearly static, and static frames make both L1 and TCD
look good for free — a model that does nothing scores well on them. The cut
version removes that inflation, so it is the more honest choice and the harder
target. If M1 misses, `mathineni/lsa64dataset` (1,899,638,618 bytes ≈ official
raw) is the alternative to try, and the direction of the miss is diagnostic: if
our numbers come out *better* than the paper's, idle frames are a prime suspect.

Both are third-party re-uploads rather than the authors' own distribution. Sizes
match the official figures, which is reassuring but not proof of bit-identity.

## D15 — Kaggle auth: legacy key vs OAuth

Kaggle now has two auth schemes, and the CLI silently prefers the worse one.

| `~/.kaggle/kaggle.json` present | `auth_method` | `datasets list/files` | `kernels push` |
|---|---|---|---|
| yes | `LEGACY_API_KEY` | works | **fails** |
| no (OAuth only) | `OAUTH` | works | works |

With the legacy key present, `kernels push` fails demanding a `KAGGLE_API_TOKEN`
— even though `kaggle auth login` has already cached a valid OAuth token in
`~/.kaggle/credentials.json`. The legacy key is not a fallback; it is an
override that disables kernel operations entirely.

**Decision:** `~/.kaggle/kaggle.json` renamed to `kaggle.json.legacy-disabled`
(2026-07-17). Reversible by renaming back. A session backup also sits at
`/tmp/kaggle.json.session-backup`, and the user keeps a separate legacy key at
`d:\Admin\kaggle.json` — note that one carries a **different** key for the same
account (`snlmhong`), so it is not a copy of the disabled file.

**Why not just set `KAGGLE_API_TOKEN` from credentials.json?** That works, and
was used for the first push, but it bypasses the refresh flow: the access token
expires in ~3 hours. For a project measured in weeks, and for training runs that
must survive unattended across session boundaries, an auth that dies mid-run is
not viable. Native OAuth refreshes itself.

**Consequence to watch:** `datasets create` / `datasets version` are the
mechanism behind checkpoint chaining. They must be confirmed to work under OAuth
before any long training run is started — an auth failure at the *end* of a 9h
session would lose the whole session's work.

## D16 — how does a kernel push its own checkpoint back?

Checkpoint chaining is a loop with two halves:

1. **read** — the next session attaches the dataset and resumes from it
2. **write** — the dying session publishes its checkpoint as a new version

Half 2 is the problem. `kaggle_harness/chain.py` shells out to the `kaggle` CLI,
but **a Kaggle kernel has no Kaggle credentials by default**. The CLI is
installed; there is nothing for it to authenticate with. So `push_checkpoint`
as written cannot run where it was designed to run.

This blocks any run longer than one session — which is every real run.

Measured by the chainprobe kernel (2026-07-17), inside a real kernel:

```
KAGGLE_USERNAME set: False
KAGGLE_KEY set     : False
~/.kaggle exists   : False
```

### Kaggle Secrets: tried, does not work for an API-driven workflow

The first choice was a Kaggle Secret holding a legacy API key. It fails:

```
(a) SECRET NOT READABLE: ConnectionError Connection error trying to communicate with service.
```

Twice — chainprobe v2 and v3 — with the secret **visibly attached** in the
notebook editor (screenshot confirms a ticked `KAGGLE_KEY` on a version-2
notebook).

Reading of the evidence: the tick in the editor's Secrets panel describes the
**draft**. A version created by `kaggle kernels push` is a separate entity and
does not inherit that binding. Consistent with the `kaggle` PyPI package
containing no reference to secrets anywhere — the API has no way to express the
attachment, so an API-pushed version has none.

Not worth further debugging: two failed probes already cost more than the
alternative, and the alternative uses only operations already proven to work.

### Chosen: local orchestration

The kernel never authenticates. The workstation drives the loop:

1. push the training kernel (API, OAuth) — proven
2. kernel trains, writes its checkpoint to `/kaggle/working` — no credentials needed
3. fetch the kernel's output locally (`kernels output`) — proven
4. push the checkpoint as a dataset version (`datasets version`) — proven
5. push the next kernel; it attaches the dataset and resumes — proven (chainprobe)

Every step is a verified-working operation. Nothing depends on a UI action that
an API push can silently drop.

**Cost:** each checkpoint round-trips through the workstation's connection. A
TPSMM checkpoint is estimated at 500MB–1GB (weights plus two Adam moments) —
**unmeasured**; measure it at the first real checkpoint and record it here. At
~10 sessions that is roughly 10–20GB of traffic. Preprocessing round-trips once
(~1–2GB) and then never again.

**Consequence for `kaggle_harness/chain.py`:** `push_checkpoint` now runs on the
workstation, not in the kernel. Its env-var requirement is satisfied by OAuth
locally. The kernel side only needs `latest_checkpoint`, which is proven against
the real mount path.

**Rejected: private dataset holding the key.** It would save the bandwidth, but
it puts a live credential in a plain file to do so. Not worth it — especially
after this account's key already leaked into a chat transcript once.

### Verified working (chainprobe, 2026-07-17)

The read half of chaining is fully proven, including our own entry point:

```
PRIVATE DATASET MOUNTS AND READS -- chaining viable
latest_checkpoint -> /kaggle/input/datasets/snlmhong/pgmm-ckpt/ckpt_step000042.pt
CHAIN ENTRY POINT OK
```

Note the local API returns **403** on this account's own private dataset
(`datasets files`, `datasets download`) while `datasets list --mine` sees it and
`datasets version` writes to it. That 403 is an API-surface quirk and is
**irrelevant to chaining**: kernels mount datasets rather than fetching them.
Do not let it trigger a false alarm later.

## D17 — TPSMM silently overrides our split

`FramesDataset.__init__` checks for `root_dir/train`; finding none it does this:

```python
print("Use random train-test split.")
train_videos, test_videos = train_test_split(self.videos, random_state=seed, test_size=0.2)
```

On LSA64 that is **2560/640**, not the paper's **2800/400** — and it discards D9
entirely in favour of an undocumented partition. No exception, no warning: one
line of stdout, and every number downstream is quietly measured against the
wrong test set.

`pgmm/data/split_clips` would have been dead code and nobody would have noticed.

**Decision:** `pgmm/data/layout.py::materialise_split` writes the D9 split to
disk as `train/` and `test/` before training sees it, and asserts the 2800/400
counts. Caught by reading the vendored source rather than trusting the config.

## D18 — "five pairs per video" == `num_repeats: 5`

The paper: *"Five pairs of source and driving images randomly selected per
video."* TPSMM's configs use `num_repeats` of 50–150. A 30x compute difference
rides on the reading.

The semantics settle it:

- `DatasetRepeater.__len__` returns `num_repeats * len(dataset)` — the epoch is
  the video list repeated `num_repeats` times.
- `FramesDataset.__getitem__` does `frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))`
  — **exactly one (source, driving) pair per item.**

So `num_repeats` *is* pairs-per-video-per-epoch, and the paper's sentence maps
onto it precisely. **`num_repeats: 5`.**

Consequence: an epoch is 2800 x 5 = 14,000 samples; at batch 28 that is 500
steps/epoch and **50,000 steps** for the full 100 epochs. Far smaller than the
spec's 8–12 week estimate assumed. D12 will measure whether that holds.

## D19 — colour jitter vs LSA64's fluorescent gloves

Every TPSMM config applies `jitter_param` (brightness/contrast/saturation/hue
0.1). D7 says inherit TPSMM's defaults, so it stays on.

But LSA64 is unusual: its signers **wear fluorescent-coloured gloves**,
deliberately, to make the hands trivially separable. Hue and saturation jitter
attacks exactly that signal. The paper says nothing either way.

**Decision:** keep jitter, on D7 inheritance grounds. If M1 misses, this is a
cheap thing to flip and worth trying early — but changing it without evidence
would be tuning toward the paper's number, which this project does not do.

## D20 — getting the prepared data to the training kernel

**Resolution: `kernel_sources`. Nothing transfers through the workstation.**

Kaggle bundles a notebook's output into a single `_output_.zip` on its own.
`kernel_sources: ["snlmhong/pgmm-preprocess"]` mounts it in the training kernel:

```
/kaggle/input/notebooks/snlmhong/pgmm-preprocess/_output_.zip   1,343,316,532B
```

The training kernel extracts that once per session and trains. This supersedes
D16's local round-trip for the *data*; D16's core requirement — that a kernel
never needs credentials — still holds, and still governs checkpoints.

### Two wrong conclusions on the way here, recorded so they are not re-derived

**"`kernels output` is impossible for 264k files."** Wrong. It was called stuck
after six minutes on the basis of `1 file, 0 bytes` on disk. That file was
`_output_.zip` being written. It finished at **1.25 GiB in ~14 minutes**, and
`zipfile.testzip()` reports no corruption: 268,034 entries, 264,831 JPEGs —
exactly the count the kernel logged. The download works; it was judged too early.

**"Kaggle's 500-file output cap silently dropped our data."** Wrong. The cap is
real and widely reported ([1](https://www.kaggle.com/product-feedback/181143),
[2](https://github.com/Kaggle/kaggle-api/issues/665)), but it does not bite here
because Kaggle's auto-zip makes the output *one file*. Nothing was dropped —
all 264,831 frames survived. A manual tarball was added to "fix" this and then
removed as redundant.

Both errors shared a cause: concluding from an intermediate observation instead
of waiting for the operation to finish. On this platform, measuring is cheap and
inference is unreliable — but a measurement read halfway is just inference.

## D21 — TPSMM resumes correctly but not reproducibly

TPSMM does its own checkpointing, and we use it rather than ours. What
`train.py` hands to `Logger.save_cpk`:

```python
model_save = {
    'inpainting_network': ..., 'dense_motion_network': ...,
    'kp_detector': ..., 'optimizer': ...,
}
# plus bg_predictor / optimizer_bg_predictor, plus epoch
```

and on resume `MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=start_epoch-1)`
puts the LR schedule back where it was.

So model, optimizer moments, epoch and LR position all survive — everything that
governs correctness. **RNG state does not.** After a resume the data order and
augmentation draws differ from an uninterrupted run.

**The plan's M0 exit criterion — "a resumed run's loss curve matches an
uninterrupted one" — is therefore unreachable with TPSMM's mechanism, and was
written without knowing the mechanism.**

**Revised M0 exit criterion:** a resume must restore weights, optimizer moments,
epoch, and LR-schedule position, and the loss must continue from where it left
off rather than jumping. Bit-identical continuation is *not* required: a
different post-resume sample path is statistically equivalent training, not
corrupted training. What would corrupt results is a reset LR schedule or lost
optimizer moments, and those are restored.

Not patching RNG persistence into the vendored tree: it would buy
reproducibility we do not need, at the cost of touching frozen code.

**Consequence:** `pgmm/train/state.py` and `tests/test_resume_equivalence.py`
(plan Tasks 6–7) were built for a training loop we turn out not to own. They are
correct and tested, but currently unused — TPSMM's loop uses TPSMM's Logger.
Wasted effort, caused by planning the harness before reading the vendored
training code. Kept for now in case PGMM's loop diverges; delete if M2 confirms
it does not.

Checkpoint cadence is per *epoch* (`log_epoch` fires `save_cpk` when
`(epoch+1) % checkpoint_freq == 0`), not per step. At `checkpoint_freq: 5` and
~500 steps/epoch, a dying session loses at most 5 epochs.

## D12 — measured costs (partial)

First real measurements, from the CPU-only smoke kernel on 2026-07-17. CPU
sessions do not consume the 30h/week GPU quota, so this cost nothing.

| Quantity | Projected (20-clip sample) | **Measured (full run)** |
|---|---|---|
| Kaggle Python | — | 3.12.13 (local: 3.11.9) |
| Kaggle torch | — | 2.10.0+cpu (local: 2.13.0+cpu) |
| Decode + crop + resize | 0.70 s/clip | **0.83–0.87 s/clip** |
| Full LSA64 preprocessing | ~37 min | **44–46 min** (+ ~15 min for the split copy) |
| Mean frames per clip | 71.2 | **82.8** (min 14, max 201) |
| Total frames | ~227,840 | **264,831** |
| Prepared size | — | **1.25 GiB**, 5.0 KiB/frame |
| Split on disk | — | **2800 train / 400 test** — matches the paper |

The 20-clip sample underestimated frame count by 16% and speed by ~20%. Good
enough to plan with; not good enough to trust. Both runs of the preprocessing
kernel agree with each other, which is the real check.

The Python and torch versions differ from local, which is why the smoke kernel
re-runs the whole test suite on Kaggle's image rather than trusting local green.
All 49 passed there.

**Still unmeasured: GPU step time** — the number the 8–12 week estimate actually
rests on. Needs a GPU session and a working training loop.

### LSA64 mount path (measured, not guessed)

```
/kaggle/input/datasets/justinvo277/lsa64-dataset/lsa64_dataset/LSA64/001/001_001_001.mp4
```

Note `/kaggle/input/datasets/<owner>/<slug>/...`, **not** the commonly documented
`/kaggle/input/<slug>`. The first smoke run died on that assumption. Code should
discover the mount by globbing rather than hardcoding either layout.

## D10 / D11 — metric implementations

Neither the LPIPS backbone nor the SSIM implementation is stated, and both shift
the third decimal — the precision at which we are claiming a match. Pinned to
`lpips(net='alex')` and `skimage.metrics.structural_similarity(data_range=1.0,
channel_axis=-1)`. Revisit only if M1 misses by a small margin.
