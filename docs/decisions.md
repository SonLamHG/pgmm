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
| D7 | `L_r` sub-loss weights | Inherited from vendored TPSMM | decided |
| D8 | FVD implementation sensitivity | Pin one I3D; report relative only | open |
| D9 | LSA64 2800/400 split undefined | Seeded random default; signer-held-out alternative | open |
| D10 | LPIPS backbone (alex vs vgg) undefined | `lpips` pkg, `net='alex'` | decided |
| D11 | SSIM implementation undefined | `skimage`, `data_range=1.0`, `channel_axis=-1` | decided |
| D12 | Real step time / timeline unmeasured | Measure at M0, re-derive the estimate | open |
| D13 | LSA64 filename convention unverified | `NNN_NNN_NNN.mp4` = sign_signer_repetition — **confirmed** against the real archive | resolved |
| D14 | Paper does not say whether LSA64 raw or cut is used | Use **cut** (`justinvo277/lsa64-dataset`); retry raw if M1 misses | decided |
| D15 | Kaggle legacy API key shadows OAuth and cannot push kernels | Disable `~/.kaggle/kaggle.json` so the CLI resolves to OAuth | decided |
| D16 | A kernel has no Kaggle credentials, so it cannot push its own checkpoint | Unresolved — Secret with a legacy key, or kernel-output chaining | **open — blocks any long run** |

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

**Chosen: Kaggle Secret holding a legacy API key.** Rationale: legacy keys work
for dataset operations (`datasets list`/`files` both succeeded under legacy) and
fail only for *kernel* operations (D15). `datasets version` is a dataset
operation, so a legacy key should suffice. The account already has two legacy
keys (`d:\Admin\kaggle.json`, and the disabled `~/.kaggle/kaggle.json.legacy-disabled`).

**Two things still unverified, both cheap to test and both fatal if wrong:**

1. The `kaggle` PyPI package contains no reference to secrets at all — the API
   cannot attach them. Secrets are UI-only. So whether a UI-attached secret
   *survives an API kernel push* is unknown, and the whole workflow is
   API-driven.
2. Whether a legacy key actually authorises `datasets version` from inside a
   kernel. Plausible, untested.

Fallbacks if the secret route fails:

- **Private dataset holding the key.** Proven to mount (chainprobe read a
  private dataset successfully). Needs no UI step. Weaker than a Secret: the key
  sits as a plain file rather than encrypted at rest.
- **Local orchestration.** Drive the loop from the workstation: push kernel →
  fetch its output → push checkpoint to the dataset → push next kernel. Needs no
  kernel credentials at all, since `kernels output` and `datasets version` both
  work locally under OAuth. Cost: every checkpoint round-trips through the
  user's connection. A TPSMM checkpoint is estimated at 500MB–1GB (weights plus
  two Adam moments) — unmeasured — so ~10 sessions means 10–20GB of traffic.

**Must be settled before any long training run.** An auth failure at the *end*
of a 9h session loses the entire session — the most expensive possible place to
discover this.

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

## D12 — measured costs (partial)

First real measurements, from the CPU-only smoke kernel on 2026-07-17. CPU
sessions do not consume the 30h/week GPU quota, so this cost nothing.

| Quantity | Measured |
|---|---|
| Kaggle Python | 3.12.13 (local: 3.11.9) |
| Kaggle torch | 2.10.0+cpu (local: 2.13.0+cpu) |
| Decode + crop + resize | 0.70 s/clip single-process |
| Full LSA64 preprocessing | ~37 min single-process |
| Mean frames per clip | 71.2 |
| Projected total frames | ~227,840 |

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
