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
| D13 | LSA64 filename convention unverified | `NNN_NNN_NNN.mp4` assumed from docs; verify against the real archive | **open — blocks M0** |

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

## D13 — LSA64 filename convention (unverified)

`pgmm/data/lsa64_prepare.py` parses `NNN_NNN_NNN.mp4` as
`<sign>_<signer>_<repetition>`. **This was taken from LSA64's documentation and
has not been checked against the actual archive** — nobody has opened the
download yet.

Everything downstream leans on it: D9's split reads the signer field out of the
filename, so a wrong convention would silently produce a split that is not the
one we documented.

**To verify once LSA64 is downloaded:**

```bash
ls <lsa64_dir> | head -5      # expect 001_001_001.mp4 style
ls <lsa64_dir> | wc -l        # expect 3200

python -c "
from pathlib import Path
from pgmm.data.lsa64_prepare import parse_lsa64_filename
names = sorted(p.name for p in Path('<lsa64_dir>').glob('*.mp4'))
print(len(names))
signs, signers, reps = zip(*(parse_lsa64_filename(n) for n in names))
print('signs', min(signs), max(signs))       # expect 1 64
print('signers', min(signers), max(signers)) # expect 1 10
print('reps', min(reps), max(reps))          # expect 1 5
"
```

If the convention differs, fix `_NAME_RE` **and the test together**. Do not make
the test pass by loosening the assertion — the assertion is the only thing
standing between us and a mislabelled split.

## D10 / D11 — metric implementations

Neither the LPIPS backbone nor the SSIM implementation is stated, and both shift
the third decimal — the precision at which we are claiming a match. Pinned to
`lpips(net='alex')` and `skimage.metrics.structural_similarity(data_range=1.0,
channel_axis=-1)`. Revisit only if M1 misses by a small margin.
