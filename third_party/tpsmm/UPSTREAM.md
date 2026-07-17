# Vendored upstream

Repo: https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model
Commit: c616878812c9870ed81ac72561be2676fd7180e2
License: MIT (see LICENSE in this directory)
Vendored: 2026-07-17

Vendored as a plain directory rather than a git submodule: later PGMM work must
patch `model.py` to add `L_pd`/`L_align`, Kaggle clones are simpler without
submodule recursion, and one clone then contains everything.

## Why this is frozen

PGMM is evaluated as a delta over TPSMM. That delta is only the paper's
contribution if the baseline's numerics are untouched — so no edit to the math
in this directory. New code lives in `pgmm/`.

## Patch log

### 2026-07-17 — `logger.py`: shim `skimage.draw.circle`

**Numerics untouched.** scikit-image removed `draw.circle` in 0.19 (renamed to
`draw.disk`); TPSMM predates that and Kaggle ships 0.26, so `import logger`
raised `ImportError` and training could not start at all.

Patched with a try/except shim that falls back to `disk` and preserves the old
call signature, rather than rewriting the call site — keeps the vendored code as
close to upstream as possible.

`circle` is used at exactly one place (`logger.py:112`), drawing keypoint markers
onto debug images. It is visualisation only and participates in no loss, no
gradient, and no metric. Verified as the *only* stale skimage API in the tree:
`transform.resize`, `color.gray2rgb`, `img_as_float32`, `img_as_ubyte` and
`io.imread` all still exist in 0.26.

Excluded from version control (see `.gitignore`), not modified:

- `assets/` — 32MB of demo GIFs (`ted.gif`, `vox.gif`) used only by the
  upstream README. They play no part in training or evaluation and would be
  re-cloned into every Kaggle session. Restore with a fresh clone of the pinned
  commit if the upstream demo is ever needed.
