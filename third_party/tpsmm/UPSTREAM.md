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

No patches to code — numerics frozen.

Excluded from version control (see `.gitignore`), not modified:

- `assets/` — 32MB of demo GIFs (`ted.gif`, `vox.gif`) used only by the
  upstream README. They play no part in training or evaluation and would be
  re-cloned into every Kaggle session. Restore with a fresh clone of the pinned
  commit if the upstream demo is ever needed.
