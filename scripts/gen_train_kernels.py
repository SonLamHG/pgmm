"""Generate the two session kernels for the M1 run from one template.

A full run is 15.5 GPU-hours (D12) against a ~9h cap, so it spans two sessions.
Session 2 mounts session 1's output via `kernel_sources` and resumes from its
checkpoint — no download, no credentials (D16, D20).

Two kernel slugs rather than one: a kernel cannot list itself in
`kernel_sources`, so the handoff needs a distinct target.

The notebooks are generated rather than hand-copied so the two cannot drift.
All real logic lives in `pgmm/train/session.py`, which is unit-tested; these are
drivers.

    python scripts/gen_train_kernels.py
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MARKDOWN = """# R0 — TPSMM baseline on LSA64 (M1 gate), session {s} of 2

Target (paper Table 1, LSA64): L1 0.01342 | SSIM 0.9208 | LPIPS 0.02261 |
FVD 182.785 | TCD 0.130.

A full run is **15.5 GPU-hours** (D12, measured at 1.07 s/step) against a ~9h
session cap, so it spans two sessions of ~50 epochs each.

`num_epochs` is a **total**, not an increment: the vendored `train.py` runs
`for epoch in trange(start_epoch, num_epochs)` after `start_epoch = load_cpk() + 1`,
and restores the LR schedule with `last_epoch=start_epoch-1`. Bounding a session
therefore means giving it a lower total.

Bounded rather than left to be killed at the cap: a killed session's output is
not reliably saved, which would cost both the epochs since the last checkpoint
and the run's log.

Logic lives in `pgmm/train/session.py` where it is unit-tested. This is a driver.
"""

CELL_ENV = """import subprocess, sys, time
import torch

print('torch', torch.__version__, '| gpus', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  [{{i}}] {{p.name}}  {{p.total_memory/2**30:.1f}} GiB')
# Quota bills session wall-clock, not GPU-hours, so a one-GPU session wastes half
# of what it costs. Needs machine_shape=NvidiaTeslaT4 in the metadata (D22).
assert torch.cuda.device_count() == 2, 'expected 2x T4 -- check machine_shape'

r = subprocess.run(['git', 'clone', '-q', '--branch', 'feat/m0-m1-harness',
                    '--depth', '1', 'https://github.com/SonLamHG/pgmm.git',
                    '/kaggle/working/repo'], capture_output=True, text=True)
assert r.returncode == 0, r.stderr
sys.path.insert(0, '/kaggle/working/repo')
print('repo cloned')"""

CELL_EXTRACT = """# Extract every mounted kernel output: the prepared data, and -- in session 2 --
# the previous session's checkpoint. Kaggle bundles each kernel's output into a
# single _output_.zip, and that is what kernel_sources mounts (D20).
import zipfile
from pathlib import Path

WORK = Path('/kaggle/working/mounted')
zips = sorted(Path('/kaggle/input').rglob('_output_.zip'))
assert zips, 'nothing mounted -- check kernel_sources'
for z in zips:
    t0 = time.time()
    with zipfile.ZipFile(z) as zf:
        zf.extractall(WORK / z.parent.name)
    print(f'{{z.parent.name:24}} {{z.stat().st_size/2**30:5.2f}} GiB '
          f'-> {{(time.time()-t0)/60:.1f}} min')"""

CELL_SETUP = """from pgmm.train.session import (build_session_config, find_prepared_data,
                                find_resume_checkpoint, steps_per_epoch)

DATA = find_prepared_data(WORK)
n_train = len(list((DATA / 'train').iterdir()))
n_test = len(list((DATA / 'test').iterdir()))
print('data  :', DATA)
print('split :', n_train, 'train /', n_test, 'test')
# Without train/ and test/ on disk, FramesDataset silently substitutes its own
# random 80/20 split -- 2560/640, not the paper's 2800/400 (D17).
assert (n_train, n_test) == (2800, 400), f'paper says 2800/400, got {{n_train}}/{{n_test}}'

RESUME = find_resume_checkpoint(WORK)
print('resume:', RESUME or 'none - starting from scratch')
assert (RESUME is not None) == {expect_resume}, (
    'session {s} expected resume={expect_resume}, found ' + str(RESUME))

MAX_EPOCHS = {max_epochs}
CFG = Path('/kaggle/working/session.yaml')
cfg = build_session_config('/kaggle/working/repo/pgmm/config/lsa64-tpsmm.yaml',
                           DATA, CFG, max_epochs=MAX_EPOCHS)
spe = steps_per_epoch(cfg)
print(f'this session -> up to epoch {{MAX_EPOCHS}} total, {{spe:.0f}} steps/epoch')
print(f'projected     {{MAX_EPOCHS*spe*1.07/3600:.1f}}h at the measured 1.07 s/step')"""

CELL_TRAIN = """argv = [sys.executable, 'run.py', '--config', str(CFG),
        '--log_dir', '/kaggle/working/log', '--device_ids', '0,1']
if RESUME is not None:
    argv += ['--checkpoint', str(RESUME)]
print(' '.join(argv))

t0 = time.time()
r = subprocess.run(argv, cwd='/kaggle/working/repo/third_party/tpsmm',
                   capture_output=True, text=True)
elapsed = time.time() - t0
print(r.stdout[-3000:])
print('--- stderr tail ---')
print((r.stderr or '')[-2500:])
print('exit:', r.returncode, f'| wall {{elapsed/3600:.2f}}h')"""

CELL_VERIFY = """# A number derived from a crash looks like evidence. Refuse to report one.
assert r.returncode == 0, f'training exited {{r.returncode}} -- see stderr above'

ckpts = sorted(Path('/kaggle/working/log').rglob('*-checkpoint.pth.tar'))
assert ckpts, 'no checkpoint written -- the next session would have nothing to resume from'
print('checkpoints:')
for p in ckpts:
    print(f'  {{p.name}}  {{p.stat().st_size/2**20:.0f}} MiB')
print(f'\\ns/step this session: {{elapsed/(MAX_EPOCHS*spe):.3f}}  (D12 measured 1.07)')
for lg in Path('/kaggle/working/log').rglob('log.txt'):
    lines = lg.read_text().strip().splitlines()
    print('first:', lines[0][:110])
    print('last :', lines[-1][:110])"""

CELL_CLEAN = """# The output must carry the checkpoint to the next session and nothing else:
# the extracted data would republish 1.25 GiB and the repo clone would ride along.
import shutil

shutil.rmtree(WORK, ignore_errors=True)
shutil.rmtree('/kaggle/working/repo', ignore_errors=True)
Path('/kaggle/working/session.yaml').unlink(missing_ok=True)
print('output root:', sorted(p.name for p in Path('/kaggle/working').iterdir()))"""

SESSIONS = [
    dict(s=1, max_epochs=50, expect_resume=False,
         sources=["snlmhong/pgmm-preprocess"]),
    dict(s=2, max_epochs=100, expect_resume=True,
         sources=["snlmhong/pgmm-preprocess", "snlmhong/pgmm-train-s1"]),
]


def build(spec: dict) -> None:
    fmt = dict(s=spec["s"], max_epochs=spec["max_epochs"],
               expect_resume=spec["expect_resume"])
    cells = [{"cell_type": "markdown", "metadata": {},
              "source": MARKDOWN.format(**fmt)}]
    for body in (CELL_ENV, CELL_EXTRACT, CELL_SETUP, CELL_TRAIN,
                 CELL_VERIFY, CELL_CLEAN):
        cells.append({"cell_type": "code", "execution_count": None,
                      "metadata": {}, "outputs": [],
                      "source": body.format(**fmt)})
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    d = REPO / "kaggle_harness" / f"train_s{spec['s']}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "train.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"snlmhong/pgmm-train-s{spec['s']}",
        "title": f"pgmm-train-s{spec['s']}",
        "code_file": "train.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": spec["sources"],
        "model_sources": [],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"s{spec['s']}: max_epochs={spec['max_epochs']} "
          f"resume={spec['expect_resume']} sources={spec['sources']}")


if __name__ == "__main__":
    for spec in SESSIONS:
        build(spec)
