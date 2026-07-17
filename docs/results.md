# Results

Our numbers against the paper's. **Runs that fail to reproduce are recorded
here too** — the output of this project is an honest replication report, not a
matching table.

Every deviation is traceable to a numbered decision in `docs/decisions.md`.

## Status

Nothing measured yet. M0 (harness) in progress; M1 (baseline gate) not started.

## M1 — TPSMM baseline, LSA64 (gate)

Paper targets from Table 1. Tolerance: ±5% relative on L1/LPIPS, ±0.005
absolute on SSIM. FVD exempt (ordering only). TCD is a calibration target for
D3, not a gate.

| Metric | Paper | Ours | Δ | Within tolerance |
|---|---|---|---|---|
| L1 ↓ | 0.01342 | — | — | ±5% |
| SSIM ↑ | 0.9208 | — | — | ±0.005 |
| LPIPS ↓ | 0.02261 | — | — | ±5% |
| FVD ↓ | 182.785 | not measured | — | deferred to the next plan |
| TCD ↓ | 0.130 | — | — | calibration target for D3 |

## M3 — PGMM, LSA64

Not started. Gated on M1.

| Metric | Paper | Ours | Δ |
|---|---|---|---|
| L1 ↓ | 0.01144 | — | — |
| SSIM ↑ | 0.9395 | — | — |
| LPIPS ↓ | 0.01638 | — | — |
| FVD ↓ | 154.726 | — | — |
| TCD ↓ | 0.125 | — | — |

## Cited, not reproduced

MRAA and DynaST are independently published methods and are not retrained;
PGMM is built on TPSMM, so TPSMM is the only control that isolates the
contribution. Their LSA64 numbers, for context:

| Method | L1↓ | SSIM↑ | LPIPS↓ | FVD↓ | TCD↓ |
|---|---|---|---|---|---|
| MRAA | 0.01196 | 0.9324 | 0.01996 | 184.668 | 0.150 |
