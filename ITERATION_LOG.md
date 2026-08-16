# Round 2 Optimization Iteration Log

This document records reproducible hypotheses, configurations, spatial-hole validation results, platform results, and decisions. Every local score uses the same 12 connected spatial holes (360 points) rather than an optimistic random split.

## Evaluation contract

- Official score: `0.4 * PAS + 0.4 * PDP + 0.2 / (1 + NMSE)`.
- Both 2-D UPA and flat 256-port DFT interpretations are monitored. The selected platform candidate uses UPA.
- Power, cosine similarity, and NMSE totals use `float64` accumulation.
- A platform score is recorded only after a real upload; local estimates are never labeled as platform results.
- `run_iteration.py` runs `evaluate_hybrid.py`, writes full JSON to `experiments/iterations.jsonl`, and appends a Markdown summary here.

Example:

```powershell
python run_iteration.py --id R2-030 --note "32-NN and 16 projections" -- `
  --checkpoint outputs_l4_upa/best.pt `
  --delay-checkpoint outputs_smooth/best.pt `
  --pas-layout upa --neighbor-policy nonzero --k 32 `
  --alphas 0.3 0.4 --iterations 16
```

## Baselines

| ID | Method | Local UPA score | Platform | Decision |
|---|---|---:|---:|---|
| R2-000 | 1-NN | 0.5283 | - | Large degradation inside spatial holes |
| R2-001 | Smooth coordinate neural field | 0.5566 | - | AI learns generalizable spectral structure |
| R2-002 | AI + 4-NN power fusion, `alpha=0.5` | 0.5913 | **0.61** | Starting point for this iteration |
| R2-003 | Flat 256-port restoration | 0.5755 under UPA | - | Retained only as metric-layout fallback |

## Current iteration results

| ID | Change | UPA-PAS | PDP | Local score | Decision |
|---|---|---:|---:|---:|---|
| R2-010 | Locally shift the horizontal-angle peak | 0.4975 | 0.7183 | 0.5863 | Reject: multipath clusters do not move rigidly |
| R2-011 | Exclude 262 all-zero neighbors; 4-NN, `alpha=0.4` | 0.5233 | 0.7186 | 0.5968 | Keep |
| R2-012 | Delaunay barycentric interpolation | 0.4813 | 0.6999 | 0.5725 | Reject: triangle vertices span too far across holes |
| R2-013 | 8-NN, `alpha=0.3` | 0.5336 | 0.7236 | 0.6029 | Keep |
| R2-014 | 16-NN, `alpha=0.2` | 0.5402 | 0.7294 | 0.6078 | Keep |
| R2-015 | 32-NN, `alpha=0.2` | 0.5448 | 0.7334 | 0.6113 | Keep |
| R2-016 | Generalized power-mean sharpening, `q=2` | 0.5432 | 0.7291 | 0.6089 | Reject: over-sharpens peaks |
| R2-017 | L2-normalize each scoring slice before fusion | 0.5436 | 0.7407 | 0.6137 | Reject: PDP gain does not offset PAS loss |
| R2-018 | UPA-only low-frequency field, `fourier_levels=4` | - | - | Proxy 0.6343 | Use as primary angle model |
| R2-019 | Lower-frequency field, `fourier_levels=3` | - | - | Proxy 0.6442 | Use in angle ensemble |
| R2-020 | L4 angle model + original smooth delay model; 4 projections | 0.5523 | 0.7330 | 0.6141 | Keep: specialist heads are complementary |
| R2-021 | 16 alternating projections | 0.5559 | 0.7330 | 0.6155 | Keep |
| R2-022 | Ensemble L4 and L3 angle-power predictions | 0.5577 | 0.7330 | 0.6163 | Keep |
| R2-023 | Finish with an additional angle constraint, `alpha=0.4` | 0.5620 | 0.7317 | **0.6175** | Current best |
| R2-024 | Expand the final neighborhood to 64 points | 0.5643 | 0.7323 | **0.6186** | Selected |
| R2-025 | Preserve 8 exact-zero decisions from the original AI coverage model | - | - | Platform-only safeguard | Selected |

## Findings

- The training set contains 262 exactly-zero channels. They dilute nonzero spectral templates, so they are excluded from the neighbor pool while coverage remains an explicit neural-network task.
- The original all-data AI model predicts exactly zero at 8 test points. These points also have the highest local zero-channel probabilities, so the optimized array preserves that learned coverage mask.
- Horizontal-angle peaks correlate strongly with BS-user geometry (`R2` about 0.72-0.84), but rigidly shifting the entire spectrum destroys independently moving reflection clusters.
- Increasing the neighborhood from 4 to 64 consistently helps connected holes. A lower-variance spectral estimate across the hole is more reliable than copying the nearest point.
- PAS is the bottleneck. Low-frequency UPA-specific fields estimate angle power, while the original smooth model estimates delay power. Ending the projection sequence with the angle constraint trades a small amount of PDP for a larger PAS gain.

## Selected candidate

```text
Angle models: outputs_l4_upa_full/best.pt + outputs_l3_upa_full/best.pt
Delay model: outputs_final/best.pt
Neighbors: nonzero training channels only, 64-NN, inverse-distance power=2
Fusion: AI alpha=0.4, neighbor alpha=0.6
Recovery: 16 PAS/PDP alternating projections plus a final PAS constraint
```

The current local gain is `0.6186 - 0.5913 = +0.0273`. The real platform gain must be measured by uploading the newly generated ZIP.
