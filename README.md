# Round 2 Physical-AI Wireless Channel Generation

This project generates complex MIMO-OFDM channels with shape `(256, 4, 192)` from receiver coordinates and the supplied scene. It uses coordinate-conditioned neural fields in sparse angle-delay space, not ray tracing. Neural predictions are fused with locally interpolated training spectra and restored through alternating PAS/PDP projections.

## Verified data properties

- Two coverage regions contain 2,000 training and 250 test points each.
- Test points form connected spatial holes, so random train/validation splits are misleading.
- The training set contains 262 exactly-zero channels.
- The first 16 delay bins retain most predictable energy.
- Antenna order is polarization-horizontal-vertical: `2 x 16 x 8`.

## Environment and cache

```powershell
pip install -r requirements.txt
python prepare_cache.py --data-dir Round2_Map --cache-dir cache --delay-bins 16
python analyze_alignment.py --data-dir Round2_Map --cache cache/alignment_summaries.npz
```

## Train the selected neural fields

Primary angle field:

```powershell
python train.py --data-dir Round2_Map --cache-dir cache `
  --output-dir outputs_l4_upa_full --train-all --epochs 12 `
  --scheduler-tmax 20 --batch-size 32 --lr 0.0003 --width 256 `
  --fourier-levels 4 --pas-mode upa --complex-weight 0.2
```

Angle ensemble field:

```powershell
python train.py --data-dir Round2_Map --cache-dir cache `
  --output-dir outputs_l3_upa_full --train-all --epochs 9 `
  --scheduler-tmax 20 --batch-size 32 --lr 0.0003 --width 256 `
  --fourier-levels 3 --pas-mode upa --complex-weight 0.2
```

The delay field is the original smooth all-data model at `outputs_final/best.pt`.

## Generate the optimized submission array

```powershell
python predict_hybrid.py --data-dir Round2_Map --cache-dir cache `
  --checkpoint outputs_l4_upa_full/best.pt `
  --angle-checkpoint-extra outputs_l3_upa_full/best.pt `
  --delay-checkpoint outputs_final/best.pt `
  --pas-layout upa --neighbor-policy nonzero --k 64 `
  --distance-power 2 --alpha 0.4 --iterations 16 --final-angle `
  --calibration-real 0.0123316781 --calibration-imag -0.0233473133 `
  --zero-mask-reference outputs_final/Round2_Test_Channel_UPA.npy `
  --output outputs_optimized/Round2_Test_Channel.npy
```

## Validate and record an iteration

The validation split contains 12 connected holes (360 points) with train-neighbor distances similar to the real test holes.

```powershell
python run_iteration.py --id R2-030 --note "candidate description" -- `
  --checkpoint outputs_l4_upa/best.pt `
  --angle-checkpoint-extra outputs_l3_upa/best.pt `
  --delay-checkpoint outputs_smooth/best.pt `
  --pas-layout upa --neighbor-policy nonzero --k 32 `
  --alphas 0.3 0.4 --iterations 16 --final-angle
```

The starting hybrid scored 0.5913 locally and 0.61 on the platform. The selected configuration scores 0.6186 on the same local validation protocol. Full accepted and rejected experiments are documented in `ITERATION_LOG.md`.

## Build the ZIP

```powershell
python make_submission.py `
  --channel outputs_optimized/Round2_Test_Channel.npy `
  --checkpoint outputs_l4_upa_full/best.pt `
  --extra-checkpoint outputs_l3_upa_full/best.pt `
  --extra-checkpoint outputs_final/best.pt `
  --output submissions/Round2_optimized_submission.zip
```

The ZIP contains `Round2_Test_Channel.npy`, source code, documentation, and all neural checkpoints required to reproduce the prediction. It is intentionally excluded from Git history because it is larger than GitHub's normal file limit.
