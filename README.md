# SiT for Diffusers

This repository is refactored to a Diffusers-style layout and API surface.

The old standalone training and sampling stack (`train.py`, `sample.py`, `sample_ddp.py`, `transport/*`, and related utilities) has been removed in favor of modular components under `src/diffusers`.

## Package Layout

- `src/diffusers/models/transformers/transformer_sit.py`: `SiTTransformer2DModel`
- `src/diffusers/schedulers/scheduling_flow_match_sit.py`: `SiTFlowMatchScheduler`
- `src/diffusers/pipelines/sit/pipeline_sit.py`: `SiTPipeline`
- `scripts/convert_sit_to_diffusers.py`: checkpoint conversion to a Diffusers pipeline directory
- `scripts/sample_sit.py`: sampling from a converted pipeline

## Install

```bash
pip install -e .
```

## Convert Legacy Checkpoint

```bash
python scripts/convert_sit_to_diffusers.py \
  --checkpoint /path/to/sit_checkpoint.pt \
  --output sit-xl-2-diffusers \
  --model-size sit-xl-2 \
  --mode ode
```

Optional:
- `--vae stabilityai/sd-vae-ft-mse`
- `--mode sde`

## Sample

```bash
python scripts/sample_sit.py \
  --model sit-xl-2-diffusers \
  --class-label 207 \
  --height 256 \
  --width 256 \
  --num-inference-steps 250 \
  --guidance-scale 4.0 \
  --output sample.png
```

## Notes

- The architecture and repository split were aligned with the Diffusers-native style used by the NiT diffusers refactor.
- The scheduler supports ODE and SDE-like stepping in a single class via `mode`.
- The code is ready to upstream into corresponding `diffusers` package directories.
