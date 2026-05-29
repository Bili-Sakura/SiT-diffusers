from pathlib import Path


def infer_resolution(model_name: str) -> int:
    parts = model_name.split("-")
    for part in parts:
        if part.isdigit():
            value = int(part)
            if value in (256, 512, 768, 1024):
                return value
    return 256


def build_readme(model_name: str) -> str:
    resolution = infer_resolution(model_name)
    return f"""---
library_name: diffusers
pipeline_tag: unconditional-image-generation
tags:
  - diffusers
  - sit
  - image-generation
  - class-conditional
inference: true
---

# {model_name}

Self-contained Diffusers checkpoint repo for SiT.

## Usage

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("./").to("cuda" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device=pipe.device).manual_seed(0)

image = pipe(
    class_labels=207,
    height={resolution},
    width={resolution},
    num_inference_steps=250,
    guidance_scale=4.0,
    generator=generator,
).images[0]
image.save("demo.png")
```

## Components

- `pipeline.py`
- `transformer/transformer_sit.py`
- `scheduler/scheduler_config.json`
- `transformer/diffusion_pytorch_model.safetensors`
- `vae/diffusion_pytorch_model.safetensors`
"""


def main() -> None:
    root = Path("D:/sakura-project/SiT-diffusers/pretrained_models")

    for repo in root.iterdir():
        if not repo.is_dir() or not (repo / "model_index.json").exists():
            continue
        (repo / "README.md").write_text(build_readme(repo.name), encoding="utf-8")
        print(f"updated {repo.name}")


if __name__ == "__main__":
    main()
