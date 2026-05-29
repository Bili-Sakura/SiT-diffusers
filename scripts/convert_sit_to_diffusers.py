import argparse
import json
from pathlib import Path
import shutil
import sys

import torch
from diffusers.models import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

sys.path.insert(0, Path(__file__).resolve().parents[1].as_posix())

from src.diffusers.models.transformers.transformer_sit import SiTTransformer2DModel
from src.diffusers.pipelines.sit.pipeline_sit import SiTPipeline


MODEL_SPECS = {
    "sit-xl-2": dict(depth=28, hidden_size=1152, patch_size=2, num_heads=16),
    "sit-xl-4": dict(depth=28, hidden_size=1152, patch_size=4, num_heads=16),
    "sit-xl-8": dict(depth=28, hidden_size=1152, patch_size=8, num_heads=16),
    "sit-l-2": dict(depth=24, hidden_size=1024, patch_size=2, num_heads=16),
    "sit-l-4": dict(depth=24, hidden_size=1024, patch_size=4, num_heads=16),
    "sit-l-8": dict(depth=24, hidden_size=1024, patch_size=8, num_heads=16),
    "sit-b-2": dict(depth=12, hidden_size=768, patch_size=2, num_heads=12),
    "sit-b-4": dict(depth=12, hidden_size=768, patch_size=4, num_heads=12),
    "sit-b-8": dict(depth=12, hidden_size=768, patch_size=8, num_heads=12),
    "sit-s-2": dict(depth=12, hidden_size=384, patch_size=2, num_heads=6),
    "sit-s-4": dict(depth=12, hidden_size=384, patch_size=4, num_heads=6),
    "sit-s-8": dict(depth=12, hidden_size=384, patch_size=8, num_heads=6),
}

SCHEDULER_CONFIG = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "_diffusers_version": "0.36.0",
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "stochastic_sampling": False,
}


def load_checkpoint(path: str):
    checkpoint = torch.load(path, map_location="cpu")
    if "ema" in checkpoint:
        checkpoint = checkpoint["ema"]
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return {k.replace("module.", ""): v for k, v in checkpoint.items()}


def infer_learn_sigma(state_dict: dict, patch_size: int, in_channels: int = 4) -> bool:
    weight = state_dict.get("final_layer.linear.weight")
    if weight is None:
        return True
    out_dim = weight.shape[0]
    base = patch_size * patch_size * in_channels
    return out_dim == base * 2


def make_self_contained_repo(output_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pipeline_src = repo_root / "src" / "diffusers" / "pipelines" / "sit" / "pipeline_sit.py"
    transformer_src = repo_root / "src" / "diffusers" / "models" / "transformers" / "transformer_sit.py"

    shutil.copy2(pipeline_src, output_path / "pipeline.py")
    shutil.copy2(transformer_src, output_path / "transformer" / "transformer_sit.py")

    scheduler_dir = output_path / "scheduler"
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    (scheduler_dir / "scheduler_config.json").write_text(json.dumps(SCHEDULER_CONFIG, indent=2) + "\n", encoding="utf-8")
    legacy_scheduler = scheduler_dir / "scheduling_flow_match_sit.py"
    if legacy_scheduler.exists():
        legacy_scheduler.unlink()

    model_index_path = output_path / "model_index.json"
    with model_index_path.open("r", encoding="utf-8") as f:
        model_index = json.load(f)
    model_index["_class_name"] = ["pipeline", "SiTPipeline"]
    model_index["transformer"] = ["transformer_sit", "SiTTransformer2DModel"]
    model_index["scheduler"] = ["diffusers", "FlowMatchEulerDiscreteScheduler"]
    with model_index_path.open("w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Convert SiT checkpoint to Diffusers pipeline layout.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SiT checkpoint (.pt)")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--model-size", type=str, choices=sorted(MODEL_SPECS.keys()), required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--vae", type=str, default="stabilityai/sd-vae-ft-mse")
    args = parser.parse_args()

    spec = MODEL_SPECS[args.model_size]
    state_dict = load_checkpoint(args.checkpoint)
    learn_sigma = infer_learn_sigma(state_dict, patch_size=spec["patch_size"], in_channels=4)
    transformer = SiTTransformer2DModel(
        input_size=args.image_size // 8,
        num_classes=args.num_classes,
        learn_sigma=learn_sigma,
        **spec,
    )
    missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")

    scheduler = FlowMatchEulerDiscreteScheduler.from_config(SCHEDULER_CONFIG)
    vae = AutoencoderKL.from_pretrained(args.vae)
    pipeline = SiTPipeline(transformer=transformer, scheduler=scheduler, vae=vae)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    pipeline.save_pretrained(output_path.as_posix())
    make_self_contained_repo(output_path)
    print(f"Saved diffusers pipeline to: {output_path}")


if __name__ == "__main__":
    main()
