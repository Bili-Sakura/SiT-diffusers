#!/usr/bin/env python3
"""Sample images from a converted SiT Diffusers pipeline directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.diffusers.pipelines.sit.pipeline_sit import SiTPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample images from a converted SiT diffusers pipeline.")
    parser.add_argument("--model", type=str, required=True, help="Path to pipeline directory")
    parser.add_argument("--class-label", type=int, default=207)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-inference-steps", type=int, default=250)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="sample.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(args.seed)

    pipeline = SiTPipeline.from_pretrained(args.model).to(device)
    image = pipeline(
        class_labels=args.class_label,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        output_type="pt",
    ).images
    save_image(image, args.output, normalize=True, value_range=(-1, 1))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
