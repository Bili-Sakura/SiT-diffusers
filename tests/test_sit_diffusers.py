import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

torch = pytest.importorskip("torch")
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from src.diffusers.models.transformers.transformer_sit import SiTTransformer2DModel
from src.diffusers.pipelines.sit.pipeline_sit import SiTPipeline


def test_transformer_forward_shape():
    model = SiTTransformer2DModel(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=32,
        depth=2,
        num_heads=4,
        num_classes=10,
        learn_sigma=False,
    )
    latents = torch.randn(2, 4, 8, 8)
    timesteps = torch.tensor([0.25, 0.5])
    class_labels = torch.tensor([1, 2])

    output = model(latents, timesteps, class_labels).sample

    assert output.shape == latents.shape


def test_scheduler_step_matches_velocity_update():
    scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0, stochastic_sampling=False)
    scheduler.set_timesteps(2)
    sample = torch.ones(1, 4, 2, 2)
    velocity = torch.full_like(sample, 0.5)
    timestep = scheduler.timesteps[0]
    output = scheduler.step(velocity, timestep, sample).prev_sample
    assert output.shape == sample.shape


class _DummyVAE(torch.nn.Module):
    class Config:
        block_out_channels = [128, 256, 512, 512]
        scaling_factor = 0.18215

    config = Config()

    def decode(self, latents):
        return type("Decoded", (), {"sample": latents})()


def test_pipeline_latent_output_smoke():
    transformer = SiTTransformer2DModel(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=32,
        depth=1,
        num_heads=4,
        num_classes=10,
        learn_sigma=False,
    )
    scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0, stochastic_sampling=False)
    pipe = SiTPipeline(
        transformer=transformer,
        scheduler=scheduler,
        vae=_DummyVAE(),
        id2label={0: "class zero", 1: "class one"},
    )
    result = pipe(
        class_labels=[1, 2],
        height=64,
        width=64,
        num_inference_steps=3,
        guidance_scale=2.0,
        output_type="latent",
    )
    assert result.images.shape == (2, 4, 8, 8)
