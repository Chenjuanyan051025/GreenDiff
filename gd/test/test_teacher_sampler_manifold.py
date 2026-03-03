import copy
import os
import sys

import pytest
import torch


def _ensure_project_root():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_project_root()

from gd.guidance.latent_guidance import LatentGuidance  # noqa: E402
from gd.inference.teacher_sampler import TeacherSampler  # noqa: E402
from gd.utils.config_utils import load_config  # noqa: E402


class _DummyDiffusion(torch.nn.Module):
    def __init__(self, T: int):
        super().__init__()
        self.T = T

    def step(self, z_t: torch.Tensor, t: torch.Tensor, cond_input: torch.Tensor, eta: float, t_prev=None) -> torch.Tensor:
        _ = (t, cond_input, eta, t_prev)
        return z_t + 0.1

    def get_alpha_sigma(self, t: torch.Tensor):
        _ = t
        alpha = torch.full_like(t, 0.8, dtype=torch.float32)
        sigma = torch.full_like(t, 0.6, dtype=torch.float32)
        return alpha, sigma


class _DummyVAE(torch.nn.Module):
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z.mean(dim=1, keepdim=True)


class _DummyLatentGreen(torch.nn.Module):
    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _ = t
        return z.abs() + 0.05


class _DummyRestart:
    def check(self, z: torch.Tensor, g_obs: torch.Tensor):
        _ = g_obs
        delta = torch.zeros(z.shape[0], device=z.device)
        epsilon = torch.zeros(z.shape[0], device=z.device)
        restart_mask = torch.zeros(z.shape[0], device=z.device, dtype=torch.bool)
        restart_mask[0] = True
        return delta, epsilon, restart_mask

    def add_restart_noise(self, z: torch.Tensor, diffusion_model: torch.nn.Module) -> torch.Tensor:
        _ = diffusion_model
        return z + 1.0


def _cfg():
    cfg = copy.deepcopy(load_config("configs/default.yaml"))
    cfg["data"]["resolution"] = 8
    cfg["data"]["K"] = 4
    cfg["data"]["sublattice_resolved_ldos"] = False
    cfg["data"]["ldos_transform"]["enabled"] = False
    cfg["vae"]["latent_channels"] = 4
    cfg["vae"]["latent_downsample"] = 1
    cfg["diffusion"]["T"] = 4
    cfg["diffusion"]["sampler"]["steps"] = 3
    cfg["guidance"]["enabled"] = True
    cfg["guidance"]["use_latent_green"] = True
    cfg["guidance"]["lambda"]["lambda0"] = 0.5
    cfg["guidance"]["lambda"]["schedule"] = "constant"
    cfg["guidance"]["lambda"]["start_step"] = 4
    cfg["guidance"]["lambda"]["grad_steps_per_iter"] = 1
    cfg["guidance"]["manifold"]["enabled"] = True
    cfg["guidance"]["manifold"]["mode"] = "first_order_projection"
    cfg["guidance"]["manifold"]["step_schedule"] = "constant"
    cfg["guidance"]["manifold"]["lambda0"] = 0.5
    cfg["guidance"]["manifold"]["start_step"] = 4
    cfg["guidance"]["manifold"]["grad_steps_per_iter"] = 1
    cfg["guidance"]["manifold"]["normalize_grad"] = False
    cfg["guidance"]["manifold"]["grad_clip"] = None
    cfg["guidance"]["manifold"]["max_step_norm"] = None
    cfg["guidance"]["risk"]["enabled"] = True
    cfg["guidance"]["risk"]["normalize_by_latent_std"] = True
    cfg["guidance"]["budget_hooks"]["enabled"] = False
    cfg["validation"]["enabled"] = False
    cfg["validation"]["restart"]["enabled"] = False
    cfg["validation"]["kpm_check"]["enabled"] = False
    return cfg


def test_latent_guidance_correct_contract_and_apply_compat():
    torch.manual_seed(0)
    cfg = _cfg()
    guidance = LatentGuidance(cfg)
    diffusion = _DummyDiffusion(cfg["diffusion"]["T"])
    latent_green = _DummyLatentGreen()
    z = torch.randn(2, cfg["vae"]["latent_channels"], cfg["data"]["resolution"], cfg["data"]["resolution"])
    g_obs = torch.rand_like(z)
    t = torch.zeros(z.shape[0], dtype=torch.long)

    result = guidance.correct(z, g_obs, t, diffusion, latent_green)
    assert set(result.keys()) == {
        "z_corrected",
        "loss_mean",
        "loss_per_sample",
        "grad_norm",
        "delta_norm",
        "step_scale",
        "risk_score",
        "query_requested",
        "applied",
    }
    assert result["z_corrected"].shape == z.shape
    assert result["loss_per_sample"].shape == (z.shape[0],)
    assert result["grad_norm"].shape == (z.shape[0],)
    assert result["delta_norm"].shape == (z.shape[0],)
    assert result["step_scale"].shape == (z.shape[0],)
    assert result["risk_score"].shape == (z.shape[0],)
    assert result["query_requested"].shape == (z.shape[0],)
    assert torch.isfinite(result["z_corrected"]).all()
    assert torch.isfinite(result["loss_per_sample"]).all()
    assert torch.isfinite(result["grad_norm"]).all()
    assert torch.isfinite(result["delta_norm"]).all()
    assert torch.all(result["grad_norm"] >= 0)
    assert torch.all(result["delta_norm"] >= 0)

    z_applied = guidance.apply(z, g_obs, t, diffusion, latent_green)
    assert isinstance(z_applied, torch.Tensor)
    assert z_applied.shape == z.shape


def test_latent_guidance_schedule_and_multistep():
    cfg = _cfg()
    diffusion = _DummyDiffusion(cfg["diffusion"]["T"])
    base = LatentGuidance(cfg)
    t = torch.zeros(2, dtype=torch.long)

    legacy_cfg = _cfg()
    legacy_cfg["guidance"]["manifold"]["step_schedule"] = "legacy"
    legacy_cfg["guidance"]["lambda"]["schedule"] = "constant"
    legacy = LatentGuidance(legacy_cfg)

    sigma2_cfg = _cfg()
    sigma2_cfg["guidance"]["manifold"]["step_schedule"] = "sigma2"
    sigma2 = LatentGuidance(sigma2_cfg)

    late_cfg = _cfg()
    late_cfg["guidance"]["manifold"]["step_schedule"] = "late_strong"
    late = LatentGuidance(late_cfg)

    constant_scale = base._compute_step_scale(t, diffusion)
    legacy_scale = legacy._compute_step_scale(t, diffusion)
    sigma2_scale = sigma2._compute_step_scale(t, diffusion)
    late_scale = late._compute_step_scale(t, diffusion)

    assert torch.allclose(legacy_scale, sigma2_scale)
    assert not torch.allclose(constant_scale, sigma2_scale)
    assert not torch.allclose(late_scale, constant_scale)

    z = torch.randn(2, cfg["vae"]["latent_channels"], cfg["data"]["resolution"], cfg["data"]["resolution"])
    g_obs = torch.rand_like(z)
    t_batch = torch.zeros(z.shape[0], dtype=torch.long)
    latent_green = _DummyLatentGreen()

    one_step = LatentGuidance(_cfg()).correct(z, g_obs, t_batch, diffusion, latent_green)["z_corrected"]
    multi_cfg = _cfg()
    multi_cfg["guidance"]["manifold"]["grad_steps_per_iter"] = 2
    multi_step = LatentGuidance(multi_cfg).correct(z, g_obs, t_batch, diffusion, latent_green)["z_corrected"]
    assert not torch.allclose(one_step, multi_step)


def test_teacher_sampler_trace_and_budget_hooks():
    torch.manual_seed(0)
    cfg = _cfg()
    cfg["guidance"]["budget_hooks"]["enabled"] = True
    cfg["guidance"]["budget_hooks"]["threshold"] = 0.0
    cfg["guidance"]["budget_hooks"]["max_queries_per_sample"] = 1
    guidance = LatentGuidance(cfg)
    teacher = TeacherSampler(
        cfg,
        diffusion_model=_DummyDiffusion(cfg["diffusion"]["T"]),
        vae=_DummyVAE(),
        condition_encoder=torch.nn.Identity(),
        latent_green=_DummyLatentGreen(),
        guidance=guidance,
    )
    g_obs = torch.rand(2, cfg["data"]["K"], cfg["data"]["resolution"], cfg["data"]["resolution"])

    v_hat = teacher.sample(g_obs)
    assert isinstance(v_hat, torch.Tensor)
    assert v_hat.shape == (g_obs.shape[0], 1, g_obs.shape[-2], g_obs.shape[-1])

    v_hat_trace, trace = teacher.sample(g_obs, return_trace=True)
    assert v_hat_trace.shape == v_hat.shape
    assert trace["num_steps"] == len(trace["steps"])
    assert "summary" in trace
    assert trace["summary"]["num_corrections"] > 0
    assert any(step["correction_applied"] for step in trace["steps"])
    assert all("risk_score_mean" in step for step in trace["steps"])
    assert all("query_requested_frac" in step for step in trace["steps"])
    assert all("teacher_reject_frac" in step for step in trace["steps"])
    assert teacher.last_trace is trace


def test_teacher_sampler_budget_non_dry_run_without_restart_raises():
    cfg = _cfg()
    cfg["guidance"]["budget_hooks"]["enabled"] = True
    cfg["guidance"]["budget_hooks"]["threshold"] = 0.0
    cfg["guidance"]["budget_hooks"]["max_queries_per_sample"] = 1
    cfg["guidance"]["budget_hooks"]["dry_run"] = False
    teacher = TeacherSampler(
        cfg,
        diffusion_model=_DummyDiffusion(cfg["diffusion"]["T"]),
        vae=_DummyVAE(),
        condition_encoder=torch.nn.Identity(),
        latent_green=_DummyLatentGreen(),
        guidance=LatentGuidance(cfg),
    )
    g_obs = torch.rand(1, cfg["data"]["K"], cfg["data"]["resolution"], cfg["data"]["resolution"])
    with pytest.raises(RuntimeError):
        teacher.sample(g_obs)


def test_teacher_sampler_budget_non_dry_run_with_restart_applies_rejection():
    cfg = _cfg()
    cfg["guidance"]["budget_hooks"]["enabled"] = True
    cfg["guidance"]["budget_hooks"]["threshold"] = 0.0
    cfg["guidance"]["budget_hooks"]["max_queries_per_sample"] = 1
    cfg["guidance"]["budget_hooks"]["dry_run"] = False
    teacher = TeacherSampler(
        cfg,
        diffusion_model=_DummyDiffusion(cfg["diffusion"]["T"]),
        vae=_DummyVAE(),
        condition_encoder=torch.nn.Identity(),
        latent_green=_DummyLatentGreen(),
        guidance=LatentGuidance(cfg),
    )
    teacher.restart = _DummyRestart()
    g_obs = torch.rand(2, cfg["data"]["K"], cfg["data"]["resolution"], cfg["data"]["resolution"])

    _v_hat, trace = teacher.sample(g_obs, return_trace=True)
    assert trace["summary"]["num_query_requested"] > 0
    assert trace["summary"]["num_teacher_rejected"] > 0
    assert any(step["teacher_reject_frac"] > 0.0 for step in trace["steps"])


def test_teacher_sampler_masked_step_preserves_unmasked_samples():
    cfg = _cfg()
    teacher = TeacherSampler(
        cfg,
        diffusion_model=_DummyDiffusion(cfg["diffusion"]["T"]),
        vae=_DummyVAE(),
        condition_encoder=torch.nn.Identity(),
        latent_green=_DummyLatentGreen(),
        guidance=LatentGuidance(cfg),
    )
    z = torch.zeros(2, cfg["vae"]["latent_channels"], cfg["data"]["resolution"], cfg["data"]["resolution"])
    g_obs = torch.rand(2, cfg["data"]["K"], cfg["data"]["resolution"], cfg["data"]["resolution"])
    mask = torch.tensor([True, False])
    trace = teacher._init_trace()
    out = teacher._sample_steps(z.clone(), g_obs, torch.tensor([1]), 0.0, mask, trace)
    assert torch.allclose(out[1], z[1])
    assert trace["steps"][0]["masked_update"] is True
