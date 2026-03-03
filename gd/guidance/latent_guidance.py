from typing import Any, Dict, Tuple

import torch

from gd.utils.ldos_transform import ldos_obs_from_linear
from gd.utils.obs_layout import g_obs_to_model_view


class LatentGuidance:
    """
    Physics-based latent guidance using a frozen LatentGreen surrogate.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.guidance_cfg = config["guidance"]
        self.lambda_cfg = self.guidance_cfg["lambda"]
        self.loss_cfg = self.guidance_cfg["loss"]
        self.manifold_cfg = self.guidance_cfg.get("manifold", {})
        self.risk_cfg = self.guidance_cfg.get("risk", {})
        self.budget_cfg = self.guidance_cfg.get("budget_hooks", {})
        self.data_cfg = config.get("data", {})

    def get_start_step(self) -> int:
        value = self.manifold_cfg.get("start_step")
        if value is None:
            value = self.lambda_cfg.get("start_step", 0)
        return int(value)

    def is_correction_active_for_step(self, t_value: int) -> bool:
        if not bool(self.guidance_cfg.get("enabled", False)):
            return False
        if not bool(self.guidance_cfg.get("use_latent_green", True)):
            return False
        if not bool(self.manifold_cfg.get("enabled", True)):
            return False
        return int(t_value) <= self.get_start_step()

    def apply(
        self,
        z: torch.Tensor,
        g_obs: torch.Tensor,
        t: torch.Tensor,
        diffusion_model: torch.nn.Module,
        latent_green: torch.nn.Module,
    ) -> torch.Tensor:
        return self.correct(z, g_obs, t, diffusion_model, latent_green)["z_corrected"]

    def correct(
        self,
        z: torch.Tensor,
        g_obs: torch.Tensor,
        t: torch.Tensor,
        diffusion_model: torch.nn.Module,
        latent_green: torch.nn.Module,
    ) -> Dict[str, Any]:
        z_base = z.detach()
        batch = z_base.shape[0]
        zero_vec = torch.zeros(batch, device=z_base.device, dtype=z_base.dtype)
        false_mask = torch.zeros(batch, device=z_base.device, dtype=torch.bool)

        if not bool(self.manifold_cfg.get("enabled", True)):
            return self._noop_result(z_base, zero_vec, false_mask)

        mode = str(self.manifold_cfg.get("mode", "first_order_projection")).lower()
        if mode != "first_order_projection":
            raise ValueError(
                f"Unsupported guidance.manifold.mode={mode!r}; only 'first_order_projection' is implemented."
            )

        steps = self._grad_steps_per_iter()
        z_work = z_base
        last_loss_per_sample = zero_vec
        last_loss_mean = z_base.new_tensor(0.0)
        last_grad_norm = zero_vec
        last_step_scale = zero_vec

        for _ in range(steps):
            z_step = z_work.detach().requires_grad_(True)
            g_pred = latent_green(z_step, t)
            loss_per_sample = self._loss_per_sample(g_pred, g_obs)
            loss_mean = loss_per_sample.mean()
            grad = torch.autograd.grad(loss_mean, z_step, create_graph=False)[0]

            raw_grad_norm, grad_normalized = self._normalize_grad_per_sample(
                grad, float(self.manifold_cfg.get("eps", 1.0e-8))
            )
            if bool(self.manifold_cfg.get("normalize_grad", False)):
                grad_out = grad_normalized
            else:
                grad_out = self._restore_grad_scale(grad_normalized, raw_grad_norm)

            if self.manifold_cfg.get("grad_clip") is not None:
                grad_out, _ = self._clip_grad_per_sample(grad_out)

            step_scale = self._compute_step_scale(t, diffusion_model).to(device=z_base.device, dtype=z_base.dtype)
            view_shape = [step_scale.shape[0]] + [1] * (z_step.dim() - 1)
            delta_step = -step_scale.view(*view_shape) * grad_out

            if self.manifold_cfg.get("max_step_norm") is not None:
                delta_step, _ = self._clip_delta_per_sample(delta_step)

            z_work = (z_step + delta_step).detach()
            last_loss_per_sample = loss_per_sample.detach()
            last_loss_mean = loss_mean.detach()
            last_grad_norm = raw_grad_norm.detach()
            last_step_scale = step_scale.detach()

        delta = z_work - z_base
        _, delta_norm = self._clip_delta_per_sample(delta, limit=None)
        risk_score = self._compute_risk_score(last_loss_per_sample, last_grad_norm, delta_norm, z_base)
        query_requested = self._budget_query_mask(risk_score)
        return {
            "z_corrected": z_work.detach(),
            "loss_mean": last_loss_mean,
            "loss_per_sample": last_loss_per_sample,
            "grad_norm": last_grad_norm,
            "delta_norm": delta_norm,
            "step_scale": last_step_scale,
            "risk_score": risk_score,
            "query_requested": query_requested,
            "applied": True,
        }

    def _noop_result(self, z: torch.Tensor, zero_vec: torch.Tensor, false_mask: torch.Tensor) -> Dict[str, Any]:
        return {
            "z_corrected": z,
            "loss_mean": z.new_tensor(0.0),
            "loss_per_sample": zero_vec,
            "grad_norm": zero_vec,
            "delta_norm": zero_vec,
            "step_scale": zero_vec,
            "risk_score": zero_vec,
            "query_requested": false_mask,
            "applied": False,
        }

    def _loss(self, g_pred: torch.Tensor, g_obs: torch.Tensor) -> torch.Tensor:
        return self._loss_per_sample(g_pred, g_obs).mean()

    def _loss_per_sample(self, g_pred: torch.Tensor, g_obs: torch.Tensor) -> torch.Tensor:
        g_pred = ldos_obs_from_linear(g_pred, self.data_cfg)
        if g_obs.dim() == 5:
            g_obs = g_obs_to_model_view(g_obs, self.data_cfg)
        reduce_dims = tuple(range(1, g_pred.dim()))
        loss_type = str(self.loss_cfg.get("type", "obs_consistency")).lower()
        if loss_type == "obs_consistency":
            return ((g_pred - g_obs) ** 2).mean(dim=reduce_dims)
        if loss_type == "charbonnier":
            eps = float(self.loss_cfg.get("charbonnier_eps", 1.0e-3))
            return torch.sqrt((g_pred - g_obs) ** 2 + eps**2).mean(dim=reduce_dims)
        if loss_type == "huber":
            delta = float(self.loss_cfg.get("huber_delta", 0.1))
            diff = g_pred - g_obs
            return torch.where(
                torch.abs(diff) < delta,
                0.5 * diff**2,
                delta * (torch.abs(diff) - 0.5 * delta),
            ).mean(dim=reduce_dims)
        return ((g_pred - g_obs) ** 2).mean(dim=reduce_dims)

    def _resolve_step_schedule(self) -> str:
        schedule = self.manifold_cfg.get("step_schedule")
        if schedule is None:
            return "legacy"
        schedule = str(schedule).lower()
        if schedule not in {"legacy", "constant", "sigma2", "late_strong"}:
            raise ValueError(
                f"Unsupported guidance.manifold.step_schedule={schedule!r}; "
                "expected one of legacy/constant/sigma2/late_strong."
            )
        return schedule

    def _lambda0(self) -> float:
        value = self.manifold_cfg.get("lambda0")
        if value is None:
            value = self.lambda_cfg.get("lambda0", 1.0)
        return float(value)

    def _grad_steps_per_iter(self) -> int:
        value = self.manifold_cfg.get("grad_steps_per_iter")
        if value is None:
            value = self.lambda_cfg.get("grad_steps_per_iter", 1)
        return max(1, int(value))

    def _compute_step_scale(self, t: torch.Tensor, diffusion_model: torch.nn.Module) -> torch.Tensor:
        alpha, sigma = diffusion_model.get_alpha_sigma(t)
        alpha = alpha.to(dtype=torch.float32)
        sigma = sigma.to(dtype=torch.float32)
        schedule = self._resolve_step_schedule()
        lambda0 = self._lambda0()
        if schedule == "legacy":
            old_schedule = str(self.lambda_cfg.get("schedule", "sigma2")).lower()
            if old_schedule == "late_strong":
                return lambda0 * (1.0 - alpha) ** 2
            return lambda0 * (sigma**2)
        if schedule == "constant":
            return torch.full_like(alpha, lambda0)
        if schedule == "late_strong":
            return lambda0 * (1.0 - alpha) ** 2
        return lambda0 * (sigma**2)

    def _normalize_grad_per_sample(self, grad: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
        grad_flat = grad.reshape(grad.shape[0], -1)
        grad_norm = torch.linalg.vector_norm(grad_flat, ord=2, dim=1).clamp_min(float(eps))
        view_shape = [grad.shape[0]] + [1] * (grad.dim() - 1)
        return grad_norm, grad / grad_norm.view(*view_shape)

    def _restore_grad_scale(self, grad_normalized: torch.Tensor, grad_norm: torch.Tensor) -> torch.Tensor:
        view_shape = [grad_normalized.shape[0]] + [1] * (grad_normalized.dim() - 1)
        return grad_normalized * grad_norm.view(*view_shape)

    def _clip_grad_per_sample(self, grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grad_norm, grad_normalized = self._normalize_grad_per_sample(
            grad, float(self.manifold_cfg.get("eps", 1.0e-8))
        )
        limit = self.manifold_cfg.get("grad_clip")
        if limit is None:
            return grad, grad_norm
        scale = torch.clamp_max(
            torch.full_like(grad_norm, float(limit)) / grad_norm,
            1.0,
        )
        view_shape = [grad.shape[0]] + [1] * (grad.dim() - 1)
        return grad_normalized * (grad_norm * scale).view(*view_shape), grad_norm

    def _clip_delta_per_sample(self, delta: torch.Tensor, limit: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
        delta_norm, delta_normalized = self._normalize_grad_per_sample(
            delta, float(self.manifold_cfg.get("eps", 1.0e-8))
        )
        if limit is None:
            limit = self.manifold_cfg.get("max_step_norm")
        if limit is None:
            return delta, delta_norm
        scale = torch.clamp_max(
            torch.full_like(delta_norm, float(limit)) / delta_norm,
            1.0,
        )
        view_shape = [delta.shape[0]] + [1] * (delta.dim() - 1)
        return delta_normalized * (delta_norm * scale).view(*view_shape), delta_norm

    def _compute_risk_score(
        self,
        loss_per_sample: torch.Tensor,
        grad_norm: torch.Tensor,
        delta_norm: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(self.risk_cfg.get("enabled", True)):
            return torch.zeros_like(loss_per_sample)
        weights = self.risk_cfg.get("weights", {})
        grad_term = grad_norm
        delta_term = delta_norm
        if bool(self.risk_cfg.get("normalize_by_latent_std", True)):
            latent_std = z.reshape(z.shape[0], -1).std(dim=1).clamp_min(float(self.manifold_cfg.get("eps", 1.0e-8)))
            grad_term = grad_term / latent_std
            delta_term = delta_term / latent_std
        return (
            float(weights.get("phys_loss", 1.0)) * loss_per_sample
            + float(weights.get("grad_norm", 0.0)) * grad_term
            + float(weights.get("delta_norm", 0.0)) * delta_term
        )

    def _budget_query_mask(self, risk_score: torch.Tensor) -> torch.Tensor:
        if not bool(self.budget_cfg.get("enabled", False)):
            return torch.zeros_like(risk_score, dtype=torch.bool)
        policy = str(self.budget_cfg.get("policy", "threshold")).lower()
        if policy != "threshold":
            raise ValueError(
                f"Unsupported guidance.budget_hooks.policy={policy!r}; only 'threshold' is implemented in Phase 1."
            )
        threshold = self.budget_cfg.get("threshold")
        if threshold is None:
            return torch.zeros_like(risk_score, dtype=torch.bool)
        max_queries = int(self.budget_cfg.get("max_queries_per_sample", 0) or 0)
        if max_queries <= 0:
            return torch.zeros_like(risk_score, dtype=torch.bool)
        return risk_score >= float(threshold)
