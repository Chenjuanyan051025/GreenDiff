from typing import Dict, Any, Optional, Union
import copy

import torch

from gd.models.vae import VAE
from gd.models.diffusion import LatentDiffusion
from gd.models.condition_encoder import ConditionEncoder
from gd.models.latent_green import LatentGreen
from gd.guidance.latent_guidance import LatentGuidance
from gd.guidance.restart import RestartSampler
from gd.utils.ldos_transform import force_linear_ldos_mode
from gd.utils.obs_layout import g_obs_to_model_view


class TeacherSampler:
    """
    High-quality sampling using the teacher diffusion model.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        diffusion_model: Optional[torch.nn.Module] = None,
        vae: Optional[torch.nn.Module] = None,
        condition_encoder: Optional[torch.nn.Module] = None,
        latent_green: Optional[torch.nn.Module] = None,
        guidance: Optional[LatentGuidance] = None,
    ):
        self.config = copy.deepcopy(config)
        force_linear_ldos_mode(self.config, verbose=False, context="teacher_sampler")
        self.diffusion = diffusion_model or LatentDiffusion(self.config)
        self.vae = vae or VAE(self.config)
        self.condition_encoder = condition_encoder or ConditionEncoder(self.config)
        self.latent_green = latent_green or LatentGreen(self.config)
        self.guidance = guidance or LatentGuidance(self.config)
        self.diff_cfg = self.config["diffusion"]
        self.guidance_cfg = self.config["guidance"]
        self.validation_cfg = self.config["validation"]
        self.restart = None
        self.last_trace: Optional[Dict[str, Any]] = None
        if self.validation_cfg["enabled"] and self.validation_cfg["kpm_check"]["enabled"]:
            try:
                from gd.data.kpm_forward import KPMForward

                __import__("kwant")
                kpm = KPMForward(
                    {
                        "kpm": self.config["physics"]["kpm"],
                        "hamiltonian": self.config["physics"]["hamiltonian"],
                        "data": self.config.get("data", {}),
                        "rng_seed": self.config["project"]["seed"],
                    }
                )
                self.restart = RestartSampler(self.config, kpm, self.vae)
            except ImportError:
                print("Warning: 'kwant' not found. Disabling KPM check and Restart Guidance.")
                self.validation_cfg["kpm_check"]["enabled"] = False
                self.validation_cfg["restart"]["enabled"] = False
                self.restart = None
        self.unscale_factor: Union[float, torch.Tensor] = 1.0

    def sample(self, g_obs: torch.Tensor, return_trace: bool = False):
        """
        Generates a sample given a condition.
        Args:
            g_obs: Measurement tensor of shape (B, C, H, W) or canonical `(B,K,2,H,W)`.
            return_trace: Whether to return step-wise sampling statistics.
        Returns:
            torch.Tensor or Tuple[torch.Tensor, Dict[str, Any]]: Reconstructed potential, optionally with trace.
        """
        device = g_obs.device
        if g_obs.dim() == 5:
            g_obs = g_obs_to_model_view(g_obs, self.config.get("data", {}))
        B = g_obs.shape[0]
        H, W = g_obs.shape[-2:]
        C = self.config["vae"]["latent_channels"]
        h = H // self.config["vae"]["latent_downsample"]
        w = W // self.config["vae"]["latent_downsample"]
        z = torch.randn(B, C, h, w, device=device)
        trace = self._init_trace(B, device)

        T = self.diff_cfg["T"]
        steps = self.diff_cfg["sampler"]["steps"]
        eta = self.diff_cfg["sampler"]["eta"]
        if steps <= 1:
            timesteps = torch.tensor([max(T - 1, 0)], device=device, dtype=torch.long)
        else:
            timesteps = torch.linspace(max(T - 1, 0), 1, steps, device=device).long()
        z = self._sample_steps(z, g_obs, timesteps, eta, None, trace)
        if self.restart is not None and self.validation_cfg["restart"]["enabled"]:
            max_restarts = self.validation_cfg["restart"]["max_restarts"]
            t_restart = self.validation_cfg["restart"]["t_restart"]
            for _ in range(max_restarts):
                with torch.no_grad():
                    _delta, _epsilon, mask = self.restart.check(z, g_obs)
                if not mask.any():
                    break
                trace["num_restart_passes"] += 1
                z_restart = self.restart.add_restart_noise(z, self.diffusion)
                z = torch.where(mask[:, None, None, None], z_restart, z)
                steps_restart = max(2, int(steps * t_restart / T))
                restart_timesteps = torch.linspace(max(t_restart, 1), 1, steps_restart, device=device).long()
                z = self._sample_steps(z, g_obs, restart_timesteps, eta, mask, trace)

        unscale = self.unscale_factor
        if isinstance(unscale, torch.Tensor):
            unscale = unscale.to(device=z.device, dtype=z.dtype)
            if unscale.numel() == 1:
                if abs(float(unscale.item()) - 1.0) > 1e-6:
                    z = z * unscale
            else:
                z = z * unscale
        else:
            if abs(float(unscale) - 1.0) > 1e-6:
                z = z * float(unscale)

        with torch.no_grad():
            v_hat = self.vae.decode(z)
        self._finalize_trace(trace)
        self.last_trace = trace
        if return_trace:
            return v_hat, trace
        return v_hat

    def _sample_steps(
        self,
        z: torch.Tensor,
        g_obs: torch.Tensor,
        timesteps: torch.Tensor,
        eta: float,
        mask: Optional[torch.Tensor],
        trace: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        B = z.shape[0]
        for idx, t in enumerate(timesteps):
            t_batch = torch.full((B,), int(t.item()), device=z.device, dtype=torch.long)
            if idx + 1 < len(timesteps):
                t_prev_val = int(timesteps[idx + 1].item())
            else:
                t_prev_val = 0
            t_prev_batch = torch.full((B,), t_prev_val, device=z.device, dtype=torch.long)
            with torch.no_grad():
                z_proposal = self.diffusion.step(z, t_batch, g_obs, eta, t_prev=t_prev_batch)

            correction_result = None
            z_updated = z_proposal
            if self.guidance.is_correction_active_for_step(int(t.item())):
                with torch.enable_grad():
                    correction_result = self.guidance.correct(z_proposal, g_obs, t_batch, self.diffusion, self.latent_green)
                z_updated = correction_result["z_corrected"]
                z_updated, query_requested, teacher_rejected = self._handle_budget_query_request(
                    correction_result, z_updated, g_obs, t_batch, trace
                )
                correction_result["query_requested"] = query_requested
                correction_result["teacher_rejected"] = teacher_rejected
                correction_result["teacher_applied"] = query_requested

            if mask is not None:
                z = torch.where(mask[:, None, None, None], z_updated, z)
            else:
                z = z_updated
            if trace is not None:
                self._append_trace_step(trace, int(t.item()), correction_result, mask, B)
        return z

    def _handle_budget_query_request(
        self,
        correction_result: Optional[Dict[str, Any]],
        z_current: torch.Tensor,
        g_obs: torch.Tensor,
        t_batch: torch.Tensor,
        trace: Optional[Dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _ = (z_current, g_obs, t_batch)
        if correction_result is None:
            empty = torch.zeros(z_current.shape[0], device=z_current.device, dtype=torch.bool)
            return z_current, empty, empty
        budget_cfg = self.guidance_cfg.get("budget_hooks", {})
        if not bool(budget_cfg.get("enabled", False)):
            empty = torch.zeros_like(correction_result["query_requested"], dtype=torch.bool)
            return z_current, empty, empty

        query_mask = correction_result["query_requested"]
        max_queries = int(budget_cfg.get("max_queries_per_sample", 0) or 0)
        if max_queries <= 0:
            empty = torch.zeros_like(query_mask, dtype=torch.bool)
            return z_current, empty, empty
        if trace is not None:
            query_counts = trace.get("_query_counts")
            if query_counts is not None and query_counts.device != query_mask.device:
                query_counts = query_counts.to(query_mask.device)
                trace["_query_counts"] = query_counts
            if query_counts is not None:
                query_mask = query_mask & (query_counts < max_queries)
        if not bool(query_mask.any().item()):
            empty = torch.zeros_like(query_mask, dtype=torch.bool)
            return z_current, empty, empty
        if trace is not None and "_query_counts" in trace:
            trace["_query_counts"][query_mask] += 1

        if bool(budget_cfg.get("dry_run", True)):
            empty = torch.zeros_like(query_mask, dtype=torch.bool)
            return z_current, query_mask, empty

        if self.restart is None:
            raise RuntimeError(
                "Real budget hooks require an available high-fidelity restart checker. "
                "Enable validation.kpm_check/restart and ensure dependencies are installed."
            )

        with torch.no_grad():
            _delta, _epsilon, restart_mask = self.restart.check(z_current, g_obs)
        teacher_rejected = query_mask & restart_mask
        if not bool(teacher_rejected.any().item()):
            return z_current, query_mask, teacher_rejected
        z_restarted = self.restart.add_restart_noise(z_current, self.diffusion)
        z_next = torch.where(teacher_rejected[:, None, None, None], z_restarted, z_current)
        return z_next, query_mask, teacher_rejected

    def _init_trace(self, batch_size: int = 0, device: Optional[torch.device] = None) -> Dict[str, Any]:
        return {
            "num_steps": 0,
            "num_restart_passes": 0,
            "steps": [],
            "summary": {},
            "_query_counts": torch.zeros(batch_size, device=device, dtype=torch.long) if batch_size > 0 else None,
            "_accum": {
                "num_corrections": 0,
                "sum_loss": 0.0,
                "sum_grad_norm": 0.0,
                "sum_delta_norm": 0.0,
                "sum_risk_score": 0.0,
                "num_query_requested": 0,
                "num_teacher_rejected": 0,
            },
        }

    def _append_trace_step(
        self,
        trace: Dict[str, Any],
        t_value: int,
        correction_result: Optional[Dict[str, Any]],
        mask: Optional[torch.Tensor],
        batch_size: int,
    ) -> None:
        active_mask = mask if mask is not None else torch.ones(batch_size, dtype=torch.bool)
        if correction_result is not None and active_mask.device != correction_result["loss_per_sample"].device:
            active_mask = active_mask.to(correction_result["loss_per_sample"].device)
        any_active = bool(active_mask.any().item()) if isinstance(active_mask, torch.Tensor) else bool(active_mask.any())

        if correction_result is None or not bool(correction_result.get("applied", False)) or not any_active:
            step = {
                "t": int(t_value),
                "proposal_applied": True,
                "correction_applied": False,
                "loss_mean": 0.0,
                "grad_norm_mean": 0.0,
                "delta_norm_mean": 0.0,
                "step_scale_mean": 0.0,
                "risk_score_mean": 0.0,
                "query_requested_frac": 0.0,
                "teacher_reject_frac": 0.0,
                "masked_update": mask is not None,
            }
            trace["steps"].append(step)
            trace["num_steps"] += 1
            return

        loss_mean = self._masked_mean(correction_result["loss_per_sample"], active_mask)
        grad_norm_mean = self._masked_mean(correction_result["grad_norm"], active_mask)
        delta_norm_mean = self._masked_mean(correction_result["delta_norm"], active_mask)
        step_scale_mean = self._masked_mean(correction_result["step_scale"], active_mask)
        if bool(self.guidance_cfg.get("risk", {}).get("emit_trace", True)):
            risk_score_mean = self._masked_mean(correction_result["risk_score"], active_mask)
        else:
            risk_score_mean = 0.0
        query_requested_frac = float(correction_result["query_requested"][active_mask].float().mean().item())
        teacher_reject_frac = float(
            correction_result.get("teacher_rejected", torch.zeros_like(correction_result["query_requested"]))[active_mask]
            .float()
            .mean()
            .item()
        )
        query_count = int(correction_result["query_requested"][active_mask].sum().item())
        reject_count = int(
            correction_result.get("teacher_rejected", torch.zeros_like(correction_result["query_requested"]))[active_mask]
            .sum()
            .item()
        )

        step = {
            "t": int(t_value),
            "proposal_applied": True,
            "correction_applied": True,
            "loss_mean": loss_mean,
            "grad_norm_mean": grad_norm_mean,
            "delta_norm_mean": delta_norm_mean,
            "step_scale_mean": step_scale_mean,
            "risk_score_mean": risk_score_mean,
            "query_requested_frac": query_requested_frac,
            "teacher_reject_frac": teacher_reject_frac,
            "masked_update": mask is not None,
        }
        trace["steps"].append(step)
        trace["num_steps"] += 1
        accum = trace["_accum"]
        accum["num_corrections"] += 1
        accum["sum_loss"] += loss_mean
        accum["sum_grad_norm"] += grad_norm_mean
        accum["sum_delta_norm"] += delta_norm_mean
        accum["sum_risk_score"] += risk_score_mean
        accum["num_query_requested"] += query_count
        accum["num_teacher_rejected"] += reject_count

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> float:
        if values.numel() == 0:
            return 0.0
        if mask.device != values.device:
            mask = mask.to(values.device)
        if not bool(mask.any().item()):
            return 0.0
        return float(values[mask].mean().item())

    def _finalize_trace(self, trace: Dict[str, Any]) -> None:
        accum = trace.pop("_accum", {})
        trace.pop("_query_counts", None)
        num_corrections = int(accum.get("num_corrections", 0))
        denom = float(max(1, num_corrections))
        trace["summary"] = {
            "num_corrections": num_corrections,
            "mean_loss": float(accum.get("sum_loss", 0.0)) / denom if num_corrections > 0 else 0.0,
            "mean_grad_norm": float(accum.get("sum_grad_norm", 0.0)) / denom if num_corrections > 0 else 0.0,
            "mean_delta_norm": float(accum.get("sum_delta_norm", 0.0)) / denom if num_corrections > 0 else 0.0,
            "mean_risk_score": float(accum.get("sum_risk_score", 0.0)) / denom if num_corrections > 0 else 0.0,
            "num_query_requested": int(accum.get("num_query_requested", 0)),
            "num_teacher_rejected": int(accum.get("num_teacher_rejected", 0)),
        }
