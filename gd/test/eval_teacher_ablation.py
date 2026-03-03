import argparse
import glob
import json
import os
import re
import sys

import torch
from torch.utils.data import DataLoader


def _ensure_project_root():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


PROJECT_ROOT = _ensure_project_root()

from gd.data.dataset import GFDataset  # noqa: E402
from gd.inference.teacher_sampler import TeacherSampler  # noqa: E402
from gd.utils.config_utils import get_latest_checkpoint_dir, load_config, resolve_config_paths  # noqa: E402
from gd.utils.ldos_transform import force_linear_ldos_mode, ldos_obs_from_linear  # noqa: E402
from gd.utils.obs_layout import g_obs_to_model_view  # noqa: E402


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def normalize_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_k = k[len("_orig_mod."):]
        elif k.startswith("module."):
            new_k = k[len("module."):]
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict


def _ckpt_step(path: str):
    base = os.path.basename(path)
    match = re.search(r"_step_(\d+)\.pt$", base)
    if not match:
        return None
    return int(match.group(1))


def find_latest_ckpt(runs_root, current_ckpt_dir, pattern, prefer_ckpt_dir=None):
    if prefer_ckpt_dir and os.path.exists(prefer_ckpt_dir):
        ckpts = [p for p in glob.glob(os.path.join(prefer_ckpt_dir, pattern)) if _ckpt_step(p) is not None]
        ckpts = sorted(ckpts, key=_ckpt_step)
        if ckpts:
            return ckpts[-1]
    latest_dir = get_latest_checkpoint_dir(runs_root, require_pattern=pattern)
    if latest_dir:
        ckpts = [p for p in glob.glob(os.path.join(latest_dir, pattern)) if _ckpt_step(p) is not None]
        ckpts = sorted(ckpts, key=_ckpt_step)
        if ckpts:
            print(f"Found {pattern} in latest run: {latest_dir}")
            return ckpts[-1]
    if os.path.exists(current_ckpt_dir):
        ckpts = [p for p in glob.glob(os.path.join(current_ckpt_dir, pattern)) if _ckpt_step(p) is not None]
        ckpts = sorted(ckpts, key=_ckpt_step)
        if ckpts:
            return ckpts[-1]
    return None


def _load_teacher_checkpoints(teacher: TeacherSampler, config: dict, current_ckpt_dir: str, runs_root: str, device: torch.device):
    diff_ckpt = find_latest_ckpt(runs_root, current_ckpt_dir, "diffusion_step_*.pt")
    if diff_ckpt:
        print(f"Loading Diffusion from {diff_ckpt}")
        state = normalize_state_dict(torch.load(diff_ckpt, map_location=device, weights_only=True))
        ema_path = diff_ckpt.replace(".pt", "_ema.pt")
        if os.path.exists(ema_path):
            print(f"Loading Diffusion EMA from {ema_path}")
            state = normalize_state_dict(torch.load(ema_path, map_location=device, weights_only=True))
        teacher.diffusion.load_state_dict(state)
    else:
        print("Warning: No Diffusion checkpoint found.")

    prefer_ckpt_dir = os.path.dirname(diff_ckpt) if diff_ckpt else None
    vae_ckpt = find_latest_ckpt(runs_root, current_ckpt_dir, "vae_step_*.pt", prefer_ckpt_dir=prefer_ckpt_dir)
    if vae_ckpt:
        print(f"Loading VAE from {vae_ckpt}")
        teacher.vae.load_state_dict(normalize_state_dict(torch.load(vae_ckpt, map_location=device, weights_only=True)))
    else:
        print("Warning: No VAE checkpoint found.")

    lg_ckpt = find_latest_ckpt(runs_root, current_ckpt_dir, "latent_green_step_*.pt", prefer_ckpt_dir=prefer_ckpt_dir)
    if lg_ckpt:
        print(f"Loading Latent Green from {lg_ckpt}")
        teacher.latent_green.load_state_dict(normalize_state_dict(torch.load(lg_ckpt, map_location=device, weights_only=True)))
    else:
        print("Warning: No Latent Green checkpoint found.")


def _fmt(x, spec=".4f"):
    return format(float(x), spec)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TeacherSampler for ablation runs")
    parser.add_argument("--config", default="gd/configs/default.yaml", help="Path to config file")
    parser.add_argument("--ckpt_dir", help="Checkpoint directory to load weights from")
    parser.add_argument("--num_batches", type=int, default=20, help="Number of validation batches to evaluate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--json_out", help="Optional path to write structured metrics as JSON")
    args = parser.parse_args()

    base_config = load_config(args.config)
    config = base_config
    if args.ckpt_dir:
        ckpt_dir = args.ckpt_dir
        run_dir = os.path.dirname(ckpt_dir)
        runs_root = os.path.dirname(run_dir)
        run_config_path = os.path.join(run_dir, "config.yaml")
        if os.path.exists(run_config_path):
            config = load_config(run_config_path)
        config["paths"]["checkpoints"] = ckpt_dir
        config["paths"]["workdir"] = run_dir
        config["paths"]["runs_root"] = runs_root
        config = resolve_config_paths(config)

    force_linear_ldos_mode(config, verbose=True, context="eval_teacher_ablation")
    device = torch.device(config["project"]["device"])
    runs_root = config.get("paths", {}).get("runs_root", config["paths"]["workdir"])
    current_ckpt_dir = config["paths"]["checkpoints"]

    dataset = GFDataset(config, split="val")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    teacher = TeacherSampler(config)
    _load_teacher_checkpoints(teacher, config, current_ckpt_dir, runs_root, device)
    teacher.condition_encoder = teacher.diffusion.condition_encoder
    teacher.diffusion.to(device).eval()
    teacher.vae.to(device).eval()
    teacher.condition_encoder.to(device).eval()
    teacher.latent_green.to(device).eval()

    total_samples = 0
    sum_mae = 0.0
    sum_mse = 0.0
    sum_rel_l2 = 0.0
    sum_proxy_rmse = 0.0
    sum_num_corrections = 0.0
    sum_num_queries = 0.0
    sum_num_rejected = 0.0
    sum_mean_risk = 0.0
    counted_batches = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.num_batches:
            break
        g_obs = batch["g_obs"].to(device)
        v_true = batch["V"].to(device).unsqueeze(1)

        with torch.no_grad():
            v_pred, trace = teacher.sample(g_obs, return_trace=True)

            diff = v_pred - v_true
            mse_per = (diff ** 2).mean(dim=(1, 2, 3))
            mae_per = diff.abs().mean(dim=(1, 2, 3))
            rel_l2_per = torch.sqrt((diff ** 2).sum(dim=(1, 2, 3))) / torch.sqrt((v_true ** 2).sum(dim=(1, 2, 3))).clamp_min(1.0e-8)

            z_pred, mu_pred, _ = teacher.vae.encode(v_pred)
            if mu_pred is not None:
                z_pred = mu_pred
            t_zeros = torch.zeros((z_pred.shape[0],), dtype=torch.long, device=device)
            g_pred_lin = teacher.latent_green(z_pred, t_zeros)
            g_pred_obs = ldos_obs_from_linear(g_pred_lin, config.get("data", {}))
            g_obs_model = g_obs_to_model_view(g_obs, config.get("data", {})) if g_obs.dim() == 5 else g_obs
            proxy_rmse_per = torch.sqrt(((g_pred_obs - g_obs_model) ** 2).mean(dim=(1, 2, 3)))

        batch_n = int(v_true.shape[0])
        total_samples += batch_n
        counted_batches += 1
        sum_mse += float(mse_per.sum().item())
        sum_mae += float(mae_per.sum().item())
        sum_rel_l2 += float(rel_l2_per.sum().item())
        sum_proxy_rmse += float(proxy_rmse_per.sum().item())
        summary = trace["summary"]
        sum_num_corrections += float(summary.get("num_corrections", 0))
        sum_num_queries += float(summary.get("num_query_requested", 0))
        sum_num_rejected += float(summary.get("num_teacher_rejected", 0))
        sum_mean_risk += float(summary.get("mean_risk_score", 0.0))

    if total_samples <= 0:
        raise RuntimeError("No validation samples were evaluated.")

    mean_mae = sum_mae / total_samples
    mean_mse = sum_mse / total_samples
    mean_rel_l2 = sum_rel_l2 / total_samples
    mean_proxy_rmse = sum_proxy_rmse / total_samples
    mean_num_corrections = sum_num_corrections / max(1, counted_batches)
    mean_num_queries = sum_num_queries / max(1, counted_batches)
    mean_num_rejected = sum_num_rejected / max(1, counted_batches)
    mean_risk = sum_mean_risk / max(1, counted_batches)

    print("\n" + "=" * 60)
    print("Teacher Ablation Evaluation")
    print("=" * 60)
    print(f"{'Samples':<28}: {total_samples}")
    print(f"{'Batches':<28}: {counted_batches}")
    print(f"{'MAE(V)':<28}: {_fmt(mean_mae)}")
    print(f"{'MSE(V)':<28}: {_fmt(mean_mse)}")
    print(f"{'Rel L2(V)':<28}: {_fmt(mean_rel_l2)}")
    print(f"{'Proxy Obs RMSE':<28}: {_fmt(mean_proxy_rmse)}")
    print(f"{'Mean #Corrections / batch':<28}: {_fmt(mean_num_corrections)}")
    print(f"{'Mean #Queries / batch':<28}: {_fmt(mean_num_queries)}")
    print(f"{'Mean #Rejected / batch':<28}: {_fmt(mean_num_rejected)}")
    print(f"{'Mean Risk Score':<28}: {_fmt(mean_risk)}")
    print("=" * 60)

    if args.json_out:
        json_dir = os.path.dirname(os.path.abspath(args.json_out))
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)
        results = {
            "config": os.path.abspath(args.config),
            "ckpt_dir": os.path.abspath(args.ckpt_dir) if args.ckpt_dir else None,
            "samples": int(total_samples),
            "batches": int(counted_batches),
            "mae_v": float(mean_mae),
            "mse_v": float(mean_mse),
            "rel_l2_v": float(mean_rel_l2),
            "proxy_obs_rmse": float(mean_proxy_rmse),
            "mean_corrections_per_batch": float(mean_num_corrections),
            "mean_queries_per_batch": float(mean_num_queries),
            "mean_rejected_per_batch": float(mean_num_rejected),
            "mean_risk_score": float(mean_risk),
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Structured metrics written to {os.path.abspath(args.json_out)}")


if __name__ == "__main__":
    main()
