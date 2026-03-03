import argparse
import copy
import glob
import os
import subprocess
import sys
import tempfile

import yaml


def _ensure_project_root():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


PROJECT_ROOT = _ensure_project_root()

from gd.utils.config_utils import load_config  # noqa: E402


BACKBONES = ("cnn", "fno", "hybrid_fno")
PRESETS = ("base", "main_hq")


def _apply_preset(cfg: dict, preset: str) -> dict:
    preset = str(preset).lower()
    if preset == "base":
        return cfg
    if preset != "main_hq":
        raise ValueError(f"Unsupported preset={preset!r}; expected one of {PRESETS}.")

    cfg = copy.deepcopy(cfg)
    lg_model = cfg.setdefault("latent_green", {}).setdefault("model", {})
    # Main experiment high-quality preset: increase width/depth first, then modes.
    lg_model["base_channels"] = 96
    lg_model["hidden_channels"] = 96
    lg_model["num_res_blocks"] = 3
    lg_model["fno_layers"] = 6
    lg_model["fno_modes_x"] = 16
    lg_model["fno_modes_y"] = 16
    lg_model["local_branch_channels"] = 96
    lg_model["local_branch_depth"] = 2
    lg_model["use_coord_grid"] = True
    lg_model["pointwise_skip"] = True
    lg_model["norm_type"] = "groupnorm"
    lg_model["spectral_dropout"] = 0.0
    return cfg


def _write_temp_config(base_cfg: dict, backbone: str, tmp_dir: str) -> str:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("latent_green", {}).setdefault("model", {})
    cfg["latent_green"]["model"]["backbone"] = backbone
    path = os.path.join(tmp_dir, f"latent_green_{backbone}.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return path


def _cfg_get_backbone(cfg: dict) -> str:
    return str(cfg.get("latent_green", {}).get("model", {}).get("backbone", "cnn")).lower()


def _latent_green_signature(cfg: dict) -> tuple:
    lg_model = cfg.get("latent_green", {}).get("model", {})
    data_cfg = cfg.get("data", {})
    vae_cfg = cfg.get("vae", {})
    backbone = str(lg_model.get("backbone", "cnn")).lower()
    base_channels = int(lg_model.get("base_channels", 64))
    hidden_channels = int(lg_model.get("hidden_channels", base_channels))
    signature = [
        backbone,
        int(data_cfg.get("resolution", 0)),
        int(data_cfg.get("K", 0)),
        bool(data_cfg.get("sublattice_resolved_ldos", False)),
        int(vae_cfg.get("latent_downsample", 0)),
        int(vae_cfg.get("latent_channels", 0)),
        base_channels,
        hidden_channels,
        int(lg_model.get("time_embed_dim", 0)),
        bool(lg_model.get("use_timestep", True)),
    ]
    if backbone == "cnn":
        signature.extend(
            [
                int(lg_model.get("num_res_blocks", 0)),
                float(lg_model.get("dropout", 0.0)),
            ]
        )
    else:
        signature.extend(
            [
                int(lg_model.get("fno_layers", 0)),
                int(lg_model.get("fno_modes_x", 0)),
                int(lg_model.get("fno_modes_y", 0)),
                bool(lg_model.get("use_coord_grid", True)),
                bool(lg_model.get("pointwise_skip", True)),
                str(lg_model.get("norm_type", "groupnorm")).lower(),
                int(lg_model.get("local_branch_channels", hidden_channels if backbone == "hybrid_fno" else 0))
                if backbone == "hybrid_fno"
                else 0,
                int(lg_model.get("local_branch_depth", 2)) if backbone == "hybrid_fno" else 0,
            ]
        )
    return tuple(signature)


def _load_yaml_quiet(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _find_matching_ckpt_dir(base_cfg: dict, backbone: str) -> str | None:
    runs_root = base_cfg.get("paths", {}).get("runs_root")
    if not runs_root or not os.path.isdir(runs_root):
        return None

    target_cfg = copy.deepcopy(base_cfg)
    target_cfg.setdefault("latent_green", {}).setdefault("model", {})
    target_cfg["latent_green"]["model"]["backbone"] = backbone
    target_sig = _latent_green_signature(target_cfg)

    candidates: list[str] = []
    for name in sorted(os.listdir(runs_root), reverse=True):
        run_dir = os.path.join(runs_root, name)
        if not os.path.isdir(run_dir):
            continue
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        if not os.path.isdir(ckpt_dir):
            continue
        if not glob.glob(os.path.join(ckpt_dir, "latent_green_step_*.pt")):
            continue
        run_cfg = _load_yaml_quiet(os.path.join(run_dir, "config.yaml"))
        if not run_cfg:
            continue
        run_backbone = _cfg_get_backbone(run_cfg)
        if run_backbone != backbone:
            continue
        if _latent_green_signature(run_cfg) != target_sig:
            continue
        candidates.append(ckpt_dir)

    return candidates[0] if candidates else None


def _build_command(args, cfg_path: str, ckpt_dir: str | None) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "gd.test.test_latent_green",
        "--config",
        cfg_path,
        "--num_batches",
        str(args.num_batches),
        "--batch_size",
        str(args.batch_size),
    ]
    if ckpt_dir:
        cmd.extend(["--ckpt_dir", ckpt_dir])
    return cmd


def _resolve_ckpt_dir(args, backbone: str, base_cfg: dict) -> tuple[str | None, str]:
    specific = getattr(args, f"ckpt_dir_{backbone}")
    if specific:
        return specific, "explicit"
    if args.ckpt_dir:
        return args.ckpt_dir, "shared"
    auto = _find_matching_ckpt_dir(base_cfg, backbone)
    if auto:
        return auto, "auto-match"
    return None, "no-match"


def main():
    parser = argparse.ArgumentParser(
        description="Run gd.test.test_latent_green sequentially for cnn/fno/hybrid_fno backbones."
    )
    parser.add_argument("--config", default="gd/configs/default.yaml", help="Base config path.")
    parser.add_argument(
        "--preset",
        default="base",
        choices=PRESETS,
        help="Apply a predefined override set before running the three backbone evaluations.",
    )
    parser.add_argument("--num_batches", type=int, default=50, help="Number of validation batches per backbone.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per backbone evaluation.")
    parser.add_argument(
        "--ckpt_dir",
        help="Common checkpoint directory fallback for all backbones. Prefer per-backbone dirs when weights differ.",
    )
    parser.add_argument("--ckpt_dir_cnn", help="Checkpoint directory for the cnn backbone.")
    parser.add_argument("--ckpt_dir_fno", help="Checkpoint directory for the fno backbone.")
    parser.add_argument("--ckpt_dir_hybrid_fno", help="Checkpoint directory for the hybrid_fno backbone.")
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately if one backbone evaluation fails.",
    )
    args = parser.parse_args()

    base_cfg = _apply_preset(load_config(args.config), args.preset)
    results: list[tuple[str, str]] = []

    with tempfile.TemporaryDirectory(prefix="latent_green_backbones_") as tmp_dir:
        for backbone in BACKBONES:
            cfg_path = _write_temp_config(base_cfg, backbone, tmp_dir)
            ckpt_dir, ckpt_source = _resolve_ckpt_dir(args, backbone, base_cfg)
            if ckpt_dir is None:
                print("\n" + "=" * 80)
                print(f"[Backbone] {backbone}")
                print(f"[Preset]   {args.preset}")
                print("[CKPT Dir] no exact matching checkpoint run found; skipped")
                print("=" * 80)
                results.append((backbone, "SKIPPED (no matching checkpoint run found)"))
                if args.stop_on_error:
                    break
                continue
            cmd = _build_command(args, cfg_path, ckpt_dir)
            lg_model = base_cfg["latent_green"]["model"]

            print("\n" + "=" * 80)
            print(f"[Backbone] {backbone}")
            print(f"[Preset]   {args.preset}")
            print(f"[Config]   {cfg_path}")
            print(f"[CKPT Dir] {ckpt_dir} ({ckpt_source})")
            print(
                "[Model]    "
                f"base={lg_model.get('base_channels')} hidden={lg_model.get('hidden_channels', lg_model.get('base_channels'))} "
                f"res={lg_model.get('num_res_blocks')} layers={lg_model.get('fno_layers')} "
                f"modes=({lg_model.get('fno_modes_x')},{lg_model.get('fno_modes_y')}) "
                f"local={lg_model.get('local_branch_channels')}"
            )
            print("=" * 80)

            completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
            results.append((backbone, "OK" if completed.returncode == 0 else f"FAILED (exit={completed.returncode})"))
            if completed.returncode != 0 and args.stop_on_error:
                break

    print("\n" + "=" * 80)
    print("Backbone Evaluation Summary")
    print("=" * 80)
    for backbone, status in results:
        print(f"{backbone:<12}: {status}")

    failed = [backbone for backbone, status in results if status.startswith("FAILED")]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
