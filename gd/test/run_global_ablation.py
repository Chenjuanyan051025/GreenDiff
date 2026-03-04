import argparse
import copy
import csv
import datetime as _dt
import json
import os
import subprocess
import sys
from typing import Any, Dict, List

import yaml


def _ensure_project_root():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


PROJECT_ROOT = _ensure_project_root()

from gd.utils.config_utils import load_config  # noqa: E402


def _set_path(cfg: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = cfg
    for key in parts[:-1]:
        cur = cur.setdefault(key, {})
    cur[parts[-1]] = value


def _with_overrides(base_cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    for path, value in overrides.items():
        _set_path(cfg, path, value)
    return cfg


def _prepare_experiment_paths(cfg: Dict[str, Any], suite_runs_dir: str, experiment_name: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    exp_run_dir = os.path.join(suite_runs_dir, experiment_name)
    _set_path(cfg, "paths.runs_root", suite_runs_dir)
    _set_path(cfg, "paths.workdir", exp_run_dir)
    _set_path(cfg, "paths.checkpoints", os.path.join(exp_run_dir, "checkpoints"))
    _set_path(cfg, "paths.logs", os.path.join(exp_run_dir, "logs"))
    return cfg


def _group_dir(parent: str, group: str) -> str:
    path = os.path.join(parent, group)
    os.makedirs(path, exist_ok=True)
    return path


def _build_specs() -> Dict[str, List[Dict[str, Any]]]:
    proxy_backbone = [
        {
            "name": "lg_backbone_cnn",
            "group": "proxy_backbone",
            "runner": "latent_green_eval",
            "description": "LatentGreen backbone ablation: cnn",
            "overrides": {
                "latent_green.model.backbone": "cnn",
            },
        },
        {
            "name": "lg_backbone_fno",
            "group": "proxy_backbone",
            "runner": "latent_green_eval",
            "description": "LatentGreen backbone ablation: fno",
            "overrides": {
                "latent_green.model.backbone": "fno",
            },
        },
        {
            "name": "lg_backbone_hybrid_fno",
            "group": "proxy_backbone",
            "runner": "latent_green_eval",
            "description": "LatentGreen backbone ablation: hybrid_fno",
            "overrides": {
                "latent_green.model.backbone": "hybrid_fno",
            },
        },
    ]

    proxy_losses = [
        {
            "name": "lg_full",
            "group": "proxy_losses",
            "runner": "latent_green_eval",
            "description": "Full LatentGreen surrogate",
            "overrides": {},
        },
        {
            "name": "lg_no_fft",
            "group": "proxy_losses",
            "runner": "latent_green_eval",
            "description": "Disable FFT loss",
            "overrides": {
                "latent_green.model.use_fft_loss": False,
                "latent_green.model.fft_loss_weight": 0.0,
            },
        },
        {
            "name": "lg_no_psd",
            "group": "proxy_losses",
            "runner": "latent_green_eval",
            "description": "Disable PSD loss",
            "overrides": {
                "latent_green.model.psd_loss_weight": 0.0,
            },
        },
        {
            "name": "lg_no_stats",
            "group": "proxy_losses",
            "runner": "latent_green_eval",
            "description": "Disable stats loss",
            "overrides": {
                "latent_green.model.stats_loss_weight": 0.0,
            },
        },
        {
            "name": "lg_no_multiscale",
            "group": "proxy_losses",
            "runner": "latent_green_eval",
            "description": "Disable multiscale loss",
            "overrides": {
                "latent_green.model.multiscale_loss_weight": 0.0,
            },
        },
        {
            "name": "lg_no_peak",
            "group": "proxy_losses",
            "runner": "latent_green_eval",
            "description": "Disable peak control",
            "overrides": {
                "latent_green.model.peak_control.enabled": False,
                "latent_green.model.peak_control.log_aux_weight": 0.0,
                "latent_green.model.peak_control.topk_loss_weight": 0.0,
                "latent_green.model.peak_control.peak_ratio_penalty_weight": 0.0,
            },
        },
        {
            "name": "lg_no_residual",
            "group": "proxy_losses",
            "runner": "latent_green_eval",
            "description": "Disable residual loss",
            "overrides": {
                "latent_green.model.residual_loss_weight": 0.0,
            },
        },
    ]

    teacher_inference = [
        {
            "name": "teacher_no_guidance",
            "group": "teacher_inference",
            "runner": "teacher_eval",
            "description": "Teacher sampler without guidance",
            "overrides": {
                "guidance.enabled": False,
                "validation.enabled": False,
                "validation.kpm_check.enabled": False,
                "validation.restart.enabled": False,
            },
        },
        {
            "name": "teacher_legacy",
            "group": "teacher_inference",
            "runner": "teacher_eval",
            "description": "Legacy heuristic guidance",
            "overrides": {
                "guidance.enabled": True,
                "guidance.manifold.enabled": True,
                "guidance.manifold.step_schedule": "legacy",
                "guidance.manifold.grad_steps_per_iter": 1,
                "guidance.budget_hooks.enabled": False,
                "validation.enabled": False,
                "validation.kpm_check.enabled": False,
                "validation.restart.enabled": False,
            },
        },
        {
            "name": "teacher_pmd1",
            "group": "teacher_inference",
            "runner": "teacher_eval",
            "description": "Projected manifold diffusion, 1-step",
            "overrides": {
                "guidance.enabled": True,
                "guidance.manifold.enabled": True,
                "guidance.manifold.step_schedule": "constant",
                "guidance.manifold.grad_steps_per_iter": 1,
                "guidance.budget_hooks.enabled": False,
                "validation.enabled": False,
                "validation.kpm_check.enabled": False,
                "validation.restart.enabled": False,
            },
        },
        {
            "name": "teacher_pmd2",
            "group": "teacher_inference",
            "runner": "teacher_eval",
            "description": "Projected manifold diffusion, 2-step",
            "overrides": {
                "guidance.enabled": True,
                "guidance.manifold.enabled": True,
                "guidance.manifold.step_schedule": "constant",
                "guidance.manifold.grad_steps_per_iter": 2,
                "guidance.budget_hooks.enabled": False,
                "validation.enabled": False,
                "validation.kpm_check.enabled": False,
                "validation.restart.enabled": False,
            },
        },
    ]

    teacher_budget = [
        {
            "name": "teacher_no_teacher",
            "group": "teacher_budget",
            "runner": "teacher_eval",
            "description": "PMD with no high-fidelity teacher",
            "overrides": {
                "guidance.enabled": True,
                "guidance.manifold.enabled": True,
                "guidance.manifold.step_schedule": "constant",
                "guidance.manifold.grad_steps_per_iter": 2,
                "guidance.budget_hooks.enabled": False,
                "validation.enabled": False,
                "validation.kpm_check.enabled": False,
                "validation.restart.enabled": False,
            },
        },
        {
            "name": "teacher_active_b1",
            "group": "teacher_budget",
            "runner": "teacher_eval",
            "description": "PMD with active high-fidelity correction, budget 1",
            "overrides": {
                "guidance.enabled": True,
                "guidance.manifold.enabled": True,
                "guidance.manifold.step_schedule": "constant",
                "guidance.manifold.grad_steps_per_iter": 2,
                "guidance.budget_hooks.enabled": True,
                "guidance.budget_hooks.policy": "threshold",
                "guidance.budget_hooks.threshold": 0.0,
                "guidance.budget_hooks.max_queries_per_sample": 1,
                "guidance.budget_hooks.dry_run": False,
                "validation.enabled": True,
                "validation.kpm_check.enabled": True,
                "validation.restart.enabled": True,
            },
        },
        {
            "name": "teacher_active_b2",
            "group": "teacher_budget",
            "runner": "teacher_eval",
            "description": "PMD with active high-fidelity correction, budget 2",
            "overrides": {
                "guidance.enabled": True,
                "guidance.manifold.enabled": True,
                "guidance.manifold.step_schedule": "constant",
                "guidance.manifold.grad_steps_per_iter": 2,
                "guidance.budget_hooks.enabled": True,
                "guidance.budget_hooks.policy": "threshold",
                "guidance.budget_hooks.threshold": 0.0,
                "guidance.budget_hooks.max_queries_per_sample": 2,
                "guidance.budget_hooks.dry_run": False,
                "validation.enabled": True,
                "validation.kpm_check.enabled": True,
                "validation.restart.enabled": True,
            },
        },
    ]

    minimal_global = [
        proxy_backbone[0],
        proxy_backbone[1],
        proxy_backbone[2],
        teacher_inference[0],
        teacher_inference[1],
        teacher_inference[2],
        teacher_inference[3],
        teacher_budget[1],
    ]

    suites = {
        "minimal_global": minimal_global,
        "proxy_backbone": proxy_backbone,
        "proxy_losses": proxy_losses,
        "teacher_inference": teacher_inference,
        "teacher_budget": teacher_budget,
    }
    all_specs: List[Dict[str, Any]] = []
    for suite_name in ("minimal_global", "proxy_backbone", "proxy_losses", "teacher_inference", "teacher_budget"):
        if suite_name == "minimal_global":
            continue
        all_specs.extend(copy.deepcopy(suites[suite_name]))
    suites["all"] = all_specs
    return suites


def _cfg_path_for(output_dir: str, index: int, name: str) -> str:
    return os.path.join(output_dir, f"{index:02d}_{name}.yaml")


def _json_metric(result: Dict[str, Any], key: str) -> Any:
    metric = result.get("metrics", {}).get(key)
    if isinstance(metric, dict):
        return metric.get("mean")
    return None


def _load_result_json(path: str | None) -> Dict[str, Any] | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_result(spec: Dict[str, Any], result: Dict[str, Any] | None) -> Dict[str, Any]:
    row = {
        "backbone": "",
        "mae_v": "",
        "mse_v": "",
        "rel_l2_v": "",
        "proxy_obs_rmse": "",
        "mean_corrections_per_batch": "",
        "mean_queries_per_batch": "",
        "mean_rejected_per_batch": "",
        "mean_risk_score": "",
        "mse_model": "",
        "rel_model": "",
        "mse_phys": "",
        "rel_phys": "",
        "mse_phys_affine": "",
        "rel_phys_affine": "",
        "psd_error": "",
        "residual": "",
        "peak_ratio": "",
        "p99_ratio": "",
        "std_ratio": "",
    }
    if not result:
        return row
    if spec["runner"] == "teacher_eval":
        row.update(
            {
                "mae_v": result.get("mae_v", ""),
                "mse_v": result.get("mse_v", ""),
                "rel_l2_v": result.get("rel_l2_v", ""),
                "proxy_obs_rmse": result.get("proxy_obs_rmse", ""),
                "mean_corrections_per_batch": result.get("mean_corrections_per_batch", ""),
                "mean_queries_per_batch": result.get("mean_queries_per_batch", ""),
                "mean_rejected_per_batch": result.get("mean_rejected_per_batch", ""),
                "mean_risk_score": result.get("mean_risk_score", ""),
            }
        )
        return row
    row.update(
        {
            "backbone": result.get("backbone", ""),
            "mse_model": _json_metric(result, "mse_model"),
            "rel_model": _json_metric(result, "rel_model"),
            "mse_phys": _json_metric(result, "mse_phys"),
            "rel_phys": _json_metric(result, "rel_phys"),
            "mse_phys_affine": _json_metric(result, "mse_phys_affine"),
            "rel_phys_affine": _json_metric(result, "rel_phys_affine"),
            "psd_error": _json_metric(result, "psd_error"),
            "residual": _json_metric(result, "residual"),
            "peak_ratio": _json_metric(result, "peak_ratio"),
            "p99_ratio": _json_metric(result, "pred_p99_over_obs_p99"),
            "std_ratio": _json_metric(result, "pred_std_over_obs_std"),
        }
    )
    return row


def _write_result_tables(manifest: Dict[str, Any], output_dir: str) -> tuple[str, str]:
    rows: List[Dict[str, Any]] = []
    for exp in manifest.get("experiments", []):
        flat = _flatten_result(exp, _load_result_json(exp.get("result_json")))
        row = {
            "name": exp.get("name", ""),
            "group": exp.get("group", ""),
            "runner": exp.get("runner", ""),
            "status": exp.get("status", ""),
            "description": exp.get("description", ""),
            "config": exp.get("config", ""),
            "run_dir": exp.get("run_dir", ""),
            "result_json": exp.get("result_json", ""),
            "log_path": exp.get("log_path", ""),
            "train_log_path": exp.get("train_log_path", ""),
            "eval_log_path": exp.get("eval_log_path", ""),
        }
        row.update(flat)
        rows.append(row)

    csv_columns = [
        "name",
        "group",
        "runner",
        "status",
        "description",
        "config",
        "run_dir",
        "result_json",
        "log_path",
        "train_log_path",
        "eval_log_path",
        "backbone",
        "mae_v",
        "mse_v",
        "rel_l2_v",
        "proxy_obs_rmse",
        "mean_corrections_per_batch",
        "mean_queries_per_batch",
        "mean_rejected_per_batch",
        "mean_risk_score",
        "mse_model",
        "rel_model",
        "mse_phys",
        "rel_phys",
        "mse_phys_affine",
        "rel_phys_affine",
        "psd_error",
        "residual",
        "peak_ratio",
        "p99_ratio",
        "std_ratio",
    ]
    csv_path = os.path.join(output_dir, "summary.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(rows)

    md_path = os.path.join(output_dir, "summary.md")
    group_order: List[str] = []
    grouped_rows: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        group = str(row.get("group", "ungrouped"))
        if group not in grouped_rows:
            grouped_rows[group] = []
            group_order.append(group)
        grouped_rows[group].append(row)

    def _cell(value: Any) -> str:
        if value in ("", None):
            return "-"
        return str(value)

    md_lines = [
        f"# Ablation Results: {manifest.get('suite', '')}",
        "",
        f"- Base Config: `{manifest.get('base_config', '')}`",
        f"- Stage: `{manifest.get('stage', '')}`",
        "",
        "## Overview",
        "",
        "| Name | Group | Status | Rel L2(V) | Residual | PSD Error | #Queries/batch | #Rejected/batch |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md_lines.append(
            "| {name} | {group} | {status} | {rel_l2_v} | {residual} | {psd_error} | {mean_queries_per_batch} | {mean_rejected_per_batch} |".format(
                name=row["name"],
                group=row["group"],
                status=row["status"],
                rel_l2_v=_cell(row["rel_l2_v"]),
                residual=_cell(row["residual"]),
                psd_error=_cell(row["psd_error"]),
                mean_queries_per_batch=_cell(row["mean_queries_per_batch"]),
                mean_rejected_per_batch=_cell(row["mean_rejected_per_batch"]),
            )
        )
    for group in group_order:
        md_lines.extend(
            [
                "",
                f"## {group}",
                "",
                "| Name | Status | Run Dir | Rel L2(V) | Residual | PSD Error | #Queries/batch | #Rejected/batch |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in grouped_rows[group]:
            md_lines.append(
                "| {name} | {status} | `{run_dir}` | {rel_l2_v} | {residual} | {psd_error} | {mean_queries_per_batch} | {mean_rejected_per_batch} |".format(
                    name=row["name"],
                    status=row["status"],
                    run_dir=row["run_dir"],
                    rel_l2_v=_cell(row["rel_l2_v"]),
                    residual=_cell(row["residual"]),
                    psd_error=_cell(row["psd_error"]),
                    mean_queries_per_batch=_cell(row["mean_queries_per_batch"]),
                    mean_rejected_per_batch=_cell(row["mean_rejected_per_batch"]),
                )
            )
    with open(md_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(md_lines) + "\n")
    return csv_path, md_path


def _build_commands(cfg_path: str, spec: Dict[str, Any], args, result_json_path: str | None = None) -> Dict[str, List[str]]:
    if spec["runner"] == "latent_green_eval":
        train_cmd = [sys.executable, "-m", "gd.train.train_latent_green", "--config", cfg_path]
        eval_cmd = [
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
        if result_json_path:
            eval_cmd.extend(["--json_out", result_json_path])
        return {"train": train_cmd, "eval": eval_cmd}
    if spec["runner"] == "teacher_eval":
        train_cmd = [sys.executable, "-m", "gd.train.train_diffusion", "--config", cfg_path]
        eval_cmd = [
            sys.executable,
            "-m",
            "gd.test.eval_teacher_ablation",
            "--config",
            cfg_path,
            "--num_batches",
            str(args.num_batches),
            "--batch_size",
            str(args.batch_size),
        ]
        if result_json_path:
            eval_cmd.extend(["--json_out", result_json_path])
        return {"train": train_cmd, "eval": eval_cmd}
    raise ValueError(f"Unsupported runner={spec['runner']!r}")


def _maybe_run(cmd: List[str], cwd: str, execute: bool, log_path: str | None = None) -> int | None:
    if not execute:
        return None
    log_handle = None
    if log_path:
        log_dir = os.path.dirname(os.path.abspath(log_path))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_handle = open(log_path, "w", encoding="utf-8")
    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if log_handle is not None:
                log_handle.write(line)
                log_handle.flush()
        process.stdout.close()
        return int(process.wait())
    finally:
        if log_handle is not None:
            log_handle.close()


def _execute_experiment(spec: Dict[str, Any], commands: Dict[str, List[str]], args, logs_root: str) -> tuple[str, str, str]:
    group = spec["group"]
    if args.stage == "print":
        log_group_dir = _group_dir(os.path.join(logs_root, "print"), group)
        return "PLANNED", os.path.join(log_group_dir, f"{spec['name']}.print.log"), ""

    if args.stage == "train":
        log_group_dir = _group_dir(os.path.join(logs_root, "train"), group)
        train_log_path = os.path.join(log_group_dir, f"{spec['name']}.train.log")
        train_rc = _maybe_run(commands["train"], PROJECT_ROOT, execute=True, log_path=train_log_path)
        status = "OK" if train_rc == 0 else f"TRAIN FAILED (exit={train_rc})"
        return status, train_log_path, ""

    if args.stage == "eval":
        log_group_dir = _group_dir(os.path.join(logs_root, "eval"), group)
        eval_log_path = os.path.join(log_group_dir, f"{spec['name']}.eval.log")
        eval_rc = _maybe_run(commands["eval"], PROJECT_ROOT, execute=True, log_path=eval_log_path)
        status = "OK" if eval_rc == 0 else f"EVAL FAILED (exit={eval_rc})"
        return status, "", eval_log_path

    if args.stage == "train_then_eval":
        train_group_dir = _group_dir(os.path.join(logs_root, "train"), group)
        eval_group_dir = _group_dir(os.path.join(logs_root, "eval"), group)
        train_log_path = os.path.join(train_group_dir, f"{spec['name']}.train.log")
        eval_log_path = os.path.join(eval_group_dir, f"{spec['name']}.eval.log")
        train_rc = _maybe_run(commands["train"], PROJECT_ROOT, execute=True, log_path=train_log_path)
        if train_rc != 0:
            return f"TRAIN FAILED (exit={train_rc})", train_log_path, ""
        eval_rc = _maybe_run(commands["eval"], PROJECT_ROOT, execute=True, log_path=eval_log_path)
        if eval_rc != 0:
            return f"EVAL FAILED (exit={eval_rc})", train_log_path, eval_log_path
        return "OK", train_log_path, eval_log_path

    raise ValueError(f"Unsupported stage={args.stage!r}")


def main():
    suites = _build_specs()
    parser = argparse.ArgumentParser(description="Generate and optionally run global ablation configs for GreenDiff.")
    parser.add_argument("--config", default="gd/configs/default.yaml", help="Base config path.")
    parser.add_argument(
        "--suite",
        default="minimal_global",
        choices=tuple(sorted(suites.keys())),
        help="Which predefined experiment suite to materialize.",
    )
    parser.add_argument("--out_dir", help="Directory to write generated configs into.")
    parser.add_argument(
        "--stage",
        default="print",
        choices=("print", "train", "eval", "train_then_eval"),
        help="Print commands only, run train only, eval only, or train then eval sequentially.",
    )
    parser.add_argument("--num_batches", type=int, default=20, help="Eval batches for generated eval commands.")
    parser.add_argument("--batch_size", type=int, default=4, help="Eval batch size for generated eval commands.")
    parser.add_argument("--stop_on_error", action="store_true", help="Stop immediately if an executed command fails.")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    if args.out_dir:
        out_dir = os.path.abspath(args.out_dir)
    else:
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "gd", "runs", "ablations", args.suite, stamp))
    os.makedirs(out_dir, exist_ok=True)
    configs_dir = os.path.join(out_dir, "configs")
    suite_runs_dir = os.path.join(out_dir, "runs")
    results_dir = os.path.join(out_dir, "results")
    result_json_dir = os.path.join(results_dir, "json")
    logs_root = os.path.join(out_dir, "logs")
    os.makedirs(configs_dir, exist_ok=True)
    os.makedirs(suite_runs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(result_json_dir, exist_ok=True)
    os.makedirs(logs_root, exist_ok=True)

    specs = [copy.deepcopy(spec) for spec in suites[args.suite]]
    manifest = {
        "suite": args.suite,
        "base_config": os.path.abspath(args.config),
        "output_dir": out_dir,
        "stage": args.stage,
        "layout": {
            "root_dir": out_dir,
            "configs_dir": configs_dir,
            "suite_runs_dir": suite_runs_dir,
            "results_dir": results_dir,
            "result_json_dir": result_json_dir,
            "logs_root": logs_root,
        },
        "experiments": [],
    }

    failures: List[str] = []
    print("\n" + "=" * 90)
    print(f"Global Ablation Suite: {args.suite}")
    print(f"Root Directory       : {out_dir}")
    print(f"Configs Directory    : {configs_dir}")
    print(f"Suite Runs Directory : {suite_runs_dir}")
    print(f"Results Directory    : {results_dir}")
    print(f"Logs Root            : {logs_root}")
    print(f"Execution Stage      : {args.stage}")
    print("=" * 90)

    for index, spec in enumerate(specs, start=1):
        group = spec["group"]
        cfg_group_dir = _group_dir(configs_dir, group)
        run_group_dir = _group_dir(suite_runs_dir, group)
        result_group_dir = _group_dir(result_json_dir, group)

        cfg = _with_overrides(base_cfg, spec["overrides"])
        cfg = _prepare_experiment_paths(cfg, run_group_dir, spec["name"])
        cfg_path = _cfg_path_for(cfg_group_dir, index, spec["name"])
        result_json_path = os.path.join(result_group_dir, f"{index:02d}_{spec['name']}.json")
        exp_run_dir = cfg["paths"]["workdir"]
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        commands = _build_commands(cfg_path, spec, args, result_json_path=result_json_path)
        status, train_log_path, eval_log_path = _execute_experiment(spec, commands, args, logs_root)
        log_path = eval_log_path or train_log_path
        if status not in {"PLANNED", "OK"}:
            failures.append(spec["name"])
            if args.stop_on_error:
                manifest["experiments"].append(
                    {
                        "name": spec["name"],
                        "group": spec["group"],
                        "description": spec["description"],
                        "runner": spec["runner"],
                        "config": cfg_path,
                        "run_dir": exp_run_dir,
                        "result_json": result_json_path,
                        "log_path": log_path,
                        "train_log_path": train_log_path,
                        "eval_log_path": eval_log_path,
                        "overrides": spec["overrides"],
                        "train_cmd": commands["train"],
                        "eval_cmd": commands["eval"],
                        "status": status,
                    }
                )
                break

        print(f"\n[{index:02d}] {spec['name']}")
        print(f"Group       : {spec['group']}")
        print(f"Description : {spec['description']}")
        print(f"Config      : {cfg_path}")
        print(f"Train Cmd   : {' '.join(commands['train'])}")
        print(f"Eval Cmd    : {' '.join(commands['eval'])}")
        print(f"Status      : {status}")
        print(f"Run Dir     : {exp_run_dir}")
        print(f"Result JSON : {result_json_path}")
        if args.stage == "print":
            print(f"Log Path    : {log_path}")
        elif train_log_path:
            print(f"Train Log   : {train_log_path}")
        if eval_log_path:
            print(f"Eval Log    : {eval_log_path}")
        if args.stage != "print" and not train_log_path and not eval_log_path:
            print(f"Log Path    : {log_path}")

        manifest["experiments"].append(
            {
                "name": spec["name"],
                "group": spec["group"],
                "description": spec["description"],
                "runner": spec["runner"],
                "config": cfg_path,
                "run_dir": exp_run_dir,
                "result_json": result_json_path,
                "log_path": log_path,
                "train_log_path": train_log_path,
                "eval_log_path": eval_log_path,
                "overrides": spec["overrides"],
                "train_cmd": commands["train"],
                "eval_cmd": commands["eval"],
                "status": status,
            }
        )

    manifest_path = os.path.join(out_dir, "manifest.yaml")
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)
    csv_path, md_path = _write_result_tables(manifest, results_dir)

    print("\n" + "=" * 90)
    print(f"Manifest written to: {manifest_path}")
    print(f"CSV summary written to: {csv_path}")
    print(f"Markdown summary written to: {md_path}")
    print("=" * 90)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
