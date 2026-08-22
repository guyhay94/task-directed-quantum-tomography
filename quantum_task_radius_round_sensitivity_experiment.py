"""Round-count sensitivity for the task-specific radius experiment.

The fixed Greedy measurements and their local and nonlinear reconstructions are
independent of the adaptive round count.  This runner reuses those published
trial errors and evaluates S-PAQT, S-SGQT, and S-OSGQT at each requested T.
Oracle rows select T by the smallest observed mean on the same paired truths;
they are descriptive and are not selection-adjusted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy import stats

import qiskit_quantum_backend as quantum
import quantum_benchmark_support as base
import quantum_greedy_spectral_experiment as benchmark
import quantum_radius_sensitivity_experiment as radius_experiment


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_DIR = base.ACTIVE_EXPERIMENT_ROOT / "quantum_radius_sensitivity"
OUTPUT_DIR = base.ACTIVE_EXPERIMENT_ROOT / "quantum_task_radius_round_sensitivity"

DEFAULT_DIMENSIONS = (6, 17)
DEFAULT_RADII = (0.01, 0.02, 0.03, 0.04, 0.08, 0.16)
DEFAULT_ROUNDS = (512, 768, 1536, 3072)
DEFAULT_METHODS = ("structured_paqt", "structured_sgqt", "structured_osgqt")
METHOD_LABELS = {
    "structured_paqt": "S-PAQT",
    "structured_sgqt": "S-SGQT",
    "structured_osgqt": "S-OSGQT",
}

TRIAL_FIELDS = (
    "dimension",
    "local_dimension",
    "radius",
    "trial",
    "budget",
    "rounds",
    "method",
    "settings",
    "copies",
    "shots_per_setting_min",
    "shots_per_setting_max",
    "greedy_local_raw_task_mse",
    "greedy_nonlinear_raw_task_mse",
    "competitor_raw_task_mse",
    "competitor_minus_local",
    "competitor_minus_nonlinear",
    "paqt_resampling_count",
    "paqt_minimum_ess",
    "paqt_final_ess",
    "elapsed_seconds",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def optional_float(row: dict[str, str], field: str) -> float:
    value = row.get(field)
    return float(value) if value not in (None, "") else float("nan")


def radius_token(radius: float) -> str:
    return f"{radius:g}".replace(".", "p")


def result_path(output_dir: Path, dimension: int, radius: float) -> Path:
    return output_dir / f"trial_results_d{dimension}_r{radius_token(radius)}.csv"


def append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def published_lookups(trials: int) -> dict[tuple[int, float, int], tuple[float, float]]:
    lookups: dict[tuple[int, float, int], tuple[float, float]] = {}
    for row in read_csv(PUBLISHED_DIR / "trial_results.csv"):
        trial = int(row["trial"])
        if trial >= trials or not math.isclose(
            float(row["joint_failure_probability"]), 0.05
        ):
            continue
        key = (int(row["dimension"]), float(row["radius"]), trial)
        lookups[key] = (
            float(row["gaussian_unscaled_task_squared_error"]),
            float(row["bayes_unscaled_task_squared_error"]),
        )
    return lookups


def refresh_fixed_columns(output_dir: Path, trials: int) -> int:
    """Refresh copied fixed local/nonlinear scores in completed adaptive shards."""

    published = published_lookups(trials)
    updated = 0
    for path in sorted(output_dir.glob("trial_results_d*_r*.csv")):
        rows = read_csv(path)
        changed = False
        for row in rows:
            trial = int(row["trial"])
            if trial >= trials:
                continue
            key = (int(row["dimension"]), float(row["radius"]), trial)
            if key not in published:
                raise RuntimeError(f"Missing refreshed fixed-design row: {key}")
            local_error, nonlinear_error = published[key]
            competitor_error = float(row["competitor_raw_task_mse"])
            row["greedy_local_raw_task_mse"] = local_error
            row["greedy_nonlinear_raw_task_mse"] = nonlinear_error
            row["competitor_minus_local"] = competitor_error - local_error
            row["competitor_minus_nonlinear"] = competitor_error - nonlinear_error
            updated += 1
            changed = True
        if changed:
            base.write_union_csv(path, rows)
    return updated


def existing_keys(path: Path) -> set[tuple[int, int, str]]:
    if not path.exists():
        return set()
    return {
        (int(row["trial"]), int(row["rounds"]), str(row["method"]))
        for row in read_csv(path)
    }


def run_paqt(
    *,
    rng_seed: int,
    truth_state: np.ndarray,
    model: benchmark.LocalModel,
    task_config: benchmark.GreedyTaskConfig,
    radius: float,
    budget: int,
    rounds: int,
    config: radius_experiment.RadiusSensitivityConfig,
) -> tuple[np.ndarray, int, int, float, float]:
    probes, counts, shots = base.collect_structured_paqt_measurements(
        rng=np.random.default_rng(rng_seed + 30),
        truth_state=truth_state,
        family=model.family,
        dimension=model.coordinate_map.shape[1],
        total_copies=budget,
        iterations=rounds,
        radius=radius,
    )
    posterior_rng = np.random.default_rng(rng_seed + 530)
    particle_thetas = base.make_particle_cloud(
        rng=posterior_rng,
        dimension=model.coordinate_map.shape[1],
        radius=radius,
        count=config.smc_particles,
    )
    posterior = base.run_liu_west_particle_posterior_from_measurements(
        rng=posterior_rng,
        family=model.family,
        particle_thetas=particle_thetas,
        probe_states=probes,
        counts=counts,
        shot_counts=shots,
        radius=radius,
        task_from_density=lambda density: benchmark.task_values(
            density, task_config
        ),
        task_from_state=lambda state: benchmark.task_values(state, task_config),
        tasks_from_states=lambda states: benchmark.batch_task_values(
            states, task_config
        ),
    )
    return (
        posterior.task_estimate,
        int(shots.size),
        posterior.resampling_count,
        posterior.minimum_ess,
        posterior.ess,
    )


def run_point_method(
    *,
    method: str,
    rng_seed: int,
    truth_state: np.ndarray,
    model: benchmark.LocalModel,
    task_config: benchmark.GreedyTaskConfig,
    radius: float,
    budget: int,
    rounds: int,
) -> tuple[np.ndarray, int]:
    result = base.run_structured_sgqt(
        rng=np.random.default_rng(rng_seed + 50),
        truth_state=truth_state,
        family=model.family,
        dimension=model.coordinate_map.shape[1],
        total_copies=budget,
        iterations=rounds,
        radius=radius,
        osgqt=method == "structured_osgqt",
    )
    return benchmark.task_values(result.state, task_config), int(result.settings)


def run_cell(
    *,
    dimension: int,
    radius: float,
    rounds_grid: tuple[int, ...],
    methods: tuple[str, ...],
    trials: int,
    output_dir: Path,
) -> None:
    config = radius_experiment.RadiusSensitivityConfig()
    budget = max(config.budgets)
    task_config = benchmark.GreedyTaskConfig(
        seed=config.seed,
        anchor_radius=config.anchor_radius,
        original_dimensions=(dimension,),
        budgets=(budget,),
        particle_radius=radius,
        truth_radius=radius,
    )
    model = benchmark.build_local_model(task_config, dimension)
    local_dimension = model.coordinate_map.shape[1]
    published = published_lookups(trials)
    path = result_path(output_dir, dimension, radius)
    completed = existing_keys(path)
    requested = {
        (trial, rounds, method)
        for trial in range(trials)
        for rounds in rounds_grid
        for method in methods
    }
    if requested.issubset(completed):
        print(f"d={dimension}, R={radius:g}: all requested rows complete", flush=True)
        return

    radius_index = radius_experiment.radius_seed_index(radius)
    for trial in range(trials):
        unit_coordinate = base.sample_ball(
            np.random.default_rng(config.seed + 100_000 * dimension + trial),
            local_dimension,
            1.0,
            1,
        )[0]
        truth_state = quantum.ground_state(model.family, radius * unit_coordinate)
        truth_task = benchmark.task_values(truth_state, task_config)
        lookup_key = (dimension, radius, trial)
        if lookup_key not in published:
            raise RuntimeError(f"Missing published fixed-design row: {lookup_key}")
        local_error, nonlinear_error = published[lookup_key]
        seed_base = (
            config.seed
            + 1_000_000 * dimension
            + 1000 * trial
            + budget
            + 100_000 * radius_index
        )

        for rounds in rounds_grid:
            shot_schedule = base.paqt_sgqt_shot_schedule(budget, rounds)
            for method in methods:
                key = (trial, rounds, method)
                if key in completed:
                    continue
                started = time.perf_counter()
                resampling_count = float("nan")
                minimum_ess = float("nan")
                final_ess = float("nan")
                if method == "structured_paqt":
                    (
                        estimate,
                        settings,
                        resampling_count,
                        minimum_ess,
                        final_ess,
                    ) = run_paqt(
                        rng_seed=seed_base,
                        truth_state=truth_state,
                        model=model,
                        task_config=task_config,
                        radius=radius,
                        budget=budget,
                        rounds=rounds,
                        config=config,
                    )
                else:
                    estimate, settings = run_point_method(
                        method=method,
                        rng_seed=seed_base,
                        truth_state=truth_state,
                        model=model,
                        task_config=task_config,
                        radius=radius,
                        budget=budget,
                        rounds=rounds,
                    )
                competitor_error = benchmark.raw_task_squared_error(
                    estimate, truth_task
                )
                elapsed = time.perf_counter() - started
                append_row(
                    path,
                    {
                        "dimension": dimension,
                        "local_dimension": local_dimension,
                        "radius": radius,
                        "trial": trial,
                        "budget": budget,
                        "rounds": rounds,
                        "method": method,
                        "settings": settings,
                        "copies": int(np.sum(shot_schedule)),
                        "shots_per_setting_min": int(np.min(shot_schedule)),
                        "shots_per_setting_max": int(np.max(shot_schedule)),
                        "greedy_local_raw_task_mse": local_error,
                        "greedy_nonlinear_raw_task_mse": nonlinear_error,
                        "competitor_raw_task_mse": competitor_error,
                        "competitor_minus_local": competitor_error - local_error,
                        "competitor_minus_nonlinear": competitor_error
                        - nonlinear_error,
                        "paqt_resampling_count": resampling_count,
                        "paqt_minimum_ess": minimum_ess,
                        "paqt_final_ess": final_ess,
                        "elapsed_seconds": elapsed,
                    },
                )
                completed.add(key)
                print(
                    f"d={dimension} R={radius:g} trial={trial + 1}/{trials} "
                    f"T={rounds} {METHOD_LABELS[method]}: "
                    f"competitor={competitor_error:.4e}, "
                    f"local={local_error:.4e}, nonlinear={nonlinear_error:.4e}, "
                    f"seconds={elapsed:.1f}",
                    flush=True,
                )


def load_all_rows(output_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(output_dir.glob("trial_results_d*_r*.csv")):
        rows.extend(read_csv(path))
    return rows


def mean_and_interval(values: np.ndarray) -> tuple[float, float, float]:
    count = values.size
    mean = float(np.mean(values))
    if count <= 1:
        return mean, float("nan"), float("nan")
    se = float(np.std(values, ddof=1) / math.sqrt(count))
    critical = float(stats.t.ppf(0.975, count - 1))
    return mean, mean - critical * se, mean + critical * se


def summarize(
    output_dir: Path,
    rounds_grid: tuple[int, ...],
) -> list[dict[str, object]]:
    rows = load_all_rows(output_dir)
    grouped: dict[tuple[int, float, int, str], list[dict[str, str]]] = {}
    for row in rows:
        rounds = int(row["rounds"])
        if rounds not in rounds_grid:
            continue
        key = (
            int(row["dimension"]),
            float(row["radius"]),
            rounds,
            str(row["method"]),
        )
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, object]] = []
    for (dimension, radius, rounds, method), group in sorted(grouped.items()):
        local = np.asarray(
            [float(row["greedy_local_raw_task_mse"]) for row in group]
        )
        nonlinear = np.asarray(
            [float(row["greedy_nonlinear_raw_task_mse"]) for row in group]
        )
        competitor = np.asarray(
            [float(row["competitor_raw_task_mse"]) for row in group]
        )
        local_difference = competitor - local
        nonlinear_difference = competitor - nonlinear
        local_mean, local_low, local_high = mean_and_interval(local_difference)
        nonlinear_mean, nonlinear_low, nonlinear_high = mean_and_interval(
            nonlinear_difference
        )
        summary.append(
            {
                "dimension": dimension,
                "radius": radius,
                "budget": int(group[0]["budget"]),
                "rounds": rounds,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "settings": int(group[0]["settings"]),
                "shots_per_setting_min": int(group[0]["shots_per_setting_min"]),
                "shots_per_setting_max": int(group[0]["shots_per_setting_max"]),
                "n_trials": len(group),
                "mean_greedy_local_mse": float(np.mean(local)),
                "mean_greedy_nonlinear_mse": float(np.mean(nonlinear)),
                "mean_competitor_mse": float(np.mean(competitor)),
                "mean_competitor_minus_local": local_mean,
                "competitor_minus_local_ci95_low": local_low,
                "competitor_minus_local_ci95_high": local_high,
                "mean_competitor_minus_nonlinear": nonlinear_mean,
                "competitor_minus_nonlinear_ci95_low": nonlinear_low,
                "competitor_minus_nonlinear_ci95_high": nonlinear_high,
                "local_win_fraction": float(np.mean(local < competitor)),
                "nonlinear_win_fraction": float(np.mean(nonlinear < competitor)),
                "mean_paqt_resampling_count": float(
                    np.mean(
                        [optional_float(row, "paqt_resampling_count") for row in group]
                    )
                )
                if method == "structured_paqt"
                else float("nan"),
                "mean_paqt_minimum_ess": float(
                    np.mean([optional_float(row, "paqt_minimum_ess") for row in group])
                )
                if method == "structured_paqt"
                else float("nan"),
                "mean_paqt_final_ess": float(
                    np.mean([optional_float(row, "paqt_final_ess") for row in group])
                )
                if method == "structured_paqt"
                else float("nan"),
            }
        )
    if summary:
        base.write_union_csv(output_dir / "summary_rows.csv", summary)
        write_oracle_outputs(output_dir, summary)
    return summary


def write_oracle_outputs(
    output_dir: Path,
    summary: list[dict[str, object]],
) -> None:
    grouped: dict[tuple[int, float, str], list[dict[str, object]]] = {}
    for row in summary:
        key = (int(row["dimension"]), float(row["radius"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)

    method_oracles: list[dict[str, object]] = []
    for key, group in sorted(grouped.items()):
        best = min(group, key=lambda row: float(row["mean_competitor_mse"]))
        method_oracles.append(
            {
                **best,
                "tested_round_counts": ",".join(
                    str(int(row["rounds"]))
                    for row in sorted(group, key=lambda row: int(row["rounds"]))
                ),
            }
        )
    base.write_union_csv(output_dir / "oracle_method_rows.csv", method_oracles)

    overall_groups: dict[tuple[int, float], list[dict[str, object]]] = {}
    for row in method_oracles:
        key = (int(row["dimension"]), float(row["radius"]))
        overall_groups.setdefault(key, []).append(row)
    overall = [
        min(group, key=lambda row: float(row["mean_competitor_mse"]))
        for _, group in sorted(overall_groups.items())
    ]
    base.write_union_csv(output_dir / "oracle_overall_rows.csv", overall)

    lines = [
        "# Task-Radius Adaptive Round-Count Oracle",
        "",
        "Each adaptive method receives its post hoc best observed T in each",
        "dimension-radius cell; the overall row then selects the smallest method mean.",
        "All selections are descriptive and use the same 30 evaluation truths.",
        "",
        "| d | R | Local | Nonlinear | Best adaptive | Method | T |",
        "| ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in overall:
        lines.append(
            "| {d} | {radius:g} | {local:.4e} | {nonlinear:.4e} | "
            "{competitor:.4e} | {method} | {rounds} |".format(
                d=int(row["dimension"]),
                radius=float(row["radius"]),
                local=float(row["mean_greedy_local_mse"]),
                nonlinear=float(row["mean_greedy_nonlinear_mse"]),
                competitor=float(row["mean_competitor_mse"]),
                method=METHOD_LABELS[str(row["method"])],
                rounds=int(row["rounds"]),
            )
        )
    (output_dir / "oracle_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def validate_complete(
    output_dir: Path,
    dimensions: tuple[int, ...],
    radii: tuple[float, ...],
    rounds_grid: tuple[int, ...],
    methods: tuple[str, ...],
    trials: int,
) -> None:
    rows = load_all_rows(output_dir)
    keys = {
        (
            int(row["dimension"]),
            float(row["radius"]),
            int(row["trial"]),
            int(row["rounds"]),
            str(row["method"]),
        )
        for row in rows
    }
    expected = {
        (dimension, radius, trial, rounds, method)
        for dimension in dimensions
        for radius in radii
        for trial in range(trials)
        for rounds in rounds_grid
        for method in methods
    }
    missing = expected - keys
    if missing:
        raise RuntimeError(f"Task-radius T sweep has {len(missing)} missing rows.")
    selected = [
        row
        for row in rows
        if int(row["dimension"]) in dimensions
        and float(row["radius"]) in radii
        and int(row["rounds"]) in rounds_grid
        and str(row["method"]) in methods
        and int(row["trial"]) < trials
    ]
    budget = max(radius_experiment.RadiusSensitivityConfig().budgets)
    if any(int(row["copies"]) != budget for row in selected):
        raise RuntimeError("At least one adaptive row misses the exact copy budget.")
    if any(int(row["settings"]) != 2 * int(row["rounds"]) for row in selected):
        raise RuntimeError("At least one adaptive row does not use 2T settings.")
    print(f"Validated {len(expected)} complete adaptive rows.", flush=True)


def main() -> None:
    config = radius_experiment.RadiusSensitivityConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", nargs="+", type=int, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--radii", nargs="+", type=float, default=DEFAULT_RADII)
    parser.add_argument("--rounds", nargs="+", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument(
        "--methods", nargs="+", choices=DEFAULT_METHODS, default=DEFAULT_METHODS
    )
    parser.add_argument("--trials", type=int, default=config.trials)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--refresh-fixed",
        action="store_true",
        help="Refresh copied fixed scores without rerunning adaptive methods.",
    )
    parser.add_argument("--validate-complete", action="store_true")
    args = parser.parse_args()

    dimensions = tuple(args.dimensions)
    radii = tuple(args.radii)
    rounds_grid = tuple(args.rounds)
    methods = tuple(args.methods)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "dimensions": dimensions,
                "radii": radii,
                "budget": max(config.budgets),
                "rounds": rounds_grid,
                "methods": methods,
                "paqt_start_at_pilot": True,
                "trials": args.trials,
                "seed": config.seed,
                "smc_particles": config.smc_particles,
                "paqt_resampler": "Liu-West",
                "paqt_liu_west_a": base.PAQT_LIU_WEST_A,
                "paqt_resample_ess_fraction": (
                    base.PAQT_RESAMPLE_ESS_FRACTION
                ),
                "task_endpoint": "posterior mean of particle Schmidt spectra",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.refresh_fixed:
        count = refresh_fixed_columns(args.output_dir, args.trials)
        print(f"Refreshed {count} copied fixed scores.", flush=True)
    if not args.aggregate_only and not args.refresh_fixed:
        for dimension in dimensions:
            for radius in radii:
                run_cell(
                    dimension=dimension,
                    radius=radius,
                    rounds_grid=rounds_grid,
                    methods=methods,
                    trials=args.trials,
                    output_dir=args.output_dir,
                )
    summary = summarize(args.output_dir, rounds_grid)
    if args.validate_complete:
        validate_complete(
            args.output_dir,
            dimensions,
            radii,
            rounds_grid,
            methods,
            args.trials,
        )
    print(f"Wrote {len(summary)} summary rows to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
