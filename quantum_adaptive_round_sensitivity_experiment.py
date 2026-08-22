"""Literature-aligned round-count comparison for adaptive competitors.

This experiment keeps the published truths, priors, gains, and total copy
budget fixed while varying only the number of adaptive rounds.  The reference
round grid is scaled with the copy budget so that every budget tests 25, 50,
100, and 150 shots per proposal setting.  Existing Greedy measurements are
reused because they do not depend on the adaptive round count.

The output is deliberately sharded by dimension so separate dimensions can be
run concurrently and safely resumed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import qiskit_quantum_backend as quantum
import quantum_benchmark_support as base
import quantum_greedy_spectral_experiment as benchmark


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_DIR = base.ACTIVE_EXPERIMENT_ROOT / "quantum_greedy_spectral"
OUTPUT_DIR = base.ACTIVE_EXPERIMENT_ROOT / "quantum_adaptive_round_sensitivity"

REFERENCE_BUDGET = 153600
REFERENCE_ROUNDS = (512, 768, 1536, 3072)
DEFAULT_BUDGETS = (19200, 38400, 76800, 153600)
DEFAULT_DIMENSIONS = (6, 12, 17)
DEFAULT_METHODS = ("structured_paqt", "structured_sgqt", "structured_osgqt")
METHOD_LABELS = {
    "structured_paqt": "S-PAQT",
    "structured_sgqt": "S-SGQT",
    "structured_osgqt": "S-OSGQT",
}
METHOD_STYLES = {
    "structured_paqt": {"marker": "s", "linestyle": "--"},
    "structured_sgqt": {"marker": "^", "linestyle": "-."},
    "structured_osgqt": {"marker": "D", "linestyle": ":"},
}


def repository_path(path: Path) -> str:
    """Serialize a repository path without embedding a contributor's home path."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


TRIAL_FIELDS = (
    "original_dimension",
    "local_dimension",
    "trial",
    "budget",
    "rounds",
    "method",
    "settings",
    "copies",
    "shots_per_setting_min",
    "shots_per_setting_max",
    "greedy_raw_task_squared_error",
    "competitor_raw_task_squared_error",
    "competitor_minus_greedy",
    "competitor_ess",
    "competitor_minimum_ess",
    "paqt_resampling_count",
    "elapsed_seconds",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def optional_float(row: dict[str, str], field: str) -> float:
    value = row.get(field)
    return float(value) if value not in (None, "") else float("nan")


def greedy_lookups(
    budget: int,
    trials: int,
    fixed_dir: Path = PUBLISHED_DIR,
) -> dict[tuple[int, int], float]:
    rows = read_csv(fixed_dir / "trial_results.csv")
    greedy: dict[tuple[int, int], float] = {}
    for row in rows:
        if int(row["budget"]) != budget or int(row["trial"]) >= trials:
            continue
        dimension = int(row["original_dimension"])
        trial = int(row["trial"])
        method = str(row["method"])
        if method == "greedy_spectral":
            greedy[(dimension, trial)] = benchmark.row_raw_task_squared_error(row)
    return greedy


def refresh_greedy_columns(
    output_dir: Path,
    trials: int,
    fixed_dir: Path = PUBLISHED_DIR,
) -> int:
    """Refresh copied fixed-design scores without rerunning adaptive methods."""

    updated = 0
    lookup_cache: dict[int, dict[tuple[int, int], float]] = {}
    for path in sorted(output_dir.glob("trial_results_b*_d*.csv")):
        rows = read_csv(path)
        changed = False
        for row in rows:
            trial = int(row["trial"])
            if trial >= trials:
                continue
            budget = int(row["budget"])
            dimension = int(row["original_dimension"])
            if budget not in lookup_cache:
                lookup_cache[budget] = greedy_lookups(budget, trials, fixed_dir)
            lookup = lookup_cache[budget]
            key = (dimension, trial)
            if key not in lookup:
                raise RuntimeError(f"Missing refreshed Greedy row: B={budget}, key={key}")
            greedy = lookup[key]
            competitor = float(row["competitor_raw_task_squared_error"])
            row["greedy_raw_task_squared_error"] = greedy
            row["competitor_minus_greedy"] = competitor - greedy
            updated += 1
            changed = True
        if changed:
            base.write_union_csv(path, rows)
    return updated


def rounds_for_budget(budget: int) -> tuple[int, ...]:
    """Scale the reference T grid to preserve shots per setting across budgets."""
    scale = budget / REFERENCE_BUDGET
    return tuple(
        sorted({max(1, int(round(rounds * scale))) for rounds in REFERENCE_ROUNDS})
    )


def result_path(output_dir: Path, budget: int, dimension: int) -> Path:
    """Return the budget-and-dimension result shard."""
    return output_dir / f"trial_results_b{budget}_d{dimension}.csv"


def existing_keys(path: Path) -> set[tuple[int, int, str]]:
    if not path.exists():
        return set()
    return {
        (int(row["trial"]), int(row["rounds"]), str(row["method"]))
        for row in read_csv(path)
    }


def append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_dimension_context(
    config: benchmark.GreedyTaskConfig,
    original_dimension: int,
) -> tuple[
    benchmark.LocalModel,
    np.ndarray,
    np.ndarray,
]:
    model = benchmark.build_local_model(config, original_dimension)
    local_dimension = model.coordinate_map.shape[1]
    particle_coordinates = base.make_particle_cloud(
        rng=np.random.default_rng(config.seed + 10_000 + original_dimension),
        dimension=local_dimension,
        radius=config.particle_radius,
        count=config.n_particles,
    )
    print(
        f"d={original_dimension}, local={local_dimension}: building "
        f"{config.n_particles} shared S-PAQT particles",
        flush=True,
    )
    particle_states = np.asarray(
        [quantum.ground_state(model.family, coordinate) for coordinate in particle_coordinates]
    )
    particle_tasks = benchmark.batch_task_values(particle_states, config)
    return model, particle_coordinates, particle_states, particle_tasks


def run_method(
    *,
    method: str,
    rng_seed: int,
    truth_state: np.ndarray,
    model: benchmark.LocalModel,
    particle_coordinates: np.ndarray,
    particle_states: np.ndarray,
    particle_tasks: np.ndarray,
    config: benchmark.GreedyTaskConfig,
    rounds: int,
    budget: int,
) -> object:
    local_dimension = model.coordinate_map.shape[1]
    if method == "structured_paqt":
        return base.run_structured_paqt(
            rng=np.random.default_rng(rng_seed + 30),
            truth_state=truth_state,
            family=model.family,
            dimension=local_dimension,
            particle_thetas=particle_coordinates,
            particle_states=particle_states,
            total_copies=budget,
            iterations=rounds,
            radius=config.particle_radius,
            task_from_density=lambda density: benchmark.task_values(density, config),
            particle_tasks=particle_tasks,
            task_from_state=lambda state: benchmark.task_values(state, config),
            tasks_from_states=lambda states: benchmark.batch_task_values(
                states, config
            ),
        )
    if method in ("structured_sgqt", "structured_osgqt"):
        result = base.run_structured_sgqt(
            rng=np.random.default_rng(rng_seed + 50),
            truth_state=truth_state,
            family=model.family,
            dimension=local_dimension,
            total_copies=budget,
            iterations=rounds,
            radius=config.particle_radius,
            osgqt=method == "structured_osgqt",
        )
        return benchmark.with_task_values(result, config)
    raise ValueError(f"Unknown method: {method}")


def run_dimension(
    *,
    original_dimension: int,
    budget_rounds: dict[int, tuple[int, ...]],
    methods: tuple[str, ...],
    trials: int,
    particles: int,
    output_dir: Path,
    fixed_dir: Path,
    base_config: benchmark.GreedyTaskConfig,
) -> None:
    config = replace(
        base_config,
        original_dimensions=(original_dimension,),
        budgets=tuple(budget_rounds),
        n_trials=trials,
        n_particles=particles,
    )
    lookups = {
        budget: greedy_lookups(budget, trials, fixed_dir)
        for budget in budget_rounds
    }
    for budget, greedy_lookup in lookups.items():
        missing_greedy = [
            trial
            for trial in range(trials)
            if (original_dimension, trial) not in greedy_lookup
        ]
        if missing_greedy:
            raise RuntimeError(
                f"Published Greedy rows are missing for d={original_dimension}, "
                f"B={budget}, trials={missing_greedy}."
            )

    output_paths = {
        budget: result_path(output_dir, budget, original_dimension)
        for budget in budget_rounds
    }
    completed_by_budget = {
        budget: existing_keys(path)
        for budget, path in output_paths.items()
    }
    all_complete = True
    for budget, rounds_grid in budget_rounds.items():
        requested = {
            (trial, rounds, method)
            for trial in range(trials)
            for rounds in rounds_grid
            for method in methods
        }
        if not requested.issubset(completed_by_budget[budget]):
            all_complete = False
            break
    if all_complete:
        print(f"d={original_dimension}: all requested cells already complete", flush=True)
        return

    model, particle_coordinates, particle_states, particle_tasks = (
        build_dimension_context(config, original_dimension)
    )
    local_dimension = model.coordinate_map.shape[1]
    for trial in range(trials):
        truth_coordinate = base.sample_ball(
            np.random.default_rng(config.seed + 100_000 * original_dimension + trial),
            local_dimension,
            config.truth_radius,
            1,
        )[0]
        truth_state = quantum.ground_state(model.family, truth_coordinate)
        truth_task = benchmark.task_values(truth_state, config)
        for budget, rounds_grid in budget_rounds.items():
            greedy_lookup = lookups[budget]
            greedy_error = greedy_lookup[(original_dimension, trial)]
            output_path = output_paths[budget]
            completed = completed_by_budget[budget]
            for rounds in rounds_grid:
                shot_schedule = base.paqt_sgqt_shot_schedule(budget, rounds)
                seed_base = (
                    config.seed
                    + 1_000_000 * original_dimension
                    + 1000 * trial
                    + budget
                )
                for method in methods:
                    key = (trial, rounds, method)
                    if key in completed:
                        continue
                    started = time.perf_counter()
                    result = run_method(
                        method=method,
                        rng_seed=seed_base,
                        truth_state=truth_state,
                        model=model,
                        particle_coordinates=particle_coordinates,
                        particle_states=particle_states,
                        particle_tasks=particle_tasks,
                        config=config,
                        rounds=rounds,
                        budget=budget,
                    )
                    elapsed = time.perf_counter() - started
                    competitor_error = benchmark.raw_task_squared_error(
                        result.task_estimate,
                        truth_task,
                    )
                    append_row(
                        output_path,
                        {
                            "original_dimension": original_dimension,
                            "local_dimension": local_dimension,
                            "trial": trial,
                            "budget": budget,
                            "rounds": rounds,
                            "method": method,
                            "settings": int(result.settings),
                            "copies": int(result.copies),
                            "shots_per_setting_min": int(np.min(shot_schedule)),
                            "shots_per_setting_max": int(np.max(shot_schedule)),
                            "greedy_raw_task_squared_error": greedy_error,
                            "competitor_raw_task_squared_error": competitor_error,
                            "competitor_minus_greedy": competitor_error - greedy_error,
                            "competitor_ess": float(result.ess),
                            "competitor_minimum_ess": float(
                                getattr(result, "minimum_ess", float("nan"))
                            ),
                            "paqt_resampling_count": int(
                                getattr(result, "resampling_count", 0)
                            ),
                            "elapsed_seconds": elapsed,
                        },
                    )
                    completed.add(key)
                    print(
                        f"d={original_dimension} trial={trial + 1}/{trials} "
                        f"B={budget} T={rounds} {METHOD_LABELS[method]}: "
                        f"competitor={competitor_error:.4e}, "
                        f"greedy={greedy_error:.4e}, seconds={elapsed:.1f}",
                        flush=True,
                    )


def summarize(
    output_dir: Path,
    budget_rounds: dict[int, tuple[int, ...]],
    fixed_dir: Path = PUBLISHED_DIR,
) -> list[dict[str, object]]:
    rows: list[dict[str, str]] = []
    for path in sorted(output_dir.glob("trial_results*.csv")):
        rows.extend(read_csv(path))
    grouped: dict[tuple[int, int, int, str], list[dict[str, str]]] = {}
    for row in rows:
        budget = int(row["budget"])
        rounds = int(row["rounds"])
        if budget not in budget_rounds or rounds not in budget_rounds[budget]:
            continue
        key = (
            int(row["original_dimension"]),
            budget,
            rounds,
            str(row["method"]),
        )
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, object]] = []
    for (dimension, budget, rounds, method), group in sorted(grouped.items()):
        competitor = np.asarray(
            [float(row["competitor_raw_task_squared_error"]) for row in group]
        )
        greedy = np.asarray(
            [float(row["greedy_raw_task_squared_error"]) for row in group]
        )
        differences = competitor - greedy
        count = differences.size
        se_competitor = float(np.std(competitor, ddof=1) / np.sqrt(count)) if count > 1 else float("nan")
        se_difference = float(np.std(differences, ddof=1) / np.sqrt(count)) if count > 1 else float("nan")
        critical = float(stats.t.ppf(0.975, count - 1)) if count > 1 else float("nan")
        mean_difference = float(np.mean(differences))
        t_result = stats.ttest_rel(competitor, greedy) if count > 1 else None
        summary.append(
            {
                "original_dimension": dimension,
                "budget": budget,
                "rounds": rounds,
                "method": method,
                "settings": int(group[0]["settings"]),
                "shots_per_setting_min": int(group[0]["shots_per_setting_min"]),
                "shots_per_setting_max": int(group[0]["shots_per_setting_max"]),
                "n_trials": count,
                "mean_greedy_mse": float(np.mean(greedy)),
                "mean_competitor_mse": float(np.mean(competitor)),
                "se_competitor_mse": se_competitor,
                "competitor_to_greedy_mean_ratio": float(np.mean(competitor) / np.mean(greedy)),
                "mean_competitor_minus_greedy": mean_difference,
                "paired_difference_ci95_low": mean_difference - critical * se_difference,
                "paired_difference_ci95_high": mean_difference + critical * se_difference,
                "greedy_win_fraction": float(np.mean(greedy < competitor)),
                "paired_ttest_two_sided_p": float(t_result.pvalue) if t_result is not None else float("nan"),
                "mean_elapsed_seconds": float(
                    np.mean([float(row["elapsed_seconds"]) for row in group])
                ),
                "mean_competitor_final_ess": float(
                    np.nanmean(
                        [
                            optional_float(row, "competitor_ess")
                            for row in group
                        ]
                    )
                )
                if any(
                    np.isfinite(optional_float(row, "competitor_ess"))
                    for row in group
                )
                else float("nan"),
                "mean_competitor_minimum_ess": float(
                    np.nanmean(
                        [
                            optional_float(row, "competitor_minimum_ess")
                            for row in group
                        ]
                    )
                )
                if any(
                    np.isfinite(
                        optional_float(row, "competitor_minimum_ess")
                    )
                    for row in group
                )
                else float("nan"),
                "mean_paqt_resampling_count": float(
                    np.nanmean(
                        [
                            optional_float(row, "paqt_resampling_count")
                            for row in group
                        ]
                    )
                )
                if method == "structured_paqt"
                else float("nan"),
            }
        )
    if summary:
        base.write_union_csv(output_dir / "summary_rows.csv", summary)
        write_markdown_summary(output_dir / "summary.md", summary)
        write_oracle_summaries(output_dir, summary, fixed_dir)
    return summary


def write_oracle_summaries(
    output_dir: Path,
    rows: list[dict[str, object]],
    fixed_dir: Path = PUBLISHED_DIR,
) -> None:
    """Write post hoc best-T summaries that intentionally favor each baseline."""
    by_method: dict[tuple[int, int, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            int(row["original_dimension"]),
            int(row["budget"]),
            str(row["method"]),
        )
        by_method.setdefault(key, []).append(row)

    oracle_method_rows: list[dict[str, object]] = []
    for (dimension, budget, method), group in sorted(by_method.items()):
        best = min(group, key=lambda row: float(row["mean_competitor_mse"]))
        oracle_method_rows.append(
            {
                **best,
                "tested_round_counts": ",".join(
                    str(int(row["rounds"]))
                    for row in sorted(group, key=lambda row: int(row["rounds"]))
                ),
                "n_tested_round_counts": len(group),
            }
        )
    base.write_union_csv(output_dir / "oracle_method_rows.csv", oracle_method_rows)

    by_cell: dict[tuple[int, int], list[dict[str, object]]] = {}
    for row in oracle_method_rows:
        key = (int(row["original_dimension"]), int(row["budget"]))
        by_cell.setdefault(key, []).append(row)
    oracle_overall_rows: list[dict[str, object]] = []
    for (dimension, budget), group in sorted(by_cell.items()):
        best = min(group, key=lambda row: float(row["mean_competitor_mse"]))
        oracle_overall_rows.append(
            {
                "original_dimension": dimension,
                "budget": budget,
                "oracle_method": best["method"],
                "oracle_rounds": best["rounds"],
                "oracle_settings": best["settings"],
                "oracle_shots_per_setting_min": best["shots_per_setting_min"],
                "oracle_shots_per_setting_max": best["shots_per_setting_max"],
                "n_tested_round_counts_per_method": best["n_tested_round_counts"],
                "tested_round_counts": best["tested_round_counts"],
                "mean_greedy_mse": best["mean_greedy_mse"],
                "mean_oracle_competitor_mse": best["mean_competitor_mse"],
                "oracle_to_greedy_mean_ratio": best[
                    "competitor_to_greedy_mean_ratio"
                ],
                "mean_oracle_minus_greedy": best["mean_competitor_minus_greedy"],
                "paired_difference_ci95_low": best["paired_difference_ci95_low"],
                "paired_difference_ci95_high": best["paired_difference_ci95_high"],
                "greedy_win_fraction": best["greedy_win_fraction"],
            }
        )
    base.write_union_csv(output_dir / "oracle_overall_rows.csv", oracle_overall_rows)
    write_oracle_markdown(output_dir / "oracle_summary.md", oracle_overall_rows)
    save_oracle_plot(
        output_dir / "adaptive_round_oracle_task_benchmark.png",
        oracle_method_rows,
        fixed_dir,
    )


def save_oracle_plot(
    path: Path,
    oracle_rows: list[dict[str, object]],
    fixed_dir: Path = PUBLISHED_DIR,
) -> None:
    """Plot Greedy against each adaptive method at its post hoc best T."""
    greedy_rows = [
        row
        for row in read_csv(fixed_dir / "paired_summary_rows.csv")
        if row["method"] == "greedy_spectral"
    ]
    greedy_lookup = {
        (int(row["original_dimension"]), int(row["budget"])): row
        for row in greedy_rows
    }
    dimensions = sorted({int(row["original_dimension"]) for row in oracle_rows})
    fig, axes = plt.subplots(1, len(dimensions), figsize=(12.0, 3.9), sharey=False)
    if len(dimensions) == 1:
        axes = [axes]
    for axis, dimension in zip(axes, dimensions):
        dimension_greedy = [
            greedy_lookup[(dimension, budget)]
            for budget in DEFAULT_BUDGETS
            if (dimension, budget) in greedy_lookup
        ]
        axis.errorbar(
            [int(row["budget"]) for row in dimension_greedy],
            [float(row["mean_raw_task_squared_error"]) for row in dimension_greedy],
            yerr=[float(row["se_raw_task_squared_error"]) for row in dimension_greedy],
            **benchmark.METHOD_STYLES["greedy_spectral"],
            markersize=5.0,
            linewidth=1.7,
            label=benchmark.METHOD_LABELS["greedy_spectral"],
        )
        for method in DEFAULT_METHODS:
            method_rows = sorted(
                [
                    row
                    for row in oracle_rows
                    if int(row["original_dimension"]) == dimension
                    and row["method"] == method
                ],
                key=lambda row: int(row["budget"]),
            )
            axis.errorbar(
                [int(row["budget"]) for row in method_rows],
                [float(row["mean_competitor_mse"]) for row in method_rows],
                yerr=[float(row["se_competitor_mse"]) for row in method_rows],
                **METHOD_STYLES[method],
                markersize=5.0,
                linewidth=1.7,
                label=f"{METHOD_LABELS[method]} (best $T$)",
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.set_title(f"original $d={dimension}$")
        axis.set_xlabel("target-state copies")
        axis.set_ylabel(r"raw Schmidt-spectrum MSE $\|\widehat{\tau}-\tau\|_2^2$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0.0, 0.18, 1.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_oracle_markdown(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    lines = [
        "# Post Hoc Oracle Round Selection",
        "",
        "For every dimension, budget, and method, T is selected by the smallest",
        "observed competitor mean on the same evaluation truths. The final oracle",
        "competitor is then the smallest of those three method-specific means.",
        "This selection is deliberately optimistic for the adaptive baselines.",
        "",
        "| d | B | Oracle competitor | T | Settings | Shots/setting | Greedy mean | Oracle mean | Oracle / Greedy | Oracle - Greedy [95% CI] |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        shot_text = str(row["oracle_shots_per_setting_min"])
        if row["oracle_shots_per_setting_min"] != row["oracle_shots_per_setting_max"]:
            shot_text += f"-{row['oracle_shots_per_setting_max']}"
        lines.append(
            "| {d} | {budget} | {method} | {rounds} | {settings} | {shots} | "
            "{greedy:.4e} | {oracle:.4e} | {ratio:.3f} | {difference:.4e} "
            "[{low:.4e}, {high:.4e}] |".format(
                d=row["original_dimension"],
                budget=row["budget"],
                method=METHOD_LABELS[str(row["oracle_method"])],
                rounds=row["oracle_rounds"],
                settings=row["oracle_settings"],
                shots=shot_text,
                greedy=float(row["mean_greedy_mse"]),
                oracle=float(row["mean_oracle_competitor_mse"]),
                ratio=float(row["oracle_to_greedy_mean_ratio"]),
                difference=float(row["mean_oracle_minus_greedy"]),
                low=float(row["paired_difference_ci95_low"]),
                high=float(row["paired_difference_ci95_high"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_summary(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Adaptive Round-Count Sensitivity",
        "",
        "Positive paired differences mean that Greedy has lower raw task MSE.",
        "The confidence intervals are paired Student-t intervals over the shared truths.",
        "",
        "| d | B | T | Method | Settings | Shots/setting | Greedy mean | Competitor mean | Ratio | Competitor - Greedy [95% CI] | Greedy wins |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        shot_text = str(row["shots_per_setting_min"])
        if row["shots_per_setting_min"] != row["shots_per_setting_max"]:
            shot_text += f"-{row['shots_per_setting_max']}"
        lines.append(
            "| {d} | {budget} | {rounds} | {method} | {settings} | {shots} | {greedy:.4e} | "
            "{competitor:.4e} | {ratio:.3f} | {difference:.4e} "
            "[{low:.4e}, {high:.4e}] | {wins:.1%} |".format(
                d=row["original_dimension"],
                budget=row["budget"],
                rounds=row["rounds"],
                method=METHOD_LABELS[str(row["method"])],
                settings=row["settings"],
                shots=shot_text,
                greedy=float(row["mean_greedy_mse"]),
                competitor=float(row["mean_competitor_mse"]),
                ratio=float(row["competitor_to_greedy_mean_ratio"]),
                difference=float(row["mean_competitor_minus_greedy"]),
                low=float(row["paired_difference_ci95_low"]),
                high=float(row["paired_difference_ci95_high"]),
                wins=float(row["greedy_win_fraction"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", nargs="+", type=int, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument(
        "--rounds",
        nargs="+",
        type=int,
        default=None,
        help="Optional common T grid override; by default T scales with each budget.",
    )
    parser.add_argument("--methods", nargs="+", choices=DEFAULT_METHODS, default=DEFAULT_METHODS)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--particles", type=int, default=8000)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--fixed-dir",
        type=Path,
        default=None,
        help="Fixed-design result directory used for paired Greedy scores.",
    )
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--refresh-fixed",
        action="store_true",
        help="Refresh copied Greedy scores and paired differences only.",
    )
    args = parser.parse_args()

    fixed_dir = (
        args.fixed_dir
        if args.fixed_dir is not None
        else PUBLISHED_DIR
    )
    base_config = benchmark.GreedyTaskConfig()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    budget_rounds = {
        budget: tuple(args.rounds) if args.rounds is not None else rounds_for_budget(budget)
        for budget in args.budgets
    }
    dimension_suffix = (
        ""
        if tuple(args.dimensions) == DEFAULT_DIMENSIONS
        else "_d" + "_".join(str(value) for value in args.dimensions)
    )
    config_path = args.output_dir / f"config{dimension_suffix}.json"
    config_path.write_text(
        json.dumps(
            {
                "dimensions": args.dimensions,
                "budget_rounds": {
                    str(budget): rounds
                    for budget, rounds in budget_rounds.items()
                },
                "methods": args.methods,
                "trials": args.trials,
                "particles": args.particles,
                "paqt_start_at_pilot": True,
                "paqt_resampler": "Liu-West",
                "paqt_liu_west_a": base.PAQT_LIU_WEST_A,
                "paqt_resample_ess_fraction": (
                    base.PAQT_RESAMPLE_ESS_FRACTION
                ),
                "seed": base_config.seed,
                "fixed_dir": repository_path(fixed_dir),
                "benchmark_config": base_config.__dict__,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.refresh_fixed:
        count = refresh_greedy_columns(
            args.output_dir,
            args.trials,
            fixed_dir,
        )
        print(f"Refreshed {count} copied Greedy scores.", flush=True)
    if not args.aggregate_only and not args.refresh_fixed:
        for dimension in args.dimensions:
            run_dimension(
                original_dimension=dimension,
                budget_rounds=budget_rounds,
                methods=tuple(args.methods),
                trials=args.trials,
                particles=args.particles,
                output_dir=args.output_dir,
                fixed_dir=fixed_dir,
                base_config=base_config,
            )
    summary = summarize(args.output_dir, budget_rounds, fixed_dir)
    print(f"Wrote {len(summary)} summary rows to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
