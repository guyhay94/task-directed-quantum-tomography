"""Decouple true displacement and assumed radius for full-state tomography.

The coupled full-state sweep sets both quantities to the same value.  Here the
true state is generated at ``truth_radius`` while the local prior, projection
ball, and S-PAQT support use ``prior_radius``.  Only upper-triangular cells with
``truth_radius <= prior_radius`` are evaluated.  S-PAQT uses the round count
selected in the matching coupled cell and holds it fixed down each column.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np
from scipy import stats

import qiskit_quantum_backend as quantum
import quantum_benchmark_support as base
import quantum_full_tomography_radius_experiment as full_state
import quantum_full_tomography_round_sensitivity_experiment as full_round
import quantum_greedy_spectral_experiment as benchmark


REPO_ROOT = Path(__file__).resolve().parents[1]
COUPLED_FIXED_DIR = base.ACTIVE_EXPERIMENT_ROOT / "quantum_full_tomography_radius"
COUPLED_ROUND_DIR = (
    base.ACTIVE_EXPERIMENT_ROOT / "quantum_full_tomography_round_sensitivity"
)
OUTPUT_DIR = (
    base.ACTIVE_EXPERIMENT_ROOT / "quantum_full_tomography_truth_prior_sensitivity"
)

DEFAULT_DIMENSIONS = (6, 17)
DEFAULT_RADII = (0.01, 0.02, 0.03, 0.04, 0.08, 0.16)
DEFAULT_ROUNDS = full_round.DEFAULT_ROUNDS
METHOD = "structured_paqt"


def valid_cell(truth_radius: float, prior_radius: float) -> bool:
    return truth_radius <= prior_radius + 1e-15


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def optional_float(row: dict[str, str], field: str) -> float:
    value = row.get(field)
    return float(value) if value not in (None, "") else float("nan")


def radius_token(radius: float) -> str:
    return f"{radius:g}".replace(".", "p")


def result_path(
    output_dir: Path,
    dimension: int,
    truth_radius: float,
    prior_radius: float,
) -> Path:
    return output_dir / (
        f"trial_results_d{dimension}_t{radius_token(truth_radius)}_"
        f"p{radius_token(prior_radius)}.csv"
    )


def selected_round_lookups() -> dict[tuple[int, float], int]:
    output: dict[tuple[int, float], int] = {}
    for row in read_csv(COUPLED_ROUND_DIR / "oracle_method_rows.csv"):
        if str(row["method"]) == METHOD:
            output[(int(row["dimension"]), float(row["radius"]))] = int(
                row["rounds"]
            )
    return output


def coupled_diagonal_lookups(
    dimension: int,
    radius: float,
    rounds: int,
) -> dict[int, dict[str, str]]:
    path = COUPLED_ROUND_DIR / (
        f"trial_results_d{dimension}_r{radius_token(radius)}.csv"
    )
    return {
        int(row["trial"]): row
        for row in read_csv(path)
        if str(row["method"]) == METHOD and int(row["rounds"]) == rounds
    }


def load_cell_rows(path: Path) -> list[dict[str, object]]:
    return list(read_csv(path)) if path.exists() else []


def run_cell(job: tuple[int, float, float, int, int, str]) -> int:
    dimension, truth_radius, prior_radius, rounds, trials, output_dir_string = job
    output_dir = Path(output_dir_string)
    path = result_path(output_dir, dimension, truth_radius, prior_radius)
    rows = load_cell_rows(path)
    completed = {int(row["trial"]) for row in rows}
    if len(completed) >= trials:
        return len(completed)

    config = full_state.FullTomographyConfig()
    budget = config.budget

    if math.isclose(truth_radius, prior_radius):
        coupled = coupled_diagonal_lookups(dimension, prior_radius, rounds)
        for trial in range(trials):
            if trial in completed:
                continue
            source = coupled[trial]
            rows.append(
                {
                    "dimension": dimension,
                    "local_dimension": int(source["local_dimension"]),
                    "truth_radius": truth_radius,
                    "prior_radius": prior_radius,
                    "trial": trial,
                    "budget": budget,
                    "rounds": rounds,
                    "method": METHOD,
                    "settings": int(source["settings"]),
                    "copies": int(source["copies"]),
                    "shots_per_setting_min": int(source["shots_per_setting_min"]),
                    "shots_per_setting_max": int(source["shots_per_setting_max"]),
                    "greedy_local_raw_state_infidelity": float(
                        source["greedy_raw_state_infidelity"]
                    ),
                    "competitor_raw_state_infidelity": float(
                        source["competitor_raw_state_infidelity"]
                    ),
                    "competitor_minus_local": float(
                        source["competitor_minus_greedy"]
                    ),
                    "paqt_resampling_count": float(
                        source["paqt_resampling_count"]
                    ),
                    "paqt_minimum_ess": float(source["paqt_minimum_ess"]),
                    "paqt_final_ess": float(source["paqt_final_ess"]),
                    "elapsed_seconds": float(source["elapsed_seconds"]),
                }
            )
        base.write_union_csv(path, rows)
        print(
            f"d={dimension} r={truth_radius:g} R={prior_radius:g}: "
            "reused coupled diagonal",
            flush=True,
        )
        return len(rows)

    base_config = benchmark.GreedyTaskConfig(
        seed=config.seed,
        anchor_radius=config.anchor_radius,
        budgets=(budget,),
    )
    prior_config = replace(
        base_config,
        particle_radius=prior_radius,
        truth_radius=prior_radius,
        budgets=(budget,),
    )
    model = benchmark.build_local_model(base_config, dimension)
    local_dimension = model.coordinate_map.shape[1]
    geometry = full_state.full_state_geometry(model)
    directions = geometry.generalized_eigenvectors
    anchor_states = benchmark.projective_geodesic_anchors(
        model,
        directions,
        config.anchor_radius,
    )
    shot_counts = benchmark.largest_remainder_allocation(
        budget,
        np.ones(local_dimension, dtype=float),
    )
    shot_schedule = base.paqt_sgqt_shot_schedule(budget, rounds)

    for trial in range(trials):
        if trial in completed:
            continue
        unit_coordinate = base.sample_ball(
            np.random.default_rng(config.seed + 100_000 * dimension + trial),
            local_dimension,
            1.0,
            1,
        )[0]
        truth_state = quantum.ground_state(
            model.family,
            truth_radius * unit_coordinate,
        )
        seed_base = config.seed + 1_000_000 * dimension + 1000 * trial + budget
        greedy, _ = benchmark.run_local_gaussian_estimator(
            rng=np.random.default_rng(seed_base + 10),
            truth_state=truth_state,
            model=model,
            config=prior_config,
            anchor_states=anchor_states,
            shot_counts=shot_counts,
            directions=directions,
            task_metric=geometry.task_metric,
        )
        greedy_error = full_state.raw_state_infidelity(greedy.state, truth_state)
        started = time.perf_counter()
        (
            competitor_error,
            settings,
            resampling_count,
            minimum_ess,
            final_ess,
        ) = full_round.run_paqt(
            rng_seed=seed_base,
            truth_state=truth_state,
            model=model,
            radius=prior_radius,
            budget=budget,
            rounds=rounds,
            config=config,
        )
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "dimension": dimension,
                "local_dimension": local_dimension,
                "truth_radius": truth_radius,
                "prior_radius": prior_radius,
                "trial": trial,
                "budget": budget,
                "rounds": rounds,
                "method": METHOD,
                "settings": settings,
                "copies": int(np.sum(shot_schedule)),
                "shots_per_setting_min": int(np.min(shot_schedule)),
                "shots_per_setting_max": int(np.max(shot_schedule)),
                "greedy_local_raw_state_infidelity": greedy_error,
                "competitor_raw_state_infidelity": competitor_error,
                "competitor_minus_local": competitor_error - greedy_error,
                "paqt_resampling_count": resampling_count,
                "paqt_minimum_ess": minimum_ess,
                "paqt_final_ess": final_ess,
                "elapsed_seconds": elapsed,
            }
        )
        base.write_union_csv(path, rows)
    print(
        f"d={dimension} r={truth_radius:g} R={prior_radius:g}: "
        f"{trials} trials complete",
        flush=True,
    )
    return len(rows)


def load_all_rows(output_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(output_dir.glob("trial_results_d*_t*_p*.csv")):
        rows.extend(read_csv(path))
    return rows


def summarize(output_dir: Path) -> list[dict[str, object]]:
    grouped: dict[tuple[int, float, float], list[dict[str, str]]] = {}
    for row in load_all_rows(output_dir):
        key = (
            int(row["dimension"]),
            float(row["truth_radius"]),
            float(row["prior_radius"]),
        )
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, object]] = []
    for (dimension, truth_radius, prior_radius), group in sorted(grouped.items()):
        local = np.asarray(
            [float(row["greedy_local_raw_state_infidelity"]) for row in group]
        )
        competitor = np.asarray(
            [float(row["competitor_raw_state_infidelity"]) for row in group]
        )
        differences = competitor - local
        count = differences.size
        se_difference = float(np.std(differences, ddof=1) / math.sqrt(count))
        critical = float(stats.t.ppf(0.975, count - 1))
        mean_difference = float(np.mean(differences))
        summary.append(
            {
                "dimension": dimension,
                "truth_radius": truth_radius,
                "prior_radius": prior_radius,
                "budget": int(group[0]["budget"]),
                "rounds": int(group[0]["rounds"]),
                "method": METHOD,
                "settings": int(group[0]["settings"]),
                "n_trials": count,
                "mean_greedy_local_infidelity": float(np.mean(local)),
                "mean_competitor_infidelity": float(np.mean(competitor)),
                "mean_competitor_minus_local": mean_difference,
                "competitor_minus_local_ci95_low": mean_difference
                - critical * se_difference,
                "competitor_minus_local_ci95_high": mean_difference
                + critical * se_difference,
                "local_win_fraction": float(np.mean(local < competitor)),
                "mean_paqt_resampling_count": float(
                    np.mean(
                        [optional_float(row, "paqt_resampling_count") for row in group]
                    )
                ),
                "mean_paqt_minimum_ess": float(
                    np.mean([optional_float(row, "paqt_minimum_ess") for row in group])
                ),
                "mean_paqt_final_ess": float(
                    np.mean([optional_float(row, "paqt_final_ess") for row in group])
                ),
            }
        )
    base.write_union_csv(output_dir / "summary_rows.csv", summary)
    return summary


def scientific_label(value: float) -> str:
    exponent = int(math.floor(math.log10(value)))
    mantissa = value / (10.0**exponent)
    return f"{mantissa:.1f}e{exponent:+d}"


def save_heatmap(
    path: Path,
    summary: list[dict[str, object]],
    truth_radii: tuple[float, ...],
    prior_radii: tuple[float, ...],
) -> None:
    dimensions = sorted({int(row["dimension"]) for row in summary})
    truth_axis = list(truth_radii)
    prior_axis = list(prior_radii)
    endpoints = (
        ("mean_greedy_local_infidelity", "Local fixed"),
        ("mean_competitor_infidelity", r"S-PAQT at coupled-selected $T^*$"),
    )
    values = np.asarray(
        [float(row[field]) for row in summary for field, _ in endpoints]
    )
    positive = values[np.isfinite(values) & (values > 0.0)]
    norm = LogNorm(vmin=float(np.min(positive)), vmax=float(np.max(positive)))
    fig, axes = plt.subplots(
        len(dimensions),
        len(endpoints),
        figsize=(8.5, 1.8 * len(dimensions)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    image = None
    for row_index, dimension in enumerate(dimensions):
        for column_index, (field, title) in enumerate(endpoints):
            axis = axes[row_index, column_index]
            matrix = np.full((len(truth_axis), len(prior_axis)), np.nan)
            for truth_index, truth_radius in enumerate(truth_axis):
                for prior_index, prior_radius in enumerate(prior_axis):
                    matches = [
                        row
                        for row in summary
                        if int(row["dimension"]) == dimension
                        and math.isclose(float(row["truth_radius"]), truth_radius)
                        and math.isclose(float(row["prior_radius"]), prior_radius)
                    ]
                    if matches:
                        matrix[truth_index, prior_index] = float(matches[0][field])
            image = axis.imshow(
                np.ma.masked_invalid(matrix),
                origin="upper",
                interpolation="nearest",
                cmap="viridis",
                norm=norm,
                aspect="auto",
            )
            axis.set_facecolor("#eeeeee")
            for truth_index, truth_radius in enumerate(truth_axis):
                for prior_index, prior_radius in enumerate(prior_axis):
                    value = matrix[truth_index, prior_index]
                    if np.isfinite(value):
                        axis.text(
                            prior_index,
                            truth_index,
                            scientific_label(value),
                            ha="center",
                            va="center",
                            fontsize=5.0,
                            color="white" if norm(value) > 0.58 else "black",
                        )
                    if math.isclose(truth_radius, prior_radius):
                        axis.add_patch(
                            Rectangle(
                                (prior_index - 0.5, truth_index - 0.5),
                                1.0,
                                1.0,
                                fill=False,
                                edgecolor="black",
                                linewidth=1.35,
                                linestyle="--",
                            )
                        )
            axis.set_xticks(
                range(len(prior_axis)),
                [f"{radius:g}" for radius in prior_axis],
            )
            axis.set_yticks(
                range(len(truth_axis)),
                [f"{radius:g}" for radius in truth_axis],
            )
            axis.tick_params(axis="x", labelrotation=45)
            axis.set_ylim(len(truth_axis) - 0.5, -0.5)
            if row_index == 0:
                axis.set_title(title)
            if column_index == 0:
                axis.set_ylabel(
                    rf"$d={dimension}$" + "\n" + r"truth radius $r_{\rm truth}$"
                )
            if row_index == len(dimensions) - 1:
                axis.set_xlabel(r"assumed radius $R_\alpha$")
    if image is None:
        raise RuntimeError("No full-state heatmap was constructed.")
    fig.subplots_adjust(left=0.10, right=0.87, bottom=0.14, top=0.92, wspace=0.14)
    colorbar_axis = fig.add_axes([0.90, 0.18, 0.016, 0.64])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("mean raw full-state infidelity")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def validate_complete(
    output_dir: Path,
    dimensions: tuple[int, ...],
    truth_radii: tuple[float, ...],
    prior_radii: tuple[float, ...],
    trials: int,
    selected_rounds: dict[tuple[int, float], int],
) -> None:
    rows = load_all_rows(output_dir)
    keys = {
        (
            int(row["dimension"]),
            float(row["truth_radius"]),
            float(row["prior_radius"]),
            int(row["trial"]),
        )
        for row in rows
    }
    expected = {
        (dimension, truth_radius, prior_radius, trial)
        for dimension in dimensions
        for truth_radius in truth_radii
        for prior_radius in prior_radii
        if valid_cell(truth_radius, prior_radius)
        for trial in range(trials)
    }
    if keys != expected:
        raise RuntimeError(
            f"Full-state decoupling grid has {len(expected - keys)} missing and "
            f"{len(keys - expected)} extra rows."
        )
    for row in rows:
        dimension = int(row["dimension"])
        prior_radius = float(row["prior_radius"])
        if int(row["rounds"]) != selected_rounds[(dimension, prior_radius)]:
            raise RuntimeError("A full-state cell uses the wrong coupled-selected T.")
        if int(row["copies"]) != full_state.FullTomographyConfig().budget:
            raise RuntimeError("A full-state decoupling row misses its copy budget.")
        if int(row["settings"]) != 2 * int(row["rounds"]):
            raise RuntimeError("A full-state decoupling row does not use 2T settings.")
    print(f"Validated {len(expected)} complete full-state rows.", flush=True)


def run_job(job: tuple[int, float, float, int, int, str]) -> int:
    # Each process owns one cell; nested BLAS threads only oversubscribe cores.
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return run_cell(job)
    with threadpool_limits(limits=1):
        return run_cell(job)


def refresh_local_results(
    *,
    dimensions: tuple[int, ...],
    truth_radii: tuple[float, ...],
    prior_radii: tuple[float, ...],
    selected_rounds: dict[tuple[int, float], int],
    trials: int,
    output_dir: Path,
) -> int:
    """Recompute fixed full-state columns while retaining S-PAQT outputs."""

    config = full_state.FullTomographyConfig()
    budget = config.budget
    updated = 0
    for dimension in dimensions:
        base_config = benchmark.GreedyTaskConfig(
            seed=config.seed,
            anchor_radius=config.anchor_radius,
            budgets=(budget,),
        )
        model = benchmark.build_local_model(base_config, dimension)
        local_dimension = model.coordinate_map.shape[1]
        geometry = full_state.full_state_geometry(model)
        directions = geometry.generalized_eigenvectors
        anchor_states = benchmark.projective_geodesic_anchors(
            model, directions, config.anchor_radius
        )
        shot_counts = benchmark.largest_remainder_allocation(
            budget, np.ones(local_dimension, dtype=float)
        )
        for prior_radius in prior_radii:
            rounds = selected_rounds[(dimension, prior_radius)]
            prior_config = replace(
                base_config,
                particle_radius=prior_radius,
                truth_radius=prior_radius,
                budgets=(budget,),
            )
            diagonal = coupled_diagonal_lookups(dimension, prior_radius, rounds)
            for truth_radius in truth_radii:
                if not valid_cell(truth_radius, prior_radius):
                    continue
                path = result_path(output_dir, dimension, truth_radius, prior_radius)
                if not path.exists():
                    raise FileNotFoundError(f"Missing completed full-state cell: {path}")
                rows = read_csv(path)
                by_trial = {int(row["trial"]): row for row in rows}
                for trial in range(trials):
                    if trial not in by_trial:
                        raise RuntimeError(
                            f"Missing full-state row d={dimension}, r={truth_radius}, "
                            f"R={prior_radius}, trial={trial}"
                        )
                    row = by_trial[trial]
                    if math.isclose(truth_radius, prior_radius):
                        greedy_error = float(
                            diagonal[trial]["greedy_raw_state_infidelity"]
                        )
                    else:
                        unit_coordinate = base.sample_ball(
                            np.random.default_rng(
                                config.seed + 100_000 * dimension + trial
                            ),
                            local_dimension,
                            1.0,
                            1,
                        )[0]
                        truth_state = quantum.ground_state(
                            model.family, truth_radius * unit_coordinate
                        )
                        seed_base = (
                            config.seed
                            + 1_000_000 * dimension
                            + 1000 * trial
                            + budget
                        )
                        greedy, _ = benchmark.run_local_gaussian_estimator(
                            rng=np.random.default_rng(seed_base + 10),
                            truth_state=truth_state,
                            model=model,
                            config=prior_config,
                            anchor_states=anchor_states,
                            shot_counts=shot_counts,
                            directions=directions,
                            task_metric=geometry.task_metric,
                        )
                        greedy_error = full_state.raw_state_infidelity(
                            greedy.state, truth_state
                        )
                    competitor_error = float(
                        row["competitor_raw_state_infidelity"]
                    )
                    row["greedy_local_raw_state_infidelity"] = greedy_error
                    row["competitor_minus_local"] = competitor_error - greedy_error
                    updated += 1
                base.write_union_csv(path, rows)
                print(
                    f"d={dimension} r={truth_radius:g} R={prior_radius:g}: "
                    "refreshed fixed rows",
                    flush=True,
                )
    return updated


def main() -> None:
    config = full_state.FullTomographyConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", nargs="+", type=int, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--truth-radii", nargs="+", type=float, default=DEFAULT_RADII)
    parser.add_argument("--prior-radii", nargs="+", type=float, default=DEFAULT_RADII)
    parser.add_argument("--trials", type=int, default=config.trials)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--refresh-local",
        action="store_true",
        help="Refresh fixed local scores without rerunning S-PAQT.",
    )
    parser.add_argument("--validate-complete", action="store_true")
    args = parser.parse_args()

    dimensions = tuple(args.dimensions)
    truth_radii = tuple(args.truth_radii)
    prior_radii = tuple(args.prior_radii)
    selected_rounds = selected_round_lookups()
    required_round_keys = {
        (dimension, prior_radius)
        for dimension in dimensions
        for prior_radius in prior_radii
    }
    if not required_round_keys.issubset(selected_rounds):
        raise RuntimeError("Missing a coupled full-state S-PAQT T selection.")
    if any(
        selected_rounds[key] not in DEFAULT_ROUNDS for key in required_round_keys
    ):
        raise RuntimeError("A selected full-state T is outside the retained grid.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "dimensions": dimensions,
                "truth_radii": truth_radii,
                "prior_radii": prior_radii,
                "budget": config.budget,
                "rounds": DEFAULT_ROUNDS,
                "selected_rounds": {
                    f"d{dimension}_r{prior_radius:g}": selected_rounds[
                        (dimension, prior_radius)
                    ]
                    for dimension, prior_radius in sorted(required_round_keys)
                },
                "round_selection": "S-PAQT T selected on the coupled diagonal",
                "methods": (METHOD,),
                "paqt_start_at_pilot": True,
                "trials": args.trials,
                "seed": config.seed,
                "smc_particles": config.smc_particles,
                "paqt_resampler": "Liu-West",
                "paqt_liu_west_a": base.PAQT_LIU_WEST_A,
                "paqt_resample_ess_fraction": (
                    base.PAQT_RESAMPLE_ESS_FRACTION
                ),
                "state_endpoint": "Bayesian mean density operator infidelity",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    jobs = [
        (
            dimension,
            truth_radius,
            prior_radius,
            selected_rounds[(dimension, prior_radius)],
            args.trials,
            str(args.output_dir),
        )
        for dimension in dimensions
        for truth_radius in truth_radii
        for prior_radius in prior_radii
        if valid_cell(truth_radius, prior_radius)
    ]
    if args.refresh_local:
        count = refresh_local_results(
            dimensions=dimensions,
            truth_radii=truth_radii,
            prior_radii=prior_radii,
            selected_rounds=selected_rounds,
            trials=args.trials,
            output_dir=args.output_dir,
        )
        print(f"Refreshed {count} fixed full-state rows.", flush=True)
    if not args.aggregate_only and not args.refresh_local:
        if args.workers == 1:
            for job in jobs:
                run_cell(job)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                list(executor.map(run_job, jobs))

    summary = summarize(args.output_dir)
    save_heatmap(
        args.output_dir / "full_state_truth_prior_heatmap.png",
        summary,
        truth_radii,
        prior_radii,
    )
    if args.validate_complete:
        validate_complete(
            args.output_dir,
            dimensions,
            truth_radii,
            prior_radii,
            args.trials,
            selected_rounds,
        )
    print(f"Wrote {len(summary)} full-state summary rows.", flush=True)


if __name__ == "__main__":
    main()
