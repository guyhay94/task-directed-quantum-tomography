"""Fixed-round baseline for decoupling truth distance and assumed radius.

The original radius sweep sets the truth-ball radius and the assumed uniform-
ball prior radius to the same value.  This experiment instead evaluates the
upper-triangular factorial design

    truth_radius <= prior_radius,

while retaining the original paired unit-ball truth draws.  For a fixed prior
radius, the task geometry, covariance-matched precision, Greedy allocation,
particle support, and adaptive projection radius are fixed; only the truth
radius changes.  Results are reported in Euclidean Schmidt-spectrum squared
error.  Current reported comparisons use
``quantum_truth_prior_round_sensitivity_experiment.py`` so that T is varied.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np

import qiskit_quantum_backend as quantum
import quantum_benchmark_support as base
import quantum_greedy_spectral_experiment as benchmark
import quantum_radius_sensitivity_experiment as radius_experiment


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = base.ACTIVE_EXPERIMENT_ROOT / "quantum_truth_prior_decoupling"

METHODS = (
    "greedy_local",
    "greedy_nonlinear",
    "structured_paqt",
    "structured_sgqt",
    "structured_osgqt",
)

METHOD_LABELS = {
    "greedy_local": "Greedy: local",
    "greedy_nonlinear": "Greedy: nonlinear",
    "structured_paqt": "S-PAQT",
    "structured_sgqt": "S-SGQT",
    "structured_osgqt": "S-OSGQT",
}


@dataclass(frozen=True)
class TruthPriorDecouplingConfig:
    seed: int = 20260731
    dimensions: tuple[int, ...] = (6, 17)
    radii: tuple[float, ...] = (0.01, 0.02, 0.03, 0.04, 0.08, 0.16)
    budget: int = 153600
    trials: int = 30
    iterations: int = 768
    scale_particles: int = 3000
    smc_particles: int = 500
    smc_ess_fraction: float = 0.55
    smc_mutation_steps: int = 3
    smc_max_temperatures: int = 60
    anchor_radius: float = math.pi / 4.0
    branch_radius_factor: float = 1.25


def valid_cell(truth_radius: float, prior_radius: float) -> bool:
    return truth_radius <= prior_radius + 1e-15


def raw_task_squared_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    difference = np.asarray(estimate, dtype=float) - np.asarray(truth, dtype=float)
    return float(difference @ difference)


def summarize(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    keys = sorted(
        {
            (
                int(row["dimension"]),
                float(row["truth_radius"]),
                float(row["prior_radius"]),
                int(row["budget"]),
                str(row["method"]),
            )
            for row in rows
        }
    )
    for dimension, truth_radius, prior_radius, budget, method in keys:
        subset = [
            row
            for row in rows
            if int(row["dimension"]) == dimension
            and math.isclose(float(row["truth_radius"]), truth_radius)
            and math.isclose(float(row["prior_radius"]), prior_radius)
            and int(row["budget"]) == budget
            and str(row["method"]) == method
        ]
        values = np.asarray(
            [float(row["raw_task_squared_error"]) for row in subset],
            dtype=float,
        )
        temperatures = np.asarray(
            [float(row["smc_temperature_count"]) for row in subset],
            dtype=float,
        )
        acceptances = np.asarray(
            [float(row["smc_acceptance_rate"]) for row in subset],
            dtype=float,
        )
        resampling_counts = np.asarray(
            [float(row.get("paqt_resampling_count", "nan")) for row in subset],
            dtype=float,
        )
        minimum_esses = np.asarray(
            [float(row.get("paqt_minimum_ess", "nan")) for row in subset],
            dtype=float,
        )
        finite_temperatures = temperatures[np.isfinite(temperatures)]
        finite_acceptances = acceptances[np.isfinite(acceptances)]
        output.append(
            {
                "dimension": dimension,
                "truth_radius": truth_radius,
                "prior_radius": prior_radius,
                "equivalent_prior_precision": float(
                    subset[0]["equivalent_prior_precision"]
                ),
                "budget": budget,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "mean_raw_task_mse": float(np.mean(values)),
                "se_raw_task_mse": float(
                    np.std(values, ddof=1) / math.sqrt(values.size)
                    if values.size > 1
                    else 0.0
                ),
                "median_raw_task_mse": float(np.median(values)),
                "mean_truth_norm": float(
                    np.mean([float(row["truth_norm"]) for row in subset])
                ),
                "mean_estimate_norm": float(
                    np.nanmean(
                        np.asarray(
                            [float(row["estimate_norm"]) for row in subset],
                            dtype=float,
                        )
                    )
                )
                if any(np.isfinite(float(row["estimate_norm"])) for row in subset)
                else float("nan"),
                "boundary_fraction": float(
                    np.nanmean(
                        np.asarray(
                            [float(row["estimate_on_prior_boundary"]) for row in subset],
                            dtype=float,
                        )
                    )
                )
                if any(
                    np.isfinite(float(row["estimate_on_prior_boundary"]))
                    for row in subset
                )
                else float("nan"),
                "mean_smc_temperature_count": float(np.mean(finite_temperatures))
                if finite_temperatures.size
                else float("nan"),
                "mean_smc_acceptance_rate": float(np.mean(finite_acceptances))
                if finite_acceptances.size
                else float("nan"),
                "mean_paqt_resampling_count": float(
                    np.nanmean(resampling_counts)
                )
                if np.any(np.isfinite(resampling_counts))
                else float("nan"),
                "mean_paqt_minimum_ess": float(np.nanmean(minimum_esses))
                if np.any(np.isfinite(minimum_esses))
                else float("nan"),
                "trials": len(subset),
            }
        )
    return output


def scientific_label(value: float) -> str:
    if not np.isfinite(value) or value <= 0.0:
        return "--"
    exponent = int(math.floor(math.log10(value)))
    mantissa = value / (10.0**exponent)
    return f"{mantissa:.1f}e{exponent:+d}"


def save_triangular_heatmap(
    path: Path,
    summary_rows: list[dict],
    config: TruthPriorDecouplingConfig,
) -> None:
    """Plot every valid cell as shared-scale raw-MSE triangular heatmaps."""

    dimensions = list(config.dimensions)
    radii = list(config.radii)
    values = np.asarray(
        [float(row["mean_raw_task_mse"]) for row in summary_rows],
        dtype=float,
    )
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        raise ValueError("The heatmap requires at least one positive finite MSE.")
    norm = LogNorm(vmin=float(np.min(positive)), vmax=float(np.max(positive)))

    methods = tuple(
        method
        for method in METHODS
        if any(str(row["method"]) == method for row in summary_rows)
    )
    fig, axes = plt.subplots(
        len(dimensions),
        len(methods),
        figsize=(3.1 * len(methods), 3.35 * len(dimensions)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    image = None
    for row_index, dimension in enumerate(dimensions):
        for column_index, method in enumerate(methods):
            axis = axes[row_index, column_index]
            matrix = np.full((len(radii), len(radii)), np.nan, dtype=float)
            for truth_index, truth_radius in enumerate(radii):
                for prior_index, prior_radius in enumerate(radii):
                    if not valid_cell(truth_radius, prior_radius):
                        continue
                    matches = [
                        row
                        for row in summary_rows
                        if int(row["dimension"]) == dimension
                        and str(row["method"]) == method
                        and math.isclose(float(row["truth_radius"]), truth_radius)
                        and math.isclose(float(row["prior_radius"]), prior_radius)
                    ]
                    if matches:
                        matrix[truth_index, prior_index] = float(
                            matches[0]["mean_raw_task_mse"]
                        )

            masked = np.ma.masked_invalid(matrix)
            image = axis.imshow(
                masked,
                origin="upper",
                interpolation="nearest",
                cmap="viridis",
                norm=norm,
                aspect="equal",
            )
            axis.set_facecolor("#eeeeee")
            for truth_index in range(len(radii)):
                for prior_index in range(len(radii)):
                    value = matrix[truth_index, prior_index]
                    if not np.isfinite(value):
                        continue
                    normalized = norm(value)
                    color = "white" if normalized > 0.58 else "black"
                    axis.text(
                        prior_index,
                        truth_index,
                        scientific_label(value),
                        ha="center",
                        va="center",
                        fontsize=6.3,
                        color=color,
                    )
                if truth_index < len(radii):
                    axis.add_patch(
                        Rectangle(
                            (truth_index - 0.5, truth_index - 0.5),
                            1.0,
                            1.0,
                            fill=False,
                            edgecolor="black",
                            linewidth=1.35,
                            linestyle="--",
                        )
                    )
            axis.set_xticks(range(len(radii)), [f"{radius:g}" for radius in radii])
            axis.set_yticks(range(len(radii)), [f"{radius:g}" for radius in radii])
            axis.tick_params(axis="x", labelrotation=45)
            axis.set_xlim(-0.5, len(radii) - 0.5)
            # Matrix-style ordering keeps r_truth <= R_alpha visibly upper
            # triangular: the smallest truth radius is the top row.
            axis.set_ylim(len(radii) - 0.5, -0.5)
            axis.grid(False)
            if row_index == 0:
                axis.set_title(METHOD_LABELS[method])
            if column_index == 0:
                axis.set_ylabel(
                    rf"$d={dimension}$" + "\n" + r"truth-ball radius $r_{\rm truth}$"
                )
            if row_index == len(dimensions) - 1:
                axis.set_xlabel(r"assumed prior radius $R_\alpha$")

    if image is None:
        raise RuntimeError("No heatmap image was constructed.")
    fig.subplots_adjust(
        left=0.07,
        right=0.90,
        bottom=0.11,
        top=0.92,
        wspace=0.12,
        hspace=0.22,
    )
    colorbar_axis = fig.add_axes([0.92, 0.16, 0.012, 0.68])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(r"mean raw MSE $\|\widehat\tau-\tau\|_2^2$")
    fig.suptitle(
        r"Truth distance versus assumed prior radius; dashed cells are $r_{\rm truth}=R_\alpha$",
        y=0.995,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict) -> tuple:
    return (
        int(row["dimension"]),
        float(row["truth_radius"]),
        float(row["prior_radius"]),
        int(row["budget"]),
        int(row["trial"]),
        str(row["method"]),
    )


def expected_full_keys(
    config: TruthPriorDecouplingConfig,
    methods: tuple[str, ...] = METHODS,
) -> set[tuple]:
    return {
        (
            dimension,
            truth_radius,
            prior_radius,
            config.budget,
            trial,
            method,
        )
        for dimension in config.dimensions
        for truth_radius in config.radii
        for prior_radius in config.radii
        if valid_cell(truth_radius, prior_radius)
        for trial in range(config.trials)
        for method in methods
    }


def merge_sharded_results(
    config: TruthPriorDecouplingConfig,
    output_dir: Path,
    *,
    shard_prefix: str = "",
    methods: tuple[str, ...] = METHODS,
) -> None:
    """Merge the resumable cell shards and require the complete factorial grid."""

    candidates: list[Path] = []
    main_trials = output_dir / "trial_results.csv"
    if main_trials.exists():
        candidates.append(main_trials)
    shard_root = output_dir / "shards"
    candidates.extend(sorted(shard_root.glob(f"{shard_prefix}*/trial_results.csv")))
    if not candidates:
        raise FileNotFoundError("No main or sharded trial files were found.")

    merged: dict[tuple, dict] = {}
    for path in candidates:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = row_key(row)
                if key in merged:
                    previous = float(merged[key]["raw_task_squared_error"])
                    current = float(row["raw_task_squared_error"])
                    if not math.isclose(previous, current, rel_tol=1e-12, abs_tol=1e-18):
                        raise RuntimeError(f"Conflicting duplicate row for key {key}.")
                    continue
                merged[key] = row

    expected = expected_full_keys(config, methods)
    actual = set(merged)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise RuntimeError(
            f"Merged grid mismatch: {len(missing)} missing and {len(extra)} extra rows."
        )
    rows = [merged[key] for key in sorted(merged)]
    summary_rows = summarize(rows)
    base.write_union_csv(main_trials, rows)
    base.write_union_csv(output_dir / "summary_rows.csv", summary_rows)
    save_triangular_heatmap(
        output_dir / "triangular_raw_mse_heatmap.png",
        summary_rows,
        config,
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "methods": methods,
                "paqt_resampler": "Liu-West",
                "paqt_liu_west_a": base.PAQT_LIU_WEST_A,
                "paqt_resample_ess_fraction": (
                    base.PAQT_RESAMPLE_ESS_FRACTION
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    valid_cells = sum(
        valid_cell(truth_radius, prior_radius)
        for truth_radius in config.radii
        for prior_radius in config.radii
    )
    lines = [
        "# Truth-radius/prior-radius decoupling",
        "",
        "The full experiment evaluates only cells with "
        "`truth_radius <= prior_radius`.",
        "Truths use the same paired unit-ball coordinate across every cell in a "
        "dimension-trial pair, scaled by `truth_radius`.",
        "The assumed uniform-ball prior, covariance-matched precision, Greedy "
        "design, posterior support, and adaptive projection radius are all set "
        "by `prior_radius`.",
        "Every endpoint is the Euclidean Schmidt-spectrum squared error.",
        "S-PAQT reports the posterior mean of the particle task values. "
        "S-SGQT and S-OSGQT use the matched alpha=0.05, beta=0.2 comparison gains.",
        "",
        f"- dimensions: `{list(config.dimensions)}`",
        f"- radii: `{list(config.radii)}`",
        f"- budget: `{config.budget}`",
        f"- trials per valid cell: `{config.trials}`",
        f"- valid cells per dimension: `{valid_cells}`",
        f"- total method rows: `{len(rows)}`",
        "- dashed heatmap cells: coupled diagonal `truth_radius = prior_radius`",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"Merged {len(candidates)} files into {len(rows)} validated method rows.",
        flush=True,
    )


def method_row(
    *,
    dimension: int,
    local_dimension: int,
    truth_radius: float,
    prior_radius: float,
    budget: int,
    trial: int,
    method: str,
    settings: int,
    truth_norm: float,
    estimate: np.ndarray,
    truth_task: np.ndarray,
    estimate_coordinate: np.ndarray | None,
    smc_temperature_count: float = float("nan"),
    smc_acceptance_rate: float = float("nan"),
    paqt_resampling_count: float = float("nan"),
    paqt_minimum_ess: float = float("nan"),
    paqt_final_ess: float = float("nan"),
) -> dict:
    estimate_norm = (
        float(np.linalg.norm(estimate_coordinate))
        if estimate_coordinate is not None
        else float("nan")
    )
    boundary = (
        float(estimate_norm >= prior_radius * (1.0 - 1e-10))
        if np.isfinite(estimate_norm)
        else float("nan")
    )
    return {
        "dimension": dimension,
        "local_dimension": local_dimension,
        "truth_radius": truth_radius,
        "prior_radius": prior_radius,
        "equivalent_prior_precision": (local_dimension + 2.0) / (prior_radius**2),
        "budget": budget,
        "trial": trial,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "settings": settings,
        "truth_norm": truth_norm,
        "estimate_norm": estimate_norm,
        "estimate_on_prior_boundary": boundary,
        "raw_task_squared_error": raw_task_squared_error(estimate, truth_task),
        "smc_temperature_count": smc_temperature_count,
        "smc_acceptance_rate": smc_acceptance_rate,
        "paqt_resampling_count": paqt_resampling_count,
        "paqt_minimum_ess": paqt_minimum_ess,
        "paqt_final_ess": paqt_final_ess,
    }


def run(
    config: TruthPriorDecouplingConfig,
    output_dir: Path,
    *,
    resume: bool,
    refresh_competitors: bool = False,
    refresh_local: bool = False,
    refresh_fixed: bool = False,
    selected_dimensions: tuple[int, ...] | None = None,
    selected_truth_radii: tuple[float, ...] | None = None,
    selected_prior_radii: tuple[float, ...] | None = None,
    methods: tuple[str, ...] = METHODS,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_path = output_dir / "trial_results.csv"
    rows = (
        load_existing_rows(trial_path)
        if resume or refresh_competitors or refresh_local or refresh_fixed
        else []
    )
    if refresh_competitors:
        rows = [
            row
            for row in rows
            if str(row["method"])
            not in {"structured_paqt", "structured_sgqt", "structured_osgqt"}
        ]
    if refresh_local:
        rows = [row for row in rows if str(row["method"]) != "greedy_local"]
    if refresh_fixed:
        rows = [
            row
            for row in rows
            if str(row["method"]) not in {"greedy_local", "greedy_nonlinear"}
        ]
    existing = {row_key(row) for row in rows}
    base_config = benchmark.GreedyTaskConfig(
        anchor_radius=config.anchor_radius,
        budgets=(config.budget,),
    )

    dimensions = selected_dimensions or config.dimensions
    truth_radii = selected_truth_radii or config.radii
    prior_radii = selected_prior_radii or config.radii
    for dimension in dimensions:
        model = benchmark.build_local_model(base_config, dimension)
        local_dimension = model.coordinate_map.shape[1]
        unit_coordinates = [
            base.sample_ball(
                np.random.default_rng(config.seed + 100_000 * dimension + trial),
                local_dimension,
                1.0,
                1,
            )[0]
            for trial in range(config.trials)
        ]

        for prior_radius in config.radii:
            if not any(math.isclose(prior_radius, value) for value in prior_radii):
                continue
            prior_config = replace(
                base_config,
                particle_radius=prior_radius,
                truth_radius=prior_radius,
                budgets=(config.budget,),
            )
            geometry = benchmark.compute_task_geometry(model, prior_config)
            designs, _ = benchmark.spectral_designs(
                model,
                geometry,
                prior_config,
                config.budget,
            )
            anchor_states, shot_counts, directions = designs["greedy_spectral"]
            print(
                f"d={dimension}, R_alpha={prior_radius:.3g}: design ready",
                flush=True,
            )

            for truth_radius in config.radii:
                if not any(math.isclose(truth_radius, value) for value in truth_radii):
                    continue
                if not valid_cell(truth_radius, prior_radius):
                    continue
                for trial, unit_coordinate in enumerate(unit_coordinates):
                    required = {
                        (
                            dimension,
                            truth_radius,
                            prior_radius,
                            config.budget,
                            trial,
                            method,
                        )
                        for method in methods
                    }
                    if required.issubset(existing):
                        continue
                    missing_methods = {
                        method
                        for method in methods
                        if (
                            dimension,
                            truth_radius,
                            prior_radius,
                            config.budget,
                            trial,
                            method,
                        )
                        not in existing
                    }

                    truth_coordinate = truth_radius * unit_coordinate
                    truth_state = quantum.ground_state(
                        model.family,
                        truth_coordinate,
                    )
                    truth_task = benchmark.task_values(truth_state, prior_config)
                    truth_norm = float(np.linalg.norm(truth_coordinate))
                    seed_base = (
                        config.seed
                        + 1_000_000 * dimension
                        + 1000 * trial
                        + config.budget
                    )

                    trial_rows: list[dict] = []
                    if {"greedy_local", "greedy_nonlinear"} & missing_methods:
                        update = radius_experiment.local_update_and_branches(
                            rng=np.random.default_rng(seed_base + 10),
                            truth_state=truth_state,
                            model=model,
                            anchor_states=anchor_states,
                            shot_counts=shot_counts,
                            directions=directions,
                            task_metric=geometry.task_metric,
                            radius=prior_radius,
                            anchor_radius=config.anchor_radius,
                            branch_radius_factor=config.branch_radius_factor,
                        )
                        if "greedy_local" in missing_methods:
                            greedy_local_state = quantum.ground_state(
                                model.family,
                                update.primary_mean,
                            )
                            trial_rows.append(
                                method_row(
                                    dimension=dimension,
                                    local_dimension=local_dimension,
                                    truth_radius=truth_radius,
                                    prior_radius=prior_radius,
                                    budget=config.budget,
                                    trial=trial,
                                    method="greedy_local",
                                    settings=int(shot_counts.size),
                                    truth_norm=truth_norm,
                                    estimate=benchmark.task_values(
                                        greedy_local_state,
                                        prior_config,
                                    ),
                                    truth_task=truth_task,
                                    estimate_coordinate=update.primary_mean,
                                )
                            )
                        if "greedy_nonlinear" in missing_methods:
                            greedy_cloud = radius_experiment.tempered_smc_cloud(
                                rng=np.random.default_rng(seed_base + 510),
                                model=model,
                                anchor_states=anchor_states,
                                counts=update.counts,
                                shot_counts=shot_counts,
                                radius=prior_radius,
                                particle_count=config.smc_particles,
                                ess_fraction=config.smc_ess_fraction,
                                mutation_steps=config.smc_mutation_steps,
                                max_temperatures=config.smc_max_temperatures,
                            )
                            greedy_states = radius_experiment.batch_ground_states(
                                model,
                                greedy_cloud.coordinates,
                            )
                            greedy_tasks = np.asarray(
                                [
                                    benchmark.task_values(state, prior_config)
                                    for state in greedy_states
                                ]
                            )
                            trial_rows.append(
                                method_row(
                                    dimension=dimension,
                                    local_dimension=local_dimension,
                                    truth_radius=truth_radius,
                                    prior_radius=prior_radius,
                                    budget=config.budget,
                                    trial=trial,
                                    method="greedy_nonlinear",
                                    settings=int(shot_counts.size),
                                    truth_norm=truth_norm,
                                    estimate=greedy_cloud.weights @ greedy_tasks,
                                    truth_task=truth_task,
                                    estimate_coordinate=(
                                        greedy_cloud.weights @ greedy_cloud.coordinates
                                    ),
                                    smc_temperature_count=greedy_cloud.temperature_count,
                                    smc_acceptance_rate=greedy_cloud.mean_acceptance_rate,
                                )
                            )

                    if "structured_paqt" in missing_methods:
                        probes, paqt_counts, paqt_shots = (
                            base.collect_structured_paqt_measurements(
                                rng=np.random.default_rng(seed_base + 30),
                                truth_state=truth_state,
                                family=model.family,
                                dimension=local_dimension,
                                total_copies=config.budget,
                                iterations=config.iterations,
                                radius=prior_radius,
                            )
                        )
                        posterior_rng = np.random.default_rng(seed_base + 530)
                        paqt_particles = base.make_particle_cloud(
                            rng=posterior_rng,
                            dimension=local_dimension,
                            radius=prior_radius,
                            count=config.smc_particles,
                        )
                        paqt_posterior = (
                            base.run_liu_west_particle_posterior_from_measurements(
                                rng=posterior_rng,
                                family=model.family,
                                particle_thetas=paqt_particles,
                                probe_states=probes,
                                counts=paqt_counts,
                                shot_counts=paqt_shots,
                                radius=prior_radius,
                                task_from_density=lambda density: (
                                    benchmark.task_values(density, prior_config)
                                ),
                                task_from_state=lambda state: benchmark.task_values(
                                    state, prior_config
                                ),
                                tasks_from_states=lambda states: (
                                    benchmark.batch_task_values(states, prior_config)
                                ),
                            )
                        )
                        trial_rows.append(
                            method_row(
                                dimension=dimension,
                                local_dimension=local_dimension,
                                truth_radius=truth_radius,
                                prior_radius=prior_radius,
                                budget=config.budget,
                                trial=trial,
                                method="structured_paqt",
                                settings=int(paqt_shots.size),
                                truth_norm=truth_norm,
                                estimate=paqt_posterior.task_estimate,
                                truth_task=truth_task,
                                estimate_coordinate=paqt_posterior.theta_estimate,
                                paqt_resampling_count=(
                                    paqt_posterior.resampling_count
                                ),
                                paqt_minimum_ess=paqt_posterior.minimum_ess,
                                paqt_final_ess=paqt_posterior.ess,
                            )
                        )

                    if {"structured_sgqt", "structured_osgqt"} & missing_methods:
                        sgqt = base.run_structured_sgqt(
                            rng=np.random.default_rng(seed_base + 50),
                            truth_state=truth_state,
                            family=model.family,
                            dimension=local_dimension,
                            total_copies=config.budget,
                            iterations=config.iterations,
                            radius=prior_radius,
                            osgqt=False,
                        )
                        osgqt = base.run_structured_sgqt(
                            rng=np.random.default_rng(seed_base + 50),
                            truth_state=truth_state,
                            family=model.family,
                            dimension=local_dimension,
                            total_copies=config.budget,
                            iterations=config.iterations,
                            radius=prior_radius,
                            osgqt=True,
                        )
                        for method, result in (
                            ("structured_sgqt", sgqt),
                            ("structured_osgqt", osgqt),
                        ):
                            if method not in missing_methods:
                                continue
                            trial_rows.append(
                                method_row(
                                    dimension=dimension,
                                    local_dimension=local_dimension,
                                    truth_radius=truth_radius,
                                    prior_radius=prior_radius,
                                    budget=config.budget,
                                    trial=trial,
                                    method=method,
                                    settings=int(result.settings),
                                    truth_norm=truth_norm,
                                    estimate=benchmark.task_values(
                                        result.state,
                                        prior_config,
                                    ),
                                    truth_task=truth_task,
                                    estimate_coordinate=None,
                                )
                            )
                    for row in trial_rows:
                        key = row_key(row)
                        if key not in existing:
                            rows.append(row)
                            existing.add(key)

                base.write_union_csv(trial_path, rows)
                print(
                    f"d={dimension}, r_truth={truth_radius:.3g}, "
                    f"R_alpha={prior_radius:.3g}: {config.trials} trials complete",
                    flush=True,
                )

    summary_rows = summarize(rows)
    base.write_union_csv(output_dir / "summary_rows.csv", summary_rows)
    save_triangular_heatmap(
        output_dir / "triangular_raw_mse_heatmap.png",
        summary_rows,
        config,
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "methods": methods,
                "paqt_resampler": "Liu-West",
                "paqt_liu_west_a": base.PAQT_LIU_WEST_A,
                "paqt_resample_ess_fraction": (
                    base.PAQT_RESAMPLE_ESS_FRACTION
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary_lines = [
        "# Truth-radius/prior-radius decoupling",
        "",
        "The experiment evaluates only cells with `truth_radius <= prior_radius`.",
        "Truths use paired unit-ball coordinates, so the diagonal reproduces the "
        "original coupled data-generating mechanism in distribution.",
        "Every reported endpoint is the Euclidean Schmidt-spectrum squared error.",
        "S-PAQT reports the posterior mean of the particle task values. "
        "S-SGQT and S-OSGQT use the matched alpha=0.05, beta=0.2 comparison gains.",
        "",
        f"- dimensions: `{list(config.dimensions)}`",
        f"- radii: `{list(config.radii)}`",
        f"- budget: `{config.budget}`",
        f"- trials per valid cell: `{config.trials}`",
        f"- valid cells per dimension: `{sum(valid_cell(a, b) for a in config.radii for b in config.radii)}`",
        "- dashed heatmap cells: coupled diagonal `truth_radius = prior_radius`",
    ]
    (output_dir / "summary.md").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )
    print(f"Saved decoupling experiment to {output_dir.resolve()}", flush=True)


def config_for_mode(mode: str) -> TruthPriorDecouplingConfig:
    if mode == "full":
        return TruthPriorDecouplingConfig()
    if mode == "smoke":
        return TruthPriorDecouplingConfig(
            dimensions=(6,),
            radii=(0.01, 0.02),
            budget=19200,
            trials=2,
            iterations=4,
            scale_particles=200,
            smc_particles=120,
            smc_mutation_steps=1,
            smc_max_temperatures=30,
        )
    raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from trial_results.csv and skip completed method rows.",
    )
    parser.add_argument(
        "--refresh-competitors",
        action="store_true",
        help="Replace only S-PAQT/S-SGQT/S-OSGQT rows.",
    )
    parser.add_argument(
        "--refresh-local",
        action="store_true",
        help="Replace only the Q-projected greedy_local rows.",
    )
    parser.add_argument(
        "--refresh-fixed",
        action="store_true",
        help="Replace both fixed-design local and nonlinear rows.",
    )
    parser.add_argument("--dimension", action="append", type=int)
    parser.add_argument("--truth-radius", action="append", type=float)
    parser.add_argument("--prior-radius", action="append", type=float)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--merge-shards",
        action="store_true",
        help="Merge output-dir/shards after validating the complete full grid.",
    )
    parser.add_argument(
        "--merge-refresh-shards",
        action="store_true",
        help="Merge only shards whose directory names start with refresh_.",
    )
    args = parser.parse_args()
    config = config_for_mode(args.mode)
    output_dir = args.output_dir or (
        OUTPUT_DIR if args.mode == "full" else OUTPUT_DIR / "smoke"
    )
    if args.merge_shards:
        merge_sharded_results(config, output_dir, methods=tuple(args.methods))
        return
    if args.merge_refresh_shards:
        merge_sharded_results(
            config,
            output_dir,
            shard_prefix="refresh_",
            methods=tuple(args.methods),
        )
        return
    run(
        config,
        output_dir,
        resume=(
            args.resume
            or args.refresh_competitors
            or args.refresh_local
            or args.refresh_fixed
        ),
        refresh_competitors=args.refresh_competitors,
        refresh_local=args.refresh_local,
        refresh_fixed=args.refresh_fixed,
        selected_dimensions=tuple(args.dimension) if args.dimension else None,
        selected_truth_radii=(
            tuple(args.truth_radius) if args.truth_radius else None
        ),
        selected_prior_radii=(
            tuple(args.prior_radius) if args.prior_radius else None
        ),
        methods=tuple(args.methods),
    )


if __name__ == "__main__":
    main()
