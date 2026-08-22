"""Exploratory radius sensitivity for the fixed Greedy quantum design.

This experiment supplies the manuscript's localization-radius sensitivity
analysis.  It sweeps the pre-data localization radius and copy budget and
compares estimators from identical counts:

1. the local geodesic point estimate;
2. a tempered-SMC approximation to the full nonlinear likelihood posterior;
   and
3. an additional confidence-conditioned posterior retained for diagnostics.

The nonlinear posterior is approximated by adaptive likelihood tempering from the
uniform-ball prior.  Resampling and Metropolis mutation prevent the
large-radius/high-budget posterior from collapsing onto a few importance
particles when the fidelity level sets become curved or multimodal.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import qiskit_quantum_backend as quantum
import quantum_benchmark_support as base
import quantum_greedy_spectral_experiment as benchmark


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = base.ACTIVE_EXPERIMENT_ROOT / "quantum_radius_sensitivity"

# Keep the original streams unchanged when inserting new radii into the sweep.
# New radii receive fresh indices rather than shifting the established points.
RADIUS_SEED_INDICES = {
    0.01: 0,
    0.02: 1,
    0.04: 2,
    0.08: 3,
    0.16: 4,
    0.03: 5,
}


def radius_seed_index(radius: float) -> int:
    for known_radius, index in RADIUS_SEED_INDICES.items():
        if math.isclose(radius, known_radius):
            return index
    raise ValueError(f"No stable seed index assigned to radius {radius:g}")


@dataclass(frozen=True)
class RadiusSensitivityConfig:
    seed: int = 20260731
    dimensions: tuple[int, ...] = (6, 17)
    radii: tuple[float, ...] = (0.01, 0.02, 0.03, 0.04, 0.08, 0.16)
    budgets: tuple[int, ...] = (153600,)
    joint_failure_probabilities: tuple[float, ...] = (0.05,)
    trials: int = 30
    scale_particles: int = 3000
    smc_particles: int = 500
    smc_ess_fraction: float = 0.55
    smc_mutation_steps: int = 3
    smc_max_temperatures: int = 60
    branch_radius_factor: float = 1.25
    anchor_radius: float = math.pi / 4.0


@dataclass(frozen=True)
class LocalUpdate:
    counts: np.ndarray
    truth_probabilities: np.ndarray
    observed_distances: np.ndarray
    precision: np.ndarray
    covariance: np.ndarray
    primary_mean: np.ndarray
    branch_means: np.ndarray


@dataclass(frozen=True)
class TemperedCloud:
    coordinates: np.ndarray
    probabilities: np.ndarray
    weights: np.ndarray
    temperature_count: int
    mean_acceptance_rate: float


def task_squared_error(
    estimate: np.ndarray,
    truth: np.ndarray,
    scale: np.ndarray,
) -> float:
    return float(np.sum(((np.asarray(estimate) - np.asarray(truth)) / scale) ** 2))


def unscaled_task_squared_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sum((np.asarray(estimate) - np.asarray(truth)) ** 2))


def local_task_loss(
    estimate: np.ndarray,
    truth: np.ndarray,
    task_metric: np.ndarray,
) -> float:
    difference = np.asarray(estimate) - np.asarray(truth)
    return float(difference @ task_metric @ difference)


def local_update_and_branches(
    *,
    rng: np.random.Generator,
    truth_state: np.ndarray,
    model: benchmark.LocalModel,
    anchor_states: np.ndarray,
    shot_counts: np.ndarray,
    directions: np.ndarray,
    task_metric: np.ndarray,
    radius: float,
    anchor_radius: float,
    branch_radius_factor: float,
) -> LocalUpdate:
    """Generate counts and construct all locally plausible inversion branches."""

    truth_probabilities = quantum.fidelities_to_state(anchor_states, truth_state)
    counts = rng.binomial(shot_counts, truth_probabilities)
    observed_probabilities = np.clip(
        counts / shot_counts,
        1e-12,
        1.0 - 1e-12,
    )
    observed_distances = np.arccos(np.sqrt(observed_probabilities))
    local_dimension = model.coordinate_map.shape[1]
    alpha = (local_dimension + 2.0) / (radius**2)
    information = 4.0
    precision = alpha * np.eye(local_dimension)
    for shots, direction in zip(shot_counts, directions.T):
        precision += information * shots * np.outer(direction, direction)
    covariance = np.linalg.inv(precision)

    branch_candidates: list[np.ndarray] = []
    primary_mean: np.ndarray | None = None
    for branch_signs in itertools.product((-1.0, 1.0), repeat=shot_counts.size):
        scores = anchor_radius + np.asarray(branch_signs) * observed_distances
        right_hand_side = np.zeros(local_dimension, dtype=float)
        for shots, direction, score in zip(shot_counts, directions.T, scores):
            right_hand_side += information * shots * direction * score
        raw_mean = np.linalg.solve(precision, right_hand_side)
        if all(sign < 0.0 for sign in branch_signs):
            primary_mean = base.project_to_ball_in_metric(
                raw_mean,
                task_metric,
                radius,
            )
        if np.linalg.norm(raw_mean) <= branch_radius_factor * radius:
            branch_candidates.append(base.project_to_ball(raw_mean, radius))

    if primary_mean is None:
        raise RuntimeError("The principal fidelity-inversion branch was not constructed.")
    branch_candidates.insert(0, primary_mean)
    unique: list[np.ndarray] = []
    for candidate in branch_candidates:
        if not any(np.linalg.norm(candidate - existing) <= 1e-9 for existing in unique):
            unique.append(candidate)
    return LocalUpdate(
        counts=np.asarray(counts, dtype=int),
        truth_probabilities=np.asarray(truth_probabilities, dtype=float),
        observed_distances=np.asarray(observed_distances, dtype=float),
        precision=precision,
        covariance=covariance,
        primary_mean=primary_mean,
        branch_means=np.asarray(unique, dtype=float),
    )




def batch_ground_states(
    model: benchmark.LocalModel,
    coordinates: np.ndarray,
) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=float)
    matrices = (
        model.family.base_hamiltonian[None, :, :]
        + np.tensordot(
            coordinates,
            np.asarray(model.family.terms, dtype=float),
            axes=(1, 0),
        )
    )
    _, eigenvectors = np.linalg.eigh(matrices)
    states = np.asarray(eigenvectors[:, :, 0], dtype=float)
    signs = np.where(np.sum(states, axis=1) < 0.0, -1.0, 1.0)
    return states * signs[:, None]


def batch_fidelity_probabilities(
    model: benchmark.LocalModel,
    coordinates: np.ndarray,
    anchor_states: np.ndarray,
) -> np.ndarray:
    states = batch_ground_states(model, coordinates)
    overlaps = states.conj() @ np.asarray(anchor_states).T
    return np.clip(np.abs(overlaps) ** 2, 1e-14, 1.0 - 1e-14)


def batch_binomial_log_likelihood(
    probabilities: np.ndarray,
    counts: np.ndarray,
    shot_counts: np.ndarray,
) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-14, 1.0 - 1e-14)
    return np.sum(
        counts[None, :] * np.log(probabilities)
        + (shot_counts - counts)[None, :] * np.log1p(-probabilities),
        axis=1,
    )


def systematic_resample(
    rng: np.random.Generator,
    weights: np.ndarray,
) -> np.ndarray:
    count = weights.size
    positions = (rng.random() + np.arange(count)) / count
    return np.searchsorted(np.cumsum(weights), positions, side="right")


def tempered_smc_cloud(
    *,
    rng: np.random.Generator,
    model: benchmark.LocalModel,
    anchor_states: np.ndarray,
    counts: np.ndarray,
    shot_counts: np.ndarray,
    radius: float,
    particle_count: int,
    ess_fraction: float,
    mutation_steps: int,
    max_temperatures: int,
) -> TemperedCloud:
    """Sample the exact uniform-ball posterior by adaptive likelihood tempering."""

    dimension = model.coordinate_map.shape[1]
    coordinates = base.sample_ball(rng, dimension, radius, particle_count)
    probabilities = batch_fidelity_probabilities(
        model,
        coordinates,
        anchor_states,
    )
    log_likelihood = batch_binomial_log_likelihood(
        probabilities,
        counts,
        shot_counts,
    )
    temperature = 0.0
    acceptance_rates: list[float] = []
    temperature_count = 0
    target_ess = ess_fraction * particle_count

    while temperature < 1.0 - 1e-12:
        def incremental_ess(next_temperature: float) -> float:
            log_weights = (next_temperature - temperature) * log_likelihood
            log_weights -= float(np.max(log_weights))
            weights = np.exp(log_weights)
            weights /= float(np.sum(weights))
            return float(1.0 / np.sum(weights**2))

        if incremental_ess(1.0) >= target_ess:
            next_temperature = 1.0
        else:
            lower = temperature
            upper = 1.0
            for _ in range(40):
                midpoint = 0.5 * (lower + upper)
                if incremental_ess(midpoint) < target_ess:
                    upper = midpoint
                else:
                    lower = midpoint
            next_temperature = lower
        log_weights = (next_temperature - temperature) * log_likelihood
        log_weights -= float(np.max(log_weights))
        weights = np.exp(log_weights)
        weights /= float(np.sum(weights))
        indices = systematic_resample(rng, weights)
        coordinates = coordinates[indices]
        probabilities = probabilities[indices]
        log_likelihood = log_likelihood[indices]
        temperature = next_temperature
        temperature_count += 1

        covariance = np.cov(coordinates, rowvar=False, ddof=1)
        covariance = 0.5 * (covariance + covariance.T)
        covariance += 1e-8 * radius**2 * np.eye(dimension)
        proposal_scale = 1.2 / math.sqrt(dimension)
        cholesky = np.linalg.cholesky(covariance)
        for _ in range(mutation_steps):
            proposals = coordinates + proposal_scale * (
                rng.normal(size=coordinates.shape) @ cholesky.T
            )
            inside = np.linalg.norm(proposals, axis=1) <= radius
            proposed_log_likelihood = np.full(particle_count, -np.inf, dtype=float)
            proposed_probabilities = np.empty_like(probabilities)
            if np.any(inside):
                proposed_probabilities[inside] = batch_fidelity_probabilities(
                    model,
                    proposals[inside],
                    anchor_states,
                )
                proposed_log_likelihood[inside] = batch_binomial_log_likelihood(
                    proposed_probabilities[inside],
                    counts,
                    shot_counts,
                )
            log_acceptance = temperature * (
                proposed_log_likelihood - log_likelihood
            )
            accepted = np.log(rng.random(particle_count)) < log_acceptance
            coordinates[accepted] = proposals[accepted]
            probabilities[accepted] = proposed_probabilities[accepted]
            log_likelihood[accepted] = proposed_log_likelihood[accepted]
            acceptance_rates.append(float(np.mean(accepted)))

        if temperature_count >= max_temperatures and temperature < 1.0 - 1e-12:
            raise RuntimeError(
                "Tempered SMC exceeded its temperature limit before reaching the posterior."
            )

    return TemperedCloud(
        coordinates=coordinates,
        probabilities=probabilities,
        weights=np.full(particle_count, 1.0 / particle_count, dtype=float),
        temperature_count=temperature_count,
        mean_acceptance_rate=float(np.mean(acceptance_rates)),
    )


def summarize(rows: list[dict]) -> list[dict]:
    keys = sorted(
        {
            (
                int(row["dimension"]),
                float(row["radius"]),
                int(row["budget"]),
                float(row["joint_failure_probability"]),
            )
            for row in rows
        }
    )
    output: list[dict] = []
    for dimension, radius, budget, failure in keys:
        subset = [
            row
            for row in rows
            if int(row["dimension"]) == dimension
            and float(row["radius"]) == radius
            and int(row["budget"]) == budget
            and float(row["joint_failure_probability"]) == failure
        ]

        def finite(field: str) -> np.ndarray:
            return np.asarray(
                [
                    float(row[field])
                    for row in subset
                    if np.isfinite(float(row[field]))
                ],
                dtype=float,
            )

        def mean(field: str) -> float:
            values = finite(field)
            return float(np.mean(values)) if values.size else float("nan")

        def standard_error(field: str) -> float:
            values = finite(field)
            if values.size > 1:
                return float(np.std(values, ddof=1) / math.sqrt(values.size))
            return 0.0 if values.size == 1 else float("nan")

        gaussian_mse = mean("gaussian_task_squared_error")
        bayes_mse = mean("bayes_task_squared_error")
        certified_mse = mean("certified_task_squared_error")
        output.append(
            {
                "dimension": dimension,
                "local_dimension": int(subset[0]["local_dimension"]),
                "radius": radius,
                "budget": budget,
                "joint_confidence_level": 1.0 - failure,
                "joint_failure_probability": failure,
                "active_settings": int(subset[0]["active_settings"]),
                "allocation": subset[0]["allocation"],
                "predicted_local_risk": float(subset[0]["predicted_local_risk"]),
                "mean_gaussian_task_mse": gaussian_mse,
                "se_gaussian_task_mse": standard_error("gaussian_task_squared_error"),
                "mean_bayes_task_mse": bayes_mse,
                "se_bayes_task_mse": standard_error("bayes_task_squared_error"),
                "mean_certified_task_mse": certified_mse,
                "se_certified_task_mse": standard_error("certified_task_squared_error"),
                "gaussian_excess_over_bayes_percent": 100.0
                * (gaussian_mse / bayes_mse - 1.0),
                "certified_change_from_bayes_percent": 100.0
                * (certified_mse / bayes_mse - 1.0),
                "certified_change_from_gaussian_percent": 100.0
                * (certified_mse / gaussian_mse - 1.0),
                "mean_gaussian_unscaled_task_mse": mean(
                    "gaussian_unscaled_task_squared_error"
                ),
                "se_gaussian_unscaled_task_mse": standard_error(
                    "gaussian_unscaled_task_squared_error"
                ),
                "mean_bayes_unscaled_task_mse": mean(
                    "bayes_unscaled_task_squared_error"
                ),
                "se_bayes_unscaled_task_mse": standard_error(
                    "bayes_unscaled_task_squared_error"
                ),
                "mean_certified_unscaled_task_mse": mean(
                    "certified_unscaled_task_squared_error"
                ),
                "se_certified_unscaled_task_mse": standard_error(
                    "certified_unscaled_task_squared_error"
                ),
                "mean_gaussian_local_loss": mean("gaussian_local_loss"),
                "mean_bayes_local_loss": mean("bayes_local_loss"),
                "mean_certified_local_loss": mean("certified_local_loss"),
                "gaussian_local_risk_ratio": mean("gaussian_local_loss")
                / float(subset[0]["predicted_local_risk"]),
                "mean_prior_volume_removed_percent": mean(
                    "prior_volume_removed_percent"
                ),
                "mean_bayes_posterior_mass_retained": mean(
                    "bayes_posterior_mass_retained"
                ),
                "empirical_truth_coverage": mean("truth_in_support"),
                "posterior_availability_rate": mean("certified_available"),
                "mean_prior_importance_ess": mean("prior_importance_ess"),
                "mean_bayes_ess": mean("bayes_ess"),
                "mean_certified_ess": mean("certified_ess"),
                "minimum_bayes_ess": float(np.min(finite("bayes_ess"))),
                "minimum_certified_ess": float(np.min(finite("certified_ess"))),
                "mean_branch_count": mean("branch_count"),
                "mean_smc_temperature_count": mean("smc_temperature_count"),
                "mean_smc_acceptance_rate": mean("smc_mean_acceptance_rate"),
                "maximum_component_standardized_residual": float(
                    np.max(finite("maximum_component_standardized_residual"))
                ),
                "minimum_truncation_probability": float(
                    np.min(finite("minimum_truncation_probability"))
                ),
                "trials": len(subset),
                "available_trials": int(np.sum(finite("certified_available"))),
            }
        )
    return output


def save_mse_plot(path: Path, rows: list[dict], config: RadiusSensitivityConfig) -> None:
    confidence_failure = 0.05
    fig, axes = plt.subplots(
        len(config.dimensions),
        len(config.budgets),
        figsize=(12.0, 6.7),
        sharex=True,
    )
    axes = np.asarray(axes).reshape(len(config.dimensions), len(config.budgets))
    for row_index, dimension in enumerate(config.dimensions):
        for column_index, budget in enumerate(config.budgets):
            axis = axes[row_index, column_index]
            subset = sorted(
                [
                    row
                    for row in rows
                    if int(row["dimension"]) == dimension
                    and int(row["budget"]) == budget
                    and math.isclose(
                        float(row["joint_failure_probability"]),
                        confidence_failure,
                    )
                ],
                key=lambda row: float(row["radius"]),
            )
            for field, error_field, label, marker in (
                (
                    "mean_gaussian_unscaled_task_mse",
                    "se_gaussian_unscaled_task_mse",
                    "Local Gaussian",
                    "o",
                ),
                (
                    "mean_bayes_unscaled_task_mse",
                    "se_bayes_unscaled_task_mse",
                    "Tempered SMC",
                    "s",
                ),
                (
                    "mean_certified_unscaled_task_mse",
                    "se_certified_unscaled_task_mse",
                    r"Tempered SMC $\mid S_{95}$",
                    "^",
                ),
            ):
                axis.errorbar(
                    [float(row["radius"]) for row in subset],
                    [float(row[field]) for row in subset],
                    yerr=[float(row[error_field]) for row in subset],
                    marker=marker,
                    linewidth=1.7,
                    capsize=2.5,
                    label=label,
                )
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.grid(alpha=0.25)
            axis.set_title(f"$d={dimension}$, $B={budget}$")
            axis.set_xlabel("localization radius $R$")
            axis.set_ylabel(
                r"raw Schmidt-spectrum MSE $\|\widehat{\tau}-\tau\|_2^2$"
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_support_plot(
    path: Path,
    rows: list[dict],
    config: RadiusSensitivityConfig,
) -> None:
    budget = max(config.budgets)
    fig, axes = plt.subplots(2, len(config.dimensions), figsize=(9.2, 6.5), sharex="col")
    if len(config.dimensions) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for column, dimension in enumerate(config.dimensions):
        for failure in config.joint_failure_probabilities:
            subset = sorted(
                [
                    row
                    for row in rows
                    if int(row["dimension"]) == dimension
                    and int(row["budget"]) == budget
                    and math.isclose(float(row["joint_failure_probability"]), failure)
                ],
                key=lambda row: float(row["radius"]),
            )
            label = f"{100 * (1.0 - failure):.0f}% confidence"
            axes[0, column].plot(
                [float(row["radius"]) for row in subset],
                [float(row["mean_prior_volume_removed_percent"]) for row in subset],
                marker="o",
                linewidth=1.7,
                label=label,
            )
            axes[1, column].plot(
                [float(row["radius"]) for row in subset],
                [float(row["certified_change_from_bayes_percent"]) for row in subset],
                marker="s",
                linewidth=1.7,
                label=label,
            )
        axes[0, column].set_xscale("log", base=2)
        axes[0, column].grid(alpha=0.25)
        axes[0, column].set_title(f"$d={dimension}$, $B={budget}$")
        axes[0, column].set_ylabel("prior volume removed (%)")
        axes[1, column].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        axes[1, column].grid(alpha=0.25)
        axes[1, column].set_xlabel("localization radius $R$")
        axes[1, column].set_ylabel(r"$S$ MSE change from SMC (%)")
        axes[1, column].set_xlim(min(config.radii) / 1.08, max(config.radii) * 1.08)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def write_results(
    config: RadiusSensitivityConfig,
    output_dir: Path,
    rows: list[dict],
    geometry_rows: list[dict],
) -> None:
    """Write aggregates and figures shared by full and local-only refreshes."""

    summary_rows = summarize(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    base.write_union_csv(output_dir / "trial_results.csv", rows)
    base.write_union_csv(output_dir / "summary_rows.csv", summary_rows)
    base.write_union_csv(output_dir / "geometry.csv", geometry_rows)
    (output_dir / "config.json").write_text(
        json.dumps(config.__dict__, indent=2),
        encoding="utf-8",
    )
    summary_lines = [
        "# Quantum Radius Sensitivity",
        "",
        "This sweep is reported in the manuscript's radius-sensitivity analysis.",
        "",
        *base.markdown_table(
            summary_rows,
            [
                "dimension",
                "radius",
                "budget",
                "joint_confidence_level",
                "active_settings",
                "mean_gaussian_task_mse",
                "mean_bayes_task_mse",
                "mean_certified_task_mse",
                "gaussian_excess_over_bayes_percent",
                "certified_change_from_bayes_percent",
                "certified_change_from_gaussian_percent",
                "gaussian_local_risk_ratio",
                "mean_prior_volume_removed_percent",
                "mean_bayes_posterior_mass_retained",
                "empirical_truth_coverage",
                "posterior_availability_rate",
                "mean_bayes_ess",
                "minimum_bayes_ess",
                "mean_certified_ess",
                "minimum_certified_ess",
                "mean_smc_temperature_count",
                "mean_smc_acceptance_rate",
            ],
        ),
    ]
    (output_dir / "summary.md").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )
    save_mse_plot(output_dir / "radius_mse.png", summary_rows, config)
    save_support_plot(output_dir / "radius_support.png", summary_rows, config)
    print(f"Saved radius sensitivity experiment to {output_dir.resolve()}", flush=True)


def run(config: RadiusSensitivityConfig, output_dir: Path) -> None:
    rows: list[dict] = []
    geometry_rows: list[dict] = []
    base_config = benchmark.GreedyTaskConfig(
        anchor_radius=config.anchor_radius,
        budgets=config.budgets,
    )
    for dimension in config.dimensions:
        model = benchmark.build_local_model(base_config, dimension)
        local_dimension = model.coordinate_map.shape[1]
        for radius in config.radii:
            radius_index = radius_seed_index(radius)
            radius_config = replace(
                base_config,
                particle_radius=radius,
                truth_radius=radius,
                budgets=config.budgets,
            )
            scale_coordinates = base.sample_ball(
                np.random.default_rng(config.seed + 10_000 * dimension + radius_index),
                local_dimension,
                radius,
                config.scale_particles,
            )
            scale_states = np.asarray(
                [
                    quantum.ground_state(model.family, coordinate)
                    for coordinate in scale_coordinates
                ]
            )
            geometry = benchmark.compute_task_geometry(model, radius_config)
            designs_by_budget: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            audit_by_budget: dict[int, dict[str, object]] = {}
            for budget in config.budgets:
                designs, audit = benchmark.spectral_designs(
                    model,
                    geometry,
                    radius_config,
                    budget,
                )
                designs_by_budget[budget] = designs["greedy_spectral"]
                audit_by_budget[budget] = audit

            print(
                f"d={dimension}, R={radius:.3g}: scale and designs ready",
                flush=True,
            )
            for trial in range(config.trials):
                unit_coordinate = base.sample_ball(
                    np.random.default_rng(config.seed + 100_000 * dimension + trial),
                    local_dimension,
                    1.0,
                    1,
                )[0]
                truth_coordinate = radius * unit_coordinate
                truth_state = quantum.ground_state(model.family, truth_coordinate)
                truth_task = benchmark.task_values(truth_state, radius_config)
                for budget_index, budget in enumerate(config.budgets):
                    anchor_states, shot_counts, directions = designs_by_budget[budget]
                    update = local_update_and_branches(
                        rng=np.random.default_rng(
                            config.seed
                            + 1_000_000 * dimension
                            + 10_000 * radius_index
                            + 100 * trial
                            + budget_index
                        ),
                        truth_state=truth_state,
                        model=model,
                        anchor_states=anchor_states,
                        shot_counts=shot_counts,
                        directions=directions,
                        task_metric=geometry.task_metric,
                        radius=radius,
                        anchor_radius=config.anchor_radius,
                        branch_radius_factor=config.branch_radius_factor,
                    )
                    cloud = tempered_smc_cloud(
                        rng=np.random.default_rng(
                            config.seed
                            + 2_000_000 * dimension
                            + 20_000 * radius_index
                            + 200 * trial
                            + budget_index
                        ),
                        model=model,
                        anchor_states=anchor_states,
                        counts=update.counts,
                        shot_counts=shot_counts,
                        radius=radius,
                        particle_count=config.smc_particles,
                        ess_fraction=config.smc_ess_fraction,
                        mutation_steps=config.smc_mutation_steps,
                        max_temperatures=config.smc_max_temperatures,
                    )
                    particle_states = batch_ground_states(model, cloud.coordinates)
                    particle_tasks = np.asarray(
                        [
                            benchmark.task_values(state, radius_config)
                            for state in particle_states
                        ]
                    )
                    particle_probabilities = cloud.probabilities
                    prior_probabilities = np.column_stack(
                        [
                            quantum.fidelities_to_state(scale_states, anchor)
                            for anchor in anchor_states
                        ]
                    )
                    bayes_weights = cloud.weights
                    bayes_task = bayes_weights @ particle_tasks
                    bayes_theta = bayes_weights @ cloud.coordinates
                    bayes_ess = float(1.0 / np.sum(bayes_weights**2))
                    gaussian_state = quantum.ground_state(
                        model.family,
                        update.primary_mean,
                    )
                    gaussian_task = benchmark.task_values(gaussian_state, radius_config)
                    audit = audit_by_budget[budget]

                    for failure in config.joint_failure_probabilities:
                        individual_failures = np.full(
                            shot_counts.size,
                            failure / shot_counts.size,
                        )
                        lower_bounds = base.exact_binomial_lower_confidence_bounds(
                            update.counts,
                            shot_counts,
                            individual_failures,
                        )
                        support_mask = np.all(
                            particle_probabilities >= lower_bounds[None, :],
                            axis=1,
                        )
                        prior_support_mask = np.all(
                            prior_probabilities >= lower_bounds[None, :],
                            axis=1,
                        )
                        prior_fraction = float(np.mean(prior_support_mask))
                        posterior_mass = float(
                            np.clip(np.sum(bayes_weights[support_mask]), 0.0, 1.0)
                        )
                        available = bool(
                            np.any(support_mask) and posterior_mass > 0.0
                        )
                        if available:
                            certified_weights = np.where(
                                support_mask,
                                bayes_weights,
                                0.0,
                            )
                            certified_weights /= posterior_mass
                            certified_task = certified_weights @ particle_tasks
                            certified_theta = certified_weights @ cloud.coordinates
                            certified_ess = float(
                                1.0 / np.sum(certified_weights**2)
                            )
                        else:
                            certified_task = np.full_like(truth_task, np.nan)
                            certified_theta = np.full_like(truth_coordinate, np.nan)
                            certified_ess = 0.0
                        rows.append(
                            {
                                "dimension": dimension,
                                "local_dimension": local_dimension,
                                "radius": radius,
                                "budget": budget,
                                "trial": trial,
                                "joint_failure_probability": failure,
                                "joint_confidence_level": 1.0 - failure,
                                "active_settings": int(shot_counts.size),
                                "allocation": str(shot_counts.tolist()),
                                "predicted_local_risk": float(
                                    audit["greedy_local_risk"]
                                ),
                                "truth_radius": float(np.linalg.norm(truth_coordinate)),
                                "gaussian_task_squared_error": task_squared_error(
                                    gaussian_task,
                                    truth_task,
                                    geometry.task_scale,
                                ),
                                "bayes_task_squared_error": task_squared_error(
                                    bayes_task,
                                    truth_task,
                                    geometry.task_scale,
                                ),
                                "certified_task_squared_error": (
                                    task_squared_error(
                                        certified_task,
                                        truth_task,
                                        geometry.task_scale,
                                    )
                                    if available
                                    else float("nan")
                                ),
                                "gaussian_unscaled_task_squared_error": (
                                    unscaled_task_squared_error(gaussian_task, truth_task)
                                ),
                                "bayes_unscaled_task_squared_error": (
                                    unscaled_task_squared_error(bayes_task, truth_task)
                                ),
                                "certified_unscaled_task_squared_error": (
                                    unscaled_task_squared_error(certified_task, truth_task)
                                    if available
                                    else float("nan")
                                ),
                                "gaussian_local_loss": local_task_loss(
                                    update.primary_mean,
                                    truth_coordinate,
                                    geometry.task_metric,
                                ),
                                "bayes_local_loss": local_task_loss(
                                    bayes_theta,
                                    truth_coordinate,
                                    geometry.task_metric,
                                ),
                                "certified_local_loss": (
                                    local_task_loss(
                                        certified_theta,
                                        truth_coordinate,
                                        geometry.task_metric,
                                    )
                                    if available
                                    else float("nan")
                                ),
                                "prior_support_fraction": prior_fraction,
                                "prior_volume_removed_percent": 100.0
                                * (1.0 - prior_fraction),
                                "bayes_posterior_mass_retained": posterior_mass,
                                "truth_in_support": int(
                                    np.all(update.truth_probabilities >= lower_bounds)
                                ),
                                "certified_available": int(available),
                                "prior_importance_ess": float(
                                    config.scale_particles
                                ),
                                "bayes_ess": bayes_ess,
                                "certified_ess": certified_ess,
                                "branch_count": 0,
                                "maximum_component_standardized_residual": 0.0,
                                "minimum_truncation_probability": 1.0,
                                "smc_temperature_count": cloud.temperature_count,
                                "smc_mean_acceptance_rate": (
                                    cloud.mean_acceptance_rate
                                ),
                            }
                        )
                if (trial + 1) % 5 == 0 or trial + 1 == config.trials:
                    print(
                        f"d={dimension}, R={radius:.3g}: "
                        f"completed {trial + 1}/{config.trials}",
                        flush=True,
                    )

            geometry_rows.append(
                {
                    "dimension": dimension,
                    "local_dimension": local_dimension,
                    "radius": radius,
                    "task_weighting": "identity (raw Euclidean spectrum loss)",
                    "task_eigenvalues": str(
                        geometry.generalized_eigenvalues[: geometry.task_rank].tolist()
                    ),
                    "allocations": json.dumps(
                        {
                            str(budget): audit_by_budget[budget]["greedy_allocation"]
                            for budget in config.budgets
                        },
                        sort_keys=True,
                    ),
                }
            )

    write_results(config, output_dir, rows, geometry_rows)


def refresh_local_results(config: RadiusSensitivityConfig, output_dir: Path) -> None:
    """Recompute only Q-projected local estimates from the original seeded counts."""

    trial_path = output_dir / "trial_results.csv"
    geometry_path = output_dir / "geometry.csv"
    if not trial_path.exists() or not geometry_path.exists():
        raise FileNotFoundError("A completed radius sweep is required for refresh.")
    with trial_path.open(newline="", encoding="utf-8") as handle:
        rows: list[dict] = list(csv.DictReader(handle))
    with geometry_path.open(newline="", encoding="utf-8") as handle:
        geometry_rows: list[dict] = list(csv.DictReader(handle))
    indexed: dict[tuple[int, float, int, int, float], dict] = {
        (
            int(row["dimension"]),
            float(row["radius"]),
            int(row["budget"]),
            int(row["trial"]),
            float(row["joint_failure_probability"]),
        ): row
        for row in rows
    }

    base_config = benchmark.GreedyTaskConfig(
        anchor_radius=config.anchor_radius,
        budgets=config.budgets,
    )
    updated = 0
    for dimension in config.dimensions:
        model = benchmark.build_local_model(base_config, dimension)
        local_dimension = model.coordinate_map.shape[1]
        for radius in config.radii:
            radius_index = radius_seed_index(radius)
            radius_config = replace(
                base_config,
                particle_radius=radius,
                truth_radius=radius,
                budgets=config.budgets,
            )
            scale_coordinates = base.sample_ball(
                np.random.default_rng(config.seed + 10_000 * dimension + radius_index),
                local_dimension,
                radius,
                config.scale_particles,
            )
            scale_states = batch_ground_states(model, scale_coordinates)
            geometry = benchmark.compute_task_geometry(model, radius_config)
            designs_by_budget = {
                budget: benchmark.spectral_designs(
                    model, geometry, radius_config, budget
                )[0]["greedy_spectral"]
                for budget in config.budgets
            }
            for trial in range(config.trials):
                unit_coordinate = base.sample_ball(
                    np.random.default_rng(config.seed + 100_000 * dimension + trial),
                    local_dimension,
                    1.0,
                    1,
                )[0]
                truth_coordinate = radius * unit_coordinate
                truth_state = quantum.ground_state(model.family, truth_coordinate)
                truth_task = benchmark.task_values(truth_state, radius_config)
                for budget_index, budget in enumerate(config.budgets):
                    anchor_states, shot_counts, directions = designs_by_budget[budget]
                    update = local_update_and_branches(
                        rng=np.random.default_rng(
                            config.seed
                            + 1_000_000 * dimension
                            + 10_000 * radius_index
                            + 100 * trial
                            + budget_index
                        ),
                        truth_state=truth_state,
                        model=model,
                        anchor_states=anchor_states,
                        shot_counts=shot_counts,
                        directions=directions,
                        task_metric=geometry.task_metric,
                        radius=radius,
                        anchor_radius=config.anchor_radius,
                        branch_radius_factor=config.branch_radius_factor,
                    )
                    gaussian_state = quantum.ground_state(model.family, update.primary_mean)
                    gaussian_task = benchmark.task_values(gaussian_state, radius_config)
                    for failure in config.joint_failure_probabilities:
                        key = (dimension, radius, budget, trial, failure)
                        if key not in indexed:
                            raise RuntimeError(f"Missing completed radius row: {key}")
                        row = indexed[key]
                        row["gaussian_task_squared_error"] = task_squared_error(
                            gaussian_task, truth_task, geometry.task_scale
                        )
                        row["gaussian_unscaled_task_squared_error"] = (
                            unscaled_task_squared_error(gaussian_task, truth_task)
                        )
                        row["gaussian_local_loss"] = local_task_loss(
                            update.primary_mean, truth_coordinate, geometry.task_metric
                        )
                        updated += 1
            print(f"d={dimension}, R={radius:.3g}: refreshed local rows", flush=True)
    write_results(config, output_dir, rows, geometry_rows)
    print(f"Refreshed {updated} Q-projected local scores.", flush=True)


def config_for_mode(mode: str) -> RadiusSensitivityConfig:
    config = RadiusSensitivityConfig()
    if mode == "smoke":
        return replace(
            config,
            dimensions=(6,),
            radii=(0.01, 0.08),
            budgets=(1200,),
            joint_failure_probabilities=(0.05,),
            trials=2,
            scale_particles=200,
            smc_particles=120,
            smc_mutation_steps=1,
        )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--refresh-local",
        action="store_true",
        help="Recompute only Q-projected local estimates in an existing sweep.",
    )
    arguments = parser.parse_args()
    config = config_for_mode(str(arguments.mode))
    output_dir = OUTPUT_DIR if arguments.mode == "full" else OUTPUT_DIR / "smoke"
    if arguments.refresh_local:
        refresh_local_results(config, output_dir)
    else:
        run(config, output_dir)


if __name__ == "__main__":
    main()
