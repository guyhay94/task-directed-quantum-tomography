"""Faithful fixed-radius test of the manuscript's greedy spectral algorithm.

The task is the odd-even Schmidt spectrum of a six-qubit ground state.  The
structured model is locally whitened by the pullback fidelity metric, the
particle prior is uniform in those whitened coordinates, and projective-
geodesic anchors are placed at one common Fubini--Study radius.  Direct
projective fidelity readout then gives the common local information coefficient
nu=4 required by the theorem.

Nonadaptive controls isolate the benefits of task directions and
diminishing-marginal-gain allocation.  Adaptive competitors are run by the
separate four-schedule round-sensitivity entry point.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import local_information_anchor_design as anchor_design
import qiskit_quantum_backend as quantum
import quantum_benchmark_support as base


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = base.ACTIVE_EXPERIMENT_ROOT / "quantum_greedy_spectral"


@dataclass(frozen=True)
class GreedyTaskConfig:
    seed: int = 20260811
    n_qubits: int = 6
    transverse_field: float = 0.97
    fixed_disorder_strength: float = 0.08
    perturbation_order: str = "x_z_zz"
    original_dimensions: tuple[int, ...] = (6, 12, 17)
    budgets: tuple[int, ...] = (19200, 38400, 76800, 153600)
    n_trials: int = 30
    n_design_trials: int = 300
    n_particles: int = 8000
    n_posterior_particles: int = 20000
    particle_radius: float = 0.02
    truth_radius: float = 0.02
    anchor_radius: float = math.pi / 4.0
    finite_difference_step: float = 0.01
    metric_relative_tolerance: float = 1e-8
    task_relative_tolerance: float = 1e-9
    task_subsystem: tuple[int, ...] = (0, 2, 4)


@dataclass(frozen=True)
class LocalModel:
    original_family: quantum.QiskitHamiltonianFamily
    family: quantum.QiskitHamiltonianFamily
    coordinate_map: np.ndarray
    center_state: np.ndarray
    tangent_matrix: np.ndarray
    raw_metric_eigenvalues: np.ndarray
    original_task_jacobian: np.ndarray


@dataclass(frozen=True)
class TaskGeometry:
    task_scale: np.ndarray
    task_jacobian: np.ndarray
    task_metric: np.ndarray
    generalized_eigenvalues: np.ndarray
    generalized_eigenvectors: np.ndarray
    task_rank: int


ANCHOR_METHODS = (
    "greedy_spectral",
    "equal_spectral",
    "lambda_spectral",
    "coordinate_equal",
    "random_equal",
    "nuisance_equal",
)

POSTERIOR_METHODS = (
    "greedy_spectral",
    "equal_spectral",
)

METHOD_LABELS = {
    "greedy_spectral": "Greedy spectral",
    "equal_spectral": "Equal spectral",
    "lambda_spectral": r"$\lambda$-proportional spectral",
    "coordinate_equal": "Coordinate anchors",
    "random_equal": "Random anchors",
    "nuisance_equal": "Nuisance anchors",
}

# Keep curves distinguishable in grayscale and for readers with color-vision
# deficiencies.  The color cycle remains as a secondary visual cue.
METHOD_STYLES = {
    "greedy_spectral": {"marker": "o", "linestyle": "-"},
    "equal_spectral": {"marker": "s", "linestyle": "--"},
    "lambda_spectral": {"marker": "^", "linestyle": "-."},
    "coordinate_equal": {"marker": "D", "linestyle": "--"},
    "random_equal": {"marker": "P", "linestyle": "-."},
    "nuisance_equal": {"marker": "X", "linestyle": ":"},
}


def task_values(
    state: np.ndarray,
    config: GreedyTaskConfig,
) -> np.ndarray:
    return quantum.subsystem_density_probabilities(
        state,
        config.n_qubits,
        config.task_subsystem,
    )


def batch_task_values(
    states: np.ndarray,
    config: GreedyTaskConfig,
) -> np.ndarray:
    """Return pure-state Schmidt spectra for a batch of statevectors."""

    states = np.asarray(states)
    hilbert_dimension = 2**config.n_qubits
    if states.ndim != 2 or states.shape[1] != hilbert_dimension:
        raise ValueError("Batched task states have the wrong Hilbert dimension.")
    subsystem = tuple(int(qubit) for qubit in config.task_subsystem)
    if (
        not subsystem
        or len(subsystem) >= config.n_qubits
        or len(set(subsystem)) != len(subsystem)
        or any(qubit < 0 or qubit >= config.n_qubits for qubit in subsystem)
    ):
        raise ValueError("The task subsystem is not a valid bipartition.")
    complement = tuple(
        qubit for qubit in range(config.n_qubits) if qubit not in subsystem
    )
    # Qiskit's statevector index uses q_{n-1},...,q_0 tensor-axis order.
    subsystem_axes = tuple(config.n_qubits - qubit for qubit in subsystem)
    complement_axes = tuple(config.n_qubits - qubit for qubit in complement)
    tensor = states.reshape((states.shape[0],) + (2,) * config.n_qubits)
    matrices = np.transpose(
        tensor,
        (0,) + subsystem_axes + complement_axes,
    ).reshape(
        states.shape[0],
        2 ** len(subsystem),
        2 ** len(complement),
    )
    singular_values = np.linalg.svd(matrices, compute_uv=False)
    probabilities = np.sort(np.clip(singular_values**2, 0.0, 1.0), axis=1)[:, ::-1]
    maximum_rank = 2 ** min(len(subsystem), len(complement))
    probabilities = probabilities[:, :maximum_rank]
    totals = np.sum(probabilities, axis=1)
    if not np.allclose(totals, 1.0, rtol=1e-9, atol=1e-11):
        raise RuntimeError("A batched Schmidt spectrum does not sum to one.")
    return probabilities / totals[:, None]


def finite_difference_tangents_and_task(
    family: quantum.QiskitHamiltonianFamily,
    config: GreedyTaskConfig,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.zeros(dimension, dtype=float)
    center_state = quantum.ground_state(family, center)
    tangent_columns: list[np.ndarray] = []
    task_dimension = 2 ** min(
        len(config.task_subsystem),
        config.n_qubits - len(config.task_subsystem),
    )
    task_jacobian = np.zeros((task_dimension, dimension), dtype=float)
    step = config.finite_difference_step
    for index in range(dimension):
        delta = np.zeros(dimension, dtype=float)
        delta[index] = step
        plus = base.align_state(center_state, quantum.ground_state(family, delta))
        minus = base.align_state(center_state, quantum.ground_state(family, -delta))
        derivative = (plus - minus) / (2.0 * step)
        derivative -= center_state * np.vdot(center_state, derivative)
        tangent_columns.append(derivative)
        task_jacobian[:, index] = (
            task_values(plus, config) - task_values(minus, config)
        ) / (2.0 * step)
    return center_state, np.column_stack(tangent_columns), task_jacobian


def whiten_identifiable_tangent_space(
    tangent_matrix: np.ndarray,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    metric = np.real_if_close(tangent_matrix.conj().T @ tangent_matrix).real
    metric = 0.5 * (metric + metric.T)
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=float)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=float)
    if eigenvalues[0] <= 0.0:
        raise ValueError("The pilot has no identifiable tangent directions.")
    keep = eigenvalues > relative_tolerance * eigenvalues[0]
    if not np.any(keep):
        raise ValueError("The metric tolerance removed every tangent direction.")
    coordinate_map = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])[None, :]
    return coordinate_map, eigenvalues


def transformed_family(
    family: quantum.QiskitHamiltonianFamily,
    coordinate_map: np.ndarray,
) -> quantum.QiskitHamiltonianFamily:
    transformed_terms = []
    for local_index in range(coordinate_map.shape[1]):
        operator = quantum.zero_operator(family.n_qubits)
        for original_index, term in enumerate(family.term_operators):
            coefficient = float(coordinate_map[original_index, local_index])
            if coefficient != 0.0:
                operator = operator + coefficient * term
        transformed_terms.append(operator.simplify())
    return quantum.QiskitHamiltonianFamily(
        n_qubits=family.n_qubits,
        base_operator=family.base_operator,
        term_operators=tuple(transformed_terms),
        term_names=tuple(f"x{index}" for index in range(coordinate_map.shape[1])),
        task_operators=tuple(),
        task_coefficients=np.zeros((0, coordinate_map.shape[1]), dtype=float),
        task_design_residual=0.0,
    )


def build_local_model(
    config: GreedyTaskConfig,
    original_dimension: int,
) -> LocalModel:
    original_family = base.build_disordered_tfi_family(
        base.TFIConfig(
            n_qubits=config.n_qubits,
            transverse_field=config.transverse_field,
            fixed_disorder_strength=config.fixed_disorder_strength,
            perturbation_order=config.perturbation_order,
        ),
        original_dimension,
    )
    center_state, original_tangents, original_task_jacobian = (
        finite_difference_tangents_and_task(
            original_family,
            config,
            original_dimension,
        )
    )
    center_spectrum = task_values(center_state, config)
    spectral_gaps = center_spectrum[:-1] - center_spectrum[1:]
    if np.any(spectral_gaps <= 1e-13):
        raise RuntimeError(
            "The pilot Schmidt spectrum is degenerate at the finite-difference scale."
        )
    if not np.allclose(
        np.sum(original_task_jacobian, axis=0),
        0.0,
        rtol=0.0,
        atol=1e-9,
    ):
        raise RuntimeError(
            "The Schmidt-probability Jacobian violates probability normalization."
        )
    coordinate_map, raw_metric_eigenvalues = whiten_identifiable_tangent_space(
        original_tangents,
        config.metric_relative_tolerance,
    )
    local_family = transformed_family(original_family, coordinate_map)
    tangent_matrix = original_tangents @ coordinate_map
    tangent_gram = np.real_if_close(tangent_matrix.conj().T @ tangent_matrix).real
    if not np.allclose(
        tangent_gram,
        np.eye(coordinate_map.shape[1]),
        rtol=1e-7,
        atol=1e-8,
    ):
        raise RuntimeError("The retained tangent coordinates were not whitened.")
    return LocalModel(
        original_family=original_family,
        family=local_family,
        coordinate_map=coordinate_map,
        center_state=center_state,
        tangent_matrix=tangent_matrix,
        raw_metric_eigenvalues=raw_metric_eigenvalues,
        original_task_jacobian=original_task_jacobian,
    )


def compute_task_geometry(
    model: LocalModel,
    config: GreedyTaskConfig,
) -> TaskGeometry:
    """Return the pullback of the reported raw Euclidean task loss."""

    jacobian = model.original_task_jacobian @ model.coordinate_map
    task_metric = jacobian.T @ jacobian
    task_metric = 0.5 * (task_metric + task_metric.T)
    eigenvalues, eigenvectors = anchor_design.canonical_generalized_task_basis(
        task_metric,
        np.eye(task_metric.shape[0]),
        config.task_relative_tolerance,
    )
    task_rank = int(
        np.sum(eigenvalues > config.task_relative_tolerance * eigenvalues[0])
    )
    if task_rank < 1:
        raise RuntimeError("The selected Schmidt-spectrum task has zero rank.")
    return TaskGeometry(
        task_scale=np.ones(jacobian.shape[0], dtype=float),
        task_jacobian=jacobian,
        task_metric=task_metric,
        generalized_eigenvalues=eigenvalues,
        generalized_eigenvectors=eigenvectors,
        task_rank=task_rank,
    )


def projective_geodesic_anchors(
    model: LocalModel,
    directions: np.ndarray,
    radius: float,
) -> np.ndarray:
    anchors: list[np.ndarray] = []
    for direction in np.asarray(directions, dtype=float).T:
        tangent = model.tangent_matrix @ direction
        tangent -= model.center_state * np.vdot(model.center_state, tangent)
        norm = float(np.linalg.norm(tangent))
        if norm <= 1e-12:
            raise ValueError("An anchor direction has zero projective tangent norm.")
        tangent /= norm
        anchor = math.cos(radius) * model.center_state + math.sin(radius) * tangent
        anchors.append(np.asarray(quantum.as_statevector(anchor).data))
    return np.asarray(anchors)


def largest_remainder_allocation(total: int, weights: np.ndarray) -> np.ndarray:
    weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if weights.ndim != 1 or weights.size == 0 or float(np.sum(weights)) <= 0.0:
        raise ValueError("Allocation weights must contain positive mass.")
    raw = total * weights / float(np.sum(weights))
    allocation = np.floor(raw).astype(int)
    remaining = int(total - np.sum(allocation))
    order = np.lexsort((np.arange(raw.size), -(raw - allocation)))
    allocation[order[:remaining]] += 1
    return allocation


def allocation_objective(
    eigenvalues: np.ndarray,
    allocations: np.ndarray,
    alpha: float,
    nu: float,
) -> float:
    return float(
        np.sum(
            np.asarray(eigenvalues, dtype=float)
            / (alpha + nu * np.asarray(allocations, dtype=float))
        )
    )


def spectral_designs(
    model: LocalModel,
    geometry: TaskGeometry,
    config: GreedyTaskConfig,
    budget: int,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    dict[str, object],
]:
    local_dimension = model.coordinate_map.shape[1]
    alpha = (local_dimension + 2.0) / (config.particle_radius**2)
    nu = 4.0
    greedy = anchor_design.greedy_spectral_allocation(
        task_metric=geometry.task_metric,
        geometry_metric=np.eye(local_dimension),
        alpha=alpha,
        nu=nu,
        budget=budget,
        retained_dimension=geometry.task_rank,
        null_eigenvalue_relative_tolerance=config.task_relative_tolerance,
    )
    directions = greedy.generalized_eigenvectors[:, : geometry.task_rank]
    eigenvalues = greedy.generalized_eigenvalues[: geometry.task_rank]
    spectral_anchors = projective_geodesic_anchors(
        model,
        directions,
        config.anchor_radius,
    )
    equal_allocation = largest_remainder_allocation(
        budget,
        np.ones(geometry.task_rank),
    )
    lambda_allocation = largest_remainder_allocation(budget, eigenvalues)

    coordinate_scores = np.diag(geometry.task_metric)
    coordinate_indices = np.argsort(coordinate_scores)[-geometry.task_rank :][::-1]
    coordinate_directions = np.eye(local_dimension)[:, coordinate_indices]
    coordinate_anchors = projective_geodesic_anchors(
        model,
        coordinate_directions,
        config.anchor_radius,
    )

    random_rng = np.random.default_rng(config.seed + 70_000 + local_dimension)
    random_basis, _ = np.linalg.qr(random_rng.normal(size=(local_dimension, local_dimension)))
    random_directions = random_basis[:, : geometry.task_rank]
    random_anchors = projective_geodesic_anchors(
        model,
        random_directions,
        config.anchor_radius,
    )

    nuisance_count = min(
        geometry.task_rank,
        local_dimension - geometry.task_rank,
    )
    if nuisance_count <= 0:
        raise RuntimeError("The nuisance ablation requires a nonempty task nullspace.")
    nuisance_directions = greedy.generalized_eigenvectors[
        :, geometry.task_rank : geometry.task_rank + nuisance_count
    ]
    nuisance_anchors = projective_geodesic_anchors(
        model,
        nuisance_directions,
        config.anchor_radius,
    )
    nuisance_allocation = largest_remainder_allocation(
        budget,
        np.ones(nuisance_count),
    )

    def active_design(
        anchors: np.ndarray,
        allocation: np.ndarray,
        design_directions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        active = np.asarray(allocation, dtype=int) > 0
        if not np.any(active):
            raise RuntimeError("A nonadaptive design received no active settings.")
        return (
            anchors[active],
            np.asarray(allocation, dtype=int)[active],
            np.asarray(design_directions, dtype=float)[:, active],
        )

    designs = {
        "greedy_spectral": active_design(
            spectral_anchors,
            greedy.shot_allocations,
            directions,
        ),
        "equal_spectral": active_design(
            spectral_anchors,
            equal_allocation,
            directions,
        ),
        "lambda_spectral": active_design(
            spectral_anchors,
            lambda_allocation,
            directions,
        ),
        "coordinate_equal": active_design(
            coordinate_anchors,
            equal_allocation,
            coordinate_directions,
        ),
        "random_equal": active_design(
            random_anchors,
            equal_allocation,
            random_directions,
        ),
        "nuisance_equal": active_design(
            nuisance_anchors,
            nuisance_allocation,
            nuisance_directions,
        ),
    }
    audit = {
        "alpha": alpha,
        "nu": nu,
        "task_rank": geometry.task_rank,
        "task_eigenvalues": eigenvalues.tolist(),
        "greedy_allocation": greedy.shot_allocations.tolist(),
        "equal_allocation": equal_allocation.tolist(),
        "lambda_allocation": lambda_allocation.tolist(),
        "nuisance_direction_count": nuisance_count,
        "nuisance_task_rayleigh_quotients": np.diag(
            nuisance_directions.T
            @ geometry.task_metric
            @ nuisance_directions
        ).tolist(),
        "terminal_marginal_gains": greedy.terminal_marginal_gains.tolist(),
        "prior_local_risk": greedy.prior_risk,
        "greedy_local_risk": greedy.final_risk,
        "equal_local_risk": allocation_objective(
            eigenvalues,
            equal_allocation,
            alpha,
            nu,
        ),
        "lambda_local_risk": allocation_objective(
            eigenvalues,
            lambda_allocation,
            alpha,
            nu,
        ),
        "coordinate_indices": coordinate_indices.tolist(),
    }
    return designs, audit


def run_local_gaussian_estimator(
    *,
    rng: np.random.Generator,
    truth_state: np.ndarray,
    model: LocalModel,
    config: GreedyTaskConfig,
    anchor_states: np.ndarray,
    shot_counts: np.ndarray,
    directions: np.ndarray,
    task_metric: np.ndarray,
    particle_coordinates: np.ndarray | None = None,
    particle_tasks: np.ndarray | None = None,
    particle_probabilities: np.ndarray | None = None,
    particle_log_importance: np.ndarray | None = None,
) -> tuple[
    base.PosteriorResult,
    base.TaskParticlePosterior | None,
]:
    """Apply the local estimator and optional ordinary nonlinear posterior.

    The data are sampled from exact quantum fidelities.  Only the estimator is
    local: each projective-geodesic setting is converted to its pilot score
    coordinate, whose one-shot precision is nu=4.  This avoids a finite
    particle grid imposing an artificial error floor on the manuscript's local
    estimator.  For the secondary nonlinear check, the same counts are processed
    by the ordinary likelihood-only particle posterior.
    """

    anchor_states = np.asarray(anchor_states)
    shot_counts = np.asarray(shot_counts, dtype=int)
    directions = np.asarray(directions, dtype=float)
    if directions.shape != (model.coordinate_map.shape[1], anchor_states.shape[0]):
        raise ValueError("The design directions do not match the anchor settings.")
    if shot_counts.shape != (anchor_states.shape[0],):
        raise ValueError("The shot allocation does not match the anchor settings.")

    truth_probabilities = quantum.fidelities_to_state(anchor_states, truth_state)
    counts = rng.binomial(shot_counts, truth_probabilities)
    pilot_probabilities = quantum.fidelities_to_state(
        anchor_states,
        model.center_state,
    )
    # Along the exact projective geodesic, p=cos^2(rho-z).  Inverting that
    # relation removes the quadratic bias of (p-p0)/p'(0), while its pilot
    # variance remains 1/(4n), hence the same manuscript coefficient nu=4.
    pilot_radii = np.arccos(np.sqrt(pilot_probabilities))
    observed_probabilities = np.clip(
        counts / shot_counts,
        1e-12,
        1.0 - 1e-12,
    )
    score_coordinates = pilot_radii - np.arccos(
        np.sqrt(observed_probabilities)
    )

    local_dimension = model.coordinate_map.shape[1]
    alpha = (local_dimension + 2.0) / (config.particle_radius**2)
    nu = 4.0
    precision = alpha * np.eye(local_dimension)
    right_hand_side = np.zeros(local_dimension, dtype=float)
    for count, direction, score in zip(
        shot_counts,
        directions.T,
        score_coordinates,
    ):
        precision += nu * count * np.outer(direction, direction)
        right_hand_side += nu * count * direction * score
    theta_estimate = np.linalg.solve(precision, right_hand_side)
    theta_estimate = base.project_to_ball_in_metric(
        theta_estimate,
        task_metric,
        config.particle_radius,
        relative_tolerance=config.task_relative_tolerance,
    )
    state_estimate = quantum.ground_state(model.family, theta_estimate)
    gaussian_result = base.PosteriorResult(
        task_estimate=task_values(state_estimate, config),
        theta_estimate=theta_estimate,
        state=state_estimate,
        weights=np.empty(0, dtype=float),
        ess=float("nan"),
        settings=int(anchor_states.shape[0]),
        copies=int(np.sum(shot_counts)),
    )
    posterior_inputs = (
        particle_coordinates,
        particle_tasks,
        particle_probabilities,
        particle_log_importance,
    )
    if all(value is None for value in posterior_inputs):
        particle_posterior = None
    elif any(value is None for value in posterior_inputs):
        raise ValueError("The particle-posterior inputs must be supplied together.")
    else:
        particle_posterior = base.task_particle_posterior_from_probabilities(
            particle_thetas=particle_coordinates,
            particle_tasks=particle_tasks,
            particle_probabilities=particle_probabilities,
            counts=counts,
            shot_counts=shot_counts,
            log_prior_over_proposal=particle_log_importance,
        )
    return gaussian_result, particle_posterior


def with_task_values(
    result: base.EstimateResult,
    config: GreedyTaskConfig,
) -> base.EstimateResult:
    return base.EstimateResult(
        task_estimate=task_values(result.state, config),
        state=result.state,
        settings=result.settings,
        copies=result.copies,
        ess=result.ess,
    )


def task_error(
    estimate: np.ndarray,
    truth: np.ndarray,
) -> float:
    """Return the Euclidean error of the reported Schmidt-spectrum task."""

    return float(np.linalg.norm(np.asarray(estimate) - np.asarray(truth)))


def raw_task_squared_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Return the unnormalized squared Euclidean Schmidt-spectrum error."""

    difference = np.asarray(estimate, dtype=float) - np.asarray(truth, dtype=float)
    return float(difference @ difference)


def row_raw_task_squared_error(row: dict) -> float:
    """Read a saved raw error, or reconstruct it from legacy trial rows."""

    saved_value = row.get("raw_task_squared_error", float("nan"))
    saved = float(saved_value) if str(saved_value).strip() else float("nan")
    if np.isfinite(saved):
        return saved
    truth = np.asarray(json.loads(str(row["truth_task"])), dtype=float)
    estimate = np.asarray(json.loads(str(row["estimated_task"])), dtype=float)
    return raw_task_squared_error(estimate, truth)


def summarize(
    rows: list[dict],
    *,
    trial_limit: int | None = None,
) -> list[dict]:
    output: list[dict] = []
    keys = sorted(
        {
            (int(row["original_dimension"]), int(row["local_dimension"]), int(row["budget"]), str(row["method"]))
            for row in rows
        }
    )
    for original_dimension, local_dimension, budget, method in keys:
        subset = [
            row
            for row in rows
            if int(row["original_dimension"]) == original_dimension
            and int(row["budget"]) == budget
            and row["method"] == method
            and (
                trial_limit is None
                or int(row["trial"]) < trial_limit
            )
        ]
        values = np.array([float(row["task_error"]) for row in subset])
        squared_values = np.array(
            [float(row["task_squared_error"]) for row in subset]
        )
        raw_squared_values = np.asarray(
            [row_raw_task_squared_error(row) for row in subset],
            dtype=float,
        )
        local_values = np.array(
            [
                float(row["local_task_squared_error"])
                for row in subset
                if np.isfinite(float(row["local_task_squared_error"]))
            ],
            dtype=float,
        )
        def finite_field(field: str) -> np.ndarray:
            def parse_optional(value: object) -> float:
                return float(value) if str(value).strip() else float("nan")

            return np.asarray(
                [
                    parse_optional(row.get(field, float("nan")))
                    for row in subset
                    if np.isfinite(parse_optional(row.get(field, float("nan"))))
                ],
                dtype=float,
            )

        bayes_squared_values = finite_field("bayes_task_squared_error")
        bayes_local_values = finite_field("bayes_local_task_squared_error")
        posterior_ess_values = finite_field("bayes_posterior_ess")
        prior_importance_ess_values = finite_field("bayes_prior_importance_ess")
        method_ess_values = finite_field("ess")
        positive_method_ess_values = method_ess_values[method_ess_values > 0.0]

        def mean_or_nan(values: np.ndarray) -> float:
            return float(np.mean(values)) if values.size else float("nan")

        def se_or_nan(values: np.ndarray) -> float:
            if values.size > 1:
                return float(np.std(values, ddof=1) / math.sqrt(values.size))
            return 0.0 if values.size == 1 else float("nan")

        output.append(
            {
                "original_dimension": original_dimension,
                "local_dimension": local_dimension,
                "task_rank": int(subset[0]["task_rank"]),
                "budget": budget,
                "method": method,
                "settings": int(subset[0]["settings"]),
                "adaptive_rounds": int(subset[0]["adaptive_rounds"]),
                "mean_task_error": float(np.mean(values)),
                "se_task_error": (
                    float(np.std(values, ddof=1) / math.sqrt(values.size))
                    if values.size > 1
                    else 0.0
                ),
                "mean_task_squared_error": float(np.mean(squared_values)),
                "se_task_squared_error": (
                    float(np.std(squared_values, ddof=1) / math.sqrt(squared_values.size))
                    if squared_values.size > 1
                    else 0.0
                ),
                "root_mean_squared_task_error": float(
                    np.sqrt(np.mean(squared_values))
                ),
                "mean_raw_task_squared_error": float(np.mean(raw_squared_values)),
                "se_raw_task_squared_error": (
                    float(
                        np.std(raw_squared_values, ddof=1)
                        / math.sqrt(raw_squared_values.size)
                    )
                    if raw_squared_values.size > 1
                    else 0.0
                ),
                "root_mean_raw_task_error": float(
                    np.sqrt(np.mean(raw_squared_values))
                ),
                "mean_local_task_squared_error": (
                    float(np.mean(local_values)) if local_values.size else float("nan")
                ),
                "se_local_task_squared_error": (
                    float(np.std(local_values, ddof=1) / math.sqrt(local_values.size))
                    if local_values.size > 1
                    else (0.0 if local_values.size == 1 else float("nan"))
                ),
                "n_local_loss_trials": int(local_values.size),
                "mean_bayes_task_squared_error": mean_or_nan(bayes_squared_values),
                "se_bayes_task_squared_error": se_or_nan(bayes_squared_values),
                "mean_bayes_local_task_squared_error": mean_or_nan(
                    bayes_local_values
                ),
                "mean_bayes_posterior_ess": mean_or_nan(posterior_ess_values),
                "mean_bayes_prior_importance_ess": mean_or_nan(
                    prior_importance_ess_values
                ),
                "mean_method_ess": mean_or_nan(positive_method_ess_values),
                "minimum_positive_method_ess": (
                    float(np.min(positive_method_ess_values))
                    if positive_method_ess_values.size
                    else float("nan")
                ),
                "n_bayes_posterior_trials": int(bayes_squared_values.size),
                "n_trials": len(subset),
            }
        )
    return output


def save_plot(path: Path, summary_rows: list[dict], config: GreedyTaskConfig) -> None:
    """Plot the proposed design against representative fixed-design controls."""

    plotted = (
        "greedy_spectral",
        "equal_spectral",
        "lambda_spectral",
        "coordinate_equal",
    )
    dimensions = list(config.original_dimensions)
    fig, axes = plt.subplots(1, len(dimensions), figsize=(12.0, 3.9), sharey=False)
    if len(dimensions) == 1:
        axes = [axes]
    for axis, dimension in zip(axes, dimensions):
        for method in plotted:
            method_rows = sorted(
                [
                    row
                    for row in summary_rows
                    if int(row["original_dimension"]) == dimension
                    and row["method"] == method
                ],
                key=lambda row: int(row["budget"]),
            )
            if not method_rows:
                continue
            axis.errorbar(
                [int(row["budget"]) for row in method_rows],
                [float(row["mean_raw_task_squared_error"]) for row in method_rows],
                yerr=[float(row["se_raw_task_squared_error"]) for row in method_rows],
                **METHOD_STYLES[method],
                markersize=5.0,
                linewidth=1.7,
                label=METHOD_LABELS[method],
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


def save_local_loss_plot(
    path: Path,
    summary_rows: list[dict],
    config: GreedyTaskConfig,
) -> None:
    """Plot fixed-design ablations using raw Schmidt-spectrum MSE."""

    plotted = (
        "greedy_spectral",
        "equal_spectral",
        "lambda_spectral",
        "coordinate_equal",
        "random_equal",
        "nuisance_equal",
    )
    dimensions = list(config.original_dimensions)
    fig, axes = plt.subplots(1, len(dimensions), figsize=(12.0, 3.9), sharey=False)
    if len(dimensions) == 1:
        axes = [axes]
    for axis, dimension in zip(axes, dimensions):
        for method in plotted:
            method_rows = sorted(
                [
                    row
                    for row in summary_rows
                    if int(row["original_dimension"]) == dimension
                    and row["method"] == method
                    and np.isfinite(float(row["mean_raw_task_squared_error"]))
                ],
                key=lambda row: int(row["budget"]),
            )
            if not method_rows:
                continue
            axis.errorbar(
                [int(row["budget"]) for row in method_rows],
                [float(row["mean_raw_task_squared_error"]) for row in method_rows],
                yerr=[float(row["se_raw_task_squared_error"]) for row in method_rows],
                **METHOD_STYLES[method],
                markersize=5.0,
                linewidth=1.7,
                label=(
                    "Greedy spectral (ours)"
                    if method == "greedy_spectral"
                    else METHOD_LABELS[method]
                ),
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


def run_dimension(
    config: GreedyTaskConfig,
    original_dimension: int,
) -> tuple[list[dict], dict]:
    model = build_local_model(config, original_dimension)
    local_dimension = model.coordinate_map.shape[1]
    geometry = compute_task_geometry(model, config)
    designs_by_budget: dict[
        int,
        dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ] = {}
    audit_by_budget: dict[int, dict[str, object]] = {}
    for budget in config.budgets:
        designs, audit = spectral_designs(model, geometry, config, budget)
        designs_by_budget[budget] = designs
        audit_by_budget[budget] = audit

    posterior_key_by_design: dict[tuple[int, str], tuple[str, int]] = {}
    posterior_clouds: dict[
        tuple[str, int],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    for budget in config.budgets:
        for method in POSTERIOR_METHODS:
            anchor_states, _, directions = designs_by_budget[budget][method]
            cloud_key = (method, int(directions.shape[1]))
            posterior_key_by_design[(budget, method)] = cloud_key
            if cloud_key in posterior_clouds:
                continue
            print(
                f"d={original_dimension}, local={local_dimension}: building "
                f"{config.n_posterior_particles} importance particles for "
                f"{method} with {directions.shape[1]} settings",
                flush=True,
            )
            posterior_coordinates, posterior_log_importance = (
                base.make_directional_importance_ball_cloud(
                    rng=np.random.default_rng(
                        config.seed
                        + 15_000
                        + 100 * original_dimension
                        + 10 * len(posterior_clouds)
                    ),
                    dimension=local_dimension,
                    radius=config.particle_radius,
                    count=config.n_posterior_particles,
                    design_directions=directions,
                )
            )
            posterior_states = np.array(
                [
                    quantum.ground_state(model.family, coordinate)
                    for coordinate in posterior_coordinates
                ]
            )
            posterior_tasks = np.array(
                [task_values(state, config) for state in posterior_states]
            )
            posterior_probabilities = np.column_stack(
                [
                    quantum.fidelities_to_state(posterior_states, anchor_state)
                    for anchor_state in anchor_states
                ]
            )
            posterior_clouds[cloud_key] = (
                posterior_coordinates,
                posterior_tasks,
                posterior_probabilities,
                posterior_log_importance,
            )

    rows: list[dict] = []
    total_trials = max(config.n_trials, config.n_design_trials)
    for trial in range(total_trials):
        truth_coordinate = base.sample_ball(
            np.random.default_rng(config.seed + 100_000 * original_dimension + trial),
            local_dimension,
            config.truth_radius,
            1,
        )[0]
        truth_state = quantum.ground_state(model.family, truth_coordinate)
        truth_task = task_values(truth_state, config)
        for budget in config.budgets:
            seed_base = config.seed + 1_000_000 * original_dimension + 1000 * trial + budget
            results: dict[str, object] = {}
            posterior_results: dict[str, base.TaskParticlePosterior] = {}
            for method_index, method in enumerate(ANCHOR_METHODS):
                anchor_states, shot_counts, directions = designs_by_budget[budget][method]
                if method in POSTERIOR_METHODS:
                    cloud_key = posterior_key_by_design[(budget, method)]
                    (
                        method_posterior_coordinates,
                        method_posterior_tasks,
                        method_particle_probabilities,
                        method_particle_log_importance,
                    ) = posterior_clouds[cloud_key]
                else:
                    method_posterior_coordinates = None
                    method_posterior_tasks = None
                    method_particle_probabilities = None
                    method_particle_log_importance = None
                try:
                    result, particle_posterior = run_local_gaussian_estimator(
                        rng=np.random.default_rng(seed_base + 10 + method_index),
                        truth_state=truth_state,
                        model=model,
                        config=config,
                        anchor_states=anchor_states,
                        shot_counts=shot_counts,
                        directions=directions,
                        task_metric=geometry.task_metric,
                        particle_coordinates=method_posterior_coordinates,
                        particle_tasks=method_posterior_tasks,
                        particle_probabilities=method_particle_probabilities,
                        particle_log_importance=method_particle_log_importance,
                    )
                except RuntimeError as error:
                    raise RuntimeError(
                        "Particle-posterior failure at "
                        f"d={original_dimension}, trial={trial}, budget={budget}, "
                        f"method={method}: {error}"
                    ) from error
                results[method] = result
                if particle_posterior is not None:
                    posterior_results[method] = particle_posterior
            for method, result in results.items():
                error = task_error(result.task_estimate, truth_task)
                particle_posterior = posterior_results.get(method)
                theta_estimate = getattr(result, "theta_estimate", None)
                if theta_estimate is not None:
                    theta_estimate = np.asarray(theta_estimate, dtype=float)
                if (
                    theta_estimate is not None
                    and theta_estimate.shape == truth_coordinate.shape
                    and np.all(np.isfinite(theta_estimate))
                ):
                    parameter_error = theta_estimate - truth_coordinate
                    local_task_squared_error = float(
                        parameter_error @ geometry.task_metric @ parameter_error
                    )
                else:
                    local_task_squared_error = float("nan")
                if particle_posterior is not None:
                    bayes_error = task_error(
                        particle_posterior.task_estimate,
                        truth_task,
                    )
                    bayes_parameter_error = (
                        particle_posterior.theta_estimate - truth_coordinate
                    )
                    bayes_local_task_squared_error = float(
                        bayes_parameter_error
                        @ geometry.task_metric
                        @ bayes_parameter_error
                    )
                    posterior_fields = {
                        "bayes_task_error": bayes_error,
                        "bayes_task_squared_error": bayes_error**2,
                        "bayes_local_task_squared_error": bayes_local_task_squared_error,
                        "bayes_posterior_ess": particle_posterior.ess,
                        "bayes_prior_importance_ess": (
                            particle_posterior.prior_importance_ess
                        ),
                    }
                else:
                    posterior_fields = {
                        "bayes_task_error": float("nan"),
                        "bayes_task_squared_error": float("nan"),
                        "bayes_local_task_squared_error": float("nan"),
                        "bayes_posterior_ess": float("nan"),
                        "bayes_prior_importance_ess": float("nan"),
                    }
                rows.append(
                    {
                        "original_dimension": original_dimension,
                        "local_dimension": local_dimension,
                        "task_rank": geometry.task_rank,
                        "trial": trial,
                        "budget": budget,
                        "method": method,
                        "settings": int(result.settings),
                        "copies": int(result.copies),
                        "adaptive_rounds": 0,
                        "truth_task": str(truth_task.tolist()),
                        "estimated_task": str(np.asarray(result.task_estimate).tolist()),
                        "task_error": error,
                        "task_squared_error": error**2,
                        "raw_task_squared_error": raw_task_squared_error(
                            result.task_estimate,
                            truth_task,
                        ),
                        "local_task_squared_error": local_task_squared_error,
                        "ess": float(result.ess),
                        **posterior_fields,
                    }
                )
        if (trial + 1) % 25 == 0 or trial + 1 == total_trials:
            print(
                f"d={original_dimension}: completed {trial + 1}/{total_trials}",
                flush=True,
            )

    metadata = {
        "original_dimension": original_dimension,
        "local_dimension": local_dimension,
        "task_rank": geometry.task_rank,
        "task_subsystem": str(config.task_subsystem),
        "task_weighting": "identity (raw Euclidean spectrum loss)",
        "nominal_schmidt_spectrum": str(task_values(model.center_state, config).tolist()),
        "minimum_nominal_spectral_gap": float(
            np.min(np.diff(task_values(model.center_state, config)[::-1]))
        ),
        "task_eigenvalues": str(
            geometry.generalized_eigenvalues[: geometry.task_rank].tolist()
        ),
        "anchor_radius": config.anchor_radius,
        "particle_radius": config.particle_radius,
        "truth_radius": config.truth_radius,
        "posterior_particles": config.n_posterior_particles,
        "posterior_methods": str(POSTERIOR_METHODS),
        "metric_relative_tolerance": config.metric_relative_tolerance,
        "raw_metric_eigenvalues": str(model.raw_metric_eigenvalues.tolist()),
        "allocations_by_budget": json.dumps(audit_by_budget, sort_keys=True),
        **quantum.version_metadata(),
    }
    return rows, metadata


def write_summary(
    path: Path,
    config: GreedyTaskConfig,
    mode: str,
    summary_rows: list[dict],
    metadata_rows: list[dict],
) -> None:
    comparison_rows: list[dict] = []
    for dimension in config.original_dimensions:
        for budget in config.budgets:
            methods = {
                str(row["method"]): row
                for row in summary_rows
                if int(row["original_dimension"]) == dimension
                and int(row["budget"]) == budget
            }
            greedy = methods["greedy_spectral"]
            equal = methods["equal_spectral"]
            greedy_exact = float(greedy["mean_raw_task_squared_error"])
            equal_exact = float(equal["mean_raw_task_squared_error"])
            comparison_rows.append(
                {
                    "original_dimension": dimension,
                    "budget": budget,
                    "greedy_raw_task_mse": greedy_exact,
                    "equal_raw_task_mse": equal_exact,
                    "greedy_vs_equal_reduction_percent": 100.0
                    * (1.0 - greedy_exact / equal_exact),
                }
            )
    lines = [
        "# Fixed-Radius Greedy Spectral Schmidt-Spectrum Benchmark",
        "",
        f"- mode: `{mode}`",
        f"- odd-even task subsystem: `{config.task_subsystem}`",
        f"- common Fubini--Study anchor radius: `{config.anchor_radius}`",
        "- direct projective fidelity coefficient: `nu = 4`",
        "- allocation: exact manuscript diminishing-marginal-gain greedy rule",
        "- estimator: local geodesic score update on exact quantum binomial data",
        "- truths: sampled from the same local uniform-ball prior used to set `alpha`",
        "- task: ordered squared Schmidt coefficients across the odd-even 3|3 cut",
        "- adaptive competitors: evaluated separately on the retained four-schedule grids",
        "",
        "## Key comparisons",
        "",
        "Greedy-versus-equal uses all fixed-design trials. All comparisons use raw, "
        "unnormalized Schmidt-spectrum MSE. Positive reduction means that greedy spectral "
        "has lower raw task MSE. "
        "The allocation advantage is expected to shrink when the retained task eigenvalues "
        "become nearly equal; the lambda-proportional control remains in the full table as an "
        "allocation ablation.",
        "",
        *base.markdown_table(
            comparison_rows,
            [
                "original_dimension",
                "budget",
                "greedy_raw_task_mse",
                "equal_raw_task_mse",
                "greedy_vs_equal_reduction_percent",
            ],
        ),
        "",
        "## Geometry and allocation",
        "",
        *base.markdown_table(
            metadata_rows,
            [
                "original_dimension",
                "local_dimension",
                "task_rank",
                "task_subsystem",
                "task_weighting",
                "nominal_schmidt_spectrum",
                "minimum_nominal_spectral_gap",
                "task_eigenvalues",
                "anchor_radius",
                "particle_radius",
                "truth_radius",
                "posterior_particles",
                "posterior_methods",
                "metric_relative_tolerance",
                "quantum_backend",
                "qiskit_version",
                "qiskit_aer_version",
                "allocations_by_budget",
            ],
        ),
        "",
        "## Aggregate results",
        "",
        *base.markdown_table(
            summary_rows,
            [
                "original_dimension",
                "local_dimension",
                "task_rank",
                "budget",
                "method",
                "settings",
                "adaptive_rounds",
                "mean_raw_task_squared_error",
                "se_raw_task_squared_error",
                "root_mean_raw_task_error",
                "mean_bayes_posterior_ess",
                "mean_bayes_prior_importance_ess",
                "mean_method_ess",
                "minimum_positive_method_ess",
                "n_bayes_posterior_trials",
                "n_trials",
            ],
        ),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def config_for_mode(mode: str) -> GreedyTaskConfig:
    config = GreedyTaskConfig()
    if mode == "smoke":
        return replace(
            config,
            original_dimensions=(6,),
            budgets=(600, 1200),
            n_trials=2,
            n_design_trials=2,
            n_particles=400,
            n_posterior_particles=2000,
        )
    if mode == "pilot":
        return replace(
            config,
            budgets=(19200, 76800),
            n_trials=8,
            n_design_trials=80,
            n_particles=3000,
            n_posterior_particles=20000,
        )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot", "full"), default="smoke")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate summaries and figures from existing trial_results.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output root; mode subdirectories are added outside full mode.",
    )
    arguments = parser.parse_args()
    mode = str(arguments.mode)
    config = config_for_mode(mode)
    output_root = (
        arguments.output_dir
        if arguments.output_dir is not None
        else OUTPUT_DIR
    )
    output_dir = output_root if mode == "full" else output_root / mode

    all_rows: list[dict] = []
    metadata_rows: list[dict] = []
    if arguments.report_only:
        with (output_dir / "trial_results.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            all_rows = [
                row
                for row in csv.DictReader(handle)
                if str(row["method"]) in ANCHOR_METHODS
            ]
        with (output_dir / "geometry_and_allocations.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            metadata_rows = list(csv.DictReader(handle))
        for metadata in metadata_rows:
            metadata["allocations_by_budget"] = metadata[
                "allocations_by_budget"
            ].replace('"kappa"', '"nu"')
            metadata["posterior_particles"] = metadata.get(
                "posterior_particles",
                metadata.get("confidence_particles", ""),
            )
            metadata["posterior_methods"] = metadata.get(
                "posterior_methods",
                str(POSTERIOR_METHODS),
            )
    else:
        for dimension in config.original_dimensions:
            rows, metadata = run_dimension(config, dimension)
            all_rows.extend(rows)
            metadata_rows.append(metadata)
    summary_rows = summarize(all_rows)
    paired_summary_rows = summarize(all_rows, trial_limit=config.n_trials)

    output_dir.mkdir(parents=True, exist_ok=True)
    if not arguments.report_only:
        base.write_union_csv(output_dir / "trial_results.csv", all_rows)
    base.write_union_csv(output_dir / "summary_rows.csv", summary_rows)
    base.write_union_csv(
        output_dir / "paired_summary_rows.csv",
        paired_summary_rows,
    )
    base.write_union_csv(output_dir / "geometry_and_allocations.csv", metadata_rows)
    if not arguments.report_only:
        (output_dir / "config.json").write_text(
            json.dumps({**config.__dict__, "mode": mode}, indent=2),
            encoding="utf-8",
        )
    write_summary(
        output_dir / "summary.md",
        config,
        mode,
        summary_rows,
        metadata_rows,
    )
    save_plot(
        output_dir / "greedy_spectral_task_benchmark.png",
        summary_rows,
        config,
    )
    save_local_loss_plot(
        output_dir / "greedy_spectral_local_loss.png",
        summary_rows,
        config,
    )
    print(f"Saved fixed-radius greedy benchmark to {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
