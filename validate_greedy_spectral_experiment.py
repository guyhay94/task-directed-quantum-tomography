"""Numerical checks for the fixed-radius greedy spectral experiment."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

import local_information_anchor_design as anchor_design
import qiskit_quantum_backend as quantum
import quantum_benchmark_support as base
import quantum_greedy_spectral_experiment as experiment


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = base.ACTIVE_EXPERIMENT_ROOT / "greedy_spectral_validation.json"


def compositions(total: int, parts: int):
    """Yield all nonnegative integer vectors of length parts summing to total."""

    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for tail in compositions(total - first, parts - 1):
            yield (first, *tail)


def validate_heap_against_brute_force() -> dict[str, object]:
    eigenvalues = np.array([9.0, 4.0, 1.0])
    alpha = 1.7
    nu = 4.0
    budget = 15
    result = anchor_design.greedy_spectral_allocation(
        task_metric=np.diag(eigenvalues),
        geometry_metric=np.eye(3),
        alpha=alpha,
        nu=nu,
        budget=budget,
    )
    candidates = list(compositions(budget, 3))
    objectives = np.array(
        [
            np.sum(eigenvalues / (alpha + nu * np.asarray(candidate)))
            for candidate in candidates
        ]
    )
    best_index = int(np.argmin(objectives))
    best_allocation = np.asarray(candidates[best_index], dtype=int)
    best_objective = float(objectives[best_index])
    if not np.array_equal(result.shot_allocations, best_allocation):
        raise AssertionError(
            f"Greedy allocation {result.shot_allocations} differs from brute-force "
            f"optimum {best_allocation}."
        )
    if not np.isclose(result.final_risk, best_objective, rtol=1e-13, atol=1e-13):
        raise AssertionError("Greedy objective differs from the brute-force optimum.")
    return {
        "budget": budget,
        "greedy_allocation": result.shot_allocations.tolist(),
        "brute_force_allocation": best_allocation.tolist(),
        "objective": best_objective,
    }


def validate_task_metric_projection() -> dict[str, object]:
    """Check the identity, singular-metric, KKT, and safety guarantees."""

    radius = 0.75
    identity_point = np.array([1.2, -0.7, 0.4])
    identity_projection = base.project_to_ball_in_metric(
        identity_point,
        np.eye(3),
        radius,
    )
    radial_projection = base.project_to_ball(identity_point, radius)
    if not np.array_equal(identity_projection, radial_projection):
        raise AssertionError("Q=I does not exactly reproduce radial projection.")

    metric = np.diag([9.0, 2.0, 0.0])
    inside_point = np.array([0.2, -0.1, 5.0])
    inside_projection = base.project_to_ball_in_metric(
        inside_point,
        metric,
        radius,
    )
    if not np.allclose(inside_projection, [0.2, -0.1, 0.0], atol=1e-14):
        raise AssertionError("The minimum-norm tie-break did not remove ker(Q).")

    outside_point = np.array([2.0, -1.0, 5.0])
    projected = base.project_to_ball_in_metric(outside_point, metric, radius)
    if not np.isclose(np.linalg.norm(projected), radius, rtol=0.0, atol=2e-12):
        raise AssertionError("The active Q-projection does not lie on the ball.")
    if abs(float(projected[2])) > 1e-14:
        raise AssertionError("The active Q-projection retained a task-null component.")
    multipliers = np.array(
        [
            metric[index, index]
            * (outside_point[index] - projected[index])
            / projected[index]
            for index in (0, 1)
        ]
    )
    if not np.allclose(multipliers, multipliers[0], rtol=2e-10, atol=2e-10):
        raise AssertionError("The projected coordinates violate the KKT multiplier.")

    rng = np.random.default_rng(20260813)
    maximum_safety_residual = -np.inf
    for _ in range(100):
        raw = rng.normal(size=3)
        truth = base.project_to_ball(rng.normal(size=3), radius)
        estimate = base.project_to_ball_in_metric(raw, metric, radius)
        raw_error = float((raw - truth) @ metric @ (raw - truth))
        projected_error = float((estimate - truth) @ metric @ (estimate - truth))
        movement = float((raw - estimate) @ metric @ (raw - estimate))
        safety_residual = projected_error + movement - raw_error
        maximum_safety_residual = max(maximum_safety_residual, safety_residual)
        if safety_residual > 2e-10 * max(1.0, raw_error):
            raise AssertionError("The task-metric projection inequality failed.")

    return {
        "identity_matches_radial_exactly": True,
        "singular_metric": np.diag(metric).tolist(),
        "inside_projection": inside_projection.tolist(),
        "active_projection": projected.tolist(),
        "active_projection_norm": float(np.linalg.norm(projected)),
        "kkt_multiplier": float(np.mean(multipliers)),
        "maximum_safety_residual": float(maximum_safety_residual),
    }


def finite_difference_fisher(
    model: experiment.LocalModel,
    anchor: np.ndarray,
    direction: np.ndarray,
    step: float,
) -> tuple[float, float, float]:
    plus = quantum.ground_state(model.family, step * direction)
    minus = quantum.ground_state(model.family, -step * direction)
    pilot_probability = quantum.state_fidelity_probability(
        anchor,
        model.center_state,
    )
    derivative = (
        quantum.state_fidelity_probability(anchor, plus)
        - quantum.state_fidelity_probability(anchor, minus)
    ) / (2.0 * step)
    fisher = derivative**2 / (pilot_probability * (1.0 - pilot_probability))
    return float(pilot_probability), float(derivative), float(fisher)


def validate_quantum_geometry() -> dict[str, object]:
    config = experiment.GreedyTaskConfig(
        original_dimensions=(6,),
        budgets=(600,),
        n_trials=1,
        n_particles=400,
    )
    model = experiment.build_local_model(config, 6)
    geometry = experiment.compute_task_geometry(model, config)
    expected_raw_metric = geometry.task_jacobian.T @ geometry.task_jacobian
    raw_metric_alignment_error = float(
        np.max(np.abs(geometry.task_metric - expected_raw_metric))
    )
    if raw_metric_alignment_error > 1e-14:
        raise AssertionError(
            "The experimental task metric is not the pullback of raw spectrum MSE."
        )
    if not np.array_equal(geometry.task_scale, np.ones_like(geometry.task_scale)):
        raise AssertionError("The raw spectrum task unexpectedly uses component scaling.")
    directions = geometry.generalized_eigenvectors[:, : geometry.task_rank]
    anchors = experiment.projective_geodesic_anchors(
        model,
        directions,
        config.anchor_radius,
    )

    tangent_gram = np.real_if_close(
        model.tangent_matrix.conj().T @ model.tangent_matrix
    ).real
    whitening_error = float(np.max(np.abs(tangent_gram - np.eye(tangent_gram.shape[0]))))
    if whitening_error > 1e-7:
        raise AssertionError(f"Tangent whitening error is {whitening_error}.")

    expected_probability = float(np.cos(config.anchor_radius) ** 2)
    fisher_rows = [
        finite_difference_fisher(model, anchor, direction, step=1e-4)
        for anchor, direction in zip(anchors, directions.T)
    ]
    probabilities = np.array([row[0] for row in fisher_rows])
    derivatives = np.array([row[1] for row in fisher_rows])
    fisher_values = np.array([row[2] for row in fisher_rows])
    if not np.allclose(probabilities, expected_probability, rtol=1e-10, atol=1e-12):
        raise AssertionError("Projective anchors are not at the requested common radius.")
    if not np.allclose(fisher_values, 4.0, rtol=8e-3, atol=8e-3):
        raise AssertionError(f"Per-shot Fisher information is not nu=4: {fisher_values}.")

    _, audit = experiment.spectral_designs(model, geometry, config, budget=600)
    greedy_allocation = np.asarray(audit["greedy_allocation"], dtype=int)
    if int(np.sum(greedy_allocation)) != 600:
        raise AssertionError("The greedy allocation does not spend the full budget.")
    if float(audit["greedy_local_risk"]) > float(audit["equal_local_risk"]) + 1e-12:
        raise AssertionError("The greedy local risk is worse than equal allocation.")
    if float(audit["greedy_local_risk"]) > float(audit["lambda_local_risk"]) + 1e-12:
        raise AssertionError("The greedy local risk is worse than lambda-proportional allocation.")
    nuisance_rayleigh = np.asarray(
        audit["nuisance_task_rayleigh_quotients"],
        dtype=float,
    )
    expected_nuisance_count = min(
        geometry.task_rank,
        model.coordinate_map.shape[1] - geometry.task_rank,
    )
    if int(audit["nuisance_direction_count"]) != expected_nuisance_count:
        raise AssertionError("The nuisance control does not use the available nullspace.")
    nuisance_relative_weight = float(
        np.max(np.abs(nuisance_rayleigh))
        / max(float(geometry.generalized_eigenvalues[0]), 1e-30)
    )
    if nuisance_relative_weight > config.task_relative_tolerance:
        raise AssertionError(
            f"A nuisance direction has non-negligible task weight: {nuisance_rayleigh}."
        )

    return {
        "original_dimension": 6,
        "local_dimension": int(model.coordinate_map.shape[1]),
        "task_rank": int(geometry.task_rank),
        "task_weighting": "identity (raw Euclidean spectrum loss)",
        "raw_task_metric_alignment_max_abs_error": raw_metric_alignment_error,
        "whitening_max_abs_error": whitening_error,
        "anchor_radius": config.anchor_radius,
        "expected_pilot_probability": expected_probability,
        "pilot_probabilities": probabilities.tolist(),
        "directional_probability_derivatives": derivatives.tolist(),
        "finite_difference_fisher_per_shot": fisher_values.tolist(),
        "expected_nu": 4.0,
        "greedy_allocation": greedy_allocation.tolist(),
        "greedy_local_risk": float(audit["greedy_local_risk"]),
        "equal_local_risk": float(audit["equal_local_risk"]),
        "lambda_local_risk": float(audit["lambda_local_risk"]),
        "nuisance_direction_count": int(audit["nuisance_direction_count"]),
        "nuisance_task_rayleigh_quotients": nuisance_rayleigh.tolist(),
        "maximum_nuisance_relative_task_weight": nuisance_relative_weight,
        "terminal_marginal_gains": audit["terminal_marginal_gains"],
    }


def validate_osgqt_correction() -> dict[str, object]:
    """Check that the published correction vanishes at the truth."""

    rng = np.random.default_rng(20260725)
    current = quantum.random_state_data(64, rng)
    direction = rng.choice([-1.0, 1.0], size=current.size)
    _, beta = base.paired_sgqt_osgqt_paper_gains(0)
    plus_state = quantum.normalized_state_data(current + beta * direction)
    minus_state = quantum.normalized_state_data(current - beta * direction)
    measured_plus = quantum.state_fidelity_probability(current, plus_state)
    measured_minus = quantum.state_fidelity_probability(current, minus_state)
    sgqt_difference = base.sgqt_fidelity_difference(
        measured_plus,
        measured_minus,
        current,
        plus_state,
        minus_state,
        osgqt=False,
    )
    osgqt_difference = base.sgqt_fidelity_difference(
        measured_plus,
        measured_minus,
        current,
        plus_state,
        minus_state,
        osgqt=True,
    )
    if not np.isclose(osgqt_difference, 0.0, atol=2e-15):
        raise AssertionError(
            f"The noiseless OSGQT correction does not vanish: {osgqt_difference}."
        )
    return {
        "standard_noiseless_numerator": sgqt_difference,
        "corrected_noiseless_numerator": osgqt_difference,
    }


def validate_competitor_paper_contracts() -> dict[str, object]:
    """Check active competitor gains, initialization, and Bayesian endpoint."""

    expected_paqt_gains = (
        10.0 / (64.0**0.602),
        0.1 / (64.0**0.101),
    )
    if not np.allclose(
        base.paqt_paper_gains(63),
        expected_paqt_gains,
        rtol=0.0,
        atol=1e-15,
    ):
        raise AssertionError("The PAQT gain schedule does not match Eq. (26).")
    if "start_at_pilot" in inspect.signature(base.run_structured_paqt).parameters:
        raise AssertionError("Structured PAQT still exposes a non-pilot start option.")
    if "start_at_pilot" in inspect.signature(
        base.collect_structured_paqt_measurements
    ).parameters:
        raise AssertionError("SMC PAQT still exposes a non-pilot start option.")

    for iteration in (0, 1, 63, 3071):
        if base.paired_sgqt_osgqt_paper_gains(iteration) != (0.05, 0.2):
            raise AssertionError("The matched SGQT/OSGQT gains drift with iteration.")

    rng = np.random.default_rng(20260822)
    particles = base.sample_ball(rng, 3, 0.2, 30_000)
    weights = np.exp(2.0 * particles[:, 0])
    weights /= float(np.sum(weights))
    posterior_mean = weights @ particles
    differences = particles - posterior_mean
    posterior_covariance = (differences * weights[:, None]).T @ differences
    resampled = base.liu_west_resample(
        rng=rng,
        particles=particles,
        weights=weights,
        radius=0.5,
    )
    resampled_covariance = np.cov(resampled, rowvar=False, ddof=0)
    mean_error = float(np.max(np.abs(np.mean(resampled, axis=0) - posterior_mean)))
    covariance_relative_error = float(
        np.linalg.norm(resampled_covariance - posterior_covariance)
        / np.linalg.norm(posterior_covariance)
    )
    if mean_error > 1e-3 or covariance_relative_error > 0.02:
        raise AssertionError("Liu--West resampling does not preserve posterior moments.")
    if np.max(np.linalg.norm(resampled, axis=1)) > 0.5:
        raise AssertionError("Liu--West resampling left the localization ball.")

    task_config = experiment.GreedyTaskConfig()
    random_states = rng.normal(size=(20, 2**task_config.n_qubits))
    random_states /= np.linalg.norm(random_states, axis=1)[:, None]
    scalar_tasks = np.asarray(
        [experiment.task_values(state, task_config) for state in random_states]
    )
    batched_tasks = experiment.batch_task_values(random_states, task_config)
    batch_task_error = float(np.max(np.abs(batched_tasks - scalar_tasks)))
    if batch_task_error > 2e-14:
        raise AssertionError("Batched pure-state task evaluation changed the endpoint.")

    local_model = experiment.build_local_model(task_config, 6)
    test_coordinates = base.sample_ball(rng, 6, 0.02, 20)
    indexed_ground_states = base.batch_ground_states(
        local_model.family,
        test_coordinates,
    )
    scalar_ground_states = np.asarray(
        [quantum.ground_state(local_model.family, point) for point in test_coordinates]
    )
    ground_state_fidelities = np.abs(
        np.sum(indexed_ground_states.conj() * scalar_ground_states, axis=1)
    ) ** 2
    minimum_ground_state_fidelity = float(np.min(ground_state_fidelities))
    if minimum_ground_state_fidelity < 1.0 - 2e-13:
        raise AssertionError("Indexed particle ground states changed the physical states.")

    states = np.zeros((2, 4), dtype=complex)
    states[0, 0] = 1.0
    states[1, 3] = 1.0
    density = base.pure_state_bayesian_mean_density(
        states,
        np.array([0.5, 0.5]),
    )
    reduced_spectrum = quantum.subsystem_density_probabilities(
        density,
        n_qubits=2,
        subsystem=(0,),
    )
    if not np.allclose(reduced_spectrum, [0.5, 0.5], atol=2e-15):
        raise AssertionError("The PAQT Bayesian-mean task endpoint is incorrect.")
    if not np.allclose(density, density.conj().T, atol=2e-15):
        raise AssertionError("The Bayesian mean density is not Hermitian.")
    if not np.isclose(np.trace(density), 1.0, atol=2e-15):
        raise AssertionError("The Bayesian mean density does not have unit trace.")

    task_particles = np.array([[0.0], [2.0]])
    task_posterior = base.run_particle_posterior_from_measurements(
        particle_thetas=np.array([[0.0], [1.0]]),
        particle_states=states,
        probe_states=np.array([[1.0, 0.0, 0.0, 1.0]]) / np.sqrt(2.0),
        counts=np.array([1]),
        shot_counts=np.array([2]),
        task_from_density=lambda _: np.array([-1.0]),
        particle_tasks=task_particles,
    )
    if not np.allclose(task_posterior.task_estimate, [1.0], atol=2e-15):
        raise AssertionError(
            "The PAQT squared-loss endpoint is not the posterior mean task."
        )
    return {
        "paqt_iteration_64_gains": list(base.paqt_paper_gains(63)),
        "structured_paqt_starts_at_pilot": True,
        "structured_paqt_start_is_not_configurable": True,
        "paired_sgqt_osgqt_alpha": 0.05,
        "paired_sgqt_osgqt_beta": 0.2,
        "paqt_resampler": "Liu-West",
        "paqt_liu_west_a": base.PAQT_LIU_WEST_A,
        "paqt_resample_ess_fraction": base.PAQT_RESAMPLE_ESS_FRACTION,
        "liu_west_mean_max_abs_error": mean_error,
        "liu_west_covariance_relative_error": covariance_relative_error,
        "batched_task_max_abs_error": batch_task_error,
        "indexed_ground_state_minimum_fidelity": minimum_ground_state_fidelity,
        "bayesian_mean_reduced_spectrum": reduced_spectrum.tolist(),
        "bayesian_mean_trace": float(np.trace(density).real),
        "posterior_mean_task": task_posterior.task_estimate.tolist(),
    }


def validate_confidence_support() -> dict[str, object]:
    """Check active confidence bounds and directional importance weights."""

    counts = np.array([80, 60], dtype=int)
    shots = np.array([100, 100], dtype=int)
    failures = np.array([0.025, 0.025], dtype=float)
    lower = base.exact_binomial_lower_confidence_bounds(counts, shots, failures)
    observed = counts / shots
    if np.any(lower < 0.0) or np.any(lower > observed):
        raise AssertionError("A one-sided binomial lower bound is outside [0, p-hat].")

    rng = np.random.default_rng(20260731)
    cloud, log_importance = base.make_directional_importance_ball_cloud(
        rng=rng,
        dimension=6,
        radius=0.01,
        count=20_000,
        design_directions=np.eye(6)[:, :2],
    )
    importance = np.exp(log_importance)
    half_ball_fraction = float(
        np.sum(importance * (cloud[:, 0] >= 0.0)) / np.sum(importance)
    )
    if abs(half_ball_fraction - 0.5) > 0.035:
        raise AssertionError(
            "Directional importance weights do not recover uniform-ball symmetry."
        )
    return {
        "joint_failure_probability": 0.05,
        "individual_failure_probabilities": failures.tolist(),
        "lower_fidelity_bounds": lower.tolist(),
        "uniform_half_ball_importance_estimate": half_ball_fraction,
        "importance_ess": float(importance.sum() ** 2 / np.sum(importance**2)),
    }


def main() -> None:
    report = {
        "status": "passed",
        "heap_vs_brute_force": validate_heap_against_brute_force(),
        "task_metric_projection": validate_task_metric_projection(),
        "quantum_geometry": validate_quantum_geometry(),
        "competitor_paper_contracts": validate_competitor_paper_contracts(),
        "osgqt_correction": validate_osgqt_correction(),
        "confidence_support": validate_confidence_support(),
        "qiskit": quantum.version_metadata(),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
