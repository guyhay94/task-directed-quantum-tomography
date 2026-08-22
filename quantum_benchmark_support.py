"""Shared physical model and competitor routines for the quantum benchmark.

This module contains only infrastructure used by the current Schmidt-spectrum
experiment. It deliberately has no experiment entry point or plotting code.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.linalg import eigh as scipy_eigh
from scipy.stats import beta as beta_distribution
from scipy.special import logsumexp
from scipy.stats import vonmises_fisher

import qiskit_quantum_backend as quantum
from quantum_experiment_utils import balanced_shot_counts


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "odd_even_xz_style"
PAQT_LIU_WEST_A = 0.98
PAQT_RESAMPLE_ESS_FRACTION = 0.5


@dataclass(frozen=True)
class TFIConfig:
    """Parameters of the six-qubit disordered transverse-field Ising family."""

    n_qubits: int = 6
    transverse_field: float = 0.97
    term_strength: float = 0.35
    fixed_disorder_strength: float = 0.08
    perturbation_order: str = "x_z_zz"


@dataclass(frozen=True)
class PosteriorResult:
    task_estimate: np.ndarray
    theta_estimate: np.ndarray
    state: np.ndarray
    weights: np.ndarray
    ess: float
    settings: int
    copies: int
    resampling_count: int = 0
    minimum_ess: float = float("nan")


@dataclass(frozen=True)
class EstimateResult:
    task_estimate: np.ndarray
    state: np.ndarray
    settings: int
    copies: int
    ess: float


@dataclass(frozen=True)
class TaskParticlePosterior:
    """Ordinary task posterior evaluated on an importance particle cloud."""

    task_estimate: np.ndarray
    theta_estimate: np.ndarray
    ess: float
    prior_importance_ess: float




def build_disordered_tfi_family(
    config: TFIConfig,
    dimension: int,
) -> quantum.QiskitHamiltonianFamily:
    """Construct the physical Hamiltonian family used by the paper benchmark."""

    hamiltonian = quantum.zero_operator(config.n_qubits)
    terms = []
    names: list[str] = []
    fixed_fields = [
        config.fixed_disorder_strength * math.sin(1.37 * (site + 1))
        for site in range(config.n_qubits)
    ]
    z_terms = [
        quantum.pauli_operator(config.n_qubits, {site: "Z"})
        for site in range(config.n_qubits)
    ]
    x_terms = [
        quantum.pauli_operator(config.n_qubits, {site: "X"})
        for site in range(config.n_qubits)
    ]
    zz_terms = [
        quantum.pauli_operator(config.n_qubits, {site: "Z", site + 1: "Z"})
        for site in range(config.n_qubits - 1)
    ]

    for term in zz_terms:
        hamiltonian -= term
    for term in x_terms:
        hamiltonian -= config.transverse_field * term
    for field, term in zip(fixed_fields, z_terms):
        hamiltonian -= field * term

    term_groups = {
        "z": tuple(
            (
                config.term_strength * quantum.rms_normalize_operator(term),
                f"hZ{site}",
            )
            for site, term in enumerate(z_terms)
        ),
        "x": tuple(
            (
                config.term_strength * quantum.rms_normalize_operator(term),
                f"gX{site}",
            )
            for site, term in enumerate(x_terms)
        ),
        "zz": tuple(
            (
                config.term_strength * quantum.rms_normalize_operator(term),
                f"JZZ{site}_{site + 1}",
            )
            for site, term in enumerate(zz_terms)
        ),
    }
    order_by_name = {
        "x_z_zz": ("x", "z", "zz"),
        "z_x_zz": ("z", "x", "zz"),
    }
    try:
        group_order = order_by_name[config.perturbation_order]
    except KeyError as error:
        raise ValueError(
            f"Unknown perturbation order: {config.perturbation_order}"
        ) from error
    for group_name in group_order:
        for term, name in term_groups[group_name]:
            terms.append(term)
            names.append(name)

    if dimension > len(terms):
        raise ValueError(
            f"Requested d={dimension}, but only {len(terms)} physical terms are defined."
        )
    return quantum.QiskitHamiltonianFamily(
        n_qubits=config.n_qubits,
        base_operator=hamiltonian,
        term_operators=tuple(terms[:dimension]),
        term_names=tuple(names[:dimension]),
        task_operators=tuple(),
        task_coefficients=np.zeros((0, len(terms)), dtype=float),
        task_design_residual=0.0,
    )


def ground_state(
    family: quantum.QiskitHamiltonianFamily,
    theta: np.ndarray,
) -> np.ndarray:
    return quantum.ground_state(family, theta)


def align_state(reference: np.ndarray, state: np.ndarray) -> np.ndarray:
    overlap = np.vdot(reference, state)
    if abs(overlap) <= 1e-15:
        return np.asarray(state)
    return np.asarray(state) * np.exp(-1j * np.angle(overlap))


def sample_ball(
    rng: np.random.Generator,
    dimension: int,
    radius: float,
    count: int,
) -> np.ndarray:
    directions = rng.normal(size=(count, dimension))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    radii = radius * (rng.uniform(size=count) ** (1.0 / dimension))
    return directions * radii[:, None]


def make_particle_cloud(
    rng: np.random.Generator,
    dimension: int,
    radius: float,
    count: int,
) -> np.ndarray:
    """Draw a particle cloud from the structured uniform-ball prior."""

    if count < 1:
        raise ValueError("The particle cloud must contain at least one point.")
    return sample_ball(rng, dimension, radius, count)


def make_directional_importance_ball_cloud(
    *,
    rng: np.random.Generator,
    dimension: int,
    radius: float,
    count: int,
    design_directions: np.ndarray,
    concentrations: tuple[float, ...] = (8.0, 24.0, 64.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a directional mixture on a ball and return uniform-prior weights.

    Every mixture component has the uniform-ball radial law.  The directional
    mixture contains a uniform component and von Mises--Fisher components aimed
    at each measured direction and their normalized positive sum.  The returned
    log weights are ``log(uniform prior / proposal)`` and therefore correct all
    posterior and support-volume calculations back to the uniform-ball prior.
    """

    if dimension < 2:
        raise ValueError("Directional importance sampling requires dimension at least two.")
    if radius <= 0.0 or count <= 0:
        raise ValueError("The radius and particle count must be positive.")
    directions = np.asarray(design_directions, dtype=float)
    if directions.ndim != 2 or directions.shape[0] != dimension or directions.shape[1] == 0:
        raise ValueError("Design directions must have shape (dimension, settings).")
    directions = directions / np.linalg.norm(directions, axis=0)[None, :]
    summed = np.sum(directions, axis=1)
    summed_norm = float(np.linalg.norm(summed))
    centers = [directions[:, index] for index in range(directions.shape[1])]
    if summed_norm > 1e-12:
        centers.append(summed / summed_norm)

    component_specs: list[tuple[np.ndarray | None, float | None]] = [(None, None)]
    component_specs.extend(
        (center, concentration)
        for concentration in concentrations
        for center in centers
    )
    component_count = len(component_specs)
    allocations = np.full(component_count, count // component_count, dtype=int)
    allocations[: count % component_count] += 1
    mixture_weights = allocations / float(count)

    sampled_directions: list[np.ndarray] = []
    for allocation, (center, concentration) in zip(allocations, component_specs):
        if allocation == 0:
            continue
        if center is None:
            draws = rng.normal(size=(allocation, dimension))
            draws /= np.linalg.norm(draws, axis=1)[:, None]
        else:
            draws = np.asarray(
                vonmises_fisher(center, concentration).rvs(
                    size=allocation,
                    random_state=rng,
                ),
                dtype=float,
            ).reshape(allocation, dimension)
        sampled_directions.append(draws)
    unit_directions = np.vstack(sampled_directions)
    radii = radius * (rng.uniform(size=count) ** (1.0 / dimension))
    particles = unit_directions * radii[:, None]

    sphere_log_area = (
        math.log(2.0)
        + 0.5 * dimension * math.log(math.pi)
        - math.lgamma(0.5 * dimension)
    )
    log_ratio_components = [
        np.full(count, math.log(mixture_weights[0]), dtype=float)
    ]
    for component_index, (center, concentration) in enumerate(
        component_specs[1:],
        start=1,
    ):
        log_ratio_components.append(
            math.log(mixture_weights[component_index])
            + vonmises_fisher(center, concentration).logpdf(unit_directions)
            + sphere_log_area
        )
    log_proposal_over_prior = logsumexp(
        np.vstack(log_ratio_components),
        axis=0,
    )
    return particles, -log_proposal_over_prior




def exact_binomial_lower_confidence_bounds(
    counts: np.ndarray,
    shot_counts: np.ndarray,
    failure_probabilities: np.ndarray,
) -> np.ndarray:
    """Return one-sided Clopper--Pearson lower bounds for binomial success rates."""

    counts = np.asarray(counts, dtype=int)
    shot_counts = np.asarray(shot_counts, dtype=int)
    failure_probabilities = np.asarray(failure_probabilities, dtype=float)
    if counts.shape != shot_counts.shape or counts.shape != failure_probabilities.shape:
        raise ValueError("Counts, shots, and failure probabilities must have one shape.")
    if np.any(shot_counts <= 0):
        raise ValueError("Every confidence constraint requires a positive shot count.")
    if np.any(counts < 0) or np.any(counts > shot_counts):
        raise ValueError("Binomial counts must lie between zero and the shot count.")
    if np.any(failure_probabilities <= 0.0) or np.any(failure_probabilities >= 1.0):
        raise ValueError("Failure probabilities must lie strictly between zero and one.")

    lower = np.zeros(counts.shape, dtype=float)
    positive = counts > 0
    lower[positive] = beta_distribution.ppf(
        failure_probabilities[positive],
        counts[positive],
        shot_counts[positive] - counts[positive] + 1,
    )
    return lower




def task_particle_posterior_from_probabilities(
    *,
    particle_thetas: np.ndarray,
    particle_tasks: np.ndarray,
    particle_probabilities: np.ndarray,
    counts: np.ndarray,
    shot_counts: np.ndarray,
    log_prior_over_proposal: np.ndarray | None = None,
) -> TaskParticlePosterior:
    """Evaluate the ordinary binomial task posterior without support truncation."""

    particle_thetas = np.asarray(particle_thetas, dtype=float)
    particle_tasks = np.asarray(particle_tasks, dtype=float)
    particle_probabilities = np.asarray(particle_probabilities, dtype=float)
    counts = np.asarray(counts, dtype=int)
    shot_counts = np.asarray(shot_counts, dtype=int)
    if particle_probabilities.ndim != 2:
        raise ValueError("Particle probabilities must have shape (particles, settings).")
    particle_count, setting_count = particle_probabilities.shape
    if particle_thetas.shape[0] != particle_count or particle_tasks.shape[0] != particle_count:
        raise ValueError("Particle coordinates, tasks, and probabilities do not align.")
    if counts.shape != (setting_count,) or shot_counts.shape != (setting_count,):
        raise ValueError("Counts and shot allocations do not match the settings.")
    if log_prior_over_proposal is None:
        log_importance = np.zeros(particle_count, dtype=float)
    else:
        log_importance = np.asarray(log_prior_over_proposal, dtype=float)
        if log_importance.shape != (particle_count,):
            raise ValueError("Importance weights do not match the particle count.")
        if not np.all(np.isfinite(log_importance)):
            raise ValueError("Importance weights must be finite.")

    stabilized_importance = np.exp(log_importance - float(np.max(log_importance)))
    prior_weights = stabilized_importance / float(np.sum(stabilized_importance))
    log_posterior = log_importance.copy()
    for setting in range(setting_count):
        log_posterior += quantum.binomial_log_likelihood(
            int(counts[setting]),
            int(shot_counts[setting]),
            particle_probabilities[:, setting],
        )
    log_posterior -= float(np.max(log_posterior))
    weights = np.exp(log_posterior)
    weights /= float(np.sum(weights))
    return TaskParticlePosterior(
        task_estimate=weights @ particle_tasks,
        theta_estimate=weights @ particle_thetas,
        ess=float(1.0 / np.sum(weights**2)),
        prior_importance_ess=float(1.0 / np.sum(prior_weights**2)),
    )


def run_particle_posterior_from_measurements(
    particle_thetas: np.ndarray,
    particle_states: np.ndarray,
    probe_states: np.ndarray,
    counts: np.ndarray,
    shot_counts: np.ndarray,
    task_from_density: Callable[[np.ndarray], np.ndarray],
    particle_tasks: np.ndarray | None = None,
) -> PosteriorResult:
    particle_thetas = np.asarray(particle_thetas, dtype=float)
    particle_states = np.asarray(particle_states, dtype=complex)
    probe_states = np.asarray(probe_states, dtype=complex)
    counts = np.asarray(counts, dtype=float)
    shot_counts = np.asarray(shot_counts, dtype=float)
    if particle_states.ndim != 2 or particle_thetas.shape[0] != particle_states.shape[0]:
        raise ValueError("Particle coordinates and states do not align.")
    if probe_states.ndim != 2 or probe_states.shape[1] != particle_states.shape[1]:
        raise ValueError("Probe and particle state dimensions do not align.")
    if counts.shape != shot_counts.shape or counts.shape != (probe_states.shape[0],):
        raise ValueError("Counts and shot allocations do not match the probes.")
    if np.any(shot_counts <= 0.0) or np.any(counts < 0.0) or np.any(counts > shot_counts):
        raise ValueError("Every binomial count must lie within a positive shot allocation.")
    log_likelihood = np.zeros(particle_states.shape[0], dtype=float)
    for probe_state, count, shots in zip(probe_states, counts, shot_counts):
        probabilities = quantum.fidelities_to_state(particle_states, probe_state)
        log_likelihood += quantum.binomial_log_likelihood(count, shots, probabilities)
    log_likelihood -= float(np.max(log_likelihood))
    weights = np.exp(log_likelihood)
    weights /= float(np.sum(weights))
    bayesian_mean_density = pure_state_bayesian_mean_density(
        particle_states,
        weights,
    )
    if particle_tasks is None:
        task_estimate = np.asarray(task_from_density(bayesian_mean_density), dtype=float)
    else:
        particle_tasks = np.asarray(particle_tasks, dtype=float)
        if particle_tasks.ndim != 2 or particle_tasks.shape[0] != particle_states.shape[0]:
            raise ValueError("Particle tasks do not align with the particle states.")
        # Under squared task loss, the Bayes action is the posterior mean of
        # the task itself.  This generally differs from applying a nonlinear
        # task map to the Bayesian mean density operator.
        task_estimate = weights @ particle_tasks
    return PosteriorResult(
        task_estimate=task_estimate,
        theta_estimate=weights @ particle_thetas,
        state=bayesian_mean_density,
        weights=weights,
        ess=float(1.0 / np.sum(weights**2)),
        settings=int(probe_states.shape[0]),
        copies=int(np.sum(shot_counts)),
        minimum_ess=float(1.0 / np.sum(weights**2)),
    )


def pure_state_bayesian_mean_density(
    particle_states: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return the PAQT Bayesian mean density operator for pure particles."""

    states = np.asarray(particle_states, dtype=complex)
    weights = np.asarray(weights, dtype=float)
    if states.ndim != 2 or weights.shape != (states.shape[0],):
        raise ValueError("Particle states and weights do not align.")
    if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("Particle weights must be finite and nonnegative.")
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValueError("Particle weights must have positive total mass.")
    normalized_weights = weights / total_weight
    density = np.einsum(
        "n,ni,nj->ij",
        normalized_weights,
        states,
        states.conj(),
        optimize=True,
    )
    density = 0.5 * (density + density.conj().T)
    density /= float(np.trace(density).real)
    return density


def batch_ground_states(
    family: quantum.QiskitHamiltonianFamily,
    coordinates: np.ndarray,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    """Evaluate ground states without materializing every Hamiltonian at once."""

    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != len(family.terms):
        raise ValueError("Particle coordinates do not match the Hamiltonian family.")
    if batch_size <= 0:
        raise ValueError("The ground-state batch size must be positive.")
    base_hamiltonian = np.asarray(family.base_hamiltonian)
    terms = np.asarray(family.terms)
    states = np.empty(
        (coordinates.shape[0], base_hamiltonian.shape[0]),
        dtype=np.result_type(base_hamiltonian, terms),
    )
    for start in range(0, coordinates.shape[0], batch_size):
        stop = min(start + batch_size, coordinates.shape[0])
        matrices = base_hamiltonian[None, :, :] + np.tensordot(
            coordinates[start:stop],
            terms,
            axes=(1, 0),
        )
        for offset, matrix in enumerate(matrices):
            # Only the ground-state eigenvector is needed.  LAPACK's indexed
            # solver avoids computing the other 63 eigenvectors for each
            # six-qubit particle Hamiltonian.
            _, eigenvectors = scipy_eigh(
                matrix,
                subset_by_index=(0, 0),
                driver="evr",
                check_finite=False,
                overwrite_a=True,
            )
            states[start + offset] = eigenvectors[:, 0]
    return np.real_if_close(states, tol=1000)


def liu_west_resample(
    *,
    rng: np.random.Generator,
    particles: np.ndarray,
    weights: np.ndarray,
    radius: float,
    a: float = PAQT_LIU_WEST_A,
    maximum_attempts: int = 1000,
) -> np.ndarray:
    """Apply constrained Liu--West resampling to a weighted particle cloud."""

    particles = np.asarray(particles, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if particles.ndim != 2 or weights.shape != (particles.shape[0],):
        raise ValueError("Liu--West particles and weights do not align.")
    if not 0.0 < a <= 1.0:
        raise ValueError("The Liu--West shrinkage parameter must lie in (0, 1].")
    if radius <= 0.0 or maximum_attempts <= 0:
        raise ValueError("The radius and maximum attempt count must be positive.")
    if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("Liu--West weights must be finite and nonnegative.")
    weights = weights / float(np.sum(weights))
    mean = weights @ particles
    differences = particles - mean
    covariance = (differences * weights[:, None]).T @ differences
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = 1e-14 * max(float(eigenvalues[-1]), radius**2, 1.0)
    eigenvalues = np.maximum(eigenvalues, tolerance)
    noise_transform = (
        math.sqrt(max(1.0 - a**2, 0.0))
        * eigenvectors
        @ np.diag(np.sqrt(eigenvalues))
    )

    output = np.empty_like(particles)
    remaining = np.arange(particles.shape[0])
    for _ in range(maximum_attempts):
        if remaining.size == 0:
            return output
        ancestors = rng.choice(
            particles.shape[0],
            size=remaining.size,
            replace=True,
            p=weights,
        )
        locations = a * particles[ancestors] + (1.0 - a) * mean
        proposals = locations + rng.normal(
            size=(remaining.size, particles.shape[1])
        ) @ noise_transform.T
        accepted = np.linalg.norm(proposals, axis=1) <= radius
        output[remaining[accepted]] = proposals[accepted]
        remaining = remaining[~accepted]
    raise RuntimeError(
        "Liu--West resampling could not draw valid particles inside the ball."
    )


def run_liu_west_particle_posterior_from_measurements(
    *,
    rng: np.random.Generator,
    family: quantum.QiskitHamiltonianFamily,
    particle_thetas: np.ndarray,
    probe_states: np.ndarray,
    counts: np.ndarray,
    shot_counts: np.ndarray,
    radius: float,
    task_from_density: Callable[[np.ndarray], np.ndarray],
    particle_states: np.ndarray | None = None,
    particle_tasks: np.ndarray | None = None,
    task_from_state: Callable[[np.ndarray], np.ndarray] | None = None,
    tasks_from_states: Callable[[np.ndarray], np.ndarray] | None = None,
    resampling_a: float = PAQT_LIU_WEST_A,
    resample_ess_fraction: float = PAQT_RESAMPLE_ESS_FRACTION,
    likelihood_batch_size: int = 128,
) -> PosteriorResult:
    """Run the sequential Liu--West particle filter used by PAQT."""

    thetas = np.asarray(particle_thetas, dtype=float).copy()
    probes = np.asarray(probe_states)
    counts = np.asarray(counts, dtype=int)
    shots = np.asarray(shot_counts, dtype=int)
    if thetas.ndim != 2 or thetas.shape[0] < 2:
        raise ValueError("PAQT requires at least two structured particles.")
    if probes.ndim != 2 or counts.shape != shots.shape or counts.shape != (probes.shape[0],):
        raise ValueError("PAQT probes, counts, and shot allocations do not align.")
    if probes.shape[1] != family.base_hamiltonian.shape[0]:
        raise ValueError("PAQT probes do not match the Hamiltonian dimension.")
    if np.any(shots <= 0) or np.any(counts < 0) or np.any(counts > shots):
        raise ValueError("Every PAQT count must lie within a positive shot allocation.")
    if not 0.0 < resample_ess_fraction < 1.0:
        raise ValueError("The PAQT resampling threshold must lie in (0, 1).")
    if likelihood_batch_size <= 0:
        raise ValueError("The PAQT likelihood batch size must be positive.")
    if np.any(np.linalg.norm(thetas, axis=1) > radius * (1.0 + 1e-12)):
        raise ValueError("An initial PAQT particle lies outside the localization ball.")

    if particle_states is None:
        states = batch_ground_states(family, thetas)
    else:
        states = np.asarray(particle_states).copy()
        if states.shape != (thetas.shape[0], probes.shape[1]):
            raise ValueError("PAQT particle states do not match the coordinates.")
    tasks = None if particle_tasks is None else np.asarray(particle_tasks, dtype=float)
    if tasks is not None and tasks.shape[0] != thetas.shape[0]:
        raise ValueError("PAQT particle tasks do not match the coordinates.")

    particle_count = thetas.shape[0]
    weights = np.full(particle_count, 1.0 / particle_count, dtype=float)
    threshold = resample_ess_fraction * particle_count
    minimum_ess = float(particle_count)
    resampling_count = 0
    setting = 0
    while setting < probes.shape[0]:
        batch_start = setting
        stop = min(batch_start + likelihood_batch_size, probes.shape[0])
        # Matrix multiplication is much faster than one matrix-vector product
        # per setting.  We still consume the resulting columns sequentially,
        # and discard unused columns if a resampling event changes the cloud,
        # so the particle-filter update order is unchanged.
        probabilities = quantum.fidelity_matrix(states, probes[batch_start:stop])
        for offset in range(stop - batch_start):
            current = batch_start + offset
            log_weights = np.full(particle_count, -np.inf, dtype=float)
            np.log(weights, out=log_weights, where=weights > 0.0)
            log_weights += quantum.binomial_log_likelihood(
                int(counts[current]),
                int(shots[current]),
                probabilities[:, offset],
            )
            log_weights -= float(np.max(log_weights))
            weights = np.exp(log_weights)
            weights /= float(np.sum(weights))
            ess = float(1.0 / np.sum(weights**2))
            minimum_ess = min(minimum_ess, ess)
            setting = current + 1
            if ess < threshold:
                thetas = liu_west_resample(
                    rng=rng,
                    particles=thetas,
                    weights=weights,
                    radius=radius,
                    a=resampling_a,
                )
                states = batch_ground_states(family, thetas)
                tasks = None
                weights.fill(1.0 / particle_count)
                resampling_count += 1
                break

    bayesian_mean_density = pure_state_bayesian_mean_density(states, weights)
    if task_from_state is not None:
        if tasks is None:
            if tasks_from_states is None:
                tasks = np.asarray([task_from_state(state) for state in states])
            else:
                tasks = np.asarray(tasks_from_states(states), dtype=float)
                if tasks.ndim != 2 or tasks.shape[0] != particle_count:
                    raise ValueError(
                        "Batched PAQT particle tasks do not match the particle cloud."
                    )
        task_estimate = weights @ tasks
    else:
        task_estimate = np.asarray(task_from_density(bayesian_mean_density), dtype=float)
    final_ess = float(1.0 / np.sum(weights**2))
    return PosteriorResult(
        task_estimate=np.asarray(task_estimate, dtype=float),
        theta_estimate=weights @ thetas,
        state=bayesian_mean_density,
        weights=weights,
        ess=final_ess,
        settings=int(probes.shape[0]),
        copies=int(np.sum(shots)),
        resampling_count=resampling_count,
        minimum_ess=minimum_ess,
    )


def paqt_sgqt_shot_schedule(total_copies: int, iterations: int) -> np.ndarray:
    iterations = max(1, int(iterations))
    if total_copies < 2 * iterations:
        iterations = max(1, total_copies // 2)
    return balanced_shot_counts(total_copies, 2 * iterations)


def paqt_paper_gains(iteration: int) -> tuple[float, float]:
    k = iteration + 1
    return 10.0 / (k**0.602), 0.1 / (k**0.101)


def project_to_ball(theta: np.ndarray, radius: float) -> np.ndarray:
    norm = float(np.linalg.norm(theta))
    return theta * (radius / norm) if norm > radius else theta


def project_to_ball_in_metric(
    theta: np.ndarray,
    task_metric: np.ndarray,
    radius: float,
    *,
    relative_tolerance: float = 1e-9,
    multiplier_tolerance: float = 1e-12,
) -> np.ndarray:
    """Select the minimum-Euclidean-norm task-metric projection onto a ball.

    This solves

        min_{||b||_2 <= radius} (b-theta)^T task_metric (b-theta)

    for a positive-semidefinite task metric.  Eigenmodes below the requested
    relative tolerance are treated as the numerical task nullspace and set to
    zero, which implements the manuscript's minimum-norm tie-break.  When the
    task metric is a positive multiple of the identity, the result is exactly
    the ordinary radial projection used by the full-state specialization.
    """

    theta = np.asarray(theta, dtype=float)
    task_metric = np.asarray(task_metric, dtype=float)
    if theta.ndim != 1:
        raise ValueError("The point to project must be a vector.")
    if task_metric.shape != (theta.size, theta.size):
        raise ValueError("The task metric shape does not match the vector.")
    if radius <= 0.0 or not np.isfinite(radius):
        raise ValueError("The projection radius must be positive and finite.")
    if relative_tolerance < 0.0 or multiplier_tolerance <= 0.0:
        raise ValueError("Projection tolerances must be nonnegative and positive.")
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(task_metric)):
        raise ValueError("The projection inputs must be finite.")

    symmetric_metric = 0.5 * (task_metric + task_metric.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_metric)
    spectral_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(eigenvalues[0]) < -1e-10 * spectral_scale:
        raise ValueError("The task metric must be positive semidefinite.")

    largest = float(eigenvalues[-1])
    if largest <= 0.0:
        return np.zeros_like(theta)
    positive = eigenvalues > relative_tolerance * largest
    if not np.any(positive):
        return np.zeros_like(theta)

    active_values = np.asarray(eigenvalues[positive], dtype=float)
    active_vectors = np.asarray(eigenvectors[:, positive], dtype=float)
    active_coordinates = active_vectors.T @ theta

    # Preserve the old full-state path exactly when Q is proportional to I.
    if np.all(positive) and np.allclose(
        active_values,
        active_values[0],
        rtol=1e-14,
        atol=1e-14 * max(1.0, abs(float(active_values[0]))),
    ):
        return project_to_ball(theta, radius)

    relevant_part = active_vectors @ active_coordinates
    if float(np.linalg.norm(relevant_part)) <= radius:
        return relevant_part

    def constraint_residual(multiplier: float) -> float:
        shrinkage = active_values / (active_values + multiplier)
        return float(np.sum((shrinkage * active_coordinates) ** 2) - radius**2)

    lower = 0.0
    upper = max(largest, 1.0)
    while constraint_residual(upper) > 0.0:
        upper *= 2.0
        if not np.isfinite(upper):
            raise RuntimeError("Failed to bracket the task-projection multiplier.")
    for _ in range(200):
        midpoint = 0.5 * (lower + upper)
        if constraint_residual(midpoint) > 0.0:
            lower = midpoint
        else:
            upper = midpoint
        if upper - lower <= multiplier_tolerance * max(1.0, upper):
            break
    multiplier = 0.5 * (lower + upper)
    projected_coordinates = (
        active_values / (active_values + multiplier) * active_coordinates
    )
    return active_vectors @ projected_coordinates


def collect_structured_paqt_measurements(
    *,
    rng: np.random.Generator,
    truth_state: np.ndarray,
    family: quantum.QiskitHamiltonianFamily,
    dimension: int,
    total_copies: int,
    iterations: int,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the paper's S-PAQT acquisition and return probes and counts."""

    theta = np.zeros(dimension, dtype=float)
    shot_counts = paqt_sgqt_shot_schedule(total_copies, iterations)
    probe_states: list[np.ndarray] = []
    counts: list[int] = []
    for iteration in range(shot_counts.size // 2):
        alpha, epsilon = paqt_paper_gains(iteration)
        delta = rng.choice([-1.0, 1.0], size=dimension).astype(float)
        # The adaptive iterate is constrained to the localization ball, while
        # the paired SPSA probes remain unclipped as specified in the paper.
        plus_state = ground_state(family, theta + epsilon * delta)
        minus_state = ground_state(family, theta - epsilon * delta)
        plus_shots = int(shot_counts[2 * iteration])
        minus_shots = int(shot_counts[2 * iteration + 1])
        plus_probability = quantum.state_fidelity_probability(
            plus_state,
            truth_state,
        )
        minus_probability = quantum.state_fidelity_probability(
            minus_state,
            truth_state,
        )
        plus_count = int(rng.binomial(plus_shots, plus_probability))
        minus_count = int(rng.binomial(minus_shots, minus_probability))
        gradient = (
            (plus_count / plus_shots - minus_count / minus_shots)
            / (2.0 * epsilon)
        ) * delta
        theta = project_to_ball(theta + alpha * gradient, radius)
        probe_states.extend((plus_state, minus_state))
        counts.extend((plus_count, minus_count))
    return (
        np.asarray(probe_states),
        np.asarray(counts, dtype=int),
        np.asarray(shot_counts, dtype=int),
    )


def run_structured_paqt(
    rng: np.random.Generator,
    truth_state: np.ndarray,
    family: quantum.QiskitHamiltonianFamily,
    dimension: int,
    particle_thetas: np.ndarray,
    particle_states: np.ndarray,
    total_copies: int,
    iterations: int,
    radius: float,
    task_from_density: Callable[[np.ndarray], np.ndarray],
    particle_tasks: np.ndarray | None = None,
    task_from_state: Callable[[np.ndarray], np.ndarray] | None = None,
    tasks_from_states: Callable[[np.ndarray], np.ndarray] | None = None,
    resampling_a: float = PAQT_LIU_WEST_A,
    resample_ess_fraction: float = PAQT_RESAMPLE_ESS_FRACTION,
) -> PosteriorResult:
    probe_states, counts, shot_counts = collect_structured_paqt_measurements(
        rng=rng,
        truth_state=truth_state,
        family=family,
        dimension=dimension,
        total_copies=total_copies,
        iterations=iterations,
        radius=radius,
    )
    return run_liu_west_particle_posterior_from_measurements(
        rng=rng,
        family=family,
        particle_thetas=particle_thetas,
        particle_states=particle_states,
        particle_tasks=particle_tasks,
        probe_states=probe_states,
        counts=counts,
        shot_counts=shot_counts,
        radius=radius,
        task_from_density=task_from_density,
        task_from_state=task_from_state,
        tasks_from_states=tasks_from_states,
        resampling_a=resampling_a,
        resample_ess_fraction=resample_ess_fraction,
    )


def paired_sgqt_osgqt_paper_gains(iteration: int) -> tuple[float, float]:
    """Matched gains used for the SGQT/OSGQT comparison of Dekkers et al."""

    if iteration < 0:
        raise ValueError("The iteration index must be nonnegative.")
    return 0.05, 0.2


def sgqt_fidelity_difference(
    measured_plus: float,
    measured_minus: float,
    current_state: np.ndarray,
    plus_state: np.ndarray,
    minus_state: np.ndarray,
    osgqt: bool,
) -> float:
    """Return the SGQT numerator, including the published OSGQT correction."""

    difference = float(measured_plus - measured_minus)
    if osgqt:
        difference -= quantum.state_fidelity_probability(current_state, plus_state)
        difference += quantum.state_fidelity_probability(current_state, minus_state)
    return difference


def run_structured_sgqt(
    rng: np.random.Generator,
    truth_state: np.ndarray,
    family: quantum.QiskitHamiltonianFamily,
    dimension: int,
    total_copies: int,
    iterations: int,
    radius: float,
    osgqt: bool,
) -> EstimateResult:
    theta = np.zeros(dimension, dtype=float)
    shot_counts = paqt_sgqt_shot_schedule(total_copies, iterations)
    directions = [
        rng.choice([-1.0, 1.0], size=dimension).astype(float)
        for _ in range(shot_counts.size // 2)
    ]
    for iteration, direction in enumerate(directions):
        alpha, beta = paired_sgqt_osgqt_paper_gains(iteration)
        current_state = ground_state(family, theta)
        plus_state = ground_state(family, theta + beta * direction)
        minus_state = ground_state(family, theta - beta * direction)
        plus_shots = int(shot_counts[2 * iteration])
        minus_shots = int(shot_counts[2 * iteration + 1])
        plus_value = rng.binomial(
            plus_shots,
            quantum.state_fidelity_probability(plus_state, truth_state),
        ) / plus_shots
        minus_value = rng.binomial(
            minus_shots,
            quantum.state_fidelity_probability(minus_state, truth_state),
        ) / minus_shots
        fidelity_difference = sgqt_fidelity_difference(
            plus_value,
            minus_value,
            current_state,
            plus_state,
            minus_state,
            osgqt,
        )
        theta += alpha * (fidelity_difference / (2.0 * beta)) * direction
        theta = project_to_ball(theta, radius)
    state = ground_state(family, theta)
    return EstimateResult(
        task_estimate=np.empty(0, dtype=float),
        state=state,
        settings=int(shot_counts.size),
        copies=int(np.sum(shot_counts)),
        ess=float("nan"),
    )


def markdown_table(rows: list[dict], columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [
            f"{row[column]:.6g}" if isinstance(row[column], float) else str(row[column])
            for column in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_union_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
