"""Generalized task coordinates and exact greedy spectral shot allocation."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import numpy as np
from scipy.linalg import eigh


Array = np.ndarray


@dataclass(frozen=True)
class GreedySpectralAllocationResult:
    """Exact integer allocation for the manuscript's fixed-radius model."""

    generalized_eigenvalues: Array
    generalized_eigenvectors: Array
    retained_dimension: int
    shot_allocations: Array
    terminal_marginal_gains: Array
    prior_risk: float
    final_risk: float


def greedy_spectral_allocation(
    *,
    task_metric: Array,
    geometry_metric: Array,
    alpha: float,
    nu: float,
    budget: int,
    retained_dimension: int | None = None,
    null_eigenvalue_relative_tolerance: float = 1e-10,
) -> GreedySpectralAllocationResult:
    """Run the fixed-radius greedy allocation stated in the manuscript.

    The calculation is performed in the generalized task basis.  It assumes
    baseline precision ``alpha * I`` after G-whitening and common rank-one
    information ``nu`` per shot.  A heap implements the manuscript's
    repeated largest-marginal-gain rule in ``O((budget + r) log r)`` time.
    """

    task_metric = np.asarray(task_metric, dtype=float)
    geometry_metric = np.asarray(geometry_metric, dtype=float)
    if task_metric.ndim != 2 or task_metric.shape[0] != task_metric.shape[1]:
        raise ValueError("The task metric must be a square matrix.")
    if geometry_metric.shape != task_metric.shape:
        raise ValueError("The geometry and task metrics must have the same shape.")
    for name, matrix in (("task", task_metric), ("geometry", geometry_metric)):
        if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
            raise ValueError(f"The {name} metric must be symmetric.")
    np.linalg.cholesky(geometry_metric)
    task_tolerance = 1e-10 * max(float(np.linalg.norm(task_metric, ord=2)), 1.0)
    if float(np.min(np.linalg.eigvalsh(task_metric))) < -task_tolerance:
        raise ValueError("The task metric must be positive semidefinite.")
    if alpha <= 0.0 or nu <= 0.0:
        raise ValueError("alpha and nu must be positive.")
    if int(budget) != budget or budget < 0:
        raise ValueError("The shot budget must be a nonnegative integer.")
    if null_eigenvalue_relative_tolerance < 0.0:
        raise ValueError("The null-eigenvalue tolerance must be nonnegative.")

    eigenvalues, eigenvectors = canonical_generalized_task_basis(
        task_metric,
        geometry_metric,
        null_eigenvalue_relative_tolerance,
    )
    dimension = eigenvalues.size
    leading = float(eigenvalues[0])
    positive_count = int(
        np.sum(eigenvalues > null_eigenvalue_relative_tolerance * leading)
    )
    if retained_dimension is None:
        retained_dimension = positive_count
    retained_dimension = int(retained_dimension)
    if not 1 <= retained_dimension <= dimension:
        raise ValueError("The retained dimension must lie between one and d.")

    retained_values = np.maximum(eigenvalues[:retained_dimension], 0.0)
    allocations = np.zeros(retained_dimension, dtype=int)

    def marginal_gain(index: int) -> float:
        count = int(allocations[index])
        value = float(retained_values[index])
        return (
            value
            * nu
            / ((alpha + nu * count) * (alpha + nu * (count + 1)))
        )

    heap = [(-marginal_gain(index), index) for index in range(retained_dimension)]
    heapq.heapify(heap)
    for _ in range(int(budget)):
        _, index = heapq.heappop(heap)
        allocations[index] += 1
        heapq.heappush(heap, (-marginal_gain(index), index))

    terminal_gains = np.array(
        [marginal_gain(index) for index in range(retained_dimension)],
        dtype=float,
    )
    prior_risk = float(np.sum(np.maximum(eigenvalues, 0.0)) / alpha)
    retained_risk = float(
        np.sum(retained_values / (alpha + nu * allocations))
    )
    tail_risk = float(
        np.sum(np.maximum(eigenvalues[retained_dimension:], 0.0)) / alpha
    )
    return GreedySpectralAllocationResult(
        generalized_eigenvalues=eigenvalues,
        generalized_eigenvectors=eigenvectors,
        retained_dimension=retained_dimension,
        shot_allocations=allocations,
        terminal_marginal_gains=terminal_gains,
        prior_risk=prior_risk,
        final_risk=retained_risk + tail_risk,
    )


def _metric_inner(first: Array, second: Array, metric: Array) -> float:
    return float(first @ metric @ second)


def canonical_generalized_task_basis(
    task_metric: Array,
    geometry_metric: Array,
    zero_relative_tolerance: float,
) -> tuple[Array, Array]:
    """Return descending generalized eigenpairs with a stable null-space basis.

    Positive task eigenspaces come directly from the symmetric generalized
    eigensolver.  When the task matrix is rank deficient, its null eigenspace
    has no preferred eigenbasis.  We complete the positive eigendirections by
    deterministic pivoted G-Gram--Schmidt on the coefficient axes.  This is an
    exact generalized eigenbasis for the numerical null space and avoids
    machine-dependent control designs.
    """

    eigenvalues, eigenvectors = eigh(task_metric, geometry_metric)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=float)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=float)
    leading = max(float(eigenvalues[0]), 0.0)
    if leading <= 0.0:
        raise ValueError("The task metric has no positive generalized eigenvalue.")
    threshold = zero_relative_tolerance * leading
    positive_count = int(np.sum(eigenvalues > threshold))

    basis: list[Array] = []
    for index in range(positive_count):
        vector = eigenvectors[:, index].copy()
        for _ in range(2):
            for previous in basis:
                vector -= previous * _metric_inner(previous, vector, geometry_metric)
        norm = math.sqrt(max(_metric_inner(vector, vector, geometry_metric), 0.0))
        if norm <= 1e-14:
            raise ValueError("The positive generalized eigenspace is numerically singular.")
        vector /= norm
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0.0:
            vector = -vector
        basis.append(vector)

    remaining_axes = list(range(task_metric.shape[0]))
    while len(basis) < task_metric.shape[0]:
        candidates: list[tuple[float, int, Array]] = []
        for axis in remaining_axes:
            vector = np.eye(1, task_metric.shape[0], axis, dtype=float).ravel()
            for _ in range(2):
                for previous in basis:
                    vector -= previous * _metric_inner(previous, vector, geometry_metric)
            norm_squared = _metric_inner(vector, vector, geometry_metric)
            candidates.append((norm_squared, axis, vector))
        norm_squared, axis, vector = max(
            candidates,
            key=lambda item: (item[0], -item[1]),
        )
        if norm_squared <= 1e-20:
            raise ValueError("Could not construct a complete G-orthonormal basis.")
        vector /= math.sqrt(norm_squared)
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0.0:
            vector = -vector
        basis.append(vector)
        remaining_axes.remove(axis)

    directions = np.column_stack(basis)
    rayleigh_values = np.array(
        [
            float(vector @ task_metric @ vector)
            for vector in directions.T
        ],
        dtype=float,
    )
    rayleigh_values[:positive_count] = eigenvalues[:positive_count]
    rayleigh_values[positive_count:] = 0.0
    return rayleigh_values, directions

