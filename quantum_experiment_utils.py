"""Small classical utilities shared by the active quantum experiments."""

from __future__ import annotations

import numpy as np


def kron_all(operators: list[np.ndarray]) -> np.ndarray:
    output = operators[0]
    for operator in operators[1:]:
        output = np.kron(output, operator)
    return output


def single_site_operator(
    local_operator: np.ndarray,
    site: int,
    n_qubits: int,
) -> np.ndarray:
    operators = [np.eye(2)] * n_qubits
    operators[site] = local_operator
    return kron_all(operators)


def two_site_operator(
    first_operator: np.ndarray,
    first_site: int,
    second_operator: np.ndarray,
    second_site: int,
    n_qubits: int,
) -> np.ndarray:
    operators = [np.eye(2)] * n_qubits
    operators[first_site] = first_operator
    operators[second_site] = second_operator
    return kron_all(operators)


def balanced_shot_counts(total_shots: int, n_settings: int) -> np.ndarray:
    if n_settings <= 0:
        raise ValueError("The number of settings must be positive.")
    if total_shots < n_settings:
        raise ValueError("Total shots must be at least the number of settings.")
    base = total_shots // n_settings
    remainder = total_shots % n_settings
    shot_counts = np.full(n_settings, base, dtype=int)
    shot_counts[:remainder] += 1
    return shot_counts
