"""Reproducibility checks for the Qiskit layer of the current quantum benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

import qiskit_quantum_backend as quantum
import quantum_benchmark_support as support
import quantum_greedy_spectral_experiment as experiment
from quantum_experiment_utils import single_site_operator, two_site_operator


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = support.ACTIVE_EXPERIMENT_ROOT / "qiskit_backend_validation.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


@dataclass(frozen=True)
class LegacyDenseFamily:
    n_qubits: int
    base_hamiltonian: np.ndarray
    terms: tuple[np.ndarray, ...]

    def hamiltonian(self, theta: np.ndarray) -> np.ndarray:
        matrix = self.base_hamiltonian.copy()
        for coefficient, term in zip(theta, self.terms):
            matrix += float(coefficient) * term
        return matrix


def legacy_dense_family(
    config: support.TFIConfig,
    dimension: int,
) -> LegacyDenseFamily:
    """Reconstruct the dense Kronecker representation independently of Qiskit."""

    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]])
    hilbert_dimension = 2**config.n_qubits
    z_terms = [
        single_site_operator(pauli_z, site, config.n_qubits)
        for site in range(config.n_qubits)
    ]
    x_terms = [
        single_site_operator(pauli_x, site, config.n_qubits)
        for site in range(config.n_qubits)
    ]
    zz_terms = [
        two_site_operator(pauli_z, site, pauli_z, site + 1, config.n_qubits)
        for site in range(config.n_qubits - 1)
    ]
    base_hamiltonian = np.zeros((hilbert_dimension, hilbert_dimension), dtype=float)
    for term in zz_terms:
        base_hamiltonian -= term
    for term in x_terms:
        base_hamiltonian -= config.transverse_field * term
    for site, term in enumerate(z_terms):
        field = config.fixed_disorder_strength * math.sin(1.37 * (site + 1))
        base_hamiltonian -= field * term
    term_groups = {
        "z": z_terms,
        "x": x_terms,
        "zz": zz_terms,
    }
    order_by_name = {
        "x_z_zz": ("x", "z", "zz"),
        "z_x_zz": ("z", "x", "zz"),
    }
    ordered_terms = tuple(
        term
        for group_name in order_by_name[config.perturbation_order]
        for term in term_groups[group_name]
    )
    terms = tuple(config.term_strength * term for term in ordered_terms[:dimension])
    return LegacyDenseFamily(config.n_qubits, base_hamiltonian, terms)


def dense_subsystem_spectrum(
    state: np.ndarray,
    n_qubits: int,
    subsystem: tuple[int, ...],
) -> np.ndarray:
    subsystem_axes = tuple(n_qubits - 1 - qubit for qubit in subsystem)
    complement_axes = tuple(
        axis for axis in range(n_qubits) if axis not in subsystem_axes
    )
    matrix = (
        np.asarray(state)
        .reshape((2,) * n_qubits)
        .transpose(subsystem_axes + complement_axes)
        .reshape(2 ** len(subsystem), -1)
    )
    singular_values = np.linalg.svd(
        matrix,
        compute_uv=False,
    )
    return np.sort(np.clip(singular_values**2, 0.0, 1.0))[::-1]


def run_validation() -> dict[str, object]:
    config = experiment.GreedyTaskConfig()
    family_config = support.TFIConfig(n_qubits=config.n_qubits)
    rng = np.random.default_rng(20260717)
    max_hamiltonian_difference = 0.0
    minimum_ground_state_fidelity = 1.0
    max_spectrum_difference = 0.0
    max_vectorized_fidelity_difference = 0.0
    reference_states: list[np.ndarray] = []

    for dimension in config.original_dimensions:
        qiskit_family = support.build_disordered_tfi_family(family_config, dimension)
        dense_family = legacy_dense_family(family_config, dimension)
        for _ in range(4):
            theta = rng.normal(scale=0.08, size=dimension)
            max_hamiltonian_difference = max(
                max_hamiltonian_difference,
                float(
                    np.max(
                        np.abs(
                            qiskit_family.hamiltonian(theta)
                            - dense_family.hamiltonian(theta)
                        )
                    )
                ),
            )
            qiskit_state = quantum.ground_state(qiskit_family, theta)
            dense_state = quantum.ground_state(dense_family, theta)
            minimum_ground_state_fidelity = min(
                minimum_ground_state_fidelity,
                quantum.state_fidelity_probability(qiskit_state, dense_state),
            )
            qiskit_spectrum = quantum.subsystem_schmidt_probabilities(
                qiskit_state,
                config.n_qubits,
                config.task_subsystem,
            )
            dense_spectrum = dense_subsystem_spectrum(
                qiskit_state,
                config.n_qubits,
                config.task_subsystem,
            )
            max_spectrum_difference = max(
                max_spectrum_difference,
                float(np.max(np.abs(qiskit_spectrum - dense_spectrum))),
            )
            reference_states.append(qiskit_state)

    state_array = np.asarray(reference_states)
    vectorized = quantum.fidelity_matrix(state_array, state_array)
    reference = np.array(
        [
            [
                quantum.state_fidelity_probability(first, second)
                for second in state_array
            ]
            for first in state_array
        ]
    )
    max_vectorized_fidelity_difference = float(np.max(np.abs(vectorized - reference)))

    target = state_array[0]
    anchor = state_array[-1]
    exact_fidelity = quantum.state_fidelity_probability(target, anchor)
    circuit = quantum.compute_uncompute_circuit(target, anchor, measure=False)
    ideal_circuit_probability = float(
        abs(Statevector.from_instruction(circuit).data[0]) ** 2
    )
    shots = 20_000
    aer_frequency = quantum.aer_fidelity_count(
        target,
        anchor,
        shots=shots,
        seed=20260717,
    ) / shots
    binomial_standard_error = math.sqrt(
        max(exact_fidelity * (1.0 - exact_fidelity), 1e-30) / shots
    )
    aer_standardized_difference = abs(aer_frequency - exact_fidelity) / max(
        binomial_standard_error,
        1.0 / shots,
    )

    local_dimensions: dict[str, int] = {}
    task_ranks: dict[str, int] = {}
    maximum_tangent_gram_error = 0.0
    for dimension in config.original_dimensions:
        model = experiment.build_local_model(config, dimension)
        local_dimension = model.coordinate_map.shape[1]
        geometry = experiment.compute_task_geometry(
            model,
            config,
        )
        tangent_gram = np.real_if_close(
            model.tangent_matrix.conj().T @ model.tangent_matrix
        ).real
        maximum_tangent_gram_error = max(
            maximum_tangent_gram_error,
            float(np.max(np.abs(tangent_gram - np.eye(local_dimension)))),
        )
        local_dimensions[str(dimension)] = int(local_dimension)
        task_ranks[str(dimension)] = int(geometry.task_rank)

    report: dict[str, object] = {
        **quantum.version_metadata(),
        "numpy_version": np.__version__,
        "dimensions": list(config.original_dimensions),
        "random_hamiltonians_per_dimension": 4,
        "max_hamiltonian_absolute_difference": max_hamiltonian_difference,
        "minimum_ground_state_fidelity": minimum_ground_state_fidelity,
        "max_schmidt_spectrum_absolute_difference": max_spectrum_difference,
        "max_vectorized_fidelity_absolute_difference": (
            max_vectorized_fidelity_difference
        ),
        "compute_uncompute_exact_fidelity": exact_fidelity,
        "compute_uncompute_ideal_probability": ideal_circuit_probability,
        "compute_uncompute_absolute_difference": abs(
            ideal_circuit_probability - exact_fidelity
        ),
        "aer_shots": shots,
        "aer_frequency": aer_frequency,
        "aer_binomial_standard_error": binomial_standard_error,
        "aer_standardized_difference": aer_standardized_difference,
        "local_dimensions": local_dimensions,
        "task_ranks": task_ranks,
        "task_weighting": "identity (raw Euclidean spectrum loss)",
        "maximum_tangent_gram_error": maximum_tangent_gram_error,
    }

    require(max_hamiltonian_difference < 2e-14, "Hamiltonian equivalence failed.")
    require(
        minimum_ground_state_fidelity > 1.0 - 2e-13,
        "Ground-state equivalence failed.",
    )
    require(max_spectrum_difference < 2e-13, "Schmidt-spectrum equivalence failed.")
    require(max_vectorized_fidelity_difference < 2e-13, "Fidelity equivalence failed.")
    require(
        abs(ideal_circuit_probability - exact_fidelity) < 2e-13,
        "Compute-uncompute circuit equivalence failed.",
    )
    require(aer_standardized_difference < 5.0, "Aer sampling validation failed.")
    require(maximum_tangent_gram_error < 2e-10, "Tangent whitening failed.")
    require(
        task_ranks == {"6": 5, "12": 6, "17": 7},
        "The current raw task ranks are stale.",
    )
    return report


def main() -> None:
    report = run_validation()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved validation report to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
