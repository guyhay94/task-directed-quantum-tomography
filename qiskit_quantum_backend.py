"""Qiskit quantum layer for the paper's six-qubit experiments.

Qiskit owns the quantum-domain objects and operations in this module:
Pauli Hamiltonians, state validation, fidelities, reduced-state spectra, and
compute--uncompute circuits.  Dense diagonalization, Monte Carlo
sampling, and Bayesian inference remain classical NumPy/SciPy operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import qiskit
import qiskit_aer
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import (
    DensityMatrix,
    SparsePauliOp,
    Statevector,
    partial_trace,
    random_statevector,
    schmidt_decomposition,
    state_fidelity as qiskit_state_fidelity,
)
from qiskit_aer import AerSimulator
from scipy.linalg import eigh
from scipy.special import xlog1py, xlogy


QISKIT_BACKEND_NAME = "qiskit"
QISKIT_VERSION = qiskit.__version__
QISKIT_AER_VERSION = qiskit_aer.__version__


@dataclass(frozen=True)
class QiskitHamiltonianFamily:
    """Local Hamiltonian family represented by Qiskit Pauli operators.

    The dense ``hamiltonian`` method is intentionally retained as the boundary
    to the classical exact eigensolver and the existing statistical baselines.
    """

    n_qubits: int
    base_operator: SparsePauliOp
    term_operators: tuple[SparsePauliOp, ...]
    term_names: tuple[str, ...]
    task_operators: tuple[np.ndarray, ...] = tuple()
    task_coefficients: np.ndarray | None = None
    task_design_residual: float = 0.0
    _base_matrix: np.ndarray = field(init=False, repr=False)
    _term_matrix_stack: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        base_matrix = np.real_if_close(self.base_operator.to_matrix(), tol=1000)
        term_matrices = [
            np.real_if_close(term.to_matrix(), tol=1000)
            for term in self.term_operators
        ]
        if np.iscomplexobj(base_matrix) or any(np.iscomplexobj(term) for term in term_matrices):
            raise ValueError("The benchmark Hamiltonian family must be real.")
        object.__setattr__(self, "_base_matrix", np.asarray(base_matrix, dtype=float))
        object.__setattr__(
            self,
            "_term_matrix_stack",
            np.asarray(term_matrices, dtype=float),
        )

    def hamiltonian_operator(self, theta: np.ndarray) -> SparsePauliOp:
        coefficients = np.asarray(theta, dtype=float)
        if coefficients.shape != (len(self.term_operators),):
            raise ValueError(
                f"Expected {len(self.term_operators)} Hamiltonian coefficients, "
                f"received shape {coefficients.shape}."
            )
        operator = self.base_operator.copy()
        for coefficient, term in zip(coefficients, self.term_operators):
            operator = operator + float(coefficient) * term
        return operator.simplify()

    def hamiltonian(self, theta: np.ndarray) -> np.ndarray:
        coefficients = np.asarray(theta, dtype=float)
        if coefficients.shape != (len(self.term_operators),):
            raise ValueError(
                f"Expected {len(self.term_operators)} Hamiltonian coefficients, "
                f"received shape {coefficients.shape}."
            )
        return self._base_matrix + np.tensordot(
            coefficients,
            self._term_matrix_stack,
            axes=(0, 0),
        )

    @property
    def base_hamiltonian(self) -> np.ndarray:
        return self._base_matrix.copy()

    @property
    def terms(self) -> tuple[np.ndarray, ...]:
        return tuple(matrix.copy() for matrix in self._term_matrix_stack)


def pauli_operator(
    n_qubits: int,
    assignments: dict[int, str],
    coefficient: float = 1.0,
) -> SparsePauliOp:
    """Construct a Pauli string using the paper's left-to-right site order."""

    label = ["I"] * n_qubits
    for site, pauli in assignments.items():
        if not 0 <= int(site) < n_qubits:
            raise ValueError(f"Qubit site {site} is outside a {n_qubits}-qubit register.")
        pauli = str(pauli).upper()
        if pauli not in {"I", "X", "Y", "Z"}:
            raise ValueError(f"Unsupported Pauli label: {pauli}")
        label[int(site)] = pauli
    return SparsePauliOp.from_list([("".join(label), complex(coefficient))])


def zero_operator(n_qubits: int) -> SparsePauliOp:
    return SparsePauliOp.from_list([("I" * n_qubits, 0.0)])


def rms_normalize_operator(operator: SparsePauliOp) -> SparsePauliOp:
    matrix = operator.to_matrix()
    dimension = matrix.shape[0]
    rms = math.sqrt(float(np.trace(matrix.conj().T @ matrix).real) / dimension)
    if rms <= 0.0:
        raise ValueError("Cannot normalize a zero quantum operator.")
    return operator / rms


def as_statevector(state: np.ndarray | Statevector) -> Statevector:
    vector = state if isinstance(state, Statevector) else Statevector(np.asarray(state, dtype=complex))
    norm = float(np.linalg.norm(vector.data))
    if not np.isclose(norm, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError(f"Quantum state has norm {norm}, not one.")
    return vector


def normalized_state_data(state: np.ndarray | Statevector) -> np.ndarray:
    """Normalize arbitrary amplitudes and return Qiskit-validated state data."""

    amplitudes = (
        np.asarray(state.data, dtype=complex)
        if isinstance(state, Statevector)
        else np.asarray(state, dtype=complex)
    )
    if amplitudes.ndim != 1:
        raise ValueError("A pure-state amplitude vector must be one-dimensional.")
    norm = float(np.linalg.norm(amplitudes))
    if not np.isfinite(norm) or norm <= 1e-15:
        raise ValueError("Cannot normalize a zero or non-finite quantum state.")
    return as_statevector(amplitudes / norm).data


def random_state_data(
    hilbert_dimension: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a complex Haar-random pure state with Qiskit."""

    if hilbert_dimension <= 0 or hilbert_dimension & (hilbert_dimension - 1):
        raise ValueError("The benchmark requires a positive power-of-two dimension.")
    return as_statevector(random_statevector(hilbert_dimension, seed=rng)).data


def ground_state(family: object, theta: np.ndarray) -> np.ndarray:
    """Return the exact ground state as validated Qiskit statevector data."""

    matrix = np.asarray(family.hamiltonian(theta))
    _, eigenvectors = eigh(matrix)
    state = np.asarray(eigenvectors[:, 0], dtype=complex)
    if abs(float(np.sum(state).imag)) <= 1e-12 and float(np.sum(state).real) < 0.0:
        state = -state
    vector = as_statevector(state).data
    vector = np.real_if_close(vector, tol=1000)
    if np.iscomplexobj(vector):
        return np.asarray(vector, dtype=complex)
    return np.asarray(vector, dtype=float)




def subsystem_schmidt_probabilities(
    state: np.ndarray | Statevector,
    n_qubits: int,
    subsystem: tuple[int, ...] | list[int],
) -> np.ndarray:
    """Ordered squared Schmidt coefficients for a pure-state bipartition.

    The returned vector has the maximum Schmidt dimension of the requested
    bipartition.  Qiskit omits exactly zero Schmidt coefficients, so zeros are
    padded explicitly to keep the downstream task dimension fixed.
    """

    subsystem = tuple(int(qubit) for qubit in subsystem)
    if not subsystem or len(subsystem) >= n_qubits:
        raise ValueError("The subsystem must be nonempty and smaller than the full register.")
    if len(set(subsystem)) != len(subsystem):
        raise ValueError("Subsystem qubit indices must be distinct.")
    if any(qubit < 0 or qubit >= n_qubits for qubit in subsystem):
        raise ValueError("A subsystem qubit index lies outside the register.")
    decomposition = schmidt_decomposition(
        as_statevector(state),
        list(subsystem),
    )
    probabilities = np.array(
        [float(coefficient) ** 2 for coefficient, _, _ in decomposition],
        dtype=float,
    )
    probabilities = np.sort(np.clip(probabilities, 0.0, 1.0))[::-1]
    maximum_rank = 2 ** min(len(subsystem), n_qubits - len(subsystem))
    if probabilities.size > maximum_rank:
        raise RuntimeError("Qiskit returned more Schmidt coefficients than the bipartition permits.")
    output = np.zeros(maximum_rank, dtype=float)
    output[: probabilities.size] = probabilities
    total = float(np.sum(output))
    if not np.isclose(total, 1.0, rtol=1e-9, atol=1e-11):
        raise RuntimeError(f"Squared Schmidt coefficients sum to {total}, not one.")
    return output / total


def subsystem_density_probabilities(
    state: np.ndarray | Statevector | DensityMatrix,
    n_qubits: int,
    subsystem: tuple[int, ...] | list[int],
) -> np.ndarray:
    """Ordered eigenvalues of a subsystem's reduced density operator.

    Unlike :func:`subsystem_schmidt_probabilities`, this function also accepts
    a mixed global state.  That distinction is required for PAQT: its reported
    Bayesian mean estimate is generally a mixed density operator even when all
    posterior particles and the true state are pure.
    """

    subsystem = tuple(int(qubit) for qubit in subsystem)
    if not subsystem or len(subsystem) >= n_qubits:
        raise ValueError("The subsystem must be nonempty and smaller than the full register.")
    if len(set(subsystem)) != len(subsystem):
        raise ValueError("Subsystem qubit indices must be distinct.")
    if any(qubit < 0 or qubit >= n_qubits for qubit in subsystem):
        raise ValueError("A subsystem qubit index lies outside the register.")

    if isinstance(state, DensityMatrix):
        density = state
    elif isinstance(state, Statevector) or np.asarray(state).ndim == 1:
        density = DensityMatrix(as_statevector(state))
    else:
        matrix = np.asarray(state, dtype=complex)
        hilbert_dimension = 2**n_qubits
        if matrix.shape != (hilbert_dimension, hilbert_dimension):
            raise ValueError(
                "A density operator must have shape "
                f"({hilbert_dimension}, {hilbert_dimension})."
            )
        density = DensityMatrix(matrix)
    if density.num_qubits != n_qubits or not density.is_valid(
        rtol=1e-9,
        atol=1e-11,
    ):
        raise ValueError("The supplied density operator is not a valid n-qubit state.")

    complement = [qubit for qubit in range(n_qubits) if qubit not in subsystem]
    reduced = partial_trace(density, complement)
    probabilities = np.linalg.eigvalsh(np.asarray(reduced.data, dtype=complex)).real
    probabilities = np.sort(np.clip(probabilities, 0.0, 1.0))[::-1]
    expected_dimension = 2 ** len(subsystem)
    if probabilities.shape != (expected_dimension,):
        raise RuntimeError("Qiskit returned a reduced state of unexpected dimension.")
    total = float(np.sum(probabilities))
    if not np.isclose(total, 1.0, rtol=1e-9, atol=1e-11):
        raise RuntimeError(f"Reduced-state eigenvalues sum to {total}, not one.")
    return probabilities / total




def state_fidelity_probability(
    first: np.ndarray | Statevector,
    second: np.ndarray | Statevector,
) -> float:
    value = float(qiskit_state_fidelity(as_statevector(first), as_statevector(second), validate=True))
    return float(np.clip(value, 0.0, 1.0))


def fidelity_matrix(states: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Vectorized pure-state fidelities for previously validated state data.

    Particle and anchor states are validated when Qiskit creates them.  This
    hot path deliberately avoids reconstructing millions of ``Statevector``
    wrappers during posterior evaluation.
    """

    left = np.asarray(states)
    right = np.asarray(references)
    if left.ndim == 1:
        left = left[None, :]
    if right.ndim == 1:
        right = right[None, :]
    if left.shape[1] != right.shape[1]:
        raise ValueError("State and reference Hilbert-space dimensions differ.")
    overlaps = left.conj() @ right.T if np.iscomplexobj(left) else left @ right.T
    return np.clip(np.abs(overlaps) ** 2, 0.0, 1.0)


def fidelities_to_state(states: np.ndarray, reference: np.ndarray | Statevector) -> np.ndarray:
    return fidelity_matrix(states, as_statevector(reference).data)[:, 0]


def binomial_log_likelihood(
    counts: np.ndarray,
    shot_counts: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Boundary-safe binomial log likelihood, omitting constant coefficients."""

    counts = np.asarray(counts)
    shot_counts = np.asarray(shot_counts)
    probabilities = np.asarray(probabilities, dtype=float)
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("Binomial probabilities must lie in [0, 1].")
    return xlogy(counts, probabilities) + xlog1py(
        shot_counts - counts,
        -probabilities,
    )


def compute_uncompute_circuit(
    target_state: np.ndarray | Statevector,
    anchor_state: np.ndarray | Statevector,
    measure: bool = True,
) -> QuantumCircuit:
    """Circuit whose all-zero probability is the target--anchor fidelity."""

    target = as_statevector(target_state)
    anchor = as_statevector(anchor_state)
    if target.num_qubits != anchor.num_qubits:
        raise ValueError("Target and anchor use different numbers of qubits.")
    circuit = QuantumCircuit(target.num_qubits, target.num_qubits if measure else 0)
    qubits = list(range(target.num_qubits))
    circuit.append(StatePreparation(target), qubits)
    circuit.append(StatePreparation(anchor).inverse(), qubits)
    if measure:
        circuit.measure(qubits, qubits)
    return circuit


def aer_fidelity_count(
    target_state: np.ndarray | Statevector,
    anchor_state: np.ndarray | Statevector,
    shots: int,
    seed: int,
) -> int:
    """Sample the compute--uncompute fidelity circuit with Qiskit Aer."""

    if shots <= 0:
        raise ValueError("The number of shots must be positive.")
    circuit = compute_uncompute_circuit(target_state, anchor_state, measure=True)
    simulator = AerSimulator()
    # Force a fully decomposed universal basis.  This mirrors the gate-level
    # circuit needed by hardware and avoids leaving opaque state-preparation
    # or multiplexer instructions for the simulator backend.
    isa_circuit = transpile(
        circuit,
        basis_gates=["u", "cx"],
        optimization_level=1,
        seed_transpiler=seed,
    )
    counts = simulator.run(isa_circuit, shots=shots, seed_simulator=seed).result().get_counts()
    return int(counts.get("0" * circuit.num_qubits, 0))


def version_metadata() -> dict[str, str]:
    return {
        "quantum_backend": QISKIT_BACKEND_NAME,
        "qiskit_version": QISKIT_VERSION,
        "qiskit_aer_version": QISKIT_AER_VERSION,
    }
