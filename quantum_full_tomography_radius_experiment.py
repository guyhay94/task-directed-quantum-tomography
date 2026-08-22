"""Fixed-design baseline for the full-state radius sweep.

Greedy is designed for the full local state rather than for the Schmidt-spectrum
task: in the whitened tangent coordinates its local Fubini--Study loss matrix is
the identity.  The adaptive methods are evaluated separately on the retained
four-schedule round grid.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path

import numpy as np

import qiskit_quantum_backend as quantum
import quantum_benchmark_support as base
import quantum_greedy_spectral_experiment as benchmark


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = base.ACTIVE_EXPERIMENT_ROOT / "quantum_full_tomography_radius"

METHODS = (
    "greedy_full_state",
)

METHOD_LABELS = {
    "greedy_full_state": "Greedy full-state",
}


@dataclass(frozen=True)
class FullTomographyConfig:
    seed: int = 20260731
    dimensions: tuple[int, ...] = (6, 17)
    radii: tuple[float, ...] = (0.01, 0.02, 0.03, 0.04, 0.08, 0.16)
    budget: int = 153600
    trials: int = 30
    anchor_radius: float = math.pi / 4.0
    smc_particles: int = 500
    smc_ess_fraction: float = 0.55
    smc_mutation_steps: int = 3
    smc_max_temperatures: int = 60


def raw_state_infidelity(estimate: np.ndarray, truth: np.ndarray) -> float:
    fidelity = quantum.state_fidelity_probability(estimate, truth)
    return float(np.clip(1.0 - fidelity, 0.0, 1.0))


def full_state_geometry(model: benchmark.LocalModel) -> benchmark.TaskGeometry:
    """Return the local pure-state loss in whitened tangent coordinates."""

    dimension = model.coordinate_map.shape[1]
    identity = np.eye(dimension, dtype=float)
    return benchmark.TaskGeometry(
        task_scale=np.ones(dimension, dtype=float),
        task_jacobian=identity,
        task_metric=identity,
        generalized_eigenvalues=np.ones(dimension, dtype=float),
        generalized_eigenvectors=identity,
        task_rank=dimension,
    )


def method_row(
    *,
    dimension: int,
    local_dimension: int,
    radius: float,
    budget: int,
    trial: int,
    method: str,
    settings: int,
    truth_state: np.ndarray,
    estimate_state: np.ndarray | None = None,
    posterior_mean_infidelity: float | None = None,
) -> dict:
    if (estimate_state is None) == (posterior_mean_infidelity is None):
        raise ValueError(
            "Supply exactly one of estimate_state and posterior_mean_infidelity."
        )
    error = (
        raw_state_infidelity(estimate_state, truth_state)
        if estimate_state is not None
        else float(posterior_mean_infidelity)
    )
    if not np.isfinite(error):
        raise RuntimeError("A tomography endpoint produced nonfinite infidelity.")
    return {
        "dimension": dimension,
        "local_dimension": local_dimension,
        "radius": radius,
        "budget": budget,
        "trial": trial,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "settings": settings,
        "raw_state_infidelity": error,
        "raw_state_fidelity": 1.0 - error,
    }


def summarize(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    keys = sorted(
        {
            (
                int(row["dimension"]),
                float(row["radius"]),
                int(row["budget"]),
                str(row["method"]),
            )
            for row in rows
        }
    )
    for dimension, radius, budget, method in keys:
        subset = [
            row
            for row in rows
            if int(row["dimension"]) == dimension
            and math.isclose(float(row["radius"]), radius)
            and int(row["budget"]) == budget
            and str(row["method"]) == method
        ]
        values = np.asarray(
            [float(row["raw_state_infidelity"]) for row in subset],
            dtype=float,
        )
        output.append(
            {
                "dimension": dimension,
                "local_dimension": int(subset[0]["local_dimension"]),
                "radius": radius,
                "budget": budget,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "mean_raw_state_infidelity": float(np.mean(values)),
                "se_raw_state_infidelity": float(
                    np.std(values, ddof=1) / math.sqrt(values.size)
                ),
                "median_raw_state_infidelity": float(np.median(values)),
                "minimum_raw_state_infidelity": float(np.min(values)),
                "maximum_raw_state_infidelity": float(np.max(values)),
                "trials": len(subset),
            }
        )
    return output


def row_key(row: dict) -> tuple:
    return (
        int(row["dimension"]),
        float(row["radius"]),
        int(row["budget"]),
        int(row["trial"]),
        str(row["method"]),
    )


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def expected_keys(config: FullTomographyConfig) -> set[tuple]:
    return {
        (dimension, radius, config.budget, trial, method)
        for dimension in config.dimensions
        for radius in config.radii
        for trial in range(config.trials)
        for method in METHODS
    }


def write_outputs(
    output_dir: Path,
    rows: list[dict],
    config: FullTomographyConfig,
) -> None:
    rows = sorted(rows, key=row_key)
    summary_rows = summarize(rows)
    base.write_union_csv(output_dir / "trial_results.csv", rows)
    base.write_union_csv(output_dir / "summary_rows.csv", summary_rows)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Structured full-state tomography radius sweep",
        "",
        "Every endpoint is raw state infidelity against the pure truth, "
        "`1 - <psi_truth|rho_hat|psi_truth>`, without normalization. For the "
        "point estimates, `rho_hat = |psi_hat><psi_hat|`.",
        "Greedy uses the identity Fubini--Study loss matrix in whitened local "
        "coordinates. Adaptive competitors are stored by the separate "
        "full-state round-sensitivity experiment.",
        "",
        f"- dimensions: `{list(config.dimensions)}`",
        f"- radii: `{list(config.radii)}`",
        f"- budget: `{config.budget}`",
        f"- trials per cell: `{config.trials}`",
        f"- method rows: `{len(rows)}`",
        "",
        "| d | R | Greedy full-state |",
        "|---:|---:|---:|",
    ]
    for dimension in config.dimensions:
        for radius in config.radii:
            cell = {
                str(row["method"]): float(row["mean_raw_state_infidelity"])
                for row in summary_rows
                if int(row["dimension"]) == dimension
                and math.isclose(float(row["radius"]), radius)
            }
            if set(cell) != set(METHODS):
                continue
            lines.append(
                f"| {dimension} | {radius:g} | "
                f"{cell['greedy_full_state']:.3e} |"
            )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(
    config: FullTomographyConfig,
    output_dir: Path,
    *,
    resume: bool,
    selected_dimensions: tuple[int, ...] | None = None,
    selected_radii: tuple[float, ...] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_path = output_dir / "trial_results.csv"
    rows = load_rows(trial_path) if resume else []
    existing = {row_key(row) for row in rows}
    base_config = benchmark.GreedyTaskConfig(
        seed=config.seed,
        anchor_radius=config.anchor_radius,
        budgets=(config.budget,),
    )
    dimensions = selected_dimensions or config.dimensions
    radii = selected_radii or config.radii

    for dimension in dimensions:
        model = benchmark.build_local_model(base_config, dimension)
        local_dimension = model.coordinate_map.shape[1]
        geometry = full_state_geometry(model)
        unit_coordinates = [
            base.sample_ball(
                np.random.default_rng(config.seed + 100_000 * dimension + trial),
                local_dimension,
                1.0,
                1,
            )[0]
            for trial in range(config.trials)
        ]
        for radius in config.radii:
            if not any(math.isclose(radius, selected) for selected in radii):
                continue
            radius_config = replace(
                base_config,
                particle_radius=radius,
                truth_radius=radius,
                budgets=(config.budget,),
            )
            # With identity full-state loss and an isotropic ball prior, every
            # whitened tangent direction has the same value.  The full-rank
            # Greedy optimum is therefore the orthogonal basis with balanced
            # copy counts.  Calling the task-specific design bundle here would
            # incorrectly require a nonempty task nullspace for an ablation.
            directions = geometry.generalized_eigenvectors
            anchor_states = benchmark.projective_geodesic_anchors(
                model,
                directions,
                radius_config.anchor_radius,
            )
            shot_counts = benchmark.largest_remainder_allocation(
                config.budget,
                np.ones(local_dimension, dtype=float),
            )
            if int(np.sum(shot_counts)) != config.budget:
                raise RuntimeError("Greedy tomography allocation lost copies.")
            print(f"d={dimension}, R={radius:.3g}: design ready", flush=True)

            for trial, unit_coordinate in enumerate(unit_coordinates):
                required = {
                    (dimension, radius, config.budget, trial, method)
                    for method in METHODS
                }
                if required.issubset(existing):
                    continue
                missing_methods = {
                    method
                    for method in METHODS
                    if (dimension, radius, config.budget, trial, method) not in existing
                }
                truth_coordinate = radius * unit_coordinate
                truth_state = quantum.ground_state(model.family, truth_coordinate)
                seed_base = (
                    config.seed
                    + 1_000_000 * dimension
                    + 1000 * trial
                    + config.budget
                )
                trial_rows: list[dict] = []
                if "greedy_full_state" in missing_methods:
                    greedy, _ = benchmark.run_local_gaussian_estimator(
                        rng=np.random.default_rng(seed_base + 10),
                        truth_state=truth_state,
                        model=model,
                        config=radius_config,
                        anchor_states=anchor_states,
                        shot_counts=shot_counts,
                        directions=directions,
                        task_metric=geometry.task_metric,
                    )
                    trial_rows.append(
                        method_row(
                            dimension=dimension,
                            local_dimension=local_dimension,
                            radius=radius,
                            budget=config.budget,
                            trial=trial,
                            method="greedy_full_state",
                            settings=greedy.settings,
                            estimate_state=greedy.state,
                            truth_state=truth_state,
                        )
                    )

                for row in trial_rows:
                    key = row_key(row)
                    if key not in existing:
                        rows.append(row)
                        existing.add(key)

            write_outputs(output_dir, rows, config)
            print(
                f"d={dimension}, R={radius:.3g}: {config.trials} trials complete",
                flush=True,
            )


def merge_shards(config: FullTomographyConfig, output_dir: Path) -> None:
    candidates = sorted((output_dir / "shards").glob("*/trial_results.csv"))
    main_path = output_dir / "trial_results.csv"
    if main_path.exists():
        candidates.insert(0, main_path)
    if not candidates:
        raise FileNotFoundError("No tomography shard trial files were found.")
    merged: dict[tuple, dict] = {}
    for path in candidates:
        for row in load_rows(path):
            key = row_key(row)
            if key in merged:
                previous = float(merged[key]["raw_state_infidelity"])
                current = float(row["raw_state_infidelity"])
                if not math.isclose(previous, current, rel_tol=1e-12, abs_tol=1e-15):
                    raise RuntimeError(f"Conflicting duplicate tomography row: {key}")
                continue
            merged[key] = row
    expected = expected_keys(config)
    actual = set(merged)
    if actual != expected:
        raise RuntimeError(
            f"Merged tomography grid mismatch: "
            f"{len(expected - actual)} missing, {len(actual - expected)} extra."
        )
    write_outputs(output_dir, list(merged.values()), config)
    print(
        f"Merged {len(candidates)} files into {len(merged)} validated rows.",
        flush=True,
    )


def config_for_mode(mode: str) -> FullTomographyConfig:
    if mode == "full":
        return FullTomographyConfig()
    if mode == "smoke":
        return FullTomographyConfig(
            dimensions=(6,),
            radii=(0.01, 0.04),
            budget=19200,
            trials=2,
            smc_particles=120,
            smc_mutation_steps=1,
            smc_max_temperatures=30,
        )
    raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dimension", action="append", type=int)
    parser.add_argument("--radius", action="append", type=float)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--merge-shards", action="store_true")
    args = parser.parse_args()
    config = config_for_mode(args.mode)
    output_dir = args.output_dir or (
        OUTPUT_DIR if args.mode == "full" else OUTPUT_DIR / "smoke"
    )
    if args.merge_shards:
        merge_shards(config, output_dir)
        return
    run(
        config,
        output_dir,
        resume=args.resume,
        selected_dimensions=tuple(args.dimension) if args.dimension else None,
        selected_radii=tuple(args.radius) if args.radius else None,
    )


if __name__ == "__main__":
    main()
