"""Literature-aligned round-count comparison for the full-state experiment.

The fixed full-state design is independent of the adaptive round count.  This
runner therefore reuses its published trial errors and varies only the number
of paired-proposal rounds used by S-PAQT, S-SGQT, and S-OSGQT.  For each
dimension, radius, and adaptive method, the reported oracle row selects the T
with the smallest observed mean over the 25--150-shot-per-setting schedules on
the same 30 evaluation truths.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import qiskit_quantum_backend as quantum
import quantum_benchmark_support as base
import quantum_full_tomography_radius_experiment as full_state
import quantum_greedy_spectral_experiment as benchmark


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_DIR = base.ACTIVE_EXPERIMENT_ROOT / "quantum_full_tomography_radius"
OUTPUT_DIR = base.ACTIVE_EXPERIMENT_ROOT / "quantum_full_tomography_round_sensitivity"

DEFAULT_ROUNDS = (512, 768, 1536, 3072)
DEFAULT_METHODS = ("structured_paqt", "structured_sgqt", "structured_osgqt")
METHOD_LABELS = {
    "structured_paqt": "S-PAQT",
    "structured_sgqt": "S-SGQT",
    "structured_osgqt": "S-OSGQT",
}
METHOD_STYLES = {
    "structured_paqt": {"marker": "s", "linestyle": "--"},
    "structured_sgqt": {"marker": "^", "linestyle": "-."},
    "structured_osgqt": {"marker": "D", "linestyle": ":"},
}

TRIAL_FIELDS = (
    "dimension",
    "local_dimension",
    "radius",
    "trial",
    "budget",
    "rounds",
    "method",
    "settings",
    "copies",
    "shots_per_setting_min",
    "shots_per_setting_max",
    "greedy_raw_state_infidelity",
    "competitor_raw_state_infidelity",
    "competitor_minus_greedy",
    "paqt_resampling_count",
    "paqt_minimum_ess",
    "paqt_final_ess",
    "elapsed_seconds",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def optional_float(row: dict[str, str], field: str) -> float:
    value = row.get(field)
    return float(value) if value not in (None, "") else float("nan")


def radius_token(radius: float) -> str:
    return f"{radius:g}".replace(".", "p")


def result_path(output_dir: Path, dimension: int, radius: float) -> Path:
    return output_dir / f"trial_results_d{dimension}_r{radius_token(radius)}.csv"


def append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def published_lookups(
    trials: int,
) -> dict[tuple[int, float, int], float]:
    greedy: dict[tuple[int, float, int], float] = {}
    for row in read_csv(PUBLISHED_DIR / "trial_results.csv"):
        trial = int(row["trial"])
        if trial >= trials:
            continue
        key = (int(row["dimension"]), float(row["radius"]), trial)
        value = float(row["raw_state_infidelity"])
        method = str(row["method"])
        if method == "greedy_full_state":
            greedy[key] = value
    return greedy


def refresh_fixed_columns(output_dir: Path, trials: int) -> int:
    """Refresh copied fixed full-state scores without rerunning competitors."""

    published = published_lookups(trials)
    updated = 0
    for path in sorted(output_dir.glob("trial_results_d*_r*.csv")):
        rows = read_csv(path)
        changed = False
        for row in rows:
            trial = int(row["trial"])
            if trial >= trials:
                continue
            key = (int(row["dimension"]), float(row["radius"]), trial)
            if key not in published:
                raise RuntimeError(f"Missing refreshed full-state row: {key}")
            greedy = published[key]
            competitor = float(row["competitor_raw_state_infidelity"])
            row["greedy_raw_state_infidelity"] = greedy
            row["competitor_minus_greedy"] = competitor - greedy
            updated += 1
            changed = True
        if changed:
            base.write_union_csv(path, rows)
    return updated


def existing_keys(path: Path) -> set[tuple[int, int, str]]:
    if not path.exists():
        return set()
    return {
        (int(row["trial"]), int(row["rounds"]), str(row["method"]))
        for row in read_csv(path)
    }


def run_paqt(
    *,
    rng_seed: int,
    truth_state: np.ndarray,
    model: benchmark.LocalModel,
    radius: float,
    budget: int,
    rounds: int,
    config: full_state.FullTomographyConfig,
) -> tuple[float, int, int, float, float]:
    local_dimension = model.coordinate_map.shape[1]
    probes, counts, shots = base.collect_structured_paqt_measurements(
        rng=np.random.default_rng(rng_seed + 30),
        truth_state=truth_state,
        family=model.family,
        dimension=local_dimension,
        total_copies=budget,
        iterations=rounds,
        radius=radius,
    )
    posterior_rng = np.random.default_rng(rng_seed + 530)
    particle_thetas = base.make_particle_cloud(
        rng=posterior_rng,
        dimension=local_dimension,
        radius=radius,
        count=config.smc_particles,
    )
    posterior = base.run_liu_west_particle_posterior_from_measurements(
        rng=posterior_rng,
        family=model.family,
        particle_thetas=particle_thetas,
        probe_states=probes,
        counts=counts,
        shot_counts=shots,
        radius=radius,
        task_from_density=lambda density: np.asarray(
            [np.vdot(truth_state, density @ truth_state).real]
        ),
    )
    fidelity = float(np.vdot(truth_state, posterior.state @ truth_state).real)
    error = float(np.clip(1.0 - fidelity, 0.0, 1.0))
    return (
        error,
        int(shots.size),
        posterior.resampling_count,
        posterior.minimum_ess,
        posterior.ess,
    )


def run_point_method(
    *,
    method: str,
    rng_seed: int,
    truth_state: np.ndarray,
    model: benchmark.LocalModel,
    radius: float,
    budget: int,
    rounds: int,
) -> tuple[float, int]:
    result = base.run_structured_sgqt(
        rng=np.random.default_rng(rng_seed + 50),
        truth_state=truth_state,
        family=model.family,
        dimension=model.coordinate_map.shape[1],
        total_copies=budget,
        iterations=rounds,
        radius=radius,
        osgqt=method == "structured_osgqt",
    )
    return full_state.raw_state_infidelity(result.state, truth_state), int(result.settings)


def run_cell(
    *,
    dimension: int,
    radius: float,
    rounds_grid: tuple[int, ...],
    methods: tuple[str, ...],
    trials: int,
    output_dir: Path,
) -> None:
    config = full_state.FullTomographyConfig()
    budget = config.budget
    base_config = benchmark.GreedyTaskConfig(
        seed=config.seed,
        anchor_radius=config.anchor_radius,
        budgets=(budget,),
    )
    model = benchmark.build_local_model(base_config, dimension)
    local_dimension = model.coordinate_map.shape[1]
    greedy_lookup = published_lookups(trials)
    path = result_path(output_dir, dimension, radius)
    completed = existing_keys(path)

    requested = {
        (trial, rounds, method)
        for trial in range(trials)
        for rounds in rounds_grid
        for method in methods
    }
    if requested.issubset(completed):
        print(f"d={dimension}, R={radius:g}: all requested rows complete", flush=True)
        return

    for trial in range(trials):
        unit_coordinate = base.sample_ball(
            np.random.default_rng(config.seed + 100_000 * dimension + trial),
            local_dimension,
            1.0,
            1,
        )[0]
        truth_state = quantum.ground_state(model.family, radius * unit_coordinate)
        greedy_key = (dimension, radius, trial)
        if greedy_key not in greedy_lookup:
            raise RuntimeError(f"Missing published Greedy row: {greedy_key}")
        greedy_error = greedy_lookup[greedy_key]
        seed_base = config.seed + 1_000_000 * dimension + 1000 * trial + budget

        for rounds in rounds_grid:
            shot_schedule = base.paqt_sgqt_shot_schedule(budget, rounds)
            for method in methods:
                key = (trial, rounds, method)
                if key in completed:
                    continue
                started = time.perf_counter()
                resampling_count = float("nan")
                minimum_ess = float("nan")
                final_ess = float("nan")
                if method == "structured_paqt":
                    (
                        competitor_error,
                        settings,
                        resampling_count,
                        minimum_ess,
                        final_ess,
                    ) = run_paqt(
                        rng_seed=seed_base,
                        truth_state=truth_state,
                        model=model,
                        radius=radius,
                        budget=budget,
                        rounds=rounds,
                        config=config,
                    )
                else:
                    competitor_error, settings = run_point_method(
                        method=method,
                        rng_seed=seed_base,
                        truth_state=truth_state,
                        model=model,
                        radius=radius,
                        budget=budget,
                        rounds=rounds,
                    )
                elapsed = time.perf_counter() - started
                append_row(
                    path,
                    {
                        "dimension": dimension,
                        "local_dimension": local_dimension,
                        "radius": radius,
                        "trial": trial,
                        "budget": budget,
                        "rounds": rounds,
                        "method": method,
                        "settings": settings,
                        "copies": int(np.sum(shot_schedule)),
                        "shots_per_setting_min": int(np.min(shot_schedule)),
                        "shots_per_setting_max": int(np.max(shot_schedule)),
                        "greedy_raw_state_infidelity": greedy_error,
                        "competitor_raw_state_infidelity": competitor_error,
                        "competitor_minus_greedy": competitor_error - greedy_error,
                        "paqt_resampling_count": resampling_count,
                        "paqt_minimum_ess": minimum_ess,
                        "paqt_final_ess": final_ess,
                        "elapsed_seconds": elapsed,
                    },
                )
                completed.add(key)
                print(
                    f"d={dimension} R={radius:g} trial={trial + 1}/{trials} "
                    f"T={rounds} {METHOD_LABELS[method]}: "
                    f"competitor={competitor_error:.4e}, greedy={greedy_error:.4e}, "
                    f"seconds={elapsed:.1f}",
                    flush=True,
                )


def load_all_rows(output_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(output_dir.glob("trial_results_d*_r*.csv")):
        rows.extend(read_csv(path))
    return rows


def summarize(
    output_dir: Path,
    rounds_grid: tuple[int, ...],
) -> list[dict[str, object]]:
    rows = load_all_rows(output_dir)
    grouped: dict[tuple[int, float, int, str], list[dict[str, str]]] = {}
    for row in rows:
        rounds = int(row["rounds"])
        if rounds not in rounds_grid:
            continue
        key = (
            int(row["dimension"]),
            float(row["radius"]),
            rounds,
            str(row["method"]),
        )
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, object]] = []
    for (dimension, radius, rounds, method), group in sorted(grouped.items()):
        competitor = np.asarray(
            [float(row["competitor_raw_state_infidelity"]) for row in group]
        )
        greedy = np.asarray(
            [float(row["greedy_raw_state_infidelity"]) for row in group]
        )
        differences = competitor - greedy
        count = differences.size
        se_competitor = (
            float(np.std(competitor, ddof=1) / math.sqrt(count))
            if count > 1
            else float("nan")
        )
        se_difference = (
            float(np.std(differences, ddof=1) / math.sqrt(count))
            if count > 1
            else float("nan")
        )
        critical = float(stats.t.ppf(0.975, count - 1)) if count > 1 else float("nan")
        mean_difference = float(np.mean(differences))
        summary.append(
            {
                "dimension": dimension,
                "radius": radius,
                "budget": int(group[0]["budget"]),
                "rounds": rounds,
                "method": method,
                "settings": int(group[0]["settings"]),
                "shots_per_setting_min": int(group[0]["shots_per_setting_min"]),
                "shots_per_setting_max": int(group[0]["shots_per_setting_max"]),
                "n_trials": count,
                "mean_greedy_infidelity": float(np.mean(greedy)),
                "mean_competitor_infidelity": float(np.mean(competitor)),
                "se_competitor_infidelity": se_competitor,
                "competitor_to_greedy_mean_ratio": float(
                    np.mean(competitor) / np.mean(greedy)
                ),
                "mean_competitor_minus_greedy": mean_difference,
                "paired_difference_ci95_low": mean_difference - critical * se_difference,
                "paired_difference_ci95_high": mean_difference + critical * se_difference,
                "greedy_win_fraction": float(np.mean(greedy < competitor)),
                "mean_paqt_resampling_count": float(
                    np.mean(
                        [optional_float(row, "paqt_resampling_count") for row in group]
                    )
                )
                if method == "structured_paqt"
                else float("nan"),
                "mean_paqt_minimum_ess": float(
                    np.mean([optional_float(row, "paqt_minimum_ess") for row in group])
                )
                if method == "structured_paqt"
                else float("nan"),
                "mean_paqt_final_ess": float(
                    np.mean([optional_float(row, "paqt_final_ess") for row in group])
                )
                if method == "structured_paqt"
                else float("nan"),
            }
        )
    if summary:
        base.write_union_csv(output_dir / "summary_rows.csv", summary)
        write_oracle_outputs(output_dir, summary)
    return summary


def write_oracle_outputs(
    output_dir: Path,
    summary: list[dict[str, object]],
) -> None:
    grouped: dict[tuple[int, float, str], list[dict[str, object]]] = {}
    for row in summary:
        key = (int(row["dimension"]), float(row["radius"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)

    oracle_rows: list[dict[str, object]] = []
    for (dimension, radius, method), group in sorted(grouped.items()):
        best = min(group, key=lambda row: float(row["mean_competitor_infidelity"]))
        oracle_rows.append(
            {
                **best,
                "tested_round_counts": ",".join(
                    str(int(row["rounds"]))
                    for row in sorted(group, key=lambda row: int(row["rounds"]))
                ),
                "n_tested_round_counts": len(group),
            }
        )
    base.write_union_csv(output_dir / "oracle_method_rows.csv", oracle_rows)

    lines = [
        "# Full-State Adaptive Round-Count Oracle",
        "",
        "Each adaptive method receives its post hoc best observed T separately",
        "for every dimension and radius. Positive differences favor Greedy.",
        "Intervals are paired Student-t intervals on the shared truths and are",
        "descriptive rather than adjusted for post-selection.",
        "",
        "| d | R | Method | T | Settings | Shots/setting | Greedy mean | Competitor mean | Competitor - Greedy [95% CI] |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in oracle_rows:
        shots = str(row["shots_per_setting_min"])
        if row["shots_per_setting_min"] != row["shots_per_setting_max"]:
            shots += f"-{row['shots_per_setting_max']}"
        lines.append(
            "| {d} | {radius:g} | {method} | {rounds} | {settings} | {shots} | "
            "{greedy:.4e} | {competitor:.4e} | {difference:.4e} "
            "[{low:.4e}, {high:.4e}] |".format(
                d=int(row["dimension"]),
                radius=float(row["radius"]),
                method=METHOD_LABELS[str(row["method"])],
                rounds=int(row["rounds"]),
                settings=int(row["settings"]),
                shots=shots,
                greedy=float(row["mean_greedy_infidelity"]),
                competitor=float(row["mean_competitor_infidelity"]),
                difference=float(row["mean_competitor_minus_greedy"]),
                low=float(row["paired_difference_ci95_low"]),
                high=float(row["paired_difference_ci95_high"]),
            )
        )
    (output_dir / "oracle_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_oracle_plot(
        output_dir / "full_state_round_oracle_benchmark.png",
        oracle_rows,
    )


def save_oracle_plot(
    path: Path,
    oracle_rows: list[dict[str, object]],
) -> None:
    """Plot the fixed full-state design against each method's best observed T."""

    fixed_lookup = {
        (int(row["dimension"]), float(row["radius"])): row
        for row in read_csv(PUBLISHED_DIR / "summary_rows.csv")
        if row["method"] == "greedy_full_state"
    }
    dimensions = sorted({int(row["dimension"]) for row in oracle_rows})
    radii = sorted({float(row["radius"]) for row in oracle_rows})
    fig, axes = plt.subplots(1, len(dimensions), figsize=(9.2, 3.8), squeeze=False)
    for axis, dimension in zip(axes[0], dimensions):
        fixed_rows = [fixed_lookup[(dimension, radius)] for radius in radii]
        axis.errorbar(
            radii,
            [float(row["mean_raw_state_infidelity"]) for row in fixed_rows],
            yerr=[float(row["se_raw_state_infidelity"]) for row in fixed_rows],
            marker="o",
            linestyle="-",
            markersize=5.0,
            linewidth=1.7,
            label="Greedy spectral",
        )
        for method in DEFAULT_METHODS:
            method_rows = sorted(
                [
                    row
                    for row in oracle_rows
                    if int(row["dimension"]) == dimension
                    and str(row["method"]) == method
                ],
                key=lambda row: float(row["radius"]),
            )
            axis.errorbar(
                [float(row["radius"]) for row in method_rows],
                [float(row["mean_competitor_infidelity"]) for row in method_rows],
                yerr=[float(row["se_competitor_infidelity"]) for row in method_rows],
                **METHOD_STYLES[method],
                markersize=5.0,
                linewidth=1.7,
                label=f"{METHOD_LABELS[method]} (best $T$)",
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(radii, [f"{radius:g}" for radius in radii])
        axis.grid(alpha=0.25)
        axis.set_title(f"original $d={dimension}$")
        axis.set_xlabel("localization radius $R$")
        axis.set_ylabel("mean raw full-state infidelity")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0.0, 0.18, 1.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def validate_complete(
    output_dir: Path,
    dimensions: tuple[int, ...],
    radii: tuple[float, ...],
    rounds_grid: tuple[int, ...],
    methods: tuple[str, ...],
    trials: int,
) -> None:
    rows = load_all_rows(output_dir)
    keys = {
        (
            int(row["dimension"]),
            float(row["radius"]),
            int(row["trial"]),
            int(row["rounds"]),
            str(row["method"]),
        )
        for row in rows
    }
    expected = {
        (dimension, radius, trial, rounds, method)
        for dimension in dimensions
        for radius in radii
        for trial in range(trials)
        for rounds in rounds_grid
        for method in methods
    }
    missing = expected - keys
    if missing:
        raise RuntimeError(f"Full-state T sweep has {len(missing)} missing rows.")
    selected = [
        row
        for row in rows
        if int(row["dimension"]) in dimensions
        and float(row["radius"]) in radii
        and int(row["rounds"]) in rounds_grid
        and str(row["method"]) in methods
        and int(row["trial"]) < trials
    ]
    if any(int(row["copies"]) != full_state.FullTomographyConfig().budget for row in selected):
        raise RuntimeError("At least one row does not spend the exact copy budget.")
    if any(int(row["settings"]) != 2 * int(row["rounds"]) for row in selected):
        raise RuntimeError("At least one adaptive row does not use 2T settings.")
    print(f"Validated {len(expected)} complete adaptive rows.", flush=True)


def main() -> None:
    config = full_state.FullTomographyConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", nargs="+", type=int, default=config.dimensions)
    parser.add_argument("--radii", nargs="+", type=float, default=config.radii)
    parser.add_argument("--rounds", nargs="+", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--methods", nargs="+", choices=DEFAULT_METHODS, default=DEFAULT_METHODS)
    parser.add_argument("--trials", type=int, default=config.trials)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--refresh-fixed",
        action="store_true",
        help="Refresh copied fixed scores without rerunning adaptive methods.",
    )
    parser.add_argument("--validate-complete", action="store_true")
    args = parser.parse_args()

    dimensions = tuple(args.dimensions)
    radii = tuple(args.radii)
    rounds_grid = tuple(args.rounds)
    methods = tuple(args.methods)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "dimensions": dimensions,
                "radii": radii,
                "budget": config.budget,
                "rounds": rounds_grid,
                "methods": methods,
                "trials": args.trials,
                "paqt_start_at_pilot": True,
                "seed": config.seed,
                "smc_particles": config.smc_particles,
                "paqt_resampler": "Liu-West",
                "paqt_liu_west_a": base.PAQT_LIU_WEST_A,
                "paqt_resample_ess_fraction": (
                    base.PAQT_RESAMPLE_ESS_FRACTION
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.refresh_fixed:
        count = refresh_fixed_columns(args.output_dir, args.trials)
        print(f"Refreshed {count} copied full-state scores.", flush=True)
    if not args.aggregate_only and not args.refresh_fixed:
        for dimension in dimensions:
            for radius in radii:
                run_cell(
                    dimension=dimension,
                    radius=radius,
                    rounds_grid=rounds_grid,
                    methods=methods,
                    trials=args.trials,
                    output_dir=args.output_dir,
                )
    summary = summarize(args.output_dir, rounds_grid)
    if args.validate_complete:
        validate_complete(
            args.output_dir,
            dimensions,
            radii,
            rounds_grid,
            methods,
            args.trials,
        )
    print(f"Wrote {len(summary)} summary rows to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
