"""Cross-check the regenerated Schmidt-spectrum benchmark against the paper."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from pathlib import Path

import quantum_benchmark_support as base


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = base.ACTIVE_EXPERIMENT_ROOT / "quantum_greedy_spectral"
ROUND_SENSITIVITY_DIR = base.ACTIVE_EXPERIMENT_ROOT / "quantum_adaptive_round_sensitivity"
RADIUS_EXPERIMENT_DIR = base.ACTIVE_EXPERIMENT_ROOT / "quantum_radius_sensitivity"
TASK_RADIUS_ROUND_DIR = (
    base.ACTIVE_EXPERIMENT_ROOT / "quantum_task_radius_round_sensitivity"
)
TRUTH_PRIOR_FIXED_DIR = (
    base.ACTIVE_EXPERIMENT_ROOT / "quantum_truth_prior_decoupling"
)
TRUTH_PRIOR_ROUND_DIR = (
    base.ACTIVE_EXPERIMENT_ROOT / "quantum_truth_prior_round_sensitivity"
)
FULL_TOMOGRAPHY_DIR = base.ACTIVE_EXPERIMENT_ROOT / "quantum_full_tomography_radius"
FULL_TOMOGRAPHY_ROUND_DIR = (
    base.ACTIVE_EXPERIMENT_ROOT / "quantum_full_tomography_round_sensitivity"
)
FULL_TOMOGRAPHY_TRUTH_PRIOR_DIR = (
    base.ACTIVE_EXPERIMENT_ROOT
    / "quantum_full_tomography_truth_prior_sensitivity"
)
PAPER_PATH = REPO_ROOT / "paper_for_saim" / "sisc-article.tex"
PAPER_APPENDIX_PATH = PAPER_PATH
PAPER_FIGURE_DIR = REPO_ROOT / "paper_for_saim" / "figures"
REPORT_PATH = base.ACTIVE_EXPERIMENT_ROOT / "final_validation_report.json"

DIMENSIONS = (6, 12, 17)
LOCAL_DIMENSIONS = {6: 6, 12: 12, 17: 15}
TASK_RANKS = {6: 5, 12: 6, 17: 7}
BUDGETS = (19200, 38400, 76800, 153600)
RADII = (0.01, 0.02, 0.03, 0.04, 0.08, 0.16)
TASK_ROUND_GRIDS = {
    19200: (64, 96, 192, 384),
    38400: (128, 192, 384, 768),
    76800: (256, 384, 768, 1536),
    153600: (512, 768, 1536, 3072),
}
TASK_ROUND_METHODS = (
    "structured_paqt",
    "structured_sgqt",
    "structured_osgqt",
)
EXPECTED_ALLOCATIONS = {
    6: {
        19200: (19200, 0, 0, 0, 0),
        38400: (36766, 1634, 0, 0, 0),
        76800: (69902, 6898, 0, 0, 0),
        153600: (136175, 17425, 0, 0, 0),
    },
    12: {
        19200: (19200, 0, 0, 0, 0, 0),
        38400: (37528, 872, 0, 0, 0, 0),
        76800: (69318, 7482, 0, 0, 0, 0),
        153600: (132898, 20702, 0, 0, 0, 0),
    },
    17: {
        19200: (17729, 1471, 0, 0, 0, 0, 0),
        38400: (31187, 7213, 0, 0, 0, 0, 0),
        76800: (58104, 18696, 0, 0, 0, 0, 0),
        153600: (111938, 41662, 0, 0, 0, 0, 0),
    },
}
ANCHOR_METHODS = (
    "greedy_spectral",
    "equal_spectral",
    "lambda_spectral",
    "coordinate_equal",
    "random_equal",
    "nuisance_equal",
)
def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_liu_west_config(config: dict, label: str) -> None:
    require(config.get("paqt_resampler") == "Liu-West", f"{label} does not use Liu-West resampling.")
    require_close(
        float(config.get("paqt_liu_west_a", float("nan"))),
        base.PAQT_LIU_WEST_A,
        f"{label} has the wrong Liu-West shrinkage",
    )
    require_close(
        float(config.get("paqt_resample_ess_fraction", float("nan"))),
        base.PAQT_RESAMPLE_ESS_FRACTION,
        f"{label} has the wrong resampling ESS threshold",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_csv_glob(directory: Path, pattern: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(directory.glob(pattern)):
        rows.extend(read_csv(path))
    return rows


def mean(values: list[float]) -> float:
    require(bool(values), "Cannot average an empty result cell.")
    return math.fsum(values) / len(values)


def standard_error(values: list[float]) -> float:
    require(len(values) > 1, "Cannot compute a sample standard error from one value.")
    center = mean(values)
    variance = math.fsum((value - center) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def require_close(
    actual: float,
    expected: float,
    message: str,
    *,
    rel_tol: float = 1e-12,
    abs_tol: float = 1e-15,
) -> None:
    require(
        math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol),
        f"{message}: expected {expected:.17g}, found {actual:.17g}.",
    )


def compact(text: str) -> str:
    return " ".join(text.split())


def compact_labeled_table(source: str, label: str) -> str:
    label_index = source.index(rf"\label{{{label}}}")
    table_start = source.rfind(r"\begin{table}", 0, label_index)
    table_end = source.index(r"\end{table}", label_index) + len(r"\end{table}")
    return compact(source[table_start:table_end])


def manuscript_latex() -> str:
    """Return the active single-file SISC manuscript as searchable source."""

    return compact(PAPER_PATH.read_text(encoding="utf-8"))


def check_experiment_configs() -> dict[str, object]:
    task = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
    expected_task = {
        "seed": 20260811,
        "n_qubits": 6,
        "transverse_field": 0.97,
        "fixed_disorder_strength": 0.08,
        "perturbation_order": "x_z_zz",
        "original_dimensions": list(DIMENSIONS),
        "budgets": list(BUDGETS),
        "n_trials": 30,
        "n_design_trials": 300,
        "n_particles": 8000,
        "n_posterior_particles": 20000,
        "particle_radius": 0.02,
        "truth_radius": 0.02,
        "finite_difference_step": 0.01,
        "metric_relative_tolerance": 1e-8,
        "task_relative_tolerance": 1e-9,
        "task_subsystem": [0, 2, 4],
        "mode": "full",
    }
    for key, expected in expected_task.items():
        require(task.get(key) == expected, f"Task configuration field {key} is stale.")
    require_close(
        float(task["anchor_radius"]),
        math.pi / 4.0,
        "Task probe radius is stale",
    )
    require(
        set(task) == {*expected_task, "anchor_radius"},
        "Task baseline configuration contains stale fields.",
    )

    task_round = json.loads(
        (ROUND_SENSITIVITY_DIR / "config.json").read_text(encoding="utf-8")
    )
    require(task_round.get("dimensions") == list(DIMENSIONS), "Wrong task round-sweep dimensions.")
    require(task_round.get("methods") == list(TASK_ROUND_METHODS), "Wrong task round-sweep methods.")
    require(task_round.get("trials") == 30, "Wrong task round-sweep trial count.")
    require(task_round.get("particles") == 8000, "Wrong task round-sweep particle count.")
    require(task_round.get("seed") == 20260811, "Wrong task round-sweep seed.")
    configured_grids = {
        int(budget): tuple(int(value) for value in rounds)
        for budget, rounds in task_round.get("budget_rounds", {}).items()
    }
    require(configured_grids == TASK_ROUND_GRIDS, "Wrong budget-scaled task round grid.")
    require(
        set(task_round)
        == {
            "dimensions",
            "budget_rounds",
            "methods",
            "trials",
            "particles",
            "paqt_start_at_pilot",
            "paqt_resampler",
            "paqt_liu_west_a",
            "paqt_resample_ess_fraction",
            "seed",
            "fixed_dir",
            "benchmark_config",
        },
        "Task round configuration contains stale fields.",
    )
    require_liu_west_config(task_round, "Primary task round sweep")

    radius = json.loads(
        (RADIUS_EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8")
    )
    expected_radius = {
        "seed": 20260731,
        "dimensions": [6, 17],
        "radii": list(RADII),
        "budgets": [153600],
        "joint_failure_probabilities": [0.05],
        "trials": 30,
        "scale_particles": 3000,
        "smc_particles": 500,
        "smc_ess_fraction": 0.55,
        "smc_mutation_steps": 3,
        "smc_max_temperatures": 60,
    }
    for key, expected in expected_radius.items():
        require(radius.get(key) == expected, f"Radius-sweep configuration field {key} is stale.")
    require_close(
        float(radius["anchor_radius"]),
        math.pi / 4.0,
        "Radius-sweep probe radius is stale",
    )

    full_state = json.loads(
        (FULL_TOMOGRAPHY_DIR / "config.json").read_text(encoding="utf-8")
    )
    expected_full_state = {
        "seed": 20260731,
        "dimensions": [6, 17],
        "radii": list(RADII),
        "budget": 153600,
        "trials": 30,
        "smc_particles": 500,
        "smc_ess_fraction": 0.55,
        "smc_mutation_steps": 3,
        "smc_max_temperatures": 60,
    }
    for key, expected in expected_full_state.items():
        require(full_state.get(key) == expected, f"Full-state configuration field {key} is stale.")
    require_close(
        float(full_state["anchor_radius"]),
        math.pi / 4.0,
        "Full-state probe radius is stale",
    )
    require(
        set(full_state) == {*expected_full_state, "anchor_radius"},
        "Full-state baseline configuration contains stale fields.",
    )
    full_round = json.loads(
        (FULL_TOMOGRAPHY_ROUND_DIR / "config.json").read_text(encoding="utf-8")
    )
    expected_full_round = {
        "dimensions": [6, 17],
        "radii": list(RADII),
        "budget": 153600,
        "rounds": [512, 768, 1536, 3072],
        "methods": list(TASK_ROUND_METHODS),
        "trials": 30,
        "paqt_start_at_pilot": True,
        "seed": 20260731,
        "smc_particles": 500,
        "paqt_resampler": "Liu-West",
        "paqt_liu_west_a": 0.98,
        "paqt_resample_ess_fraction": 0.5,
    }
    for key, expected in expected_full_round.items():
        require(full_round.get(key) == expected, f"Full-state round configuration field {key} is stale.")
    require(
        set(full_round) == set(expected_full_round),
        "Full-state round configuration contains stale fields.",
    )

    return {
        "task_seed": task["seed"],
        "task_round_grids": configured_grids,
        "radius_seed": radius["seed"],
        "full_state_seed": full_state["seed"],
        "full_state_rounds": full_round["rounds"],
    }


def check_task_vector(value: str, label: str) -> None:
    vector = [float(item) for item in ast.literal_eval(value)]
    require(len(vector) == 8, f"{label} does not contain eight Schmidt probabilities.")
    require(abs(sum(vector) - 1.0) < 2e-9, f"{label} is not normalized.")
    require(
        all(vector[index] + 1e-14 >= vector[index + 1] for index in range(7)),
        f"{label} is not ordered.",
    )


def raw_task_squared_error(row: dict[str, str]) -> float:
    saved = row.get("raw_task_squared_error", "").strip()
    if saved:
        return float(saved)
    truth = [float(value) for value in ast.literal_eval(row["truth_task"])]
    estimate = [float(value) for value in ast.literal_eval(row["estimated_task"])]
    return math.fsum((left - right) ** 2 for left, right in zip(estimate, truth))


def check_geometry() -> dict[str, object]:
    rows = read_csv(EXPERIMENT_DIR / "geometry_and_allocations.csv")
    require(len(rows) == 3, "Expected one geometry row per physical dimension.")
    expected_task_spectra = {
        6: (
            0.3531494575549166,
            0.008910485164567902,
            0.00024992138559567963,
            1.192039002167407e-06,
            5.951567137754782e-10,
        ),
        12: (
            0.5971011544863793,
            0.025813813234723773,
            0.00030562392224111744,
            0.0001481994725165974,
            4.5225834125879664e-07,
            2.024894518282758e-08,
        ),
        17: (
            0.7516595035021085,
            0.1367993875792259,
            0.004756297283421314,
            0.0003228582522900206,
            1.719790661902089e-05,
            2.490045575414395e-07,
            3.513213255391655e-08,
        ),
    }
    report: dict[str, object] = {}
    for row in rows:
        dimension = int(row["original_dimension"])
        require(dimension in DIMENSIONS, f"Unexpected physical dimension {dimension}.")
        require(
            int(row["local_dimension"]) == LOCAL_DIMENSIONS[dimension],
            f"Stale local dimension for d={dimension}.",
        )
        require(
            int(row["task_rank"]) == TASK_RANKS[dimension],
            f"Wrong Schmidt task rank for d={dimension}.",
        )
        require(
            row["task_weighting"] == "identity (raw Euclidean spectrum loss)",
            f"Task weighting is not raw Euclidean for d={dimension}.",
        )
        require(
            math.isclose(float(row["anchor_radius"]), math.pi / 4.0, rel_tol=0.0, abs_tol=1e-12),
            f"Stale anchor radius for d={dimension}.",
        )
        require(
            ast.literal_eval(row["task_subsystem"]) == (0, 2, 4),
            "Wrong odd-even subsystem.",
        )
        nominal = [float(value) for value in ast.literal_eval(row["nominal_schmidt_spectrum"])]
        require(len(nominal) == 8 and abs(sum(nominal) - 1.0) < 2e-9, "Invalid nominal spectrum.")
        require(all(nominal[i] > nominal[i + 1] for i in range(7)), "Nominal spectrum is degenerate.")
        eigenvalues = [float(value) for value in ast.literal_eval(row["task_eigenvalues"])]
        require(
            len(eigenvalues) == TASK_RANKS[dimension]
            and all(value > 0.0 for value in eigenvalues),
            "Invalid task spectrum.",
        )
        for value, expected in zip(eigenvalues, expected_task_spectra[dimension]):
            require_close(
                value,
                expected,
                f"Stale task-spectrum table entry for d={dimension}",
            )
        require_close(
            float(row["minimum_nominal_spectral_gap"]),
            0.00015680835184566852,
            f"Stale nominal Schmidt gap for d={dimension}",
        )

        audits = json.loads(row["allocations_by_budget"])
        dimension_report: dict[str, object] = {}
        for budget in BUDGETS:
            audit = audits[str(budget)]
            greedy = [int(value) for value in audit["greedy_allocation"]]
            equal = [int(value) for value in audit["equal_allocation"]]
            static = [int(value) for value in audit["lambda_allocation"]]
            require(
                len(greedy) == len(equal) == len(static) == TASK_RANKS[dimension],
                "Allocation rank is stale.",
            )
            require(sum(greedy) == sum(equal) == sum(static) == budget, "Saved allocation misses budget.")
            require(
                tuple(greedy) == EXPECTED_ALLOCATIONS[dimension][budget],
                f"Stale paper allocation for d={dimension}, B={budget}.",
            )
            require(
                float(audit["greedy_local_risk"]) <= float(audit["equal_local_risk"]) + 1e-12,
                "Greedy predicted risk exceeds equal allocation.",
            )
            require(
                float(audit["greedy_local_risk"]) <= float(audit["lambda_local_risk"]) + 1e-12,
                "Greedy predicted risk exceeds lambda-proportional allocation.",
            )
            dimension_report[str(budget)] = {
                "greedy_allocation": greedy,
                "greedy_local_risk": float(audit["greedy_local_risk"]),
                "equal_local_risk": float(audit["equal_local_risk"]),
            }
        largest = audits[str(BUDGETS[-1])]
        largest_values = [float(value) for value in largest["task_eigenvalues"]]
        largest_allocations = [int(value) for value in largest["greedy_allocation"]]
        alpha = float(largest["alpha"])
        nu = float(largest["nu"])
        risk_terms = [
            value / (alpha + nu * allocation)
            for value, allocation in zip(largest_values, largest_allocations)
        ]
        unmeasured_percent = 100.0 * math.fsum(
            term
            for term, allocation in zip(risk_terms, largest_allocations)
            if allocation == 0
        ) / math.fsum(risk_terms)
        expected_percent = {6: 1.70, 12: 1.01, 17: 5.20}[dimension]
        require(
            round(unmeasured_percent, 2) == expected_percent,
            f"Stale unmeasured-risk percentage for d={dimension}.",
        )
        dimension_report["largest_budget_unmeasured_percent"] = unmeasured_percent
        report[f"d{dimension}"] = dimension_report
    return report


def check_trials() -> dict[str, object]:
    rows = read_csv(EXPERIMENT_DIR / "trial_results.csv")
    require(len(rows) == 21600, "The fixed-design benchmark must contain 21600 trial-method rows.")
    counts = {method: 0 for method in ANCHOR_METHODS}
    keys: set[tuple[int, int, int, str]] = set()
    grouped_raw: dict[tuple[int, int, str], list[float]] = {}
    grouped_local: dict[tuple[int, int, str], list[float]] = {}
    for index, row in enumerate(rows):
        method = row["method"]
        require(method in counts, f"Unknown method {method}.")
        counts[method] += 1
        budget = int(row["budget"])
        require(budget in BUDGETS and int(row["copies"]) == budget, "A row misses its copy budget.")
        key = (int(row["original_dimension"]), budget, int(row["trial"]), method)
        require(key not in keys, f"Duplicate primary trial row: {key}.")
        keys.add(key)
        group_key = (int(row["original_dimension"]), budget, method)
        grouped_raw.setdefault(group_key, []).append(raw_task_squared_error(row))
        local_value = float(row["local_task_squared_error"])
        if math.isfinite(local_value):
            grouped_local.setdefault(group_key, []).append(local_value)
        require(int(row["trial"]) < 300, "Anchor design has an unexpected trial index.")
        require(
            1 <= int(row["settings"]) <= TASK_RANKS[int(row["original_dimension"])],
            "Fixed design setting count is invalid.",
        )
        require(int(row["adaptive_rounds"]) == 0, "Fixed design is marked adaptive.")
        if index % 500 == 0:
            check_task_vector(row["truth_task"], "Truth task")
            check_task_vector(row["estimated_task"], "Estimated task")
    for method in ANCHOR_METHODS:
        require(counts[method] == 3600, f"{method} does not have 3600 rows.")
    summary_lookup = {
        (int(row["original_dimension"]), int(row["budget"]), row["method"]): row
        for row in read_csv(EXPERIMENT_DIR / "summary_rows.csv")
    }
    require(set(summary_lookup) == set(grouped_raw), "Primary summary cells do not match trial cells.")
    for key, values in grouped_raw.items():
        summary = summary_lookup[key]
        require_close(
            float(summary["mean_raw_task_squared_error"]),
            mean(values),
            f"Raw-task summary does not reproduce trials for {key}",
        )
        local_values = grouped_local.get(key, [])
        if local_values:
            require_close(
                float(summary["mean_local_task_squared_error"]),
                mean(local_values),
                f"Local-task summary does not reproduce trials for {key}",
            )
    return {
        "trial_rows": len(rows),
        "method_counts": counts,
    }


def acquisition_cost_ratio(budget: int, rounds: int, fixed_settings: int) -> float:
    # Ito's illustrative VQA latencies are 10 us per shot, 100 ms per
    # circuit switch, and 4 s per communication round.  Normalize by 10 us.
    shot_cost, setting_cost, feedback_cost = 1.0, 10000.0, 400000.0
    fixed = budget * shot_cost + fixed_settings * setting_cost
    adaptive = (
        budget * shot_cost
        + 2 * rounds * setting_cost
        + rounds * feedback_cost
    )
    return adaptive / fixed


def check_task_round_sensitivity() -> dict[str, object]:
    latex = manuscript_latex()
    config = json.loads(
        (ROUND_SENSITIVITY_DIR / "config.json").read_text(encoding="utf-8")
    )
    require(
        config.get("paqt_start_at_pilot") is True,
        "The task round sweep must initialize S-PAQT at the common pilot.",
    )
    rows = read_csv_glob(ROUND_SENSITIVITY_DIR, "trial_results*.csv")
    expected_rows = len(DIMENSIONS) * len(BUDGETS) * 4 * len(TASK_ROUND_METHODS) * 30
    require(
        len(rows) == expected_rows,
        f"The task round sweep must contain exactly the {expected_rows} retained rows.",
    )
    keys: set[tuple[int, int, int, int, str]] = set()
    for row in rows:
        dimension = int(row["original_dimension"])
        budget = int(row["budget"])
        trial = int(row["trial"])
        rounds = int(row["rounds"])
        method = row["method"]
        key = (dimension, budget, trial, rounds, method)
        require(key not in keys, f"Duplicate task round-sweep row: {key}.")
        keys.add(key)
        require(dimension in DIMENSIONS, "Unexpected task round-sweep dimension.")
        require(budget in BUDGETS, "Unexpected task round-sweep budget.")
        require(0 <= trial < 30, "Unexpected task round-sweep trial index.")
        require(method in TASK_ROUND_METHODS, "Unexpected task round-sweep method.")
        require(
            rounds in TASK_ROUND_GRIDS[budget],
            f"Removed task round count T={rounds} remains at B={budget}.",
        )
        require(int(row["copies"]) == budget, "A task round-sweep row misses its copy budget.")
        require(int(row["settings"]) == 2 * rounds, "A task round-sweep row does not use 2T settings.")
    grouped: dict[tuple[int, int, int, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            int(row["original_dimension"]),
            int(row["budget"]),
            int(row["rounds"]),
            row["method"],
        )
        grouped.setdefault(key, []).append(row)
    require(len(grouped) == 144, "The reported task round sweep must contain 144 cells.")
    require(all(len(group) == 30 for group in grouped.values()), "A task round cell is not paired on 30 truths.")

    summary_rows = read_csv(ROUND_SENSITIVITY_DIR / "summary_rows.csv")
    summary_lookup = {
        (int(row["original_dimension"]), int(row["budget"]), int(row["rounds"]), row["method"]): row
        for row in summary_rows
    }


    require(set(summary_lookup) == set(grouped), "Task round summaries do not match the configured trial cells.")
    for key, group in grouped.items():
        competitor = [float(row["competitor_raw_task_squared_error"]) for row in group]
        greedy = [float(row["greedy_raw_task_squared_error"]) for row in group]
        summary = summary_lookup[key]
        require_close(
            float(summary["mean_competitor_mse"]),
            mean(competitor),
            f"Task competitor summary does not reproduce trials for {key}",
        )
        require_close(
            float(summary["mean_greedy_mse"]),
            mean(greedy),
            f"Task Greedy summary does not reproduce trials for {key}",
        )

    greedy_lower = sum(float(row["mean_competitor_minus_greedy"]) > 0.0 for row in summary_lookup.values())
    positive_intervals = sum(float(row["paired_difference_ci95_low"]) > 0.0 for row in summary_lookup.values())
    higher = [row for key, row in summary_lookup.items() if key[0] in (12, 17)]
    higher_greedy_lower = sum(float(row["mean_competitor_minus_greedy"]) > 0.0 for row in higher)
    higher_positive_intervals = sum(float(row["paired_difference_ci95_low"]) > 0.0 for row in higher)
    require((greedy_lower, positive_intervals) == (144, 141), "Stale all-cell task comparison counts.")
    require(
        (higher_greedy_lower, higher_positive_intervals) == (96, 93),
        "Stale higher-dimensional task comparison counts.",
    )

    oracle_rows = read_csv(ROUND_SENSITIVITY_DIR / "oracle_method_rows.csv")
    require(len(oracle_rows) == 36, "Expected 36 method-specific task oracle rows.")
    oracle_lookup = {
        (int(row["original_dimension"]), int(row["budget"]), row["method"]): row
        for row in oracle_rows
    }
    for key, oracle in oracle_lookup.items():
        candidates = [
            row
            for summary_key, row in summary_lookup.items()
            if summary_key[0] == key[0] and summary_key[1] == key[1] and summary_key[3] == key[2]
        ]
        expected = min(candidates, key=lambda row: float(row["mean_competitor_mse"]))
        require(
            int(oracle["rounds"]) == int(expected["rounds"]),
            f"Wrong task oracle round selection for {key}.",
        )
        require_close(
            float(oracle["mean_competitor_mse"]),
            float(expected["mean_competitor_mse"]),
            f"Wrong task oracle mean for {key}",
        )

    geometry_rows = {
        int(row["original_dimension"]): row
        for row in read_csv(EXPERIMENT_DIR / "geometry_and_allocations.csv")
    }
    cost_ratios: list[float] = []
    for row in oracle_rows:
        dimension = int(row["original_dimension"])
        if dimension not in (12, 17):
            continue
        budget = int(row["budget"])
        audit = json.loads(geometry_rows[dimension]["allocations_by_budget"])[str(budget)]
        fixed_settings = sum(int(value) > 0 for value in audit["greedy_allocation"])
        cost_ratios.append(
            acquisition_cost_ratio(budget, int(row["rounds"]), fixed_settings)
        )
    require(round(min(cost_ratios), 2) == 686.20, "Stale minimum task acquisition-cost ratio.")
    require(round(max(cost_ratios), 2) == 7433.14, "Stale maximum task acquisition-cost ratio.")
    require("lower mean in all 144" in latex, "Task all-cell count is missing from the paper.")
    require(
        r"687\)--\(7434" in latex,
        "The upward-rounded task acquisition-cost range is stale in the paper.",
    )

    mean_ess: list[float] = []
    minimum_ess: list[float] = []
    mean_resampling_counts: list[float] = []
    for dimension in DIMENSIONS:
        selected_rounds = int(
            oracle_lookup[(dimension, 153600, "structured_paqt")]["rounds"]
        )
        values = [
            float(row["competitor_ess"])
            for row in rows
            if int(row["original_dimension"]) == dimension
            and int(row["budget"]) == 153600
            and int(row["rounds"]) == selected_rounds
            and row["method"] == "structured_paqt"
        ]
        minimum_values = [
            float(row["competitor_minimum_ess"])
            for row in rows
            if int(row["original_dimension"]) == dimension
            and int(row["budget"]) == 153600
            and int(row["rounds"]) == selected_rounds
            and row["method"] == "structured_paqt"
        ]
        resampling_values = [
            float(row["paqt_resampling_count"])
            for row in rows
            if int(row["original_dimension"]) == dimension
            and int(row["budget"]) == 153600
            and int(row["rounds"]) == selected_rounds
            and row["method"] == "structured_paqt"
        ]
        require(len(values) == 30, f"Missing oracle S-PAQT ESS rows for d={dimension}.")
        mean_ess.append(mean(values))
        minimum_ess.append(min(minimum_values))
        mean_resampling_counts.append(mean(resampling_values))
    require([round(value, 1) for value in mean_ess] == [6064.8, 5882.3, 5968.9], "Stale oracle S-PAQT mean final ESS values.")
    require([round(value, 1) for value in minimum_ess] == [3093.3, 3028.4, 2780.4], "Stale oracle S-PAQT minimum ESS values.")
    require(
        [round(value, 1) for value in mean_resampling_counts] == [9.8, 9.7, 8.7],
        "Stale oracle S-PAQT mean resampling counts.",
    )

    return {
        "stored_rows": len(rows),
        "reported_rows": len(rows),
        "reported_cells": len(grouped),
        "greedy_lower_cells": greedy_lower,
        "positive_intervals": positive_intervals,
        "higher_dimension_greedy_lower_cells": higher_greedy_lower,
        "higher_dimension_positive_intervals": higher_positive_intervals,
        "cost_ratio_range": [min(cost_ratios), max(cost_ratios)],
        "largest_budget_paqt_mean_ess": mean_ess,
        "largest_budget_paqt_minimum_ess": minimum_ess,
        "largest_budget_paqt_mean_resampling_count": mean_resampling_counts,
    }


def check_graph_data_and_assets() -> dict[str, object]:
    """Recompute every plotted point/error bar and verify the paper PNG copies."""

    primary_trials = read_csv(EXPERIMENT_DIR / "trial_results.csv")
    plotted_fixed_methods = set(ANCHOR_METHODS)
    fixed_point_count = 0
    for filename, trial_limit in (
        ("summary_rows.csv", 300),
        ("paired_summary_rows.csv", 30),
    ):
        summaries = read_csv(EXPERIMENT_DIR / filename)
        require(
            len(summaries) == len(DIMENSIONS) * len(BUDGETS) * len(plotted_fixed_methods),
            f"Unexpected plotted fixed-design summary count in {filename}.",
        )
        for row in summaries:
            dimension = int(row["original_dimension"])
            budget = int(row["budget"])
            method = row["method"]
            require(method in plotted_fixed_methods, f"Unplotted method remains in {filename}.")
            values = [
                raw_task_squared_error(trial)
                for trial in primary_trials
                if int(trial["original_dimension"]) == dimension
                and int(trial["budget"]) == budget
                and trial["method"] == method
                and int(trial["trial"]) < trial_limit
            ]
            require(len(values) == trial_limit, f"Incomplete plotted fixed-design cell in {filename}.")
            require_close(
                float(row["mean_raw_task_squared_error"]),
                mean(values),
                f"Graph mean is stale for d={dimension}, B={budget}, {method}",
            )
            require_close(
                float(row["se_raw_task_squared_error"]),
                standard_error(values),
                f"Graph error bar is stale for d={dimension}, B={budget}, {method}",
            )
        if filename == "summary_rows.csv":
            fixed_point_count = len(summaries)

    round_trials = read_csv_glob(ROUND_SENSITIVITY_DIR, "trial_results*.csv")
    oracle_rows = read_csv(ROUND_SENSITIVITY_DIR / "oracle_method_rows.csv")
    require(len(oracle_rows) == 36, "The task graph must contain 36 adaptive oracle points.")
    for row in oracle_rows:
        dimension = int(row["original_dimension"])
        budget = int(row["budget"])
        rounds = int(row["rounds"])
        method = row["method"]
        expected_rounds = TASK_ROUND_GRIDS[budget]
        require(rounds in expected_rounds, f"Graph selects removed T={rounds} at B={budget}.")
        require(
            row["tested_round_counts"] == ",".join(str(value) for value in expected_rounds),
            f"Graph oracle metadata has a stale T grid at B={budget}.",
        )
        selected = [
            trial
            for trial in round_trials
            if int(trial["original_dimension"]) == dimension
            and int(trial["budget"]) == budget
            and int(trial["rounds"]) == rounds
            and trial["method"] == method
        ]
        require(len(selected) == 30, "A plotted adaptive oracle point is not based on 30 trials.")
        competitor = [float(trial["competitor_raw_task_squared_error"]) for trial in selected]
        greedy = [float(trial["greedy_raw_task_squared_error"]) for trial in selected]
        require_close(
            float(row["mean_competitor_mse"]),
            mean(competitor),
            f"Adaptive graph mean is stale for d={dimension}, B={budget}, {method}",
        )
        require_close(
            float(row["se_competitor_mse"]),
            standard_error(competitor),
            f"Adaptive graph error bar is stale for d={dimension}, B={budget}, {method}",
        )
        require_close(
            float(row["mean_greedy_mse"]),
            mean(greedy),
            f"Adaptive graph Greedy mean is stale for d={dimension}, B={budget}",
        )

    asset_pairs = (
        (
            ROUND_SENSITIVITY_DIR / "adaptive_round_oracle_task_benchmark.png",
            PAPER_FIGURE_DIR / "quantum_adaptive_round_oracle_task_benchmark.png",
        ),
    )
    asset_hashes: dict[str, str] = {}
    for generated, paper_copy in asset_pairs:
        require(generated.is_file(), f"Missing generated graph {generated.name}.")
        require(paper_copy.is_file(), f"Missing paper graph {paper_copy.name}.")
        generated_hash = hashlib.sha256(generated.read_bytes()).hexdigest()
        paper_hash = hashlib.sha256(paper_copy.read_bytes()).hexdigest()
        require(generated_hash == paper_hash, f"Paper graph {paper_copy.name} is a stale copy.")
        asset_hashes[paper_copy.name] = paper_hash

    return {
        "fixed_ablation_points": fixed_point_count,
        "paired_greedy_points": len(DIMENSIONS) * len(BUDGETS),
        "adaptive_oracle_points": len(oracle_rows),
        "paper_asset_sha256": asset_hashes,
    }




def latex_raw_token(value: float) -> str:
    """Format the numeric body used by the manuscript's raw-error tables."""

    if abs(value) >= 0.001:
        return f"{value:.3f}"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return rf"{mantissa:.2f}{{\times}}10^{{{exponent}}}"


def latex_scientific_body(value: float, digits: int) -> str:
    """Format a scientific-notation table body with a chosen precision."""

    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return rf"{mantissa:.{digits}f}{{\times}}10^{{{exponent}}}"




def check_summary_and_latex() -> dict[str, object]:
    latex = manuscript_latex()
    paper_source = PAPER_PATH.read_text(encoding="utf-8")
    table_label_index = paper_source.index(r"\label{tab:compact-ablation}")
    table_start = paper_source.rfind(r"\begin{table}", 0, table_label_index)
    table_end = paper_source.index(r"\end{table}", table_label_index) + len(
        r"\end{table}"
    )
    compact_ablation_table = compact(paper_source[table_start:table_end])
    rows = read_csv(EXPERIMENT_DIR / "summary_rows.csv")
    lookup = {
        (int(row["original_dimension"]), int(row["budget"]), row["method"]): row
        for row in rows
    }
    oracle_rows = read_csv(ROUND_SENSITIVITY_DIR / "oracle_method_rows.csv")
    require(len(oracle_rows) == 36, "Expected three best-T competitor rows in each dimension-budget cell.")
    oracle_lookup = {
        (int(row["original_dimension"]), int(row["budget"]), row["method"]): row
        for row in oracle_rows
    }
    task_acquisition_table = compact_labeled_table(
        paper_source, "tab:task-acquisition-summary"
    )
    require(
        "Greedy spectral wins" not in task_acquisition_table,
        "The task acquisition-summary table still contains the redundant win-count column.",
    )
    geometry_rows = {
        int(row["original_dimension"]): row
        for row in read_csv(EXPERIMENT_DIR / "geometry_and_allocations.csv")
    }
    adaptive_method_order = (
        "structured_paqt",
        "structured_sgqt",
        "structured_osgqt",
    )
    reductions: list[float] = []
    table_entries: dict[str, list[str]] = {}
    for dimension in DIMENSIONS:
        entries: list[str] = []
        for budget in BUDGETS:
            first_oracle = oracle_lookup[(dimension, budget, adaptive_method_order[0])]
            greedy_token = latex_raw_token(float(first_oracle["mean_greedy_mse"]))
            require(
                compact(greedy_token) in latex,
                f"Stale Greedy oracle-table entry for d={dimension}, B={budget}.",
            )
            entries.append(greedy_token)
            for method in adaptive_method_order:
                row = oracle_lookup[(dimension, budget, method)]
                mean_token = latex_raw_token(float(row["mean_competitor_mse"]))
                rounds_token = rf"\;({int(row['rounds'])})"
                require(
                    compact(mean_token) in latex and compact(rounds_token) in latex,
                    f"Stale best-T table entry for d={dimension}, B={budget}, {method}.",
                )
                entries.append(f"{mean_token} ({int(row['rounds'])})")
        table_entries[f"d{dimension}"] = entries

        fixed_settings: list[int] = []
        dimension_cost_ratios: list[float] = []
        dimension_oracles = [
            row
            for row in oracle_rows
            if int(row["original_dimension"]) == dimension
        ]
        for budget in BUDGETS:
            audit = json.loads(geometry_rows[dimension]["allocations_by_budget"])[
                str(budget)
            ]
            settings = sum(
                int(value) > 0 for value in audit["greedy_allocation"]
            )
            fixed_settings.append(settings)
            dimension_cost_ratios.extend(
                acquisition_cost_ratio(
                    budget,
                    int(row["rounds"]),
                    settings,
                )
                for row in dimension_oracles
                if int(row["budget"]) == budget
            )
        rounds = [int(row["rounds"]) for row in dimension_oracles]
        def integer_range(values: list[int]) -> str:
            low, high = min(values), max(values)
            return rf"\({low}\)" if low == high else rf"\({low}\)--\({high}\)"

        expected_acquisition_row = compact(
            rf"{dimension} & {integer_range(fixed_settings)} & "
            rf"{integer_range(rounds)} & "
            rf"\({math.ceil(min(dimension_cost_ratios))}\)--"
            rf"\({math.ceil(max(dimension_cost_ratios))}\)"
        )
        require(
            expected_acquisition_row in task_acquisition_table,
            f"Stale task acquisition-summary row for d={dimension}.",
        )

        greedy_row = lookup[(dimension, 153600, "greedy_spectral")]
        equal_row = lookup[(dimension, 153600, "equal_spectral")]
        greedy = float(greedy_row["mean_raw_task_squared_error"])
        equal = float(equal_row["mean_raw_task_squared_error"])
        reductions.append(100.0 * (1.0 - greedy / equal))

    require(
        "12 &" not in compact_ablation_table,
        "The compact ablation table should omit the middle dimension d=12.",
    )
    for label in (
        r"\textbf{Greedy spectral (ours)}",
        "Equal",
        r"\(\lambda\)-proportional",
        "Coordinate",
        "Random",
        "Nuisance",
    ):
        require(
            compact_ablation_table.count(f"& {label} &") == 2,
            f"The compact ablation table does not contain exactly two {label} rows.",
        )
    require(
        compact_ablation_table.count("---") == 6,
        "The compact ablation table should leave Psi_r blank only for the six nonspectral rows.",
    )

    metadata_lookup = {
        int(row["original_dimension"]): row
        for row in read_csv(EXPERIMENT_DIR / "geometry_and_allocations.csv")
    }
    spectral_methods = (
        "greedy_spectral",
        "equal_spectral",
        "lambda_spectral",
    )
    for dimension in (6, 17):
        audit = json.loads(metadata_lookup[dimension]["allocations_by_budget"])[
            "153600"
        ]
        predicted = {
            "greedy_spectral": float(audit["greedy_local_risk"]),
            "equal_spectral": float(audit["equal_local_risk"]),
            "lambda_spectral": float(audit["lambda_local_risk"]),
        }
        for method in spectral_methods:
            require(
                latex_scientific_body(predicted[method], 4)
                in compact_ablation_table,
                f"Stale predicted compact-ablation entry for d={dimension}, {method}.",
            )
        predicted_best = min(spectral_methods, key=predicted.__getitem__)
        require(
            rf"\mathbf{{{latex_scientific_body(predicted[predicted_best], 4)}}}"
            in compact_ablation_table,
            f"The compact ablation table does not bold the predicted minimum for d={dimension}.",
        )

        dimension_rows = {
            method: lookup[(dimension, 153600, method)] for method in ANCHOR_METHODS
        }
        local_values = {
            method: float(row["mean_local_task_squared_error"])
            for method, row in dimension_rows.items()
        }
        raw_values = {
            method: float(row["mean_raw_task_squared_error"])
            for method, row in dimension_rows.items()
        }
        for method in ANCHOR_METHODS:
            require(
                latex_scientific_body(local_values[method], 4)
                in compact_ablation_table,
                f"Stale local-error compact-ablation entry for d={dimension}, {method}.",
            )
            require(
                compact(latex_raw_token(raw_values[method]))
                in compact_ablation_table,
                f"Stale raw-MSE compact-ablation entry for d={dimension}, {method}.",
            )
        local_best = min(ANCHOR_METHODS, key=local_values.__getitem__)
        raw_best = min(ANCHOR_METHODS, key=raw_values.__getitem__)
        require(
            rf"\mathbf{{{latex_scientific_body(local_values[local_best], 4)}}}"
            in compact_ablation_table,
            f"The compact ablation table does not bold the local-error minimum for d={dimension}.",
        )
        require(
            rf"\mathbf{{{compact(latex_raw_token(raw_values[raw_best]))}}}"
            in compact_ablation_table,
            f"The compact ablation table does not bold the raw-MSE minimum for d={dimension}.",
        )

    for reduction in reductions:
        require(rf"{reduction:.1f}\%" in latex, "A reported greedy-versus-equal reduction is stale.")
    for filename in (
        "quantum_adaptive_round_oracle_task_benchmark.png",
    ):
        require((PAPER_FIGURE_DIR / filename).is_file(), f"Missing paper figure {filename}.")
    return {
        "maximum_budget_table_entries": table_entries,
        "greedy_vs_equal_reduction_percent": reductions,
    }


def check_radius_sensitivity_and_latex() -> dict[str, object]:
    config = json.loads((RADIUS_EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
    require(
        math.isclose(float(config["anchor_radius"]), math.pi / 4.0, rel_tol=0.0, abs_tol=1e-12),
        "The radius sweep does not use rho=pi/4.",
    )
    trials = read_csv(RADIUS_EXPERIMENT_DIR / "trial_results.csv")
    radii = tuple(float(value) for value in config["radii"])
    expected_cells = len(config["dimensions"]) * len(radii)
    expected_trials = expected_cells * int(config["trials"])
    require(
        len(trials) == expected_trials,
        f"The full radius sweep must contain {expected_trials} paired trial rows.",
    )
    trial_groups: dict[tuple[int, float], list[dict[str, str]]] = {}
    for row in trials:
        dimension = int(row["dimension"])
        radius = float(row["radius"])
        budget = int(row["budget"])
        allocation = [int(value) for value in ast.literal_eval(row["allocation"])]
        require(budget == 153600 and sum(allocation) == budget, "A radius-sweep row misses its copy budget.")
        require(len(allocation) == int(row["active_settings"]), "A radius-sweep setting count is inconsistent.")
        trial_groups.setdefault((dimension, radius), []).append(row)
    require(
        len(trial_groups) == expected_cells
        and all(len(group) == int(config["trials"]) for group in trial_groups.values()),
        "A task-radius cell is not paired on the configured truths.",
    )

    summary = read_csv(RADIUS_EXPERIMENT_DIR / "summary_rows.csv")
    require(len(summary) == expected_cells, "Unexpected radius-summary cell count.")
    radius_lookup = {
        (int(row["dimension"]), float(row["radius"])): row
        for row in summary
    }
    require(set(radius_lookup) == set(trial_groups), "Task-radius summaries do not match trial cells.")
    for key, group in trial_groups.items():
        summary_row = radius_lookup[key]
        for trial_field, summary_field in (
            ("gaussian_unscaled_task_squared_error", "mean_gaussian_unscaled_task_mse"),
            ("bayes_unscaled_task_squared_error", "mean_bayes_unscaled_task_mse"),
        ):
            require_close(
                float(summary_row[summary_field]),
                mean([float(row[trial_field]) for row in group]),
                f"Task-radius summary does not reproduce trials for {key}, {summary_field}",
            )
    return {
        "anchor_radius": float(config["anchor_radius"]),
        "trial_rows": len(trials),
    }


def check_task_radius_round_sensitivity() -> dict[str, object]:
    config = json.loads(
        (TASK_RADIUS_ROUND_DIR / "config.json").read_text(encoding="utf-8")
    )
    dimensions = tuple(int(value) for value in config["dimensions"])
    radii = tuple(float(value) for value in config["radii"])
    rounds = tuple(int(value) for value in config["rounds"])
    methods = tuple(str(value) for value in config["methods"])
    trials_per_cell = int(config["trials"])
    budget = int(config["budget"])
    require(dimensions == (6, 17), "Unexpected task-radius dimensions.")
    require(radii == RADII, "Unexpected task-radius grid.")
    require(
        rounds == TASK_ROUND_GRIDS[153600],
        "The task-radius experiment does not use the retained T grid.",
    )
    require(methods == TASK_ROUND_METHODS, "Unexpected task-radius methods.")
    require(
        config.get("paqt_start_at_pilot") is True,
        "The task-radius sweep must initialize S-PAQT at the common pilot.",
    )
    require_liu_west_config(config, "Task-radius sweep")

    rows = read_csv_glob(TASK_RADIUS_ROUND_DIR, "trial_results_d*_r*.csv")
    fixed_lookup = {
        (int(row["dimension"]), float(row["radius"]), int(row["trial"])): (
            float(row["gaussian_unscaled_task_squared_error"]),
            float(row["bayes_unscaled_task_squared_error"]),
        )
        for row in read_csv(RADIUS_EXPERIMENT_DIR / "trial_results.csv")
        if math.isclose(float(row["joint_failure_probability"]), 0.05)
        and int(row["trial"]) < trials_per_cell
    }
    expected_rows = (
        len(dimensions)
        * len(radii)
        * len(rounds)
        * len(methods)
        * trials_per_cell
    )
    require(
        len(rows) == expected_rows,
        f"The task-radius T sweep must contain {expected_rows} rows.",
    )
    groups: dict[tuple[int, float, int, str], list[dict[str, str]]] = {}
    keys: set[tuple[int, float, int, int, str]] = set()
    for row in rows:
        key = (
            int(row["dimension"]),
            float(row["radius"]),
            int(row["trial"]),
            int(row["rounds"]),
            str(row["method"]),
        )
        require(key not in keys, f"Duplicate task-radius round row: {key}.")
        keys.add(key)
        require(int(row["copies"]) == budget, "A task-radius row misses its budget.")
        require(
            int(row["settings"]) == 2 * int(row["rounds"]),
            "A task-radius row does not use 2T settings.",
        )
        fixed_key = key[:3]
        require(fixed_key in fixed_lookup, f"Missing coupled fixed endpoint {fixed_key}.")
        require_close(
            float(row["greedy_local_raw_task_mse"]),
            fixed_lookup[fixed_key][0],
            f"Coupled local endpoint changed for {fixed_key}",
        )
        require_close(
            float(row["greedy_nonlinear_raw_task_mse"]),
            fixed_lookup[fixed_key][1],
            f"Coupled nonlinear endpoint changed for {fixed_key}",
        )
        groups.setdefault((key[0], key[1], key[3], key[4]), []).append(row)
    require(
        all(len(group) == trials_per_cell for group in groups.values()),
        "A task-radius T cell is not paired on all truths.",
    )

    summary = read_csv(TASK_RADIUS_ROUND_DIR / "summary_rows.csv")
    summary_lookup = {
        (
            int(row["dimension"]),
            float(row["radius"]),
            int(row["rounds"]),
            str(row["method"]),
        ): row
        for row in summary
    }
    require(set(summary_lookup) == set(groups), "Task-radius summaries do not match trials.")
    for key, group in groups.items():
        require_close(
            float(summary_lookup[key]["mean_competitor_mse"]),
            mean([float(row["competitor_raw_task_mse"]) for row in group]),
            f"Task-radius summary does not reproduce trials for {key}",
        )

    method_oracles = read_csv(TASK_RADIUS_ROUND_DIR / "oracle_method_rows.csv")
    require(
        len(method_oracles) == len(dimensions) * len(radii) * len(methods),
        "Unexpected task-radius method-oracle count.",
    )
    method_lookup = {
        (int(row["dimension"]), float(row["radius"]), str(row["method"])): row
        for row in method_oracles
    }
    for key, oracle in method_lookup.items():
        candidates = [
            row
            for summary_key, row in summary_lookup.items()
            if summary_key[0] == key[0]
            and summary_key[1] == key[1]
            and summary_key[3] == key[2]
        ]
        expected = min(candidates, key=lambda row: float(row["mean_competitor_mse"]))
        require(int(oracle["rounds"]) == int(expected["rounds"]), f"Wrong oracle T for {key}.")
        require_close(
            float(oracle["mean_competitor_mse"]),
            float(expected["mean_competitor_mse"]),
            f"Wrong task-radius oracle mean for {key}",
        )

    overall = read_csv(TASK_RADIUS_ROUND_DIR / "oracle_overall_rows.csv")
    require(
        len(overall) == len(dimensions) * len(radii),
        "Unexpected task-radius overall-oracle count.",
    )
    for row in overall:
        key = (int(row["dimension"]), float(row["radius"]))
        candidates = [
            candidate
            for method_key, candidate in method_lookup.items()
            if method_key[:2] == key
        ]
        expected = min(candidates, key=lambda candidate: float(candidate["mean_competitor_mse"]))
        require(
            str(row["method"]) == str(expected["method"])
            and int(row["rounds"]) == int(expected["rounds"]),
            f"Wrong overall task-radius oracle for {key}.",
        )
    return {
        "trial_rows": len(rows),
        "rounds": list(rounds),
        "method_oracle_rows": len(method_oracles),
        "overall_oracle_rows": len(overall),
    }


def check_truth_prior_round_sensitivity() -> dict[str, object]:
    appendix_source = PAPER_APPENDIX_PATH.read_text(encoding="utf-8")
    appendix = compact(appendix_source)
    appendix_a_source = appendix_source.split(
        r"\section{Benchmark Specification and Numerical Stability}", 1
    )[0]
    require(
        "tab:task-radius-sensitivity" not in appendix_a_source,
        "Appendix A still contains the removed coupled-radius table.",
    )
    config = json.loads(
        (TRUTH_PRIOR_ROUND_DIR / "config.json").read_text(encoding="utf-8")
    )
    dimensions = tuple(int(value) for value in config["dimensions"])
    truth_radii = tuple(float(value) for value in config["truth_radii"])
    prior_radii = tuple(float(value) for value in config["prior_radii"])
    rounds = tuple(int(value) for value in config["rounds"])
    methods = tuple(str(value) for value in config["methods"])
    selected_rounds = {
        (int(key.split("_r", 1)[0][1:]), float(key.split("_r", 1)[1])): int(value)
        for key, value in config["selected_rounds"].items()
    }
    trials_per_cell = int(config["trials"])
    budget = int(config["budget"])
    valid_cells = [
        (dimension, truth_radius, prior_radius)
        for dimension in dimensions
        for truth_radius in truth_radii
        for prior_radius in prior_radii
        if truth_radius <= prior_radius + 1e-15
    ]
    require(dimensions == (6, 17), "Unexpected decoupled-radius dimensions.")
    require(truth_radii == RADII and prior_radii == RADII, "Unexpected decoupled-radius grid.")
    require(rounds == TASK_ROUND_GRIDS[153600], "Unexpected decoupled-radius T grid.")
    require(
        methods == ("structured_paqt",),
        "The decoupled-radius diagnostic should vary T only for S-PAQT.",
    )
    require(
        config.get("paqt_start_at_pilot") is True,
        "The decoupled task-radius sweep must initialize S-PAQT at the common pilot.",
    )
    require_liu_west_config(config, "Decoupled task-radius sweep")

    rows = read_csv_glob(TRUTH_PRIOR_ROUND_DIR, "trial_results_d*_t*_p*.csv")
    fixed_rows = read_csv(TRUTH_PRIOR_FIXED_DIR / "trial_results.csv")
    fixed_partial: dict[tuple[int, float, float, int], dict[str, float]] = {}
    for row in fixed_rows:
        method = str(row["method"])
        if method not in {"greedy_local", "greedy_nonlinear"}:
            continue
        fixed_key = (
            int(row["dimension"]),
            float(row["truth_radius"]),
            float(row["prior_radius"]),
            int(row["trial"]),
        )
        fixed_partial.setdefault(fixed_key, {})[method] = float(
            row["raw_task_squared_error"]
        )
    fixed_lookup = {
        key: (values["greedy_local"], values["greedy_nonlinear"])
        for key, values in fixed_partial.items()
        if set(values) == {"greedy_local", "greedy_nonlinear"}
    }
    require(
        set(selected_rounds)
        == {(dimension, prior_radius) for dimension in dimensions for prior_radius in prior_radii},
        "The decoupled diagnostic is missing a coupled-selected T.",
    )
    expected_rows = len(valid_cells) * len(methods) * trials_per_cell
    require(
        len(rows) == expected_rows,
        f"The decoupled-radius T sweep must contain {expected_rows} rows.",
    )
    groups: dict[tuple[int, float, float, int, str], list[dict[str, str]]] = {}
    keys: set[tuple[int, float, float, int, int, str]] = set()
    for row in rows:
        key = (
            int(row["dimension"]),
            float(row["truth_radius"]),
            float(row["prior_radius"]),
            int(row["trial"]),
            int(row["rounds"]),
            str(row["method"]),
        )
        require(key not in keys, f"Duplicate decoupled-radius row: {key}.")
        keys.add(key)
        require(int(row["copies"]) == budget, "A decoupled-radius row misses its budget.")
        require(
            int(row["settings"]) == 2 * int(row["rounds"]),
            "A decoupled-radius row does not use 2T settings.",
        )
        require(
            int(row["rounds"])
            == selected_rounds[(int(row["dimension"]), float(row["prior_radius"]))],
            "A decoupled-radius row does not use its coupled-selected T.",
        )
        fixed_key = key[:4]
        require(fixed_key in fixed_lookup, f"Missing decoupled fixed endpoint {fixed_key}.")
        require_close(
            float(row["greedy_local_raw_task_mse"]),
            fixed_lookup[fixed_key][0],
            f"Decoupled local endpoint changed for {fixed_key}",
        )
        require_close(
            float(row["greedy_nonlinear_raw_task_mse"]),
            fixed_lookup[fixed_key][1],
            f"Decoupled nonlinear endpoint changed for {fixed_key}",
        )
        groups.setdefault((key[0], key[1], key[2], key[4], key[5]), []).append(row)
    require(
        all(len(group) == trials_per_cell for group in groups.values()),
        "A decoupled-radius cell is not paired on all truths.",
    )

    summary = read_csv(TRUTH_PRIOR_ROUND_DIR / "summary_rows.csv")
    summary_lookup = {
        (
            int(row["dimension"]),
            float(row["truth_radius"]),
            float(row["prior_radius"]),
            int(row["rounds"]),
            str(row["method"]),
        ): row
        for row in summary
    }
    require(set(summary_lookup) == set(groups), "Decoupled-radius summaries do not match trials.")
    for key, group in groups.items():
        require_close(
            float(summary_lookup[key]["mean_competitor_mse"]),
            mean([float(row["competitor_raw_task_mse"]) for row in group]),
            f"Decoupled-radius summary does not reproduce trials for {key}",
        )

    method_oracles = read_csv(TRUTH_PRIOR_ROUND_DIR / "oracle_method_rows.csv")
    overall = read_csv(TRUTH_PRIOR_ROUND_DIR / "oracle_overall_rows.csv")
    require(
        len(method_oracles) == len(valid_cells) * len(methods),
        "Unexpected decoupled-radius method-oracle count.",
    )
    require(len(overall) == len(valid_cells), "Unexpected decoupled-radius overall-oracle count.")
    overall_lookup = {
        (
            int(row["dimension"]),
            float(row["truth_radius"]),
            float(row["prior_radius"]),
        ): row
        for row in overall
    }
    paqt_mean_wins = {
        key
        for key, row in overall_lookup.items()
        if float(row["mean_competitor_mse"])
        < float(row["mean_greedy_nonlinear_mse"])
    }
    require(
        paqt_mean_wins
        == {
            (6, 0.01, 0.08),
            (6, 0.01, 0.16),
            (6, 0.02, 0.16),
            (6, 0.03, 0.16),
            (6, 0.04, 0.16),
            (6, 0.08, 0.16),
            (6, 0.16, 0.16),
        },
        "The decoupled task-loss mean boundary changed.",
    )
    # Narrative conclusions may be reworded without changing the reported result.
    # Validate the underlying comparisons and quoted graph values instead.
    for token in (
        r"2.3{\times}10^{-6}",
        r"9.6{\times}10^{-5}",
        r"5.4{\times}10^{-6}",
        r"1.9{\times}10^{-4}",
        r"1.3{\times}10^{-4}",
        r"2.2{\times}10^{-4}",
        r"6.0{\times}10^{-4}",
        r"7.6{\times}10^{-4}",
        r"6.7{\times}10^{-7}",
        r"2.0{\times}10^{-6}",
        r"8.9{\times}10^{-6}",
        r"6.8{\times}10^{-6}",
        r"5.0{\times}10^{-5}",
    ):
        require(token in appendix, f"Appendix A.1 is missing the graph value {token}.")
    generated_heatmap = TRUTH_PRIOR_ROUND_DIR / "truth_prior_ratio_heatmap.png"
    paper_heatmap = PAPER_FIGURE_DIR / "quantum_truth_prior_ratio_heatmap.png"
    require(generated_heatmap.is_file(), "Missing decoupled task ratio heatmap.")
    require(paper_heatmap.is_file(), "Missing manuscript task ratio heatmap.")
    generated_hash = hashlib.sha256(generated_heatmap.read_bytes()).hexdigest()
    paper_hash = hashlib.sha256(paper_heatmap.read_bytes()).hexdigest()
    require(
        generated_hash == paper_hash,
        "The manuscript decoupled task ratio heatmap is stale.",
    )
    require(
        compact(r"\rho_{\rm task}=E_{\rm fixed}/E_{\rm S\text{-}PAQT}")
        in appendix,
        "Appendix A.1 does not define its plotted performance ratio.",
    )
    require(
        compact(r"T^*(d,R_\alpha)") in appendix
        and compact(r"(3072,3072,512,1536,3072,768)") in appendix
        and compact(r"(1536,1536,512,512,1536,3072)") in appendix,
        "Appendix A does not document the decoupled selected-T rule.",
    )
    return {
        "valid_cells": len(valid_cells),
        "trial_rows": len(rows),
        "method_oracle_rows": len(method_oracles),
        "overall_oracle_rows": len(overall),
        "paper_heatmap_sha256": paper_hash,
    }


def check_full_tomography_and_latex() -> dict[str, object]:
    latex = manuscript_latex()
    paper_source = PAPER_PATH.read_text(encoding="utf-8")
    config = json.loads((FULL_TOMOGRAPHY_DIR / "config.json").read_text(encoding="utf-8"))
    round_config = json.loads(
        (FULL_TOMOGRAPHY_ROUND_DIR / "config.json").read_text(encoding="utf-8")
    )
    baseline_methods = ("greedy_full_state",)
    adaptive_methods = TASK_ROUND_METHODS
    require(
        round_config.get("paqt_start_at_pilot") is True,
        "The full-state round sweep must initialize S-PAQT at the common pilot.",
    )
    require_liu_west_config(round_config, "Full-state round sweep")
    expected_cells = len(config["dimensions"]) * len(config["radii"])
    trials = read_csv(FULL_TOMOGRAPHY_DIR / "trial_results.csv")
    require(
        len(trials) == expected_cells * int(config["trials"]) * len(baseline_methods),
        "The full-tomography grid is incomplete.",
    )
    baseline_groups: dict[tuple[int, float, str], list[float]] = {}
    baseline_keys: set[tuple[int, float, int, str]] = set()
    for row in trials:
        key = (int(row["dimension"]), float(row["radius"]), int(row["trial"]), row["method"])
        require(key not in baseline_keys, f"Duplicate full-state baseline row: {key}.")
        baseline_keys.add(key)
        require(int(row["budget"]) == int(config["budget"]), "A full-state baseline row has the wrong budget.")
        baseline_groups.setdefault((key[0], key[1], key[3]), []).append(
            float(row["raw_state_infidelity"])
        )
    summary = read_csv(FULL_TOMOGRAPHY_DIR / "summary_rows.csv")
    require(len(summary) == expected_cells * len(baseline_methods), "Unexpected tomography summary count.")
    baseline_lookup = {
        (int(row["dimension"]), float(row["radius"]), row["method"]): row
        for row in summary
    }
    require(set(baseline_lookup) == set(baseline_groups), "Full-state summaries do not match trial cells.")
    for key, values in baseline_groups.items():
        require_close(
            float(baseline_lookup[key]["mean_raw_state_infidelity"]),
            mean(values),
            f"Full-state summary does not reproduce trials for {key}",
        )

    all_round_trials = read_csv_glob(
        FULL_TOMOGRAPHY_ROUND_DIR,
        "trial_results_d*_r*.csv",
    )
    expected_round_rows = (
        expected_cells
        * int(config["trials"])
        * len(round_config["rounds"])
        * len(adaptive_methods)
    )
    require(
        len(all_round_trials) == expected_round_rows,
        f"The full-state round sweep must contain exactly the {expected_round_rows} retained rows.",
    )
    all_round_keys: set[tuple[int, float, int, int, str]] = set()
    configured_rounds = {int(value) for value in round_config["rounds"]}
    for row in all_round_trials:
        key = (
            int(row["dimension"]),
            float(row["radius"]),
            int(row["trial"]),
            int(row["rounds"]),
            row["method"],
        )
        require(key not in all_round_keys, f"Duplicate full-state round-sweep row: {key}.")
        all_round_keys.add(key)
        require(
            int(row["rounds"]) in configured_rounds,
            f"Removed full-state round count T={row['rounds']} remains.",
        )
        require(int(row["copies"]) == int(config["budget"]), "A stored full-state round row misses its budget.")
        require(int(row["settings"]) == 2 * int(row["rounds"]), "A stored full-state round row does not use 2T settings.")

    round_trials = all_round_trials
    require(
        len(round_trials) == expected_round_rows,
        f"The full-state round sweep must contain {expected_round_rows} rows.",
    )
    for row in round_trials:
        require(
            int(row["copies"]) == int(config["budget"]),
            "A full-state round-sweep row misses the exact copy budget.",
        )
        require(
            int(row["settings"]) == 2 * int(row["rounds"]),
            "A full-state round-sweep row does not use 2T settings.",
        )
    oracle_rows = read_csv(FULL_TOMOGRAPHY_ROUND_DIR / "oracle_method_rows.csv")
    require(
        len(oracle_rows) == expected_cells * len(adaptive_methods),
        "Unexpected full-state best-T summary count.",
    )
    oracle_lookup = {
        (int(row["dimension"]), float(row["radius"]), row["method"]): row
        for row in oracle_rows
    }
    full_state_acquisition_table = compact_labeled_table(
        paper_source, "tab:full-state-acquisition-summary"
    )
    require(
        "Greedy spectral wins" not in full_state_acquisition_table,
        "The full-state acquisition-summary table still contains the redundant win-count column.",
    )
    table_label_index = paper_source.index(r"\label{tab:full-state-round-sensitivity}")
    table_start = paper_source.rfind(r"\begin{table}", 0, table_label_index)
    table_end = paper_source.index(r"\end{table}", table_label_index) + len(
        r"\end{table}"
    )
    full_state_table = compact(paper_source[table_start:table_end])
    for dimension in config["dimensions"]:
        for radius in config["radii"]:
            fixed = float(
                baseline_lookup[(int(dimension), float(radius), "greedy_full_state")][
                    "mean_raw_state_infidelity"
                ]
            )
            fixed_token = latex_scientific_body(fixed, 2)
            require(
                fixed_token in full_state_table,
                f"Stale fixed full-state table entry for d={dimension}, R={radius}.",
            )
            cell_values = {"fixed": fixed}
            for method in adaptive_methods:
                oracle = oracle_lookup[(int(dimension), float(radius), method)]
                value = float(oracle["mean_competitor_infidelity"])
                token = latex_scientific_body(value, 2)
                entry = compact(rf"{token}\;({int(oracle['rounds'])})")
                bold_entry = compact(
                    rf"\mathbf{{{token}}}\;({int(oracle['rounds'])})"
                )
                require(
                    entry in full_state_table or bold_entry in full_state_table,
                    f"Stale full-state table entry for d={dimension}, R={radius}, {method}.",
                )
                cell_values[method] = value
            best = min(cell_values, key=cell_values.__getitem__)
            best_token = latex_scientific_body(cell_values[best], 2)
            require(
                rf"\mathbf{{{best_token}}}" in full_state_table,
                f"The full-state table does not bold the minimum for d={dimension}, R={radius}.",
            )
    local_radii = [float(radius) for radius in config["radii"] if float(radius) <= 0.04]
    for dimension_value in config["dimensions"]:
        dimension = int(dimension_value)
        local_oracles = [
            row
            for row in oracle_rows
            if int(row["dimension"]) == dimension
            and float(row["radius"]) <= 0.04
        ]
        fixed_settings = int(
            baseline_lookup[(dimension, local_radii[0], "greedy_full_state")][
                "local_dimension"
            ]
        )
        rounds = [int(row["rounds"]) for row in local_oracles]
        cost_ratios = [
            acquisition_cost_ratio(
                int(row["budget"]), int(row["rounds"]), fixed_settings
            )
            for row in local_oracles
        ]

        def integer_range(values: list[int]) -> str:
            low, high = min(values), max(values)
            return rf"\({low}\)" if low == high else rf"\({low}\)--\({high}\)"

        expected_acquisition_row = compact(
            rf"{dimension} & \({fixed_settings}\) & "
            rf"{integer_range(rounds)} & "
            rf"\({math.ceil(min(cost_ratios))}\)--"
            rf"\({math.ceil(max(cost_ratios))}\)"
        )
        require(
            expected_acquisition_row in full_state_acquisition_table,
            f"Stale full-state acquisition-summary row for d={dimension}.",
        )
    configured_groups: dict[tuple[int, float, int, str], list[dict[str, str]]] = {}
    for row in round_trials:
        key = (
            int(row["dimension"]),
            float(row["radius"]),
            int(row["rounds"]),
            row["method"],
        )
        configured_groups.setdefault(key, []).append(row)
    round_summary_lookup = {
        (int(row["dimension"]), float(row["radius"]), int(row["rounds"]), row["method"]): row
        for row in read_csv(FULL_TOMOGRAPHY_ROUND_DIR / "summary_rows.csv")
        if int(row["rounds"]) in configured_rounds
    }
    require(set(round_summary_lookup) == set(configured_groups), "Full-state round summaries do not match trials.")
    for key, group in configured_groups.items():
        require(len(group) == int(config["trials"]), f"Full-state round cell {key} is not paired on 30 truths.")
        require_close(
            float(round_summary_lookup[key]["mean_competitor_infidelity"]),
            mean([float(row["competitor_raw_state_infidelity"]) for row in group]),
            f"Full-state round summary does not reproduce trials for {key}",
        )
    for key, oracle in oracle_lookup.items():
        candidates = [
            row
            for summary_key, row in round_summary_lookup.items()
            if summary_key[0] == key[0] and summary_key[1] == key[1] and summary_key[3] == key[2]
        ]
        expected = min(candidates, key=lambda row: float(row["mean_competitor_infidelity"]))
        require(int(oracle["rounds"]) == int(expected["rounds"]), f"Wrong full-state oracle T for {key}.")
        require_close(
            float(oracle["mean_competitor_infidelity"]),
            float(expected["mean_competitor_infidelity"]),
            f"Wrong full-state oracle mean for {key}",
        )
    localized_oracles = [
        row
        for row in oracle_rows
        if float(row["radius"]) <= 0.04
    ]
    require(len(localized_oracles) == 24, "The full-state localized regime must contain 24 comparisons.")
    require(
        all(float(row["mean_competitor_minus_greedy"]) > 0.0 for row in localized_oracles),
        "A localized full-state adaptive mean is not above the fixed-design mean.",
    )
    require(
        all(float(row["paired_difference_ci95_low"]) > 0.0 for row in localized_oracles),
        "A localized full-state paired interval does not exclude zero.",
    )
    full_state_cost_ratios = [
        acquisition_cost_ratio(
            int(row["budget"]),
            int(row["rounds"]),
            LOCAL_DIMENSIONS[int(row["dimension"])],
        )
        for row in localized_oracles
    ]
    require(round(min(full_state_cost_ratios), 2) == 708.81, "Stale minimum full-state cost ratio.")
    require(round(max(full_state_cost_ratios), 2) == 6041.17, "Stale maximum full-state cost ratio.")
    require(
        r"709\)--\(6042" in latex,
        "The upward-rounded full-state acquisition-cost range is stale in the paper.",
    )
    headline_rows = {
        (6, 0.08): oracle_lookup[(6, 0.08, "structured_paqt")],
        (17, 0.08): oracle_lookup[(17, 0.08, "structured_paqt")],
        (6, 0.16): oracle_lookup[(6, 0.16, "structured_paqt")],
        (17, 0.16): oracle_lookup[(17, 0.16, "structured_paqt")],
    }
    require(
        round(float(headline_rows[(6, 0.08)]["greedy_win_fraction"]) * 30) == 6
        and round(float(headline_rows[(17, 0.08)]["greedy_win_fraction"]) * 30) == 24
        and round(float(headline_rows[(6, 0.16)]["greedy_win_fraction"]) * 30) == 1
        and round(float(headline_rows[(17, 0.16)]["greedy_win_fraction"]) * 30) == 5,
        "A full-state headline paired-win count changed.",
    )
    require(
        float(headline_rows[(6, 0.08)]["paired_difference_ci95_high"]) < 0.0
        and float(headline_rows[(17, 0.08)]["paired_difference_ci95_low"]) < 0.0
        < float(headline_rows[(17, 0.08)]["paired_difference_ci95_high"])
        and float(headline_rows[(6, 0.16)]["paired_difference_ci95_high"]) < 0.0
        and float(headline_rows[(17, 0.16)]["paired_difference_ci95_high"]) < 0.0,
        "A full-state headline paired interval changed.",
    )
    generated_figure = (
        FULL_TOMOGRAPHY_ROUND_DIR / "full_state_round_oracle_benchmark.png"
    )
    paper_figure = PAPER_FIGURE_DIR / "quantum_full_state_round_oracle_benchmark.png"
    require(generated_figure.is_file(), "Missing generated full-state benchmark figure.")
    require(paper_figure.is_file(), "Missing manuscript full-state benchmark figure.")
    generated_hash = hashlib.sha256(generated_figure.read_bytes()).hexdigest()
    paper_hash = hashlib.sha256(paper_figure.read_bytes()).hexdigest()
    require(generated_hash == paper_hash, "The manuscript full-state benchmark figure is stale.")
    return {
        "baseline_trial_rows": len(trials),
        "stored_round_sensitivity_rows": len(all_round_trials),
        "round_sensitivity_rows": len(round_trials),
        "localized_oracle_comparisons": len(localized_oracles),
        "localized_cost_ratio_range": [
            min(full_state_cost_ratios),
            max(full_state_cost_ratios),
        ],
        "coupled_radius_cells": expected_cells,
        "paper_figure_sha256": paper_hash,
    }


def check_full_tomography_truth_prior_sensitivity() -> dict[str, object]:
    appendix = compact(PAPER_APPENDIX_PATH.read_text(encoding="utf-8"))
    config = json.loads(
        (FULL_TOMOGRAPHY_TRUTH_PRIOR_DIR / "config.json").read_text(
            encoding="utf-8"
        )
    )
    expected_config = {
        "dimensions": [6, 17],
        "truth_radii": list(RADII),
        "prior_radii": list(RADII),
        "budget": 153600,
        "rounds": list(TASK_ROUND_GRIDS[153600]),
        "round_selection": "S-PAQT T selected on the coupled diagonal",
        "methods": ["structured_paqt"],
        "paqt_start_at_pilot": True,
        "trials": 30,
        "seed": 20260731,
        "smc_particles": 500,
        "paqt_resampler": "Liu-West",
        "paqt_liu_west_a": 0.98,
        "paqt_resample_ess_fraction": 0.5,
        "state_endpoint": "Bayesian mean density operator infidelity",
    }
    for key, expected in expected_config.items():
        require(
            config.get(key) == expected,
            f"Full-state decoupling configuration field {key} is stale.",
        )
    require_liu_west_config(config, "Full-state decoupling sweep")
    require(
        set(config) == {*expected_config, "selected_rounds"},
        "Full-state decoupling configuration contains stale fields.",
    )

    dimensions = tuple(int(value) for value in config["dimensions"])
    truth_radii = tuple(float(value) for value in config["truth_radii"])
    prior_radii = tuple(float(value) for value in config["prior_radii"])
    valid_cells = {
        (dimension, truth_radius, prior_radius)
        for dimension in dimensions
        for truth_radius in truth_radii
        for prior_radius in prior_radii
        if truth_radius <= prior_radius + 1e-15
    }
    selected_rounds = {
        (int(key.split("_r", 1)[0][1:]), float(key.split("_r", 1)[1])): int(
            value
        )
        for key, value in config["selected_rounds"].items()
    }
    coupled_oracle = {
        (int(row["dimension"]), float(row["radius"])): int(row["rounds"])
        for row in read_csv(FULL_TOMOGRAPHY_ROUND_DIR / "oracle_method_rows.csv")
        if row["method"] == "structured_paqt"
    }
    require(
        selected_rounds == coupled_oracle,
        "The full-state decoupling grid does not use the coupled S-PAQT oracle T.",
    )

    rows = read_csv_glob(
        FULL_TOMOGRAPHY_TRUTH_PRIOR_DIR,
        "trial_results_d*_t*_p*.csv",
    )
    expected_rows = len(valid_cells) * int(config["trials"])
    require(
        len(rows) == expected_rows,
        f"The full-state decoupling grid must contain {expected_rows} rows.",
    )
    groups: dict[tuple[int, float, float], list[dict[str, str]]] = {}
    row_lookup: dict[tuple[int, float, float, int], dict[str, str]] = {}
    for row in rows:
        cell = (
            int(row["dimension"]),
            float(row["truth_radius"]),
            float(row["prior_radius"]),
        )
        key = (*cell, int(row["trial"]))
        require(cell in valid_cells, f"Unexpected full-state decoupling cell: {cell}.")
        require(key not in row_lookup, f"Duplicate full-state decoupling row: {key}.")
        row_lookup[key] = row
        groups.setdefault(cell, []).append(row)
        require(row["method"] == "structured_paqt", "Unexpected full-state method.")
        require(int(row["budget"]) == 153600, "A full-state row misses its budget.")
        require(int(row["copies"]) == 153600, "A full-state row misses its copies.")
        require(
            int(row["settings"]) == 2 * int(row["rounds"]),
            "A full-state decoupling row does not use 2T settings.",
        )
        require(
            int(row["rounds"]) == selected_rounds[(cell[0], cell[2])],
            "A full-state decoupling row does not hold the selected T by column.",
        )
    require(
        set(groups) == valid_cells
        and all(len(group) == int(config["trials"]) for group in groups.values()),
        "A full-state decoupling cell is not paired on all 30 truths.",
    )

    summary = read_csv(FULL_TOMOGRAPHY_TRUTH_PRIOR_DIR / "summary_rows.csv")
    summary_lookup = {
        (
            int(row["dimension"]),
            float(row["truth_radius"]),
            float(row["prior_radius"]),
        ): row
        for row in summary
    }
    require(
        set(summary_lookup) == valid_cells,
        "Full-state decoupling summaries do not match the trial cells.",
    )
    for cell, group in groups.items():
        require_close(
            float(summary_lookup[cell]["mean_greedy_local_infidelity"]),
            mean(
                [float(row["greedy_local_raw_state_infidelity"]) for row in group]
            ),
            f"Full-state local summary does not reproduce trials for {cell}",
        )
        require_close(
            float(summary_lookup[cell]["mean_competitor_infidelity"]),
            mean([float(row["competitor_raw_state_infidelity"]) for row in group]),
            f"Full-state S-PAQT summary does not reproduce trials for {cell}",
        )
    paqt_mean_wins = {
        cell
        for cell, row in summary_lookup.items()
        if float(row["mean_competitor_infidelity"])
        < float(row["mean_greedy_local_infidelity"])
    }
    require(
        paqt_mean_wins
        == {
            (6, 0.08, 0.08),
            (6, 0.08, 0.16),
            (6, 0.16, 0.16),
            (17, 0.08, 0.16),
            (17, 0.16, 0.16),
        },
        "The full-state decoupled mean boundary changed.",
    )
    # Do not make a particular prose formulation part of the data contract.
    for token in (
        r"6.0{\times}10^{-4}",
        r"3.6{\times}10^{-3}",
        r"2.8{\times}10^{-3}",
        r"3.2{\times}10^{-3}",
        r"7.9{\times}10^{-2}",
        r"5.5{\times}10^{-3}",
        r"3.6{\times}10^{-1}",
        r"1.5{\times}10^{-2}",
    ):
        require(token in appendix, f"Appendix A.2 is missing the graph value {token}.")

    coupled_rows = {
        (
            int(row["dimension"]),
            float(row["radius"]),
            int(row["trial"]),
        ): row
        for row in read_csv_glob(FULL_TOMOGRAPHY_ROUND_DIR, "trial_results_d*_r*.csv")
        if row["method"] == "structured_paqt"
        and int(row["rounds"])
        == selected_rounds[(int(row["dimension"]), float(row["radius"]))]
    }
    for dimension in dimensions:
        for radius in truth_radii:
            for trial in range(int(config["trials"])):
                decoupled = row_lookup[(dimension, radius, radius, trial)]
                coupled = coupled_rows[(dimension, radius, trial)]
                require_close(
                    float(decoupled["greedy_local_raw_state_infidelity"]),
                    float(coupled["greedy_raw_state_infidelity"]),
                    "The full-state decoupling diagonal changed its fixed endpoint",
                )
                require_close(
                    float(decoupled["competitor_raw_state_infidelity"]),
                    float(coupled["competitor_raw_state_infidelity"]),
                    "The full-state decoupling diagonal changed its S-PAQT endpoint",
                )

    generated_heatmap = (
        FULL_TOMOGRAPHY_TRUTH_PRIOR_DIR
        / "full_state_truth_prior_ratio_heatmap.png"
    )
    paper_heatmap = (
        PAPER_FIGURE_DIR / "quantum_full_state_truth_prior_ratio_heatmap.png"
    )
    require(generated_heatmap.is_file(), "Missing full-state ratio heatmap.")
    require(paper_heatmap.is_file(), "Missing manuscript full-state ratio heatmap.")
    generated_hash = hashlib.sha256(generated_heatmap.read_bytes()).hexdigest()
    paper_hash = hashlib.sha256(paper_heatmap.read_bytes()).hexdigest()
    require(
        generated_hash == paper_hash,
        "The manuscript full-state ratio heatmap is stale.",
    )
    require(
        compact(r"\rho_{\rm state}=E_{\rm fixed}/E_{\rm S\text{-}PAQT}")
        in appendix,
        "Appendix A.2 does not define its plotted performance ratio.",
    )
    require(
        "raw full-state infidelity" in appendix.lower()
        and compact(r"(1536,3072,3072,1536,1536,512)") in appendix
        and compact(r"(512,512,1536,1536,3072,3072)") in appendix,
        "Appendix A does not document the full-state decoupling endpoint and T rule.",
    )
    return {
        "valid_cells": len(valid_cells),
        "trial_rows": len(rows),
        "summary_rows": len(summary),
        "paper_heatmap_sha256": paper_hash,
        "paqt_mean_win_cells": sorted(paqt_mean_wins),
    }


def main() -> None:
    report = {
        "status": "passed",
        "configs": check_experiment_configs(),
        "geometry": check_geometry(),
        "trials": check_trials(),
        "task_round_sensitivity": check_task_round_sensitivity(),
        "graphs": check_graph_data_and_assets(),
        "paper": check_summary_and_latex(),
        "radius_sensitivity": check_radius_sensitivity_and_latex(),
        "task_radius_round_sensitivity": check_task_radius_round_sensitivity(),
        "truth_prior_round_sensitivity": check_truth_prior_round_sensitivity(),
        "full_tomography": check_full_tomography_and_latex(),
        "full_tomography_truth_prior_sensitivity": (
            check_full_tomography_truth_prior_sensitivity()
        ),
    }
    serialized = json.dumps(report, indent=2)
    print(serialized)
    REPORT_PATH.write_text(serialized + "\n", encoding="utf-8")
    print(f"Wrote validation report to {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()
