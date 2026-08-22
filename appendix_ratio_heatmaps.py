"""Generate the final Appendix A fixed/S-PAQT ratio heatmaps."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOT = REPO_ROOT / "experiments" / "odd_even_xz_style"
PAPER_FIGURE_ROOT = REPO_ROOT / "paper_for_saim" / "figures"
RADII = (0.01, 0.02, 0.03, 0.04, 0.08, 0.16)
DIMENSIONS = (6, 17)
LOG_RATIO_EXTENT = 2.75


@dataclass(frozen=True)
class Result:
    dimension: int
    truth_radius: float
    prior_radius: float
    fixed: float
    paqt: float


def read_results(path: Path, fixed_field: str, paqt_field: str) -> list[Result]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        Result(
            dimension=int(row["dimension"]),
            truth_radius=float(row["truth_radius"]),
            prior_radius=float(row["prior_radius"]),
            fixed=float(row[fixed_field]),
            paqt=float(row[paqt_field]),
        )
        for row in rows
    ]


def ratio_label(value: float) -> str:
    if value < 0.1:
        return f"{value:.2f}x"
    if value < 10.0:
        return f"{value:.1f}x"
    return f"{value:.0f}x"


def save_ratio_heatmap(
    rows: list[Result],
    generated_path: Path,
    paper_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    norm = TwoSlopeNorm(
        vmin=-LOG_RATIO_EXTENT,
        vcenter=0.0,
        vmax=LOG_RATIO_EXTENT,
    )
    image = None
    for column_index, dimension in enumerate(DIMENSIONS):
        axis = axes[column_index]
        records = {
            (row.truth_radius, row.prior_radius): row
            for row in rows
            if row.dimension == dimension
        }
        matrix = np.full((len(RADII), len(RADII)), np.nan)
        ratios = np.full_like(matrix, np.nan)
        for truth_index, truth_radius in enumerate(RADII):
            for prior_index, prior_radius in enumerate(RADII):
                record = records.get((truth_radius, prior_radius))
                if record is None:
                    continue
                ratio = record.fixed / record.paqt
                ratios[truth_index, prior_index] = ratio
                matrix[truth_index, prior_index] = np.log10(ratio)
        image = axis.imshow(
            np.ma.masked_invalid(matrix),
            cmap="RdBu_r",
            norm=norm,
            origin="upper",
            interpolation="nearest",
        )
        axis.set_facecolor("#eeeeee")
        for truth_index in range(len(RADII)):
            for prior_index in range(len(RADII)):
                ratio = ratios[truth_index, prior_index]
                if not np.isfinite(ratio):
                    continue
                log_ratio = np.log10(ratio)
                axis.text(
                    prior_index,
                    truth_index,
                    ratio_label(ratio),
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    color="white" if abs(log_ratio) > 1.0 else "black",
                )
                if truth_index == prior_index:
                    axis.add_patch(
                        plt.Rectangle(
                            (prior_index - 0.5, truth_index - 0.5),
                            1,
                            1,
                            fill=False,
                            edgecolor="black",
                            linewidth=0.85,
                            linestyle="--",
                        )
                    )
        indices = np.arange(len(RADII))
        axis.set_xticks(
            indices,
            [f"{value:g}" for value in RADII],
            rotation=35,
        )
        axis.set_yticks(indices, [f"{value:g}" for value in RADII])
        axis.set_xlabel(r"assumed radius $R_\alpha$")
        axis.set_ylabel(
            r"truth radius $r_{\rm truth}$" if column_index == 0 else ""
        )
        axis.set_title(f"$d={dimension}$", pad=5)
        axis.set_aspect("equal")
    colorbar = figure.colorbar(image, ax=axes, fraction=0.035, pad=0.04)
    colorbar.set_label(r"ratio $\rho=$ fixed error / S-PAQT error")
    ticks = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([ratio_label(10.0**value) for value in ticks])
    figure.subplots_adjust(left=0.09, right=0.87, bottom=0.21, top=0.93, wspace=0.16)
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    paper_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(generated_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    shutil.copyfile(generated_path, paper_path)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
        }
    )
    task_root = ACTIVE_ROOT / "quantum_truth_prior_round_sensitivity"
    task_rows = read_results(
        task_root / "oracle_overall_rows.csv",
        "mean_greedy_nonlinear_mse",
        "mean_competitor_mse",
    )
    save_ratio_heatmap(
        task_rows,
        task_root / "truth_prior_ratio_heatmap.png",
        PAPER_FIGURE_ROOT / "quantum_truth_prior_ratio_heatmap.png",
    )

    full_state_root = (
        ACTIVE_ROOT / "quantum_full_tomography_truth_prior_sensitivity"
    )
    full_state_rows = read_results(
        full_state_root / "summary_rows.csv",
        "mean_greedy_local_infidelity",
        "mean_competitor_infidelity",
    )
    save_ratio_heatmap(
        full_state_rows,
        full_state_root / "full_state_truth_prior_ratio_heatmap.png",
        PAPER_FIGURE_ROOT / "quantum_full_state_truth_prior_ratio_heatmap.png",
    )


if __name__ == "__main__":
    main()
