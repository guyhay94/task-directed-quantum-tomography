# Reproducing the paper figures

Run every command below from the repository root. No experiment output is
included in the repository. The scripts create their output directories and
print the full location when they finish.

## Figure 1 — task error versus copy budget

Run:

~~~console
python -B src/quantum_greedy_spectral_experiment.py --mode full
python -B src/quantum_adaptive_round_sensitivity_experiment.py
~~~

The second command creates **adaptive_round_oracle_task_benchmark.png**. This
is the graph used as
[Figure 1](../paper_for_saim/figures/quantum_adaptive_round_oracle_task_benchmark.png).
These two commands also produce the task-comparison, allocation, ablation, and
round-sensitivity results reported in the surrounding tables.

## Figure 2 — full-state infidelity versus localization radius

Run:

~~~console
python -B src/quantum_full_tomography_radius_experiment.py --mode full
python -B src/quantum_full_tomography_round_sensitivity_experiment.py --validate-complete
~~~

The second command creates **full_state_round_oracle_benchmark.png**. This is
the graph used as
[Figure 2](../paper_for_saim/figures/quantum_full_state_round_oracle_benchmark.png).
The same run produces the full-state comparison and round-sensitivity results
reported in the surrounding tables.

## Figure 3 — task truth-radius/prior-radius heatmap

Run these commands in order:

~~~console
python -B src/quantum_radius_sensitivity_experiment.py --mode full
python -B src/quantum_task_radius_round_sensitivity_experiment.py --validate-complete
python -B src/quantum_truth_prior_decoupling_experiment.py --mode full --methods greedy_local greedy_nonlinear
python -B src/quantum_truth_prior_round_sensitivity_experiment.py --validate-complete
~~~

These commands produce the task results needed for the Appendix A.1 heatmap.
After also completing Figure 4, run the final heatmap command below. It creates
**truth_prior_ratio_heatmap.png** and copies it to
[Figure 3](../paper_for_saim/figures/quantum_truth_prior_ratio_heatmap.png).

## Figure 4 — full-state truth-radius/prior-radius heatmap

Run these commands in order:

~~~console
python -B src/quantum_full_tomography_radius_experiment.py --mode full
python -B src/quantum_full_tomography_round_sensitivity_experiment.py --validate-complete
python -B src/quantum_full_tomography_truth_prior_sensitivity_experiment.py --workers 8 --validate-complete
~~~

If Figure 2 has already been reproduced, skip the first two commands. After
both appendix experiments are complete, run:

~~~console
python -B src/appendix_ratio_heatmaps.py
~~~

This creates **full_state_truth_prior_ratio_heatmap.png** and copies it to
[Figure 4](../paper_for_saim/figures/quantum_full_state_truth_prior_ratio_heatmap.png).
The same command also creates and copies Figure 3.

## Checking a complete reproduction

After all four workflows finish, run:

~~~console
python -B src/validate_qiskit_backend.py
python -B src/validate_greedy_spectral_experiment.py
python -B src/validate_reported_results.py
~~~

The last command checks the generated trials, summaries, manuscript values,
and figure copies against one another.
