# Next Steps

## Immediate Batch

- No additional search is required for the current task. The notebook has been updated to the selected configuration and executed end-to-end.

## Decision Gates

- If we want a more theory-driven selection rule, the next experiment should scan physical lag `tau` directly and track `near1`, `gap_2`, and macro-cluster frequency mismatch as functions of `tau`.
- If we want a more robust macro basis inside the near-1 subspace, the next step should test an additional rotation criterion inside the top-2 singular subspace, such as maximizing cluster-mean correlation or sparsity.

## Resume Command Notes

- Re-run the tuned sweep:
  - `/opt/anaconda3/envs/py311/bin/python experiments/kuramoto/run_whitened_macro_sweep.py --config-file experiments/kuramoto/tuning/batch4_configs.json --results-file experiments/kuramoto/tuning/batch4_results.jsonl`
- Re-execute the finalized notebook:
  - `/opt/anaconda3/envs/py311/bin/python -m nbclient --help`
  - Current artifact to inspect: `experiments/kuramoto/kuramoto_2group_gram_v2.ipynb`
