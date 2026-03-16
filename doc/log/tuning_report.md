# Tuning Report

## 1. Optimization objective

- Tune the Kuramoto two-group notebook so that the correctly double-whitened Koopman analysis still identifies a reasonable macro dimension.
- Success criteria:
  - the whitened singular spectrum should show a visible cutoff at the true group count `2`
  - the final macro spectral analysis should align the two macro frequencies with the two cluster frequencies
  - the notebook should be updated and re-executed with the selected setting

## 2. Repository audit summary

- The relevant notebook is [kuramoto_2group_gram_v2.ipynb](/Users/yangmingzhe/Desktop/code/github/koopCE/experiments/kuramoto/kuramoto_2group_gram_v2.ipynb).
- The repository already contains helpers for correct lagged covariance estimation and double whitening in [tools.py](/Users/yangmingzhe/Desktop/code/github/koopCE/tools/tools.py).
- A reusable tuning script was added at [run_whitened_macro_sweep.py](/Users/yangmingzhe/Desktop/code/github/koopCE/experiments/kuramoto/run_whitened_macro_sweep.py) so the notebook-facing parameters could be searched systematically instead of by hand.

## 3. Identified tuning parameters and reasoning

- `library_kind`: controls how redundant the observable space is before whitening.
- `sample_stride` and `lag_steps`: control the physical lag `tau = dt * sample_stride * lag_steps`, which strongly affects whether short-time reversible microstructure dominates the singular spectrum.
- `preprocess` and `pca_dim`: allow a light data-driven compression before whitening; this turned out to be the key knob.
- `n_trajectories`, `weights_mode`, and `center_features`: tested as secondary stabilizers once the main failure mode was reproduced.

## 4. Experiment strategy

- Baseline: reproduce the notebook-like `identity+fourier`, `lag=1` setting under correct double whitening.
- Batch 1: sweep `library_kind`, `sample_stride`, and `lag_steps`.
- Batch 2: refine the identity-only region with centering and multiple trajectories.
- Batch 3: introduce PCA preprocessing and search `pca_dim x sample_stride x lag_steps`.
- Batch 4: refine the promising `PCA-3` region and choose a balanced final configuration for the notebook.

## 5. Experiment history summary

- Baseline reproduced the original issue: the correctly whitened spectrum had a broad near-1 plateau instead of a cutoff at 2.
- Identity-only sweeps shrank the plateau from `32` to `4`, but never to `2`.
- PCA preprocessing split the search space sharply:
  - `PCA-2`: trivially rank-2, but both macro frequencies collapsed to the same cluster frequency.
  - `PCA-3`: robustly produced `near1=2` and a nontrivial gap after the second singular value.
  - `PCA-4/5`: the near-1 plateau returned at dimension `4`.
- Refining the `PCA-3` region showed a family of exact frequency-match solutions once the physical lag became moderate to large.

## 6. Best-performing configurations

- Score-only winner: `b4_pca3_lag120_s15`
  - `sigma = [0.9965, 0.9962, 0.0400]`
  - `gap_2 = 0.9562`
  - `near_one_count_099 = 2`
  - `best_freq_mismatch = 0.0000`
  - downside: `best_cluster_corr = 0.5424`, weaker than more balanced alternatives
- Final selected notebook configuration: `b4_pca3_lag60_s15`
  - `sample_stride = 15`
  - `lag_steps = 60`
  - `tau = 9.0`
  - `preprocess = pca`, `pca_dim = 3`
  - `sigma = [0.9934, 0.9928, 0.2008]`
  - `gap_2 = 0.7919`
  - `near_one_count_099 = 2`
  - `best_freq_mismatch = 0.0000`
  - `best_cluster_corr = 0.7245`

## 7. Performance comparisons

- Baseline notebook-like setting:
  - `identity+fourier`, `sample_stride=1`, `lag_steps=1`
  - `near_one_count_099 = 32`
  - `gap_2 ≈ 0`
  - frequency information exists, but the spectrum does not isolate the cluster count
- Best identity-only setting found:
  - `near_one_count_099 = 4`
  - still no cutoff at `2`
- Final selected setting:
  - `near_one_count_099 = 2`
  - exact macro-cluster frequency alignment
  - clear singular-value drop after `sigma_2`

## 8. Observed parameter sensitivities

- The dominant control variable is the physical lag `tau`, not merely `lag_steps` or `sample_stride`.
- Whitening at very short lag preserves too much one-step invertibility and keeps a large near-1 plateau.
- `PCA-3` is the smallest preprocessing dimension that still allows a genuine cutoff after `2`; higher dimensions reopen extra near-lossless directions.
- Pushing `tau` too far can sharpen the gap further, but tends to weaken direct macro-cluster correlation.

## 9. Failed or ineffective strategies

- `identity+fourier` under double whitening
- identity-only sweeps without PCA
- multi-trajectory averaging and centering as primary fixes
- `PCA-2`, because it forces the dimension rather than revealing it and loses one cluster frequency
- `PCA-4/5`, because they restore a 4-mode near-1 plateau

## 10. Final recommended configuration

- Use the tuned notebook configuration now written into [kuramoto_2group_gram_v2.ipynb](/Users/yangmingzhe/Desktop/code/github/koopCE/experiments/kuramoto/kuramoto_2group_gram_v2.ipynb):
  - `library = identity`
  - `sample_stride = 15`
  - `lag_steps = 60`
  - `preprocess = PCA(3)`
  - `rank = 2`
- Final notebook outputs after re-execution:
  - whitened singular values: `[0.9934, 0.9928, 0.2008]`
  - aligned cluster frequencies: `[0.2667, 0.2000]`
  - aligned macro frequencies: `[0.2667, 0.2000]`
  - cluster loading summary:
    - `y1`: stronger on `cluster_2`
    - `y2`: stronger on `cluster_1`

## 11. Possible future improvements

- Replace the grid over `sample_stride` and `lag_steps` with a direct scan over physical lag `tau`.
- Test noisy trajectories or multiple seeds to see whether the selected `PCA-3` setting remains stable.
- Explore a secondary rotation inside the extracted top-2 singular subspace to improve interpretability without changing the spectral cutoff.
