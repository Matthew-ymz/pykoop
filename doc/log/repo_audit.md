# Repository Audit

## Objective

- Tune `experiments/kuramoto/kuramoto_2group_gram_v2.ipynb` so that, under double whitening, the empirical whitened Koopman spectrum shows a clear truncation at the true macro dimension (`n_clusters = 2`) and the resulting two macro time series align with the two cluster frequencies in the FFT analysis.

## Training Entry Points

- Main interactive entry point: [experiments/kuramoto/kuramoto_2group_gram_v2.ipynb](/Users/yangmingzhe/Desktop/code/github/koopCE/experiments/kuramoto/kuramoto_2group_gram_v2.ipynb)
- Data generation lives in [data_generators/data_func.py](/Users/yangmingzhe/Desktop/code/github/koopCE/data_generators/data_func.py), mainly `generate_kuramoto_cluster_data_sin_cos` and `plot_clustered_kuramoto`.
- Koopman fitting and whitening helpers live in [tools/tools.py](/Users/yangmingzhe/Desktop/code/github/koopCE/tools/tools.py), mainly `compute_transition_covariances`, `fit_data_koopman_operator`, and `whiten_operator_matrix`.

## Evaluation Pipeline

- Generate two-group Kuramoto trajectories in sin-cos embedding.
- Choose a lifted feature library and optional temporal subsampling / lag.
- Fit a lifted SINDy model for comparison, but evaluate macro structure primarily with the empirical whitened Koopman matrix `K_bar`.
- Compute SVD of `K_bar`, extract `rank = 2` left singular subspace, and form macro coordinates with `C00^{-1/2} U_r`.
- Evaluate:
- `spectral_gap_2 = sigma_2 - sigma_3`
- whether `sigma_1, sigma_2` remain near 1
- FFT peak alignment between the two macro signals and the two cluster reference signals
- cluster-level interpretability of the macro coordinates as a secondary diagnostic

## Configuration Surfaces

- Data slicing: burn-in / transient cut, total duration used, sample stride
- Pairing lag for whitening: `lag_steps`
- Library family: `identity`, `fourier`, `identity+fourier`, `identity+polynomial`
- Fourier depth / polynomial degree
- Optional feature centering before covariance estimation
- Covariance weighting mode (`uniform`, `traj`) if multiple trajectories are used
- Number of trajectories used for empirical covariance estimation
- SINDy optimizer and regularization for the model-comparison branch

## Tunable Parameters

- Highest-impact first-pass knobs:
- `lag_steps`
- `sample_stride`
- `library_kind`
- `fourier_n_frequencies`
- `center_features`
- `n_trajectories`
- `burn_in`
- Lower-priority knobs:
- SINDy optimizer choice and regularization
- covariance ridge / epsilon stabilization

## Bottlenecks And Risks

- `lag=1` with a rich redundant lift produces a large near-isometric plateau (`~32` singular values at 1), so the top-2 subspace is not identifiable.
- Many equal singular values imply non-unique singular vectors, so rank-2 macro coordinates become unstable unless the degeneracy is broken by lag, averaging, or a leaner observable set.
- FFT alignment alone can be misleading if both macro variables collapse onto the same cluster frequency.
- The notebook mixes exploratory plotting with analysis logic, so reproducible tuning requires a script that mirrors the notebook computations.
