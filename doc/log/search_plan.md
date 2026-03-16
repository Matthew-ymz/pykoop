# Search Plan

## Objective

- Find a parameter setting for the Kuramoto two-group notebook such that the empirical whitened Koopman spectrum has a clear cutoff at rank 2 and the two macro signals align with the two cluster frequencies in the FFT plot.

## Constraints

- Keep the underlying Kuramoto system fixed (`N=10`, `n_clusters=2`, `K_intra=5`, `K_inter=0.11`, `noise=0`, `dt=0.01`).
- Prefer analysis-side changes over changing the physical system itself.
- Stay within repo-local scripts and outputs.
- Final winning configuration must be written back into the notebook and the notebook must execute end-to-end.

## Baseline

- Current notebook baseline:
- single trajectory
- `burn_in=1000`
- `library = identity + fourier(n_frequencies=1)`
- `lag_steps = 1`
- no feature centering
- expected failure mode: many leading singular values of the empirical whitened Koopman matrix are exactly 1, so rank 2 is not spectrally identifiable

## Planned Batches

| Batch | Hypothesis | Runs | Status |
|---|---|---:|---|
| 1 | Increasing lag and/or subsampling will collapse the near-1 plateau and expose a rank-2 gap. | 0 | pending |
| 2 | A leaner library than `identity + fourier` will suppress redundant near-isometric directions and improve rank-2 identifiability. | 0 | pending |
| 3 | Mild trajectory averaging and optional centering will stabilize the top-2 subspace without destroying cluster frequencies. | 0 | pending |
| 4 | Local refinement around the best lag/library region will maximize both spectral gap and macro frequency alignment. | 0 | pending |

## Stop Conditions

- A clear winner shows:
- `sigma_2 - sigma_3` substantially larger than baseline
- `sigma_1, sigma_2` remain close to 1
- both macro FFT peaks align with the two cluster reference frequencies
- the resulting top-2 macro coordinates are interpretable enough to keep in the notebook
- Or: no tested configuration can produce a stable rank-2 gap without materially changing the underlying system.
