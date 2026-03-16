# Notes

## Observations

- The original `identity+fourier`, `lag=1` setting keeps a large near-1 plateau under double whitening: `near_one_count_099=32`, so the singular spectrum does not identify the 2-cluster macro dimension.
- Restricting the library to `identity` and increasing lag/stride removes part of the degeneracy, but the plateau bottoms out at 4 near-1 modes rather than 2.
- A light PCA precompression is the first change that reliably restores a cutoff at the cluster count under double whitening.
- `PCA-3` is the sweet spot. `PCA-2` forces rank 2 by construction, while `PCA-4/5` reintroduce a 4-mode near-1 plateau.
- Within the `PCA-3` family, the best configurations occur at larger physical lags. They preserve exact frequency alignment with the two cluster frequencies while keeping `sigma_3` well below `sigma_2`.

## Sensitivities

- The physical lag `tau = dt * sample_stride * lag_steps` matters more than `lag_steps` or `sample_stride` alone. Exact macro-frequency alignment starts appearing once `tau` is pushed beyond the short-lag regime.
- `PCA-3` at `tau` around `8-9` already gives a clear cutoff at 2. Increasing `tau` further can sharpen the gap, but may weaken direct macro-cluster correlation.
- The chosen notebook setting `sample_stride=15`, `lag_steps=60`, `tau=9.0` is a compromise point: exact frequency match, `near1=2`, and materially stronger cluster alignment than the score-only winner.

## Failed Ideas

- `identity+fourier` keeps too many nearly lossless modes, so whitening alone cannot reveal the cluster count.
- `identity` without PCA never reached `near1=2` in the explored lag/stride/multi-trajectory region.
- `PCA-2` gives excellent spectral numbers but collapses both macro frequencies to the same cluster frequency, so it is not a satisfactory explanation of the two-group dynamics.
- Higher PCA dimensions (`4` and `5`) consistently restore a 4-mode near-1 plateau and lose the desired cutoff after 2.
