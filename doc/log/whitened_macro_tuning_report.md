# Kuramoto Whitened Macro Tuning Report

## Goal

Tune the corrected double-whitening workflow so that the Kuramoto two-group example still reveals a macro dimension of `2`, and ensure the resulting macro spectra line up with the two cluster frequencies.

## What was wrong with the original setting

- The original notebook-style setting effectively used a very short physical lag and a redundant `identity+fourier` observable space.
- Under correct double whitening, this produced a large near-1 plateau rather than a clean cutoff at the cluster count.
- Baseline result:
  - `near_one_count_099 = 32`
  - `gap_2 ≈ 0`
  - dominant frequencies existed, but the spectrum itself did not tell us that the macro dimension should be `2`.

## Search summary

### 1. Lag and library only

- Restricting to `identity` and increasing lag/stride reduced the near-1 plateau from `32` to `4`.
- This was progress, but still not enough: whitening still preserved four nearly lossless directions instead of two.

### 2. PCA preprocessing

- `PCA-2` produced only two singular values, but both macro variables locked onto the same cluster frequency. That makes the result look good spectrally while failing the dynamical interpretation test.
- `PCA-4` and `PCA-5` brought back a 4-mode near-1 plateau.
- `PCA-3` was the first setting that consistently gave:
  - `near_one_count_099 = 2`
  - a visible drop after `sigma_2`
  - exact or near-exact macro-cluster frequency alignment, depending on lag

## Selected configuration

- `library = identity`
- `sample_stride = 15`
- `lag_steps = 60`
- `dt = 0.01`
- physical lag `tau = 9.0`
- `preprocess = PCA(3)`
- `rank = 2`

This is not the score-only winner. It was chosen because it gives a better balance between:

- a real cutoff after the second singular value
- exact macro-cluster frequency alignment
- stronger macro-cluster correlation than the largest-gap alternatives

## Final results

From the re-executed notebook:

- whitened singular values:
  - `sigma = [0.9934, 0.9928, 0.2008]`
- cutoff after the second mode:
  - `gap_2 = 0.7919`
- near-1 mode count:
  - `near_one_count_099 = 2`
- aligned cluster frequencies:
  - `[0.2667, 0.2000]`
- aligned macro frequencies:
  - `[0.2667, 0.2000]`

The recovered raw-space macro loadings are also cluster-structured:

- `y1` loads more strongly on `cluster_2`
- `y2` loads more strongly on `cluster_1`

This means the selected setting does not merely force the macro rank to `2`; it also recovers two dynamical macro coordinates whose spectra match the two synchronized groups.

## Interpretation

The tuning result suggests that the failure of the original whitened spectrum was not that double whitening is inappropriate. The real issue was that, at very short lag in a highly redundant observable space, the system retains too many nearly lossless micro-level directions. A mild data-driven compression with a longer physical lag removes those extra reversible directions and makes the two-group macro structure visible again.
