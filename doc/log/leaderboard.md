# Leaderboard

| Rank | Run | Primary Metric | Secondary Metrics | Notes |
|---:|---|---:|---|---|
| 1 | `b4_pca3_lag60_s15` | `near1=2`, `gap_2=0.7919`, `freq_mismatch=0.0000` | `corr=0.7245`, `top=[0.9934, 0.9928, 0.2008]` | Chosen notebook configuration. Best balance between a real cutoff after 2 and cluster-aligned macro frequencies. |
| 2 | `b4_pca3_lag120_s15` | `near1=2`, `gap_2=0.9562`, `freq_mismatch=0.0000` | `corr=0.5424`, `top=[0.9965, 0.9962, 0.0400]` | Largest exact-match spectral gap, but cluster correlation is weaker than the selected config. |
| 3 | `b4_pca3_lag40_s20` | `near1=2`, `gap_2=0.8804`, `freq_mismatch=0.0000` | `corr=0.6328`, `top=[0.9924, 0.9921, 0.1117]` | Strong alternative with exact frequency alignment at physical lag `tau=8.0`. |
| 4 | `b2_id_lag50_s20_c1_nt3_wuniform` | `near1=4`, `gap_2≈0` | `corr=0.8945`, `top=[1.0000, 1.0000, 1.0000, 0.9997]` | Identity-only plus larger lag shrinks the plateau from 32 to 4, but still cannot isolate 2 macro modes. |
| 5 | `kuramoto_baseline_lag1_id_fourier1` | `near1=32`, `gap_2≈0` | `corr=0.6997`, `top≈[1,1,1,1,1]` | Original notebook-style setting. Exact frequency information exists, but no spectral cutoff at the cluster count. |
