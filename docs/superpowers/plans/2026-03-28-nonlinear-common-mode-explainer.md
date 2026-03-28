# Nonlinear Common Mode Explainer Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Chinese-language notebook in `exp/toy_nonlinear` that explains the nonlinear common-mode toy case and visualizes the phase portraits in both `(z, r)` and `(x1, x2)` coordinates.

**Architecture:** Keep the notebook self-contained. Start with markdown-heavy teaching cells, then add compact simulation helpers and deterministic phase-field plotting, and finish with a small sanity-check validation section.

**Tech Stack:** Jupyter notebook, Python 3, NumPy, Matplotlib

---

## Chunk 1: Notebook Skeleton And Narrative

### Task 1: Create the notebook file and section outline

**Files:**
- Create: `exp/toy_nonlinear/nonlinear_common_mode_explainer.ipynb`

- [ ] Scaffold a tutorial notebook in `exp/toy_nonlinear`.
- [ ] Replace the template text with Chinese explanations for the case setup, coordinate change, phase portraits, and sanity checks.
- [ ] Keep the top-to-bottom flow teaching-first before moving into validation.

## Chunk 2: Simulation And Visualization

### Task 2: Add reusable simulation helpers

**Files:**
- Modify: `exp/toy_nonlinear/nonlinear_common_mode_explainer.ipynb`

- [ ] Define the toy-system parameters in a compact dataclass.
- [ ] Implement `zr_to_x`, `x_to_zr`, deterministic and noisy one-step updates, and a small trajectory simulator.
- [ ] Use fixed seeds so the notebook remains reproducible.

### Task 3: Plot both two-dimensional phase portraits

**Files:**
- Modify: `exp/toy_nonlinear/nonlinear_common_mode_explainer.ipynb`

- [ ] Plot the deterministic one-step drift in `(z, r)` with sampled trajectories overlaid.
- [ ] Plot the same system mapped back to `(x1, x2)`.
- [ ] Keep legends outside the axes when legends are needed and avoid covering trajectories or vector fields.

## Chunk 3: Lightweight Validation

### Task 4: Add sanity-check diagnostics

**Files:**
- Modify: `exp/toy_nonlinear/nonlinear_common_mode_explainer.ipynb`

- [ ] Compute a few simple correlations to show that `z` carries the dominant predictive structure.
- [ ] Simulate paired trajectories with the same `z0` but different `r0`.
- [ ] Summarize in markdown why this supports treating `z` as the macro variable.

### Task 5: Execute and verify

**Files:**
- Modify: `exp/toy_nonlinear/nonlinear_common_mode_explainer.ipynb`

- [ ] Run the notebook with `jupyter nbconvert --to notebook --execute`.
- [ ] Check that all cells finish and that the plots render without legend overlap.
- [ ] If execution fails, fix the notebook and rerun the full command before reporting completion.
