# Single-Basin PySINDy Analysis Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing toy nonlinear notebook with a single-basin example, observation-function selection, PySINDy fitting, and estimation of a global observation-space Jacobian and residual covariance matrix.

**Architecture:** Move the core simulation and estimation logic into a small Python helper module under `exp/toy_nonlinear/` so it can be tested directly. Keep the notebook focused on explanation, plots, model comparison, and reported matrices.

**Tech Stack:** Python 3, NumPy, Matplotlib, PySINDy, pytest, Jupyter notebook

---

## Chunk 1: Analysis Helpers

### Task 1: Define the single-basin system and observable maps

**Files:**
- Create: `exp/toy_nonlinear/nonlinear_common_mode_analysis.py`
- Test: `tests/test_nonlinear_common_mode_analysis.py`

- [ ] **Step 1: Write failing tests**
- [ ] **Step 2: Run `pytest tests/test_nonlinear_common_mode_analysis.py -v` and confirm failure**
- [ ] **Step 3: Implement a single-basin simulator plus identity and polynomial observable builders**
- [ ] **Step 4: Re-run the targeted tests and confirm they pass**

### Task 2: Add PySINDy-based global linear proxy estimation

**Files:**
- Modify: `exp/toy_nonlinear/nonlinear_common_mode_analysis.py`
- Modify: `tests/test_nonlinear_common_mode_analysis.py`

- [ ] **Step 1: Add failing tests for residual covariance symmetry/PSD and for selecting the polynomial observable mode**
- [ ] **Step 2: Run `pytest tests/test_nonlinear_common_mode_analysis.py -v` and confirm failure**
- [ ] **Step 3: Implement observation-space fitting utilities that estimate a global linear Jacobian proxy and residual covariance**
- [ ] **Step 4: Re-run the targeted tests and confirm they pass**

## Chunk 2: Notebook Integration

### Task 3: Extend the notebook narrative and analysis cells

**Files:**
- Modify: `exp/toy_nonlinear/nonlinear_common_mode_explainer.ipynb`

- [ ] Add markdown explaining why the example is redesigned to have a single basin.
- [ ] Compare `g_id` and `g_poly` in the notebook and motivate the selected observation function.
- [ ] Report the fitted global Jacobian matrix and covariance matrix in the notebook output.
- [ ] Add one or two compact plots comparing fit quality in observation space.

## Chunk 3: Verification

### Task 4: Run automated and notebook verification

**Files:**
- Modify: `exp/toy_nonlinear/nonlinear_common_mode_explainer.ipynb`
- Modify: `exp/toy_nonlinear/nonlinear_common_mode_analysis.py`
- Modify: `tests/test_nonlinear_common_mode_analysis.py`

- [ ] Run `pytest tests/test_nonlinear_common_mode_analysis.py -v`
- [ ] Run `jupyter nbconvert --to notebook --execute exp/toy_nonlinear/nonlinear_common_mode_explainer.ipynb --output nonlinear_common_mode_explainer.executed.ipynb`
- [ ] Inspect the reported matrices and comparison metrics before reporting completion
