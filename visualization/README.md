# Visualization Module

Scripts for analyzing and comparing experiment results.

## File Structure

```
visualization/
├── README.md                  # This file
├── plot_macro_comparison.py   # Macro feature ratio comparison
└── figures/                   # Output figures
    └── macro_feature_ratio_comparison.png
```

## Scripts

### plot_macro_comparison.py

Compare `curriculum_only` (baseline) vs `curriculum_reg` feature learning performance.

**Core Metrics**:
```
M1 Feature Ratio = ||W @ M1||² / ||W||²
M2 Feature Ratio = ||W @ M2||² / ||W||²
```

These metrics measure what fraction of the learned weights W align with M1 (signal features) vs M2 (noise features).

**Usage**:
```bash
cd visualization
python plot_macro_comparison.py
```

**Output**:
- `figures/macro_feature_ratio_comparison.png` - Comparison plot
- Terminal output with final feature ratio values

## Expected Results

If Group L1 regularization is effective:
- `curriculum_reg` should have **higher M1 ratio** (better signal feature learning)
- `curriculum_reg` should have **lower M2 ratio** (better noise suppression)
- The two curves should diverge over training

## Requirements

- Python 3.7+
- matplotlib
- numpy
- Experiment results in `../results/`:
  - `results_curriculum_only.pkl`
  - `results_curriculum_reg.pkl`

---

[Back to main README](../README.md)
