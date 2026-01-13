# Silhouette Score Comparison

Compare Silhouette Scores of hidden layer features for M1/M2 classification across different training strategies.

## Metrics

**Silhouette Score** measures how well the model separates different feature classes in hidden space:
- Score close to 1: Good clustering (features well separated)
- Score close to 0: Overlapping clusters
- Score close to -1: Misclassified samples

## Data Generation

- **M1 Test Data**: One-hot z1 (single M1 feature per sample), M2 signal zeroed out
- **M2 Test Data**: One-hot z2 (single M2 feature per sample), full signal with both M1 and M2

## Feature Extraction

- **M1**: $h = Wx$ (linear, no ReLU)
- **M2**: $h = \text{ReLU}(Wx + b)$

## Experiments

- **baseline**: C:OFF, R:OFF
- **curriculum_only**: C:ON, R:OFF (Denoise curriculum)
- **reg_only**: C:OFF, R:ON (Sparsity curriculum)
- **curriculum_reg**: C:ON, R:ON (Joint curriculum)

## Usage

```bash
python plot_silhouette_score.py
```

## Outputs

- `outputs/silhouette_score.png`: Side-by-side comparison of M1 and M2 Silhouette Scores
