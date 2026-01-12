# Top Neuron Feature Ratio Comparison

Compare feature alignment ratios for Top-K neurons across different training strategies.

## Metrics

For each neuron $i$ with weight vector $w_i$:

- **M1 Ratio**: $\max_j |\langle w_i, M_{1,j} \rangle| / ||w_i||$
- **M2 Ratio**: $\max_j |\langle w_i, M_{2,j} \rangle| / ||w_i||$
- **Perp Ratio**: $\sqrt{1 - \text{M1}^2 - \text{M2}^2}$ (approximately)

These ratios are averaged over the Top-K neurons, ranked by either M1 or M2 similarity.

## Experiments

- **baseline**: C:OFF, R:OFF
- **curriculum_only**: C:ON, R:OFF (Denoise curriculum)
- **reg_only**: C:OFF, R:ON (Sparsity curriculum)
- **curriculum_reg**: C:ON, R:ON (Joint curriculum)

## Usage

```bash
python plot_top_neuron_ratio.py
```

## Outputs

- `outputs/top20_by_m1_ratio.png`: Feature ratios for Top-20 neurons ranked by M1
- `outputs/top20_by_m2_ratio.png`: Feature ratios for Top-20 neurons ranked by M2
