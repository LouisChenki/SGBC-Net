# SGBC-Net

Research code for *State-Conditioned Correction of Aggregated Neighborhood Features for Multi-Step Traffic Flow Forecasting*. Kaiqi Chen, Jia Duan, Min Deng and Wang Zhang.

SGBC-Net predicts the next twelve five-minute freeway flow observations from the preceding twelve observations of flow, occupancy and speed. It learns feature-wise corrections after spatial aggregation and retains detector-level features when adding separately transformed neighborhood context.

## Run the released model

The two PyTorch checkpoints correspond to the PeMS04 and PeMS08 results in the manuscript. The implementation in `sgbc.py` builds the seven-block network and loads the checkpoint parameters strictly. No training or external model repository is required for inference. Install Python 3.10 or later, PyTorch 2.5 or later and NumPy. For CUDA, install the PyTorch build appropriate to the device.

Download the original flow array and distance edge list from the [ASTGCN data distribution](https://github.com/Davidham3/ASTGCN/tree/master/data). Preserve detector order. Each input directory contains `pems04.npz` or `pems08.npz` and its corresponding `distance.csv`.

```sh
python evaluate.py PeMS04 data/PEMS04 --device cuda
python evaluate.py PeMS08 data/PEMS08 --device cuda
```

Use `--device cpu` for CPU inference or `--batch 8` to reduce memory use. Output is CSV, with the pooled result followed by the twelve lead-specific results. MAE and RMSE are in original flow units and MAPE is a percentage.

The evaluation uses chronological 60/20/20 partitions of windows, preserves the original training-flow scaler for each checkpoint, aggregates with the released weighted road graph and masks zero targets in original units. Calendar inputs are indices derived from the position in the series. Consecutive partitions share eleven target time positions at their boundary because windows were formed before splitting. The comparison in the paper uses the same partitions for every model. Full-precision archived test statistics are in `reproduction/`.

## Rebuild the experimental displays

Install NumPy and Matplotlib, then run from this repository:

```sh
python source/plot_comparisons.py
python source/rebuild_displays.py
```

| Item | Inputs and interpretation |
|---|---|
| Table I(a), I(b) | `comparison.csv` holds pooled test errors for the 13 models. `horizon_records.csv` holds all 312 model–dataset–lead records. The second command selects step 12, converts MAPE to percent and writes `reproduction/table_I.csv`. |
| Figure 3 | `inputs/relative_horizon_gains.csv` and `inputs/horizons.csv`. Plot error reductions against STID, AGCRN and STLAformer by lead. |
| Figure 4 | `inputs/paired_nodes.csv` and `inputs/detectors.csv`. Match detector indices within each dataset, then plot MAE differences and degree-group medians. |
| Figure 5 | `distance_counts.csv`. Compare the percentage of selected non-self sources and available sources in each graph-distance class. Each receiver contributes five selections. |
| Table II | `table_II.csv` holds the four archived PeMS04 configurations, original-unit errors and change in MAE relative to the complete configuration. |
| Figure 6 | `inputs/scenario_effects.csv` and `inputs/variation.csv`. Compare baseline-relative MAE across quartiles of recent flow variation using paired 24-hour block intervals. |
| Figure 7 | `correction_distributions.csv` contains detector and window quantiles for every correction operator. `gates.csv` contains paired block intervals for fixed-gate interventions at frozen weights. |

The scripts write PDF and PNG files under `figures/`. The summaries describe test performance of the archived checkpoints. The source column in the horizon records identifies the original evaluation run. The author-supplied review supplement also contains STLAformer reproduction and supplementary sensitivity analyses.
