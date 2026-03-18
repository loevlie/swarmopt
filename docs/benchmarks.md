# Benchmarks

## Setup

All methods get the **exact same evaluation budget** on the **same 14-parameter CNN search space**. This is a large, mixed search space with complex interactions — the kind where domain knowledge matters.

| Parameter | Type | Range |
|-----------|------|-------|
| n_blocks | int | 2–8 |
| base_channels | int | 16–128 |
| channel_growth | float | 1.0–2.5 |
| kernel_size | choice | 3, 5 |
| activation | choice | relu, gelu, leaky_relu, silu |
| use_residual | bool | — |
| use_batchnorm | bool | — |
| dropout | float | 0.0–0.5 |
| pool_every | int | 1–4 |
| pool_type | choice | max, avg |
| fc_hidden | int | 0–512 |
| lr | log float | 1e-4 to 0.1 |
| wd | log float | 1e-6 to 0.01 |
| optimizer | choice | sgd, adam, adamw |

**Task:** FashionMNIST (5k subset, 10 epochs per eval, ResNet-style CNN)

Grid search can't participate here — even 2 values per parameter gives 2^14 = 16,384 combinations, far exceeding any reasonable budget.

## Run the benchmark yourself

```bash
python examples/benchmark.py --n-evals 30
python examples/benchmark.py --n-evals 30 --skip-qwen  # skip local model
```

## Why this space favors LLM search

With 14 parameters, random search and Optuna TPE are essentially exploring blind. TPE builds a surrogate model from `(config, score)` pairs, but with 14 dimensions it needs many samples before the surrogate is useful.

The LLM starts with prior knowledge:

- *"GELU generally outperforms ReLU"* — immediately focuses on better activations
- *"Dropout hurts if the model is underfitting"* — reads the training curves to decide
- *"Residual connections help past 4-5 blocks"* — knows the interaction between depth and skip connections
- *"AdamW with lr ~1e-3 is a safe starting point"* — doesn't waste evals on lr=0.08 with Adam

Optuna would eventually learn these patterns from data, but in 30 evals it barely has enough signal.

## Claude vs Qwen

The benchmark also compares Claude API (Haiku) against a local Qwen 2.5 1.5B model running on CPU:

- **Claude**: faster responses (~1s), stronger reasoning, costs ~$0.01 per 30-eval run
- **Qwen**: free, offline, ~3s per call, but weaker at complex JSON generation and may fall back to random more often

Both should outperform Optuna and random on this search space, with Claude likely producing more consistent results due to stronger instruction following.
