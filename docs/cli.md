# CLI Reference

## `swarmopt run`

Run LLM-guided search on a training script.

```bash
swarmopt run train.py
swarmopt run train.py --backend claude
swarmopt run train.py --backend none -n 50 --log results.jsonl
```

Your script must define:

- `train_fn(config)` — returns dict with at least `{"score": float}`
- `search_space` dict **or** `model` (nn.Module) — one of the two

Optional: `ml_context` string with domain knowledge for the LLM.

| Option | Default | Description |
|--------|---------|-------------|
| `--backend` | auto | `auto`, `claude`, `openai`, `qwen`, `none` |
| `--log` | search.jsonl | Log file path (supports resume) |
| `-b` / `--batch-size` | 3 | Configs proposed per LLM call |
| `-n` / `--max-evals` | unlimited | Stop after N experiments |
| `--device` | auto | `cuda`, `mps`, `cpu` |
| `--timeout` | 600 | Max seconds per experiment |

## `swarmopt inspect`

Show what swarmopt would search over for a given model.

```bash
swarmopt inspect train.py
```

```
Model: 11,689,512 parameters
  Activations: ReLU (9 layers)
  BatchNorm: 20 layers

Search space (5 params):
  activation: Categorical(['relu', 'gelu', 'silu', 'leaky_relu'])
  use_batchnorm: Categorical([True, False])
  lr: LogUniform(0.0001, 0.1)
  wd: LogUniform(1e-06, 0.01)
  optimizer: Categorical(['sgd', 'adam', 'adamw'])
```

Only works with scripts that define a `model` variable.

## `swarmopt results`

Analyze a search log.

```bash
swarmopt results search.jsonl
swarmopt results search.jsonl --top 20
```

Shows total experiments, top N results with configs, and convergence over time.
