# Experiments

The paper benchmark is deterministic Python. It does not call the multi-agent layer and does not need an API key.

Commands:

```bash
bash scripts/run_smoke.sh
bash scripts/run_airpassengers.sh
bash scripts/run_ablation_airpassengers.sh
bash scripts/run_aggregate.sh
```

## Primary method

`dynamic_hard_router` is origin-level dynamic hard routing.

At each outer origin it selects one expert and one window type, the pair with the highest compute-aware utility on inner rolling validation. That expert is refit on the outer training prefix and scored on the outer test block. The implementation is `evaluation/nested_evaluation.py`. Utility comes from `routing/validation_router.py`.

## Fixed expert baselines

Every expert that completes inner validation is also refit on the same outer training prefix and scored on the same outer test block. A fixed expert is `model__window_type`, for example `ets__expanding`. Experts that miss an outer origin are left out of the comparison table.

## Soft-routing ablation

`soft_overall_router` uses one definition: **full softmax, no top-k truncation**.

The weights are the `routing_weight` column from `compute_routing_scores`, taken over every validated expert. `routing/soft_ensemble.py` mixes their outer forecasts. This is `SOFT_ROUTING = "full_softmax"` in `evaluation/nested_evaluation.py`.

`experiments/compare_routing_strategies.py` is a legacy debug script. It still keeps the top 3 experts and renormalizes those weights. Do not use that script as the paper soft-routing ablation.

The state-aware ablation, `state_soft_router`, is separate. It uses the full per-horizon softmax from `build_state_horizon_routing_table`, also without top-k truncation.

## Horizon-specific hard-routing ablation

`horizon_hard_router` picks, for each lead time, the expert with the lowest mean inner absolute error at that lead. It is an ablation. In the current prototype it is noisier than origin-level hard routing.

## Oracle diagnostic

`analysis/oracle_analysis.py` scores the ex-post best fixed expert on each outer origin after the forecasts exist. It reports oracle MAE, selection accuracy, router regret, the oracle gap, and a horizon-specific oracle upper bound.

These numbers are post-hoc. They are not used to select an expert.

## Nested rolling evaluation

```text
outer train
    -> diagnostics and candidate filtering on the outer train
    -> inner rolling-origin validation on the outer train
    -> routing decision
    -> refit on the outer train
    -> score the outer test block
```

Run it with:

```bash
python -m experiments.run_benchmark --config configs/airpassengers.yaml
```

The dataset, horizon, windows, and seed come from the YAML file. The runner does not hard-code AirPassengers settings.

## No test leakage

The outer test block is not an input to diagnostics, candidate filtering, inner validation, utility, or expert selection. Oracle code reads completed predictions and does not call the router.

## Current AirPassengers result

The current AirPassengers nested result is preliminary. It is one monthly series. It is not evidence that dynamic hard routing is better in general. Broader multi-dataset evaluation is ongoing.

## Result fields

`predictions.csv`, `router_selections.csv`, and `runtime.csv` record:

- dataset
- config name
- seed
- horizon
- outer origin
- model
- window type
- validation MAE (`validation_mae`, also stored as `validation_loss`)
- runtime
- selected expert

`summary.csv` and `model_results.csv` aggregate MAE and RMSE by method.

## Legacy scripts

These files are debug scripts. Importing them does not run an experiment. They are not the paper benchmark:

- `experiments/nested_routing_evaluation.py`
- `experiments/compare_routing_strategies.py`
- `experiments/router_sensitivity.py`
- `compare_models.py`
- `analysis/router_oracle_diagnostic.py`
- `analysis/hard_soft_shrinkage.py`
