# Agentic Time-Series Forecasting

This repository has two separate layers.

## A. Forecasting research engine

```text
Time series
    ↓
Diagnostics
    ↓
Candidate expert filtering
    ↓
Inner rolling-origin validation
    ↓
Compute-aware utility
    ↓
Dynamic hard expert selection
    ↓
Outer OOS forecast
    ↓
Benchmark / ablation / oracle diagnostics
```

The research engine is deterministic Python. It does not call an LLM and does not need an API key. Diagnostics, candidate filtering, rolling validation, routing utility, expert selection, forecasts, and metrics stay in `diagnostics/`, `models/`, `evaluation/`, and `routing/`.

## B. Agent orchestration layer

```text
User
  ↓
Planner Agent
  ↓
Forecasting Agent
  ↓
Critic Agent
  ↓
Final Answer
```

The agent layer in `agent/` is a user-facing interface. Qwen plans, selects the forecasting tool, and reviews the tool output. It does not generate numerical forecasts or routing weights. Benchmark numbers are never taken from the agent.

`agent/graph.py` runs that sequence in LangGraph. FastAPI and the FastMCP server both call `services/forecasting_service.py`. The research scripts do not call Qwen.

## System Architecture

Application-facing path:

```text
Web / Application
      ↓
REST API
      ↓
FastAPI
      ↓
LangGraph
Planner → Forecasting → Critic
      ↓
Forecasting Service
      ↓
Model Zoo / Validation / Router
```

Agent-facing path:

```text
Cursor / Codex / Qwen
      ↓
MCP
      ↓
FastMCP
      ↓
same Forecasting Service
```

- LangChain provides the Qwen chat model, prompt templates, tool definitions, and message objects.
- LangGraph runs the stateful planner, forecasting, and critic workflow. The critic may send the forecasting node back once.
- REST API is the HTTP interface used by an application.
- FastAPI is the Python framework that implements those endpoints.
- MCP is the tool protocol used by an agent client.
- FastMCP is the Python framework that implements the MCP server.

The LLM does not calculate forecasts, validation scores, routing weights, or metrics. Those values come from the research engine.

## Quick Start

1. Create the environment and install the existing forecasting stack, then the application dependencies:

```bash
pip install -r requirements.txt
```

2. Copy the environment template and fill in the Qwen settings. Do not commit `.env`.

```bash
cp .env.example .env
```

3. Run the CLI. With no arguments this is the AirPassengers demo:

```bash
python -m agent.multi_agent
python -m agent.multi_agent --dataset airpassengers --horizon 12
python -m agent.multi_agent --csv data/AirPassengers.csv --target "#Passengers" --horizon 12
```

4. Run the API:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

5. Test a REST endpoint:

```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"values": [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140], "horizon": 12, "target_name": "passengers"}'
```

6. Run the MCP server:

```bash
python -m mcp_server.server
```

## Research Question

Can validation-driven dynamic expert routing across classical, deep-learning, and time-series foundation models outperform fixed forecasting experts under rolling out-of-sample evaluation?

## Method

At each outer origin the evaluator cuts a training prefix and a future test block. Diagnostics and candidate filtering use the training prefix. Each candidate is scored with inner rolling-origin validation on that prefix. `routing/validation_router.py` turns inner MAE and runtime into a compute-aware utility. The dynamic hard router selects the highest-utility expert, refits it on the outer training prefix, and is scored only on the outer test block.

The outer test block is not used to choose the expert.

## Dynamic Expert Routing

The primary method is origin-level dynamic hard routing: one expert, and one training-window type, for the whole horizon at that origin.

The current prototype evidence suggests that this hard rule is more robust than soft weighting, and that horizon-specific hard routing is noisier. Soft overall routing, horizon-specific hard routing, state-aware soft routing, removing the compute penalty, and removing diagnostic candidate filtering are ablations. They are not the primary method.

The soft-routing ablation is a full softmax. `routing_weight` from `compute_routing_scores` is used for every validated expert. There is no top-k truncation. The constant is `SOFT_ROUTING = "full_softmax"` in `evaluation/nested_evaluation.py`. The legacy script `experiments/compare_routing_strategies.py` still uses a top-3 truncation and is not this ablation.

State-aware routing is a separate ablation. It uses the full per-horizon softmax from the existing state-horizon router, also without top-k truncation. Its bandwidth and recency settings match the legacy comparison script.

## Nested Rolling Evaluation

`evaluation/nested_evaluation.py` implements the protocol that was previously embedded in `experiments/run_benchmark.py`:

```text
outer train
    ↓
inner rolling CV
    ↓
candidate expert evaluation
    ↓
routing decision
    ↓
refit selected expert
    ↓
outer test forecast
```

Fixed experts are scored on the same outer test blocks. An expert that does not finish every origin is excluded from the comparison table.

## Model Zoo

The benchmark can score the enabled models in `models/registry.py`:

- seasonal_naive
- auto_arima
- ets
- xgboost
- dlinear
- patchtst
- timesnet
- itransformer, when the series is multivariate
- chronos2
- timesfm3
- moirai2
- dynamic_hard_router

`dynamic_hard_router` is the routing method. The others are fixed experts. Which experts are actually fitted depends on the candidate router, the compute budget, and `model_allowlist`.

## Multi-Agent Interface

`agent/planner_agent.py`, `agent/forecasting_agent.py`, `agent/critic_agent.py`, and `agent/graph.py` orchestrate the same deterministic tools for a short AirPassengers demo. `python -m agent.multi_agent` is the CLI. These modules are not part of the benchmark, ablation, oracle, or table pipeline. Add `--verbose` to show the long report and framework logs.

## Reproducibility

Each YAML config has one `seed`. `utils/reproducibility.py` applies it to Python, NumPy, and PyTorch before fitting. This does not make every GPU or MPS kernel bit-identical.

A run writes a new directory:

```text
results/<experiment_name>/<timestamp>_<config_hash>/
    config.yaml
    predictions.csv
    model_results.csv
    router_selections.csv
    summary.csv
    runtime.csv
```

Existing result directories are not overwritten. Point-level result rows record the dataset, config name, seed, horizon, outer origin, model, window type, validation MAE, runtime, and selected expert.

## Running Experiments

Protocol notes are in `EXPERIMENTS.md`.

```bash
bash scripts/run_smoke.sh
bash scripts/run_airpassengers.sh
bash scripts/run_ablation_airpassengers.sh
bash scripts/run_aggregate.sh
```

The full AirPassengers benchmark is:

```bash
python -m experiments.run_benchmark --config configs/airpassengers.yaml
```

Previous CLI flags still work:

```bash
python -m experiments.run_benchmark --dataset airpassengers --outer-windows 2 --inner-windows 2 --fast
```

After a run:

```bash
python -m analysis.oracle_analysis --results <experiment_dir>
python -m analysis.significance_tests --results <experiment_dir>
python -m analysis.aggregate_results --results results/benchmark
python -m analysis.make_tables --benchmark <experiment_dir> --ablation <ablation_dir>
python -m analysis.make_figures --results <experiment_dir>
```

Oracle outputs are post-hoc diagnostics. They are not used for routing. Add `--verbose` to a benchmark command for per-model logs. The default console report is the comparison table and the selected experts.

## Current Results

In the current AirPassengers prototype experiment, dynamic hard routing achieved lower MAE than every fixed expert. Broader multi-dataset evaluation is ongoing.

That statement refers to the existing nested out-of-sample prototype. It is not a general claim. Horizon-specific hard routing was noisier in that prototype, and origin-level hard routing was more robust there than soft weighting.

## Limitations

- The reported comparison is one monthly series, AirPassengers.
- The number of outer origins is small, so a significance test on this series should not be treated as a general result.
- ETTh1, Electricity, and Traffic are registered but not loaded yet.
- Fast model settings are for smoke tests. The AirPassengers YAML keeps `fast_model_config: false` for the full run.
- Foundation-model results depend on the installed checkpoints and can move with library versions.
- A global seed does not remove all hardware non-determinism.

## Roadmap

- Add loaders for ETTh1, Electricity, and Traffic through `benchmarks/registry.py`.
- Repeat the same nested protocol on those datasets before comparing average rank.
- Keep the dynamic hard router as the primary method and report the other routers as ablations.
- Interpret Wilcoxon tests and bootstrap intervals only when the number of outer origins is large enough for that claim.
