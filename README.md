# Social Norm Formation among LLM Agents

Code for studying social norm formation in LLM societies through common-pool
resource and job-market signaling games.

## Setup

```sh
uv sync --locked --extra llm --extra visualization --extra cpr-analysis
```

## Run simulations

```sh
uv run signaling abm --output output/signaling
uv run cpr abm --seed 42 --output output/fishing
```

Preview an LLM experiment:

```sh
uv run cpr llm --model gpt-4o --seed 42 --growth-rate 0.6 \
  --altruism-ratio 0.5 --output output/fishing-llm --dry-run
```

To run it, remove `--dry-run` and supply `--max-requests`. This is a cumulative
request limit, not a dollar budget. Credentials come from the environment or an
existing `.env`; `--resume` reuses saved answers.

## Analyze and plot results

```sh
uv run cpr analyze --data /path/to/cpr-results --output output/cpr-analysis
uv run cpr plot --results output/cpr-analysis --output output/cpr-figures
```

Use `signaling --help` or `cpr --help` for commands and options.

## Code structure

| Package | Simulation and API calls | Offline analysis and figures |
|---|---|---|
| `src/signaling` | `simulation/`, `providers.py`, `prompts.py` | `analysis/`, `visualization/` |
| `src/common_pool_resource` | `simulation/`, `providers.py`, `prompts.py` | `analysis/`, `visualization/` |

Both commands also support `python -m signaling` / `python -m common_pool_resource`.
