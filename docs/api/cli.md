# Command-line entry points

The three console scripts and the functions behind them. Every flag is listed on the [CLI reference](../reference/cli.md); this page is the Python side, for calling the same code from a script.

| Script | Function |
|---|---|
| `pinnrl-train` | `pinnrl.training.train.main` |
| `pinnrl-benchmark` | `pinnrl.benchmarks.cli.main` |
| `pinnrl-dashboard` | `pinnrl.main.main` |

## `pinnrl-train`

::: training.train.main

## `pinnrl-benchmark`

::: benchmarks.cli.main

::: benchmarks.cli.build_parser

## `pinnrl-dashboard`

::: main.main

::: main.parse_args

::: main.run_dashboard

::: main.bind_server
