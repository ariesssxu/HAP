# RSI Environment–Harness Co-evolution

This repository explores recursive self-improvement (RSI) as a controlled
co-evolution problem. Instead of updating model weights, it evolves two
explicit, auditable objects around a fixed student:

- an **environment specification** that creates useful learning pressure;
- a **harness** containing the student's prompt, memory, skills, and workflow.

The research loop is:

```text
history ──> environment designer ──> validated EnvironmentSpec
                                            │
                                            v
                                      deterministic compiler
                                            │
                                            v
student + harness ─────────────────────> rollout result
                                            │
                                            v
                                      failure diagnosis
                                            │
                                            v
                                      harness evolution
```

The core safety boundary is: **a model proposes data; trusted code validates
and compiles it**. An LLM does not directly write or execute environment code.

## Quick start

The project targets Python 3.10 or newer. The built-in toy experiment uses only
the standard library:

```sh
python -m rsi.cli --config rsi/configs/smoke.json
python -m unittest discover -s rsi/tests -v
```

Install the optional MiniGrid benchmark and run its scenario curriculum:

```sh
python -m pip install -e '.[minigrid]'
python -m rsi.cli \
  --domain minigrid \
  --config rsi/configs/minigrid.json \
  --policy-backend my_policy:Policy
```

The benchmark spans navigation, door/key planning, dynamic obstacles, lava,
multi-room exploration, memory, object manipulation, and unlock tasks. The
curriculum models evidence separately for every scenario × difficulty arm.

For long-horizon language actions, install the optional ALFWorld adapter:

```sh
python -m pip install -e '.[alfworld]'
alfworld-download
python -m rsi.cli --domain alfworld \
  --alfworld-config /path/to/base_config.yaml \
  --config rsi/configs/alfworld.json \
  --policy-backend my_policy:Policy
```

For an editable install and the `rsi` command:

```sh
python -m pip install -e .
rsi --config rsi/configs/toy.json
```

Results are written as one JSONL trace per condition plus `summary.json` under
the configured output directory.

## Controlled comparison

Every suite can run the same 2 × 2 experiment:

| Condition | Environment | Harness | Isolates |
|---|---|---|---|
| `fixed_env__fixed_harness` | fixed | fixed | static control |
| `evolving_env__fixed_harness` | evolving | fixed | curriculum effect |
| `fixed_env__evolving_harness` | fixed | evolving | harness repair effect |
| `evolving_env__evolving_harness` | evolving | evolving | interaction effect |

All conditions receive isolated designer and harness state. Runs are seeded,
and every proposal, rollout, diagnosis, mutation candidate, and validation
decision is recorded.

The adaptive designer uses a compact competence-frontier acquisition function:
scores are shrunk toward a monotone difficulty prior, then balanced by target
learnability, epistemic uncertainty, and recent learning or forgetting. Harness
edits are not applied directly. Each candidate is compared with its incumbent
on identical validation seeds and accepted only when its gain exceeds a minimum
improvement threshold plus a complexity penalty. Repeatable `--regression-spec`
tasks add an anti-forgetting gate. Structured skills retain procedures,
triggers, provenance, and validation statistics; `--retention both` compares
persistent experience against a freshly reset Harness.

## Repository layout

```text
rsi/
├── specs.py          # environment DSL, validation, compiler registry
├── designers.py      # fixed, random, adaptive, and LLM designers
├── toy_env.py        # deterministic reference compiler/environment
├── minigrid_env.py   # validated multi-scenario MiniGrid compiler
├── alfworld_env.py   # long-horizon text-action benchmark adapter
├── policy.py         # observation-conditioned student policy interface
├── harness.py        # prompt, memory, skills, and workflow state
├── diagnosis.py      # oracle, rule-based, and LLM diagnosis
├── evolver.py        # bounded and auditable harness mutation
├── experiment.py     # co-evolution loop and factorial baselines
├── results.py        # typed rollout and diagnosis records
├── llm.py            # provider-neutral text backend interface
├── cli.py            # command-line composition root
├── configs/          # reproducible run configurations
├── examples/         # validated environment specifications
└── tests/            # zero-dependency tests
```

See [rsi/README.md](rsi/README.md) for experiment design, metrics, extension
points, and the staged research roadmap.

## Extending the system

Add a domain by implementing an `EnvironmentCompiler` and registering it in
the CLI composition root. Add an LLM through an object exposing
`complete(prompt: str) -> str`, then pass it as `--llm-backend module:object`.
Keep provider credentials, retry policy, caching, and cost controls inside that
backend, and never commit secrets.

The old HAP paper experiments are intentionally absent from this branch. They
remain available on the `main` branch; this branch is scoped only to the RSI
research direction.
