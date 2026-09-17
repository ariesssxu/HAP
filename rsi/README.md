# Environment and Harness Co-evolution Prototype

This package is a research scaffold for studying how a system can design both
an executable environment and the harness around a student agent.

The core toy path uses only the Python standard library, is deterministic under
a seed, and is intended to make causal claims and implementation boundaries
easy to inspect.

## Research question

The project asks:

> Can a designer generate valid environments that create useful learning
> pressure while a second mechanism improves the prompt, memory, skills, and
> workflow used by a fixed student model?

At round `t`, the loop is:

```text
(history, E[t-1]) -> environment designer -> EnvironmentSpec
EnvironmentSpec -> validator -> deterministic compiler -> executable E[t]
(E[t], H[t]) -> rollout -> result -> diagnosis
(H[t], diagnosis) -> harness evolver -> H[t+1]
```

Keeping model weights fixed at first makes attribution easier. Improvements can
then be assigned to environment selection, harness mutation, or their
interaction instead of being confounded with weight updates.

## Safety and reproducibility boundary

An LLM never writes or executes environment Python in the reference design. It
proposes a JSON `EnvironmentSpec`. `rsi/specs.py` rejects unknown fields and
invalid bounds, after which a registered deterministic compiler constructs the
runtime. This implements the rule:

> LLM proposes; validator checks; deterministic code compiles.

The DSL controls domain, goal, difficulty, horizon, seed, distractors, partial
observability, required skills, hidden rules, action space, and JSON metadata.
See `examples/toy_spec.json` for a complete example.

## Repository map

```text
rsi/
├── README.md                 # this guide
├── cli.py                    # command-line entry point
├── run.py                    # alias for the CLI
├── specs.py                  # JSON DSL, validation, compiler registry
├── toy_env.py                # deterministic, no-API local environment
├── minigrid_env.py           # allow-listed multi-scenario MiniGrid runtime
├── alfworld_env.py            # optional text-action household benchmark
├── policy.py                 # observation-conditioned student policies
├── designers.py              # fixed/random/difficulty/progress/LLM designers
├── harness.py                # prompt + memory + skills + workflow
├── diagnosis.py              # oracle/rule-based/LLM diagnosers
├── evolver.py                # bounded, auditable harness mutations
├── experiment.py             # co-evolution loop and four baselines
├── results.py                # rollout and diagnosis records
├── llm.py                    # provider-neutral complete(prompt) interface
├── configs/
│   ├── smoke.json
│   ├── toy.json
│   ├── minigrid.json
│   └── alfworld.json
├── examples/
│   ├── toy_spec.json
│   ├── minigrid_door_key.json
│   └── alfworld_pick_and_place.json
└── tests/
    └── test_smoke.py
```

## Quick start: no API and no installation

From the repository root, use Python 3.10 or newer:

```sh
python -m rsi.cli --config rsi/configs/smoke.json
python -m unittest discover -s rsi/tests -v
```

Run the longer toy configuration:

```sh
python -m rsi.cli --config rsi/configs/toy.json
```

Results are written to the configured output directory as one JSONL file per
baseline plus `summary.json`. Generated `results/` directories are already
ignored by the repository.

Useful command variations:

```sh
# Print the exact baseline names
python -m rsi.cli --list-baselines

# Use the random designer and observable-only rule-based diagnosis
python -m rsi.cli --designer random --diagnoser rule_based --rounds 20

# Run one condition and one hand-written environment spec
python -m rsi.cli \
  --spec rsi/examples/toy_spec.json \
  --baseline evolving_env__evolving_harness

# Equivalent entry point for students following older notes
python -m rsi.run --config rsi/configs/smoke.json
```

No installation or third-party dependency is needed for these commands.

## MiniGrid benchmark

Install the maintained optional dependency:

```sh
python -m pip install -e '.[minigrid]'
python -m rsi.cli --domain minigrid --config rsi/configs/minigrid.json \
  --policy-backend my_policy:Policy
```

The registered scenario families include Empty, FourRooms, MultiRoom,
DoorKey, KeyCorridor, DynamicObstacles, LavaCrossing, Memory, Fetch, PutNear,
RedBlueDoors, UnlockPickup, and BlockedUnlockPickup. Each validated spec maps
to an allow-listed Gymnasium environment ID; arbitrary environment loading is
not accepted.

A student policy implements `reset()` and
`act(observation, harness, actions, step) -> action`. Unlike the toy workflow,
this interface chooses an action after every partial observation. The default
workflow policy only checks integration plumbing; meaningful experiments must
provide a trained, planning, or language-model policy. Use `--llm-policy` with
`--llm-backend` for the built-in language-model adapter.

## ALFWorld benchmark

The optional text adapter covers ALFWorld's six task families: pick-and-place,
look-at-in-light, clean/heat/cool-then-place, and pick-two-then-place. It uses
the official admissible-command interface and supports train, in-distribution,
and out-of-distribution evaluation splits.

```sh
python -m pip install -e '.[alfworld]'
alfworld-download
python -m rsi.cli --domain alfworld \
  --alfworld-config /path/to/base_config.yaml \
  --config rsi/configs/alfworld.json \
  --retention both \
  --policy-backend my_policy:Policy
```

## The four required baselines

Every suite runs a 2 × 2 factorial comparison:

| Baseline | Environment | Harness | What it isolates |
|---|---|---|---|
| `fixed_env__fixed_harness` | fixed | fixed | static control |
| `evolving_env__fixed_harness` | evolving | fixed | environment curriculum alone |
| `fixed_env__evolving_harness` | fixed | evolving | harness repair alone |
| `evolving_env__evolving_harness` | evolving | evolving | the proposed interaction |

All conditions start with a fresh harness. Stateful designers are copied so one
condition cannot leak curriculum state into another. Use the same seeds and
evaluation set across conditions when reporting results.

## Environment designers

- `FixedDesigner` returns one validated spec and acts as the control.
- `RandomDesigner` samples a seeded difficulty level.
- `DifficultyDesigner` moves up after a score at or above its target and down
  after a lower score.
- `LearningProgressDesigner` uses a shrinkage estimate and an
  uncertainty-aware acquisition score to target the competence frontier while
  still valuing recent learning or forgetting.
- `LLMDesigner` requests a JSON spec from any configured text backend. The same
  strict validator is applied to its output. A fallback spec makes classroom
  demos robust to malformed responses.

The generated difficulty ladder changes horizon, distractors, partial
observability, required skills, and hidden constraints—not just one `easy/hard`
flag.

## Harness and diagnosis

`Harness` contains four first-class mutable components:

- `prompt`: global behavioral instructions;
- `memory`: learned facts or hidden constraints;
- `skills`: declared capabilities/tools plus a structured procedure library;
- `workflow`: ordered actions taken during a rollout.

The oracle diagnoser reads privileged failure signals and provides an upper
bound for correct attribution. The rule-based diagnoser sees the public spec and
observable trace. The LLM diagnoser is a provider-neutral stub. The reference
evolver only makes bounded, deduplicated edits and increments a revision number,
which keeps every change auditable in the JSONL logs. The experiment evaluates
each candidate and incumbent on paired seeds, penalizes added complexity, and
rejects changes without sufficient validation gain.

Each stored procedure records its trigger, instruction, provenance, and online
success/failure counts. Retrieval ranks task relevance and empirical
reliability without requiring an embedding service. Repeatable
`--regression-spec path.json` arguments protect held-out tasks from regression.
Use `--retention both` to compare persistent experience with a Harness reset at
every round, isolating whether later gains actually depend on retained state.

## Connecting an LLM

Create a class or object with this interface in an importable module:

```python
class MyBackend:
    def complete(self, prompt: str) -> str:
        # Call a local model or provider here and return its text response.
        ...
```

Then run:

```sh
python -m rsi.cli \
  --designer llm \
  --diagnoser llm \
  --llm-backend my_backend:MyBackend
```

Provider credentials, retry policy, caching, cost limits, and structured-output
handling belong inside the backend. Do not commit secrets. For experiments,
record model name, model version, temperature, prompt, retry count, token usage,
and raw proposal before validation.

## Metrics and experimental protocol

Each episode record includes the full environment spec, harness revision,
trajectory, observations, reward, normalized score, success flag, failure
reason, oracle signals, diagnosis, candidate harness, and paired validation
decision. Summaries include success rate, normalized area under the learning
curve, difficulty coverage, and mutation acceptance. At minimum, report:

1. success rate and mean normalized score;
2. area under the learning curve and rounds to a target success rate;
3. held-out generalization across unseen seeds/specs/difficulty levels;
4. environment validity/compile rate and proposal diversity;
5. diagnosis accuracy against oracle labels;
6. number and type of harness mutations, memory size, and regressions;
7. environment difficulty/coverage over time;
8. wall-clock time, model calls, tokens, and estimated cost for LLM methods.

Use multiple random seeds and confidence intervals. Keep a fixed held-out test
set that is never visible to designers or evolvers. Report both final performance
and compute-normalized performance; a method that makes many model calls is not
directly comparable to the local baselines otherwise.

Recommended ablations include removing memory, prompt mutation, skills,
workflow mutation, validation, learning-progress selection, or privileged
oracle signals. Also compare mutation rollback versus unconditional mutation,
and environment novelty objectives versus pure difficulty.

## Student milestones

### Milestone 0 — understand the scaffold

Run the smoke configuration and tests. Read one JSONL trace. Draw the information
available to the environment designer, diagnoser, and evolver; accidental oracle
leakage is a common experimental bug.

### Milestone 1 — establish reproducible baselines

Run all four conditions for at least five seeds. Add a script that aggregates
mean, confidence interval, learning curve, environment coverage, and mutation
counts. Freeze the evaluation set before developing a novel method.

### Milestone 2 — replace one component

Implement one idea behind an existing protocol rather than rewriting the loop:

- a bandit/Bayesian/novelty environment designer in `designers.py`;
- causal or uncertainty-aware attribution in `diagnosis.py`;
- candidate generation plus rollback in `evolver.py`;
- a richer deterministic environment compiler registered through `specs.py`;
- an actual student policy in the environment adapter.

Add focused tests for the new component and preserve the four factorial
baselines.

### Milestone 3 — add LLM synthesis carefully

Implement a backend, cache raw responses, measure invalid proposal rate, cap
cost, and compare against parameter-matched random generation. Use a repair pass
only as a separately reported ablation; silently repairing invalid specs changes
the measured method.

### Milestone 4 — scale to real environments

Choose one external domain and define a narrow DSL plus deterministic compiler
for it. Keep the optional adapter in its own dependency extra or integration
package so the zero-dependency reference path remains small.

### Milestone 5 — credible research claim

Evaluate held-out transfer, robustness to diagnoser mistakes, diversity-collapse
failure modes, and total compute. Release configs, seeds, raw JSONL logs, prompt
versions, validation failures, and analysis code.

## Where novel work should go

The stable interfaces are intentionally small:

- environment novelty/curricula: implement `EnvironmentDesigner.propose`;
- new DSL domain: implement `EnvironmentCompiler.compile` and register it;
- failure attribution: implement `Diagnoser.diagnose`;
- search/reflection/rollback: replace or wrap `HarnessEvolver.evolve`;
- model/provider integration: implement `TextBackend.complete`;
- real agent behavior: implement an environment adapter's `run` method.

Avoid placing a proposed algorithm directly in `cli.py` or `experiment.py`.
Those files should remain boring orchestration code so methods can be exchanged
without changing the evaluation protocol.

## Known limitations and TODOs

- The toy environment is a transparent unit test for co-evolution mechanics,
  not evidence of real-world self-improvement.
- The reference evolver proposes only one diagnosis-conditioned candidate; it
  has paired validation and rollback but no beam search or persistent held-out
  validation set.
- The frontier model uses a compact shrinkage/UCB heuristic rather than a
  calibrated Bayesian ability model.
- LLM responses are only JSON-parsed and schema-validated; production code also
  needs timeouts, retries, caching, budgets, and prompt-injection boundaries.
- MiniGrid exposes a dynamic policy interface, but the included workflow policy
  is intentionally not a capable solver.
- Designers are not yet optimized for novelty, coverage, learnability, or
  adversarial robustness.
- Summary statistics are intentionally minimal; add multi-seed aggregation and
  plots before drawing conclusions.

These limitations are deliberate openings for student projects, not hidden
claims that the prototype already solves environment synthesis.
