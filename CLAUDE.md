# CLAUDE.md — Loom

> Automated interaction protocol design for faithful long-horizon persona simulation via API-accessed models.

Loom generates runtime scaffolding configurations that control how LLMs simulate personas over extended multi-turn conversations. It is **not** a fine-tuning tool. It produces structured prompts, injection schedules, and evaluation pipelines that work with any model accessible through a chat completion API.

### Directory Context
```
C:\Users\timf3\VSCode\long-term-personas\
├── loom/     # ← THIS REPO (Loom)
├── bloom/              # Adjacent: Bloom eval generation framework (design inspiration)
└── petri/              # Adjacent: Petri automated auditing tool (design inspiration)

C:\Users\timf3\VSCode\MentalHealthLLMs\
└── psychosis-bench-2.0/  # The dynamic mania benchmark that Loom primarily serves
```

Loom is standalone — it does not import from adjacent repos. It draws design inspiration from Bloom (config patterns, seed configs, judge pipeline) and Petri (auditor agents, transcript viewer). The mania bench profiles in psychosis-bench-2.0 will be ported to Loom configs via a migration script.

---

## Implementation Status

### M0: Schema + Validation — COMPLETE
- 20 Pydantic models in `schema/config.py` (all design review fixes applied)
- YAML loader with human-readable validation errors
- `loom validate` CLI command working
- 4 example configs (mania, depression, student, consistency-only)
- 38 unit tests passing

### M1: Prompt Assembler — COMPLETE
- `IntensityResolver`: sigmoid, linear, delayed_ramp, step curve interpolation
- `InjectionScheduler`: full/reminder/none scheduling per turn
- `PromptBuilder`: 3-block prompt assembly (MUST_DO / HOW / CONTEXT)
- `loom dry-run` CLI command working
- Versioned templates (`TEMPLATE_VERSION = "0.1.0"`)
- 46 unit tests passing (cumulative with M0 assembler tests)

### M2: Rollout Runner + Runtime Monitors — COMPLETE
- `RolloutRunner`: multi-turn conversation orchestrator
- `ModelClient` ABC with `OpenAIClient`, `AnthropicClient`, `MockModel`
- `StagnationMonitor`: TF-IDF cosine similarity (self-sim + convergence)
- `RepetitionMonitor`: banned patterns + structural patterns (question streaks, gratitude loops)
- `Transcript` model with `TurnRecord`, `MonitorEvent`, JSON serialization
- `loom run` CLI command wired up
- 119 unit tests passing, 3 integration tests written (need API keys)

### M3: Scoring Pipeline — NOT STARTED
- `loom score` command (post-hoc scoring of existing transcripts)
- `JudgeClient`: sends transcript window to judge model
- 5 scoring dimensions: persona_adherence, trajectory, naturalness, stagnation, fidelity
- Sliding window scoring (last N turns, not full transcript)

### M4: Adversaries — NOT STARTED
- `ContradictionTrap`: probes persona for internal inconsistencies
- `HelpfulnessCoaxer`: probes for capability drift / register shift
- Adversary configs in `configs/adversaries/`

### M5: Report + Mania Bench Port — NOT STARTED
- `loom report`: JSON + Markdown report generation
- `loom view`: terminal transcript viewer
- `scripts/port_mania_bench.py`: migration from psychosis-bench-2.0

### What Still Needs Testing
- **Live integration tests** (`pytest --live`): require `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY`. Tests are in `tests/integration/test_rollout.py`. They run a 3-turn rollout with real APIs (student_persona config, cheap models). Budget target: < €3 total.
- **Cross-provider rollout**: persona on OpenAI, interlocutor on Anthropic (needs both keys).
- All unit tests pass without API keys: `python -m pytest tests/unit/ -v`

---

## Repo Map

```
loom/
├── CLAUDE.md                    # This file — read first
├── PLAN.md                      # Scope, milestones, architecture
├── pyproject.toml               # Package config (Python 3.11+, click, pydantic, pyyaml)
├── .gitignore
├── loom/
│   ├── __init__.py              # Package root, __version__ = "0.1.0"
│   ├── cli.py                   # CLI: validate, dry-run, run  [IMPLEMENTED]
│   ├── schema/
│   │   ├── __init__.py          # Exports LoomConfig, load_config
│   │   ├── config.py            # 20 Pydantic models + validators  [IMPLEMENTED]
│   │   ├── loader.py            # YAML → LoomConfig with clear errors  [IMPLEMENTED]
│   │   └── defaults.py          # Enums, constants, SCHEMA_VERSION  [IMPLEMENTED]
│   ├── assembler/
│   │   ├── __init__.py          # Exports PromptBuilder, IntensityResolver, etc.
│   │   ├── prompt_builder.py    # Config × turn → PromptBlocks (must_do/how/context)  [IMPLEMENTED]
│   │   ├── intensity.py         # Curve interpolation (sigmoid/linear/delayed_ramp/step)  [IMPLEMENTED]
│   │   ├── injection.py         # Full/reminder/none injection scheduling  [IMPLEMENTED]
│   │   └── templates.py         # Versioned f-string templates, TEMPLATE_VERSION  [IMPLEMENTED]
│   ├── runner/
│   │   ├── __init__.py          # Exports RolloutRunner, MockModel, etc.
│   │   ├── rollout.py           # Multi-turn conversation orchestrator  [IMPLEMENTED]
│   │   ├── api_client.py        # OpenAI + Anthropic + MockModel clients  [IMPLEMENTED]
│   │   └── transcript.py        # Transcript/TurnRecord/MonitorEvent + JSON  [IMPLEMENTED]
│   ├── adversaries/
│   │   ├── __init__.py          # [STUB]
│   │   ├── base.py              # Adversary ABC  [NOT STARTED]
│   │   ├── contradiction.py     # Contradiction-trap adversary  [NOT STARTED]
│   │   └── helpfulness.py       # Helpfulness-coaxer adversary  [NOT STARTED]
│   ├── monitors/
│   │   ├── __init__.py          # [STUB]
│   │   ├── base.py              # Monitor ABC, MonitorResult, MonitorAction  [IMPLEMENTED]
│   │   ├── stagnation.py        # TF-IDF cosine similarity detection  [IMPLEMENTED]
│   │   └── repetition.py        # Banned patterns + structural patterns  [IMPLEMENTED]
│   ├── scoring/
│   │   ├── __init__.py          # [STUB]
│   │   ├── judge.py             # LLM-as-judge scaffold  [NOT STARTED]
│   │   ├── persona_adherence.py # Identity/trait/boundary consistency  [NOT STARTED]
│   │   ├── trajectory.py        # Prescribed vs actual intensity  [NOT STARTED]
│   │   ├── naturalness.py       # Anti-mannequin scoring  [NOT STARTED]
│   │   ├── stagnation.py        # Post-hoc agreement loop analysis  [NOT STARTED]
│   │   └── fidelity.py          # Scaffolding compliance measurement  [NOT STARTED]
│   ├── report/
│   │   ├── __init__.py          # [STUB]
│   │   ├── generator.py         # JSON + Markdown report  [NOT STARTED]
│   │   └── viewer.py            # Terminal transcript viewer  [NOT STARTED]
│   └── utils/
│       ├── __init__.py          # [STUB]
│       ├── logging.py           # Structured logging  [NOT STARTED]
│       └── seeds.py             # Seed management  [NOT STARTED]
├── configs/
│   ├── examples/
│   │   ├── mania_patient.yaml   # Marcus Chen, 3 dims, 5 phases  [IMPLEMENTED]
│   │   ├── depressed_patient.yaml  # Sarah Okonkwo, 3 dims, 4 phases  [IMPLEMENTED]
│   │   ├── student_persona.yaml # Jake Rivera, 2 dims, 3 phases  [IMPLEMENTED]
│   │   └── consistency_only.yaml  # Evelyn Hart, 1 dim flat, 1 phase  [IMPLEMENTED]
│   └── adversaries/             # [NOT STARTED]
├── tests/
│   ├── conftest.py              # Fixtures + --live flag handling  [IMPLEMENTED]
│   ├── unit/
│   │   ├── test_schema.py       # 38 tests: config validation  [IMPLEMENTED]
│   │   ├── test_assembler.py    # 27 tests: prompt assembly  [IMPLEMENTED]
│   │   ├── test_intensity.py    # 12 tests: curve interpolation  [IMPLEMENTED]
│   │   ├── test_injection.py    # 7 tests: injection scheduling  [IMPLEMENTED]
│   │   ├── test_monitors.py     # 17 tests: stagnation + repetition  [IMPLEMENTED]
│   │   └── test_rollout.py      # 9 tests: end-to-end with MockModel  [IMPLEMENTED]
│   ├── integration/
│   │   └── test_rollout.py      # 3 live tests (need API keys)  [IMPLEMENTED]
│   └── golden/                  # [NOT STARTED]
├── results/                     # Git-ignored; output directory for runs
└── scripts/                     # [NOT STARTED]
```

---

## Design Decisions Made During Implementation

These were resolved from the design review open questions:

| # | Decision | Where documented |
|---|----------|------------------|
| 1 | `DimensionSpec` has both `start_value`/`end_value` (trajectory) AND `min_value`/`max_value` (safety bounds) | `schema/config.py:DimensionSpec` |
| 2 | Curve params are a sub-model: `curve: {type: "sigmoid", midpoint_pct: 0.5}` | `schema/config.py:CurveParams` |
| 3 | `PhaseSpec.revelations: list[RevelationSpec]` with `topic` + `variants` dict | `schema/config.py:RevelationSpec` |
| 4 | Scoring is post-hoc only by default. `loom run` → transcripts. `loom score` → adds scores. | `cli.py` (--score flag placeholder) |
| 5 | TF-IDF for stagnation detection, no embedding API dependency for v0 | `monitors/stagnation.py` |
| 6 | Persona-only scoring for v0. Interlocutor scoring deferred. | PLAN.md |
| 7 | Revelation variant selection: intensity < 0.33 → subtle, 0.33-0.66 → moderate, >= 0.66 → direct | `assembler/prompt_builder.py:_intensity_to_variant_label()` |
| 8 | Injection: full = rebuild system prompt, reminder = append to user message, none = no change | `runner/rollout.py:_build_persona_messages()` |
| 9 | `fixed_length` only for v0. `phase_gated` and `open_ended` deferred. | `schema/defaults.py:TrajectoryMode` |
| 10 | Monitor re-prompting: up to `max_retries` with "[SYSTEM: vary your output]" injection | `runner/rollout.py` (RE_PROMPT handling) |

---

## Conventions

### Naming
- **Persona**: the simulated human (patient, student, user). Never "character" or "agent."
- **Interlocutor**: the model being evaluated (therapist, teacher, assistant). Or **target** when discussing evaluation.
- **Adversary**: the model probing for persona failures. Two types: contradiction-trap, helpfulness-coaxer.
- **Monitor**: a lightweight runtime check that runs during rollouts to detect failure modes (stagnation, repetition) and trigger interventions. Distinct from scorers, which run post-hoc.
- **Scaffolding config**: the YAML file specifying persona + trajectory + evaluation. The primary input to Loom.
- **Rollout**: a single multi-turn conversation produced by the runner.
- **Fidelity**: how well the persona model follows the prescribed config. Distinct from the interlocutor's performance.

### Code Style
- Python 3.11+. Type hints everywhere. Pydantic v2 for config models.
- `click` for CLI. No argparse.
- All prompts stored as templates in `assembler/templates.py` or in config YAML, never inline in logic code.
- No hardcoded model names in library code. Model selection is always via config or CLI arg.
- Functions that call LLM APIs must accept a `seed: int | None` parameter and pass it through.

### File Formats
- **Input configs**: YAML (validated by Pydantic on load).
- **Transcripts**: JSON (one file per rollout, includes full message history + metadata + per-turn prescribed intensity values).
- **Reports**: JSON (machine-readable) + Markdown (human-readable), both generated.
- **Scores**: embedded in transcript JSON under a `scores` key per turn.

### Versioning
- Every scaffolding config gets a `schema_version` field. Loom validates against the declared version.
- Prompt templates are versioned in `templates.py` with a `TEMPLATE_VERSION` constant. Template changes require version bumps.
- Results directories include the config hash in their name for reproducibility.

---

## Golden Path Workflow

The intended workflow for using Loom:

### 1. Define your persona config
```bash
cp configs/examples/mania_patient.yaml configs/my_persona.yaml
# Edit: persona identity, capability bounds, trajectory dimensions, phases
```

### 2. Validate the config
```bash
loom validate configs/my_persona.yaml
# Checks: schema validity, phase coverage, dimension consistency,
#          trajectory curve endpoints, required fields present
```

### 3. Preview prompts (no API calls)
```bash
loom dry-run configs/my_persona.yaml --turns 10
# Shows: assembled system prompt at each turn, injection type, phase, intensities
```

### 4. Run a small evaluation (1-3 rollouts)
```bash
loom run configs/my_persona.yaml \
  --target anthropic/claude-sonnet-4 \
  --persona-model openai/gpt-4.1 \
  --turns 30 \
  --rollouts 1 \
  --seed 42 \
  --output results/my_test/
```

### 5. Inspect the transcript
```bash
# Transcripts are JSON files in the output directory:
# results/my_test/rollout_000.json
```

### 6. (Future) Score, report, adversary modes
```bash
loom score results/my_test/ --judge-model anthropic/claude-opus-4-20250514
loom report results/my_test/ --output results/my_test/report.md
loom run configs/my_persona.yaml --adversary contradiction ...
```

---

## Running Tests

```bash
# All unit tests (no API keys needed):
python -m pytest tests/unit/ -v

# Live integration tests (needs API keys):
python -m pytest tests/ --live -v

# Current test count: 119 passed, 3 skipped (live)
```

---

## Safety Notes: Mental Health Persona Simulation

Loom can be used to simulate personas experiencing mental health crises (psychosis, mania, depression, suicidal ideation). This requires specific safeguards:

### For researchers using Loom
1. **Loom produces simulated conversations, not clinical tools.** Transcripts should never be used as therapeutic guidance or presented to patients.
2. **Simulated personas are approximations.** Even well-configured personas do not faithfully represent real clinical presentations. They are useful for *stress-testing AI systems*, not for training clinicians.
3. **Review transcripts for harmful content.** LLMs simulating distressed personas may generate content involving self-harm, substance abuse, or dangerous behavior. This content exists to test how interlocutor models respond — it should be handled with appropriate care.
4. **Do not publish raw transcripts without review.** Simulated crisis conversations can be distressing and could be misused. Redact or summarize when sharing results.

### For Loom's implementation
1. **Safety boundaries in configs.** Every persona config includes a `safety` section specifying hard limits (topics the persona will never simulate, escalation ceilings). The prompt assembler enforces these as top-priority instructions.
2. **No real clinical data.** Example configs use fictional personas. Loom never ingests, stores, or processes real patient data.
3. **Trajectory ceilings.** Intensity dimensions have a configurable `max_value` (default 0.9, not 1.0) to prevent the persona from reaching maximally extreme states that could produce harmful or unrealistic content.
4. **Interlocutor safety is out of scope for v0.** Loom scores the persona's fidelity, not the interlocutor's responses. Evaluating whether the interlocutor appropriately handles crisis content is the job of downstream benchmarks (like psychosis-bench) that consume Loom's persona simulations.

---

## Rules

### API Keys
- All API keys via environment variables. Never in config files, never in code.
- Required env vars: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` (only those needed for chosen models).
- CLI prints a clear error if a required key is missing before making any API calls.

### Local Mock Model for Tests
- All tests that would call an LLM API use a `MockModel` that returns deterministic responses.
- `MockModel` is configured via fixture, supports: canned responses, echo mode (returns input), and scripted multi-turn sequences.
- Integration tests that require real API calls are marked with `@pytest.mark.live` and skipped by default. Run with `pytest --live`.
- **No test should ever make a real API call unless explicitly opted in.**

### Deterministic Seeds
- Every `loom run` invocation requires or auto-generates a `--seed` value.
- Seeds are passed to: model API calls (where supported), random selections (phase jitter, revelation ordering), and transcript filenames.
- Seeds are recorded in transcript metadata for reproducibility.
- Note: deterministic seeds do not guarantee identical outputs across different model versions or API updates. They guarantee identical *scaffolding behavior* (same prompts, same injection schedule, same adversary actions).

### Versioned Prompts and Configs
- The `schema_version` field in configs follows semver (e.g., `"0.1.0"`).
- Prompt templates in `templates.py` have a `TEMPLATE_VERSION` string.
- Both versions are recorded in every transcript's metadata.
- Breaking changes to the schema or templates require a major version bump and a migration note in the changelog.

### Dependencies on Adjacent Repos
- Loom is a standalone tool. It does not import from `dynamic-mania-bench`, `bloom`, or `petri`.
- It draws *design inspiration* from these repos (config patterns from Bloom, judge patterns from Petri, intensity control from dynamic-mania-bench).
- `scripts/port_mania_bench.py` is a one-time migration script that reads dynamic-mania-bench YAML profiles and outputs Loom-format configs. It is not part of the library.
