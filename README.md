# 🧬 Hermes Agent Self-Evolution

**Evolutionary self-improvement for [Hermes Agent](https://github.com/NousResearch/hermes-agent).**

Hermes Agent Self-Evolution uses DSPy + GEPA (Genetic-Pareto Prompt Evolution) to automatically evolve and optimize Hermes Agent's skills, tool descriptions, system prompts, and code — producing measurably better versions through reflective evolutionary search.

**No GPU training required.** Everything operates via API calls — mutating text, evaluating results, and selecting the best variants. ~$2-10 per optimization run.

## How It Works

```
Read current skill/prompt/tool ──► Generate eval dataset
                                        │
                                        ▼
                                   GEPA Optimizer ◄── Execution traces
                                        │                    ▲
                                        ▼                    │
                                   Candidate variants ──► Evaluate
                                        │
                                   Constraint gates (tests, size limits, benchmarks)
                                        │
                                        ▼
                                   Best variant ──► PR against hermes-agent
```

GEPA reads execution traces to understand *why* things fail (not just that they failed), then proposes targeted improvements. ICLR 2026 Oral, MIT licensed.

## Quick Start

```bash
# Install
git clone https://github.com/NousResearch/hermes-agent-self-evolution.git
cd hermes-agent-self-evolution
pip install -e ".[dev]"

# Point at your hermes-agent repo
export HERMES_AGENT_REPO=~/.hermes/hermes-agent

# Evolve a skill (synthetic eval data)
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --eval-source synthetic

# Or use real session history from Claude Code, Copilot, and Hermes
python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --iterations 10 \
    --eval-source sessiondb
```

## How Tasks Are Evaluated

By default (`--executor hermes`), every train, validation and holdout task is run by the real Hermes agent with the candidate skill loaded:

```bash
hermes chat -q "<task>" --oneshot --format stream-json -s <skill>
```

- **A throwaway profile per run.** Each run uses a temporary `HERMES_HOME` that holds only the candidate skill (plus the skill's own `scripts/` and `references/`). Into it are copied your `config.yaml`, `.env`, `auth.json` and `SOUL.md`, from `--hermes-home` (default: `$HERMES_HOME`, else `~/.hermes`).
- **Your real setup, untouched.** The task therefore runs with your model, your credentials and Hermes's real tools. The evaluation's sessions and memories never touch your profile.
- **What gets scored.** The scorer grades the final answer; the semantic judge also sees the tool calls. GEPA's reflection reads the full transcript of what Hermes did.

Use `--hermes-command` when `hermes` is not on `PATH`, for example `'/path/to/venv/bin/python -m hermes_cli.main'`. `--hermes-model`, `--hermes-provider`, `--hermes-toolsets`, `--hermes-max-turns` and `--hermes-timeout` tune the runs.

**Cost and speed.** Every example is a full agent run, with several model calls and real tool execution, so a run costs and takes far more than a bare LLM call. Start with a small dataset and few iterations.

**Safety.** Tools run for real on this machine, in a scratch working directory. Hermes is not started with `--yolo`, so its usual command approval rules apply. Restrict the run with `--hermes-toolsets`, or point `--hermes-home` at a profile that uses a sandboxed terminal backend.

`--executor dspy` replaces the agent with a single LLM call that reads the skill text and has no tools. It is kept for offline experiments only: it does not measure what users get.

## What It Optimizes

| Phase | Target | Engine | Status |
|-------|--------|--------|--------|
| **Phase 1** | Skill files (SKILL.md) | DSPy + GEPA | ✅ Implemented |
| **Phase 2** | Tool descriptions | DSPy + GEPA | 🔲 Planned |
| **Phase 3** | System prompt sections | DSPy + GEPA | 🔲 Planned |
| **Phase 4** | Tool implementation code | Darwinian Evolver | 🔲 Planned |
| **Phase 5** | Continuous improvement loop | Automated pipeline | 🔲 Planned |

## Fitness Signals

Evolution is only as trustworthy as the score it optimizes. The pipeline picks the strongest signal available for the target skill:

| Signal | How it scores | Trust |
|--------|--------------|-------|
| **Objective verifier** | Checks output against verifiable ground truth (IDs, titles, authors, dates); pure Python, zero API cost per grade | Highest: cannot be gamed by echoing rubric vocabulary |
| **Semantic judge** (`--scorer semantic`, default) | An LLM judge scores task outcome, procedure, calibration and evaluator gaming; one judge call per grade | Good: judges behavior, not shared words |
| **Keyword overlap** (`--scorer keyword`) | Lexical overlap between output and rubric text | Legacy: an evolved skill can win by echoing rubric words |

Skills with a registered verifier (currently: `arxiv`) evolve against real facts by default; other skills use `--scorer`. Pass `--fitness scorer` to use `--scorer` even when a verifier exists. Verifier feedback explains exactly what was wrong ("expected ID 1706.03762, response said 1706.03799"), which is the reflection signal GEPA mutates on.

```bash
# Objective fitness is picked automatically when a verifier exists
python -m evolution.skills.evolve_skill --skill arxiv --iterations 10

# Cross-check the verifier's embedded ground truth against the live arXiv API
python -m evolution.verifiers.arxiv_verifier --validate

# See how the grader scores correct, wrong, and evasive answers
python -m evolution.verifiers.arxiv_verifier --demo
```

To add a verifier for another skill, subclass `Verifier` in `evolution/verifiers/`, decorate it with `@register_verifier`, and implement `build_dataset()` plus `score()`.

## Engines

| Engine | What It Does | License |
|--------|-------------|---------|
| **[DSPy](https://github.com/stanfordnlp/dspy) + [GEPA](https://github.com/gepa-ai/gepa)** | Reflective prompt evolution — reads execution traces, proposes targeted mutations | MIT |
| **[Darwinian Evolver](https://github.com/imbue-ai/darwinian_evolver)** | Code evolution with Git-based organisms | AGPL v3 (external CLI only) |

## Guardrails

Every evolved variant must pass:
1. **Full test suite** — `pytest tests/ -q` must pass 100%
2. **Size limits** — Skills ≤15KB, tool descriptions ≤500 chars
3. **Caching compatibility** — No mid-conversation changes
4. **Semantic preservation** — Must not drift from original purpose
5. **PR review** — All changes go through human review, never direct commit

## Full Plan

See [PLAN.md](PLAN.md) for the complete architecture, evaluation data strategy, constraints, benchmarks integration, and phased timeline.

## License

MIT — © 2026 Nous Research
