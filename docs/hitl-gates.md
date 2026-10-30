# HITL gates for the REINVENT4 Agent MVP

Standing charter (2026-10-29): the REINVENT path is a **virtual lab team**
tool chain with three human gates — not a personal-assistant swarm.

```text
goal → validate config → [approve-config] → [approve-run] → one generation
     → RDKit analysis → HTML report → [approve-claim]
```

| Gate | CLI flag | What the human is signing |
|------|----------|---------------------------|
| Config | `--approve-config` | Project / preset `reinvent.toml` identity is OK. Agents may only name a preset ID. |
| Run | `--approve-run` | Launch the predefined `reinvent …` argv (plus confirm / `--yes`). |
| Claim | `--approve-claim` | Interpretive / outbound language about this run is allowed. |

## Soft policy (this PR)

- **Hard block stays on `--approve-run` only** (existing behaviour).
- Missing `--approve-config` while launching → Critic `WARNING` + checklist pending.
- Missing `--approve-claim` → Critic `WARNING` + HTML banner: mechanical counts only.
- `--from-run` **never** copies any of the three approve flags (or `--yes` / `--skip-reinvent`).

Hard-requiring config before run is a PI decision for a later one-line change
(`hard_block_run_without_config`).

## Offline / dry-run

`--skip-reinvent` does not need the run gate for launch. Config and claim
gates still appear on the checklist so offline CSV analysis is not mistaken
for an approved scientific claim.

## Tests

`tests/test_hitl_checklist.py` — no GPU, no network, no API keys.
