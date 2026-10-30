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
  `--approve-claim` is not a hard refuse.
- Missing `--approve-config` while launching → Critic `WARNING` + checklist pending.
- Missing `--approve-claim` on an approved launch, a dry-run, or
  `--skip-reinvent` → Critic `WARNING` (same soft pattern as missing
  `--approve-config`: PASS becomes WARNING, status stays FAIL if it already
  failed) plus the HTML banner. Counts stay mechanical.
- `--from-run` **never** copies any of the three approve flags (or `--yes` / `--skip-reinvent`).

Hard-requiring config before run is a PI decision for a later one-line change
(`hard_block_run_without_config`). Claim stays soft.

## Offline / dry-run

`--skip-reinvent` does not need the run gate for launch. Config and claim
gates still appear on the checklist.

A dry-run (no `--approve-run`) or `--skip-reinvent` **without**
`--approve-claim` is a Critic `WARNING` (exit 0), not checklist/HTML-only.
Passing `--approve-claim` clears that issue. It does not launch REINVENT.
Normal approved launches are unchanged: missing claim is still that same
soft WARNING; with `--approve-claim` the claim issue is absent.

## Tests

`tests/test_hitl_checklist.py` and `tests/smoke/test_claim_gate.py` — no GPU,
no network, no API keys. Smoke covers missing vs present `--approve-claim`
on dry-run and `--skip-reinvent`.
