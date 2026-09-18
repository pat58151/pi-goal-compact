# pi-goal-compact

Goal-aware context compaction for [pi](https://pi.dev).

One extension that makes pi's existing compactions preserve working state instead of
re-summarizing it away, and keeps a `/goal` session's summary focused on the goal.

## What it does

**It adds no compaction trigger of its own and never calls `ctx.compact()`.** pi already
compacts: mid-run before each next assistant response, again when a run ends, on context
overflow, and on manual `/compact`. This extension improves what happens during those
compactions:

1. **Pre-compaction checkpoint.** A fixed number of tokens *before* pi's threshold
   (`contextWindow - compaction.reserveTokens`), the live model is asked to hand off its
   working state through a `save_checkpoint` tool (or a plain-text `## Checkpoint` message
   as a fallback). The text is captured out-of-band as a durable session entry and prepended
   **verbatim** to the next compaction summary, so in-progress work, touched files,
   decisions, next steps, and exact paths/errors survive byte-for-byte.

2. **Goal-aware summaries.** While a `/goal` is active, the summary prompt carries the goal
   objective, status, and budget, and the summary is generated with a bounded
   `SUMMARY_MAX_TOKENS` (8192) output budget instead of pi's `reserveTokens`-derived budget,
   which can be as large as the model's max output (384,000 on deepseek-flash).

Hooks used: `session_start`, `turn_end` (request/capture the checkpoint only), and
`session_before_compact` / `session_compact` (supply the summary, re-arm the prompt).

## Requirements

- pi 0.85.0 or later.
- For the goal-aware parts: the [`pi-codex-goal`](https://github.com/fitchmultz/pi-codex-goal)
  package, which provides `/goal` and the goal session entries this extension reads.
  Without it the extension still loads and still captures checkpoints, but no goal is ever
  "active", so the goal-aware summary path is inert.

## Install

```sh
pi install git:github.com/pat58151/pi-goal-compact@v1.1.0
pi install npm:pi-codex-goal          # for the goal-aware behavior
```

Or drop the file in by hand:

```sh
cp extensions/goal-compact.ts ~/.pi/agent/extensions/
```

(`~/.pi/agent` is the default agent dir; if `PI_CODING_AGENT_DIR` is set, use
`$PI_CODING_AGENT_DIR/extensions/`.) Then run `/reload` inside pi.

## Configuration

Size pi's compaction trigger in `settings.json`:

```json
{
  "compaction": {
    "reserveTokens": 616000
  }
}
```

With a 1M-token window, `reserveTokens: 616000` triggers compaction at ~384k. pi's default is
16384. The checkpoint prompt fires `PI_GOAL_PREP_LEAD_TOKENS` earlier:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PI_GOAL_PREP_LEAD_TOKENS` | `16000` | Headroom between the checkpoint prompt and pi's compaction trigger. |

So the documented setup asks for a checkpoint at ~368k and pi compacts at ~384k.

## Notes and caveats

- The extension registers a `save_checkpoint` tool. If another extension registers a tool
  with the same name, the last registration wins.
- The checkpoint prompt only fires on tool-use turns while a goal is active, and never once
  context is already past the threshold - asking then would land after the compaction.
- A captured checkpoint is consumed by the next compaction and is never reused. If the
  summarization call fails, the checkpoint alone is used as the summary rather than losing it.
- The `## Checkpoint` plain-text fallback is only captured when the prompt was actually
  injected, so models that call `save_checkpoint` voluntarily at progress milestones do not
  disturb anything.
- Goal state is read from pi session entries (`pi-codex-goal` custom entries), so it survives
  resume, fork, and reload without external state.

## Known upstream issue (pi 0.85.1)

pi's mid-run threshold compaction can silently do nothing when the newest turn's tool results
alone exceed `compaction.keepRecentTokens` (default 20000): `findCutPoint` falls back to the
oldest cut point, `prepareCompaction` finds nothing to summarize, and `_runAutoCompaction`
returns before emitting `compaction_start` - no compaction, no error, no notice. The post-run
check masks this in normal sessions, but a `/goal` run never reaches a run boundary, so
context can overshoot the threshold.

Reported as [earendil-works/pi#9740](https://github.com/earendil-works/pi/issues/9740) with a
deterministic repro and an 8-line fix in `findCutPoint`. On an unpatched pi, a goal session can
still overshoot; this extension's checkpoint machinery applies to whatever compactions do run.

## License

MIT
