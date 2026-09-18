# pi-goal-compact

Goal-aware context compaction for [pi](https://pi.dev).

A single pi extension that keeps a long-running `/goal` session from blowing past
its context window, and makes every compaction preserve the working state
byte-for-byte instead of re-summarizing it away.

## What it does

Three mechanisms, all in one file:

1. **Mid-run compaction trigger.** A `/goal` keeps the agent inside one long
   tool-call run, so pi's native threshold compaction (which only runs at
   agent-run boundaries) rarely fires. This extension checks the context on
   every tool-use turn and compacts once it crosses
   `contextWindow - compaction.reserveTokens`, then resumes the paused goal
   automatically via `/goal resume`.

2. **Pre-compaction checkpoint.** A fixed amount of tokens *before* that
   trigger, the live model is asked to hand off its working state through a
   `save_checkpoint` tool (or a plain-text `## Checkpoint` message as a
   fallback). The text is captured out-of-band and prepended **verbatim** to the
   compaction summary, so in-progress work, touched files, decisions, and next
   steps survive exactly.

3. **Goal-aware summaries.** When a goal is active, compaction summaries are
   generated with the objective, status, and budget injected into the
   summarization prompt, so the summary keeps goal-relevant progress and next
   steps. Every compaction (threshold, `/compact`, or overflow) is summarized
   with a bounded 8192-token output budget.

## Requirements

- pi 0.85.0 or later (uses `session_before_compact`, `getContextUsage`,
  `deliverAs: "steer"`).
- For the goal-aware parts: the [`pi-codex-goal`](https://github.com/fitchmultz/pi-codex-goal)
  package, which provides the `/goal` command and the goal session entries this
  extension reads.

Without `pi-codex-goal` installed the extension still loads and still does
threshold compaction, but the goal-aware summary and the mid-run goal logic are
inert (no goal is ever "active").

## Install

From git:

```sh
pi install git:github.com/pat58151/pi-goal-compact@v1.0.0
```

From a local checkout or a zip you were sent:

```sh
pi install /absolute/path/to/pi-goal-compact
```

Or just drop the file in by hand:

```sh
cp extensions/goal-compact.ts ~/.pi/agent/extensions/
```

(`~/.pi/agent` is the default agent dir; if `PI_CODING_AGENT_DIR` is set, use
`$PI_CODING_AGENT_DIR/extensions/`.) Then run `/reload` inside pi.

Install `pi-codex-goal` too if you want the goal-aware behavior:

```sh
pi install npm:pi-codex-goal
```

## Configuration

Size the compaction trigger in `settings.json` (global `~/.pi/agent/settings.json`
or project `.pi/settings.json`):

```json
{
  "compaction": {
    "reserveTokens": 616000
  }
}
```

With a 1M-token context window, `reserveTokens: 616000` triggers at ~384k.
The default when unset is 16384.

Environment overrides:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PI_GOAL_MIDRUN_RESERVE_TOKENS` | value from settings | Override the reserve for the mid-run trigger only. |
| `PI_GOAL_PREP_LEAD_TOKENS` | `16000` | Headroom between the checkpoint prompt and the compaction trigger. |

With the example above, the checkpoint prompt fires at ~368k and compaction at
~384k.

## Notes and caveats

- The extension registers a `save_checkpoint` tool. If another extension
  registers a tool with the same name, the last registration wins.
- The `## Checkpoint` plain-text fallback only triggers when the prompt was
  actually injected, so models that call `save_checkpoint` voluntarily at
  progress milestones do not trigger an early compaction.
- A captured checkpoint is consumed by the next compaction and never reused.
- Goal state is read from pi session entries (`pi-codex-goal` custom entries),
  so it survives resume, fork, and reload without any external state.

## License

MIT
