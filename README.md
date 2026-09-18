# pi-goal-compact

Goal-aware context compaction for [pi](https://pi.dev).

Makes pi's compactions preserve working state instead of re-summarizing it away, and keeps a
`/goal` session from overshooting its context window.

## What it does

1. **Mid-run trigger.** pi's own mid-run threshold check silently does nothing when the newest
   turn's tool results alone exceed `compaction.keepRecentTokens` (default 20000) - the cut-point
   search degenerates, nothing is left to summarize, and `_runAutoCompaction` returns before
   emitting anything ([upstream #9740](https://github.com/earendil-works/pi/issues/9740)). The
   post-run check masks that in ordinary sessions, but a `/goal` run never ends, so the context
   overshoots. This extension compacts itself once the context crosses
   `contextWindow - reserveTokens`, then resumes the paused goal.

2. **Turn-boundary recovery.** `ctx.compact()` can only cut at a user or assistant message, never
   at a tool result. If compaction fails with "Nothing to compact" because the last entry is a
   tool result, the extension asks the model for one short plain assistant turn, which gives pi a
   cut point, and compacts on the following turn.

3. **Pre-compaction checkpoint.** A fixed number of tokens before the trigger, the live model is
   asked to hand off its working state through a `save_checkpoint` tool (or a plain-text
   `## Checkpoint` message). The text is captured out-of-band as a durable session entry and
   prepended **verbatim** to the compaction summary, so in-progress work, touched files, decisions,
   next steps and exact paths/errors survive byte-for-byte.

4. **Goal-aware summaries.** While a `/goal` is active, the summary prompt carries the goal
   objective, status and budget, and the summary is generated with a bounded `SUMMARY_MAX_TOKENS`
   (8192) output budget instead of pi's `reserveTokens`-derived budget, which can be as large as
   the model's max output (384,000 on deepseek-flash).

Mechanisms 1 and 2 are a workaround for a pi bug. Once that fix reaches a pi release you install,
they duplicate work pi does correctly - expect one extra summarization per threshold crossing, and
a brief goal pause/resume. Mechanisms 3 and 4 stay useful either way. There is no version sniffing:
the extension always carries its own trigger, so it behaves the same on every pi 0.85.x.

## Requirements

- pi 0.85.0 or later.
- For the goal-aware parts: the [`pi-codex-goal`](https://github.com/fitchmultz/pi-codex-goal)
  package, which provides `/goal` and the goal session entries this extension reads. Without it the
  extension still loads and captures checkpoints, but no goal is ever "active".

## Install

```sh
pi install git:github.com/pat58151/pi-goal-compact@v1.1.1
pi install npm:pi-codex-goal          # for the goal-aware behavior
```

Or drop the file in by hand:

```sh
cp extensions/goal-compact.ts ~/.pi/agent/extensions/
```

(`~/.pi/agent` is the default agent dir; if `PI_CODING_AGENT_DIR` is set, use
`$PI_CODING_AGENT_DIR/extensions/`.) Then run `/reload` inside pi.

## Configuration

Size the trigger in `settings.json`:

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
| `PI_GOAL_PREP_LEAD_TOKENS` | `16000` | Headroom between the checkpoint prompt and the trigger. |

So the documented setup asks for a checkpoint at ~368k and compacts at ~384k. Keep
`compaction.keepRecentTokens` well below the trigger; the default (20000) is fine.

## Verification

Tested with an SDK harness driving a real goal-shaped run (a `pi-codex-goal` goal entry plus large
tool results), on stock 0.85.1 and on a 0.85.1 with the upstream fix applied locally:

| Scenario | Stock 0.85.1 | With the fix applied |
| --- | --- | --- |
| Checkpoint captured, folded into the summary | yes, verbatim | yes, verbatim |
| Context crossing the threshold mid-goal | compacts (abort + resume) | compacts (native, then the trigger's pass) |
| Repeated crossings in a continuing run | nothing unbounded | nothing unbounded |

The turn-boundary recovery path is defensive: it is the branch v1.0.0 hit in practice
("Nothing to compact"), and it did not trigger in these runs because pi's abort added an assistant
message that supplied a cut point.

## Notes and caveats

- The extension registers a `save_checkpoint` tool. If another extension registers a tool with the
  same name, the last registration wins.
- `ctx.compact()` aborts the current run first, which pauses the goal; the extension resumes it with
  `/goal resume` when compaction completes. Expect a brief pause at each trigger.
- The trigger only fires on tool-use turns while a goal is active, at most once per context level,
  and re-arms once the context drops back below the threshold.
- A captured checkpoint is consumed by the next compaction and is never reused. If summarization
  fails, the checkpoint alone becomes the summary rather than being lost.
- Goal state is read from pi session entries, so it survives resume, fork, and reload without
  external state.

## License

MIT
