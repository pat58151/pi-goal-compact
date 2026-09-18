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
   at a tool result. If compaction fails with "Nothing to compact" and the branch really ends at a
   tool result, the extension asks the model for one short plain assistant turn, which gives pi a
   cut point, and compacts on the following turn. This is attempted at most once per context level.
   Any other refusal is terminal: pi is then saying the history itself is not summarizable at that
   size, so the extension stops attempting a compaction until the context changes instead of
   spending a model turn per retry.

3. **Pre-compaction checkpoint.** A fixed number of tokens before the trigger, the live model is
   asked to hand off its working state through a `save_checkpoint` tool (or a plain-text
   `## Checkpoint` message). The text is captured out-of-band as a durable session entry and
   prepended **verbatim** to the next compaction summary, so in-progress work, touched files,
   decisions, next steps and exact paths/errors survive byte-for-byte. Capturing a checkpoint does
   not itself trigger compaction: it waits for the real threshold crossing, so it always lands in a
   compaction pi agrees to perform.

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
pi install git:github.com/pat58151/pi-goal-compact@v1.1.2
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

Tested with SDK harnesses driving goal-shaped runs (a `pi-codex-goal` goal entry plus large tool
results), on stock 0.85.1 and on a 0.85.1 with the upstream fix applied locally:

| Scenario | Stock 0.85.1 | With the fix applied |
| --- | --- | --- |
| Checkpoint captured, folded into the summary | yes, verbatim | yes, verbatim |
| Context crossing the threshold mid-goal | compacts (abort + resume) | compacts (native, then the trigger's pass) |
| A refusal pi will not perform | one attempt, then quiet | one attempt, then quiet |
| Repeated crossings in a continuing run | nothing unbounded | nothing unbounded |

The turn-boundary recovery path is defensive. It is the branch v1.0.0 hit in practice ("Nothing to
compact"), and it rarely fires on its own because the abort that precedes `ctx.compact()` appends
an assistant message, which already supplies a cut point. v1.1.2 therefore also treats a refusal as
terminal rather than retrying it.

Fixed in v1.1.2: capturing a checkpoint used to compact immediately, at a context size pi can
refuse ("Nothing to compact (session too small)"). Each refusal asked for a new turn boundary, the
boundary turn re-entered the same path, and the cycle repeated without limit - one model turn per
attempt, context drifting upward, nothing ever compacted. Reproduced deterministically with a
context below `keepRecentTokens` at the prep point, which is what happens whenever
`keepRecentTokens >= contextWindow - reserveTokens - PI_GOAL_PREP_LEAD_TOKENS`.

## Notes and caveats

- The extension registers a `save_checkpoint` tool. If another extension registers a tool with the
  same name, the last registration wins.
- `ctx.compact()` aborts the current run first, which pauses the goal; the extension resumes it with
  `/goal resume` when compaction completes. Expect a brief pause at each trigger.
- The trigger only fires on tool-use turns while a goal is active, at most once per context level,
  and re-arms once the context drops back below the threshold.
- A refused compaction is not retried. The extension records the context size where pi refused and
  stays quiet until the context drops back below it, or until a compaction happens for another
  reason (`/compact`, pi's own post-run check). The goal is resumed either way, because the abort
  that precedes `ctx.compact()` pauses it.
- A captured checkpoint is folded into the next compaction that actually completes, and is
  cleared then. A cancelled or failed compaction does not consume it, so the text is never lost;
  if summarization fails outright, the checkpoint alone becomes the summary.
- Goal state is read from pi session entries, so it survives resume, fork, and reload without
  external state.

## License

MIT
