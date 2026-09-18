/**
 * Goal-aware compaction
 *
 * pi compacts on its own: threshold compaction runs before each next assistant
 * response inside a run, again after a run ends, on context overflow, and on manual
 * `/compact`. This extension makes those compactions preserve working state, and it
 * also carries its own mid-run trigger, because pi cannot always do the job:
 *
 * 1. Mid-run trigger. pi's mid-run threshold check silently does nothing when the
 *    newest turn's tool results alone exceed `compaction.keepRecentTokens` (default
 *    20000): the cut-point search degenerates, nothing is left to summarize, and
 *    `_runAutoCompaction` returns before emitting anything (upstream issue #9740).
 *    The post-run check masks that in ordinary sessions, but a `/goal` run never ends,
 *    so the context overshoots. This extension therefore compacts itself once the
 *    context crosses `contextWindow - reserveTokens`, aborts + compacts, and resumes
 *    the paused goal afterwards.
 *
 * 2. Turn-boundary recovery. `ctx.compact()` is the manual compaction path, and pi can
 *    only cut at a user/assistant message - never at a tool result. When the last entry
 *    is a tool result (the usual state mid-run) compaction fails with "Nothing to
 *    compact". The extension then asks the model for one short plain assistant turn,
 *    which gives pi a valid cut point, and compacts on the following turn. This is
 *    attempted at most once per context level and only while the last entry really is a
 *    tool result; any other refusal is terminal, because pi is then saying the history
 *    itself is not summarizable and no turn boundary can change that.
 *
 *    The checkpoint capture path does not compact on its own. It waits until the context
 *    is genuinely over the threshold, so a captured checkpoint is folded into a real
 *    threshold compaction instead of forcing one pi would refuse.
 *
 * 3. Pre-compaction checkpoint. A fixed amount of tokens *before* the trigger, the live
 *    model is asked to hand off its working state through the `save_checkpoint` tool (or
 *    a plain-text `## Checkpoint` message). The text is captured out-of-band as a durable
 *    custom entry and prepended **verbatim** to the compaction summary, so in-progress
 *    work, touched files, decisions and next steps survive byte-for-byte.
 *
 * 4. Goal-aware summaries. While a /goal (pi-goal) is active, the summary prompt carries
 *    the goal objective, status and budget, and the summary is produced with a bounded
 *    output budget (SUMMARY_MAX_TOKENS) instead of pi's reserveTokens-derived budget.
 *
 * Once upstream #9740 is fixed, mechanisms 1 and 2 become redundant (they duplicate a
 * compaction pi would then do correctly). Mechanisms 3 and 4 stay useful either way.
 *
 * Configuration lives in pi's own `compaction` settings; see README.md.
 */

import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { convertToLlm, serializeConversation } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

/** Custom entry type used by the pi-goal package (identifier is not renamed for persistence). */
const GOAL_ENTRY_TYPE = "pi-codex-goal";
/** Max output tokens for the generated summary. */
const SUMMARY_MAX_TOKENS = 8192;
/** Fallback reserve when no `compaction.reserveTokens` is configured (pi's default). */
const DEFAULT_RESERVE_TOKENS = 16384;
/**
 * Tokens of headroom between the checkpoint prompt and the compaction trigger.
 * With the documented 616000 reserve on a 1M window (trigger ~384k) this fires at
 * ~368k. Override with `PI_GOAL_PREP_LEAD_TOKENS`.
 */
const PREP_LEAD_TOKENS_DEFAULT = 16000;
/** Tool the model calls to hand off its working state before compaction. */
const CHECKPOINT_TOOL_NAME = "save_checkpoint";
/** Durable custom entry type used to persist checkpoints across the session. */
const CHECKPOINT_ENTRY_TYPE = "pi-goal-checkpoint";
/** Hard cap on the checkpoint length (chars) to bound the verbatim block. */
const MAX_CHECKPOINT_CHARS = 16000;
/** Matches the `## Checkpoint` heading the model emits as a plain-text fallback. */
const CHECKPOINT_MARKER = /^##\s*Checkpoint\b/m;
/** pi's error when there is no valid cut point to compact at. */
const NOTHING_TO_COMPACT = "Nothing to compact";
/** pi's error when a compaction already ran and the context did not need another. */
const ALREADY_COMPACTED = "Already compacted";

interface CompactionConfig {
	enabled: boolean;
	reserveTokens: number;
}

function prepLeadTokens(): number {
	const override = Number(process.env.PI_GOAL_PREP_LEAD_TOKENS);
	return Number.isFinite(override) && override > 0 ? override : PREP_LEAD_TOKENS_DEFAULT;
}

/**
 * Bound a checkpoint to MAX_CHECKPOINT_CHARS, truncating at the last newline so
 * a line is never severed mid-token.
 */
function capCheckpoint(text: string): string {
	if (text.length <= MAX_CHECKPOINT_CHARS) {
		return text;
	}
	const cut = text.lastIndexOf("\n", MAX_CHECKPOINT_CHARS);
	return `${text.slice(0, cut > 0 ? cut : MAX_CHECKPOINT_CHARS)}\n[truncated]`;
}

/**
 * If an assistant message is a plain-text checkpoint (no tool calls, and it
 * contains the `## Checkpoint` heading), return its text; otherwise null.
 */
function extractCheckpointText(message: { role?: string; content?: unknown }): string | null {
	if (message.role !== "assistant" || !Array.isArray(message.content)) {
		return null;
	}
	const parts: string[] = [];
	for (const block of message.content) {
		if (!block || typeof block !== "object") {
			continue;
		}
		const type = (block as { type?: unknown }).type;
		if (type === "toolCall") {
			// A reply that also calls tools is not a clean plain-text checkpoint.
			return null;
		}
		if (type === "text" && typeof (block as { text?: unknown }).text === "string") {
			parts.push((block as { text: string }).text);
		}
	}
	const text = parts.join("\n").trim();
	if (!text) {
		return null;
	}
	return CHECKPOINT_MARKER.test(text) ? text : null;
}

/**
 * True when the branch ends at a tool result, which is the only state a one-line
 * assistant turn can fix: pi can cut at a user/assistant message and never at a tool
 * result, so the cut-point search has nothing to land on. Metadata entries (pi-goal
 * appends its own goal entries on every turn and on abort) carry no context message,
 * so they are skipped exactly like pi's own cut-point scan skips them.
 */
function lastEntryIsToolResult(ctx: ExtensionContext): boolean {
	const branch = ctx.sessionManager.getBranch();
	for (let i = branch.length - 1; i >= 0; i--) {
		const entry = branch[i] as { type?: string; message?: { role?: string } };
		if (entry.type !== "message") {
			continue;
		}
		return entry.message?.role === "toolResult";
	}
	return false;
}

function readJson(path: string): Record<string, unknown> | null {
	try {
		return JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>;
	} catch {
		return null;
	}
}

/**
 * Read `compaction.reserveTokens` the same way pi does: global settings first,
 * then project `.pi/settings.json` overrides, matching pi's precedence.
 */
function compactionSettings(cwd: string): CompactionConfig {
	const agentDir = process.env.PI_CODING_AGENT_DIR ?? join(homedir(), ".pi", "agent");
	const global = (readJson(join(agentDir, "settings.json"))?.compaction ?? {}) as Record<string, unknown>;
	const project = (readJson(join(cwd, ".pi", "settings.json"))?.compaction ?? {}) as Record<string, unknown>;
	const merged = { ...global, ...project };
	return {
		enabled: merged.enabled !== false,
		reserveTokens: typeof merged.reserveTokens === "number" ? merged.reserveTokens : DEFAULT_RESERVE_TOKENS,
	};
}

// ---------------------------------------------------------------------------
// Goal state (mirrors pi-codex-goal's reconstructGoal over session entries)
// ---------------------------------------------------------------------------

interface GoalUsageLike {
	tokensUsed: number;
	activeSeconds: number;
}

interface ThreadGoalLike {
	goalId: string;
	objective: string;
	status: string;
	tokenBudget: number | null;
	usage: GoalUsageLike;
	createdAt: number;
	updatedAt: number;
}

interface GoalCustomEntryLike {
	version?: number;
	kind?: string;
	goal?: ThreadGoalLike;
	goalId?: string;
	status?: string;
	usage?: GoalUsageLike;
	updatedAt?: number;
	clearedGoalId?: string | null;
	at?: number;
}

function readGoal(ctx: ExtensionContext): ThreadGoalLike | null {
	let goal: ThreadGoalLike | null = null;
	for (const entry of ctx.sessionManager.getBranch()) {
		if (entry.type !== "custom" || entry.customType !== GOAL_ENTRY_TYPE) {
			continue;
		}
		const data = entry.data as GoalCustomEntryLike | undefined;
		if (!data || typeof data !== "object") {
			continue;
		}
		if (data.kind === "clear") {
			goal = null;
		} else if (data.kind === "set" && data.goal) {
			goal = { ...data.goal };
		} else if (data.kind === "usage" && goal && data.goalId === goal.goalId) {
			const current: ThreadGoalLike = goal;
			goal = {
				...current,
				status: data.status ?? current.status,
				usage: data.usage ?? current.usage,
				updatedAt: data.updatedAt ?? current.updatedAt,
			};
		}
	}
	return goal;
}

// ---------------------------------------------------------------------------
// Prompts
// ---------------------------------------------------------------------------

function goalBlock(goal: ThreadGoalLike): string {
	return [
		"## Active Goal",
		`Objective: ${goal.objective}`,
		`Status: ${goal.status}`,
		`Token budget: ${goal.tokenBudget === null ? "unlimited" : goal.tokenBudget.toLocaleString()}`,
		`Tokens used: ${goal.usage.tokensUsed.toLocaleString()}`,
	].join("\n");
}

function buildSummaryPrompt(
	goal: ThreadGoalLike | null,
	conversationText: string,
	previousContext: string,
	checkpoint: string | null = null,
): string {
	const checkpointBlock = checkpoint
		? `\nA pre-compaction checkpoint was already captured verbatim by the main model and will be prepended to your summary. Do NOT reproduce its contents; summarize only what it does not already cover.\n\n<checkpoint>\n${checkpoint}\n</checkpoint>\n`
		: "";
	if (goal) {
		return `You are a conversation summarizer. A goal is being actively tracked for this session. Keep the goal at the center of the summary: preserve everything needed to continue making progress toward it, and drop details that do not matter for the goal.

${goalBlock(goal)}
${previousContext}
${checkpointBlock}
Summarize the conversation below as structured markdown with these sections:

## Goal
Restate the active goal objective above verbatim.

## Constraints & Preferences
- Requirements the user stated that affect the goal

## Progress
### Done
- [x] Completed work toward the goal

### In Progress
- [ ] Current work

### Blocked
- Issues, if any

## Key Decisions
- **[Decision]**: [Rationale] — only decisions that matter for the goal

## Next Steps
1. Concrete next actions toward the goal

## Critical Context
- Data, file paths, and facts needed to continue the goal work

Be thorough but concise. This summary replaces the summarized conversation, so it must stand alone.

<conversation>
${conversationText}
</conversation>`;
	}

	return `You are a conversation summarizer. Create a structured context checkpoint summary that another LLM will use to continue the work.

${previousContext}
${checkpointBlock}
Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages.

<conversation>
${conversationText}
</conversation>`;
}

/**
 * Prompt injected shortly before the trigger, asking the live model (which still
 * holds its full working state) to hand off a checkpoint via the
 * `save_checkpoint` tool. The tool captures the text out-of-band so it can be
 * carried verbatim through compaction instead of being re-summarized.
 */
function buildPrepPrompt(goal: ThreadGoalLike | null): string {
	const goalContext = goal ? `${goalBlock(goal)}\n\n` : "";
	return `## Pre-compaction checkpoint

Your context window is nearly full and will be compacted soon. Before that happens, record a complete checkpoint of your current working state using the \`${CHECKPOINT_TOOL_NAME}\` tool, then stop. You will be resumed automatically after compaction.

${goalContext}IMPORTANT: Call the \`${CHECKPOINT_TOOL_NAME}\` tool in this turn, passing the full structured checkpoint as its argument. If that tool is unavailable, instead output the checkpoint as a plain-text message beginning with the line \`## Checkpoint\`. Either way, do not call any other tool.

Write the checkpoint in this EXACT structure:

## Checkpoint
- **Active goal**: [one-line objective]
- **Status**: [2-3 sentences on current progress]
- **Done**: [bulleted list of completed work]
- **In progress**: [what you are doing right now, including the tool you just ran]
- **Files touched**: [exact paths and what changed in each]
- **Key decisions**: [decisions and rationale]
- **Constraints / preferences**: [requirements the user stated]
- **Immediate next steps**: [numbered list of concrete next actions]
- **Critical facts**: [data, paths, error messages, IDs needed to continue]

Be precise and complete but concise. Preserve exact file paths, function names, command outputs, and error messages verbatim. This checkpoint will be carried verbatim through compaction, so it must stand alone.`;
}

/**
 * Ask for one short plain assistant turn. pi can only cut at a user/assistant
 * message, never at a tool result, so a run whose last entry is a tool result has
 * nothing to compact at. A single assistant message creates that cut point.
 */
function buildTurnBoundaryPrompt(goal: ThreadGoalLike | null): string {
	const goalContext = goal ? `${goalSummaryLine(goal)}\n\n` : "";
	return `## Compaction needs a turn boundary

Your context is over the compaction threshold, but the last entry in this session is a tool result, and pi cannot summarize from there. Do not call any tools. Reply with one short sentence describing what you are doing right now, then stop. You will be resumed right after compaction.

${goalContext}If it helps the summary, include in that sentence the exact file paths you are currently working on. Nothing else.`;
}

function goalSummaryLine(goal: ThreadGoalLike): string {
	return `Active goal: ${goal.objective}`;
}

export default function goalCompactExtension(pi: ExtensionAPI) {
	// Set only while a compaction this extension requested is in flight, so a
	// compaction pi triggered on its own (or a user /compact) is never followed
	// by a resume.
	let midRunCompactionArmed = false;
	// Context size at the last mid-run trigger. Guards against a compact/resume
	// loop when a compaction completes but the context does not drop below the
	// threshold (e.g. a huge kept tail). Re-armed when context falls back under
	// the threshold.
	let lastTriggerTokens: number | null = null;
	// Whether the pre-compaction checkpoint prompt has been injected for the
	// current compaction cycle. Re-armed when context drops back below the prep
	// threshold (i.e. after a compaction).
	let prepInjected = false;
	// Checkpoint text captured by the save_checkpoint tool, awaiting the next
	// compaction so it can be carried verbatim into the summary.
	let pendingCheckpoint: string | null = null;
	// Set after a compaction failed for lack of a cut point; cleared once a short
	// assistant turn has been requested (and again after a successful compaction).
	let awaitingAssistantTurn = false;
	// Set when pi refuses a compaction ("Nothing to compact") in a way a turn boundary
	// cannot fix. While this is set no trigger path attempts another compaction: the
	// refusal is not retryable, and retrying costs one model turn per attempt. Lifted
	// once the context drops back below blockedAtTokens, or by a reset event (successful
	// compaction, /compact, session start).
	let compactionBlocked = false;
	let blockedAtTokens: number | null = null;
	// One-shot latch for turn-boundary recovery, so a model that answers the nudge with
	// another tool call cannot re-enter the nudge path forever.
	let turnBoundaryRequested = false;
	let settings: CompactionConfig | null = null;

	const clearCompactionLatch = () => {
		compactionBlocked = false;
		blockedAtTokens = null;
		turnBoundaryRequested = false;
	};

	// The model hands off its working state through this tool. Capturing it here
	// (rather than as a plain message) keeps it out of the summarization path:
	// the text is stored verbatim and prepended to the compaction summary.
	pi.registerTool({
		name: CHECKPOINT_TOOL_NAME,
		label: "Save Checkpoint",
		description:
			"Record a pre-compaction checkpoint of the current working state, carried verbatim through context compaction.",
		promptSnippet:
			"Before context compaction, record goal status, completed/in-progress work, touched files, decisions, constraints, next steps, and critical facts so work resumes seamlessly.",
		parameters: Type.Object({
			checkpoint: Type.String({
				description: "Structured markdown checkpoint of the current working state.",
			}),
		}),
		async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
			const text = typeof params.checkpoint === "string" ? params.checkpoint.trim() : "";
			if (text) {
				pendingCheckpoint = capCheckpoint(text);
				pi.appendEntry(CHECKPOINT_ENTRY_TYPE, {
					text: pendingCheckpoint,
					goalId: readGoal(ctx)?.goalId ?? null,
					at: Date.now(),
				});
			}
			return {
				content: [{ type: "text", text: "Checkpoint saved; compaction will proceed." }],
				details: {},
			};
		},
	});

	// Shared compaction trigger: notify, abort + compact, then resume the paused
	// goal once compaction completes (or if pi's native pass wins the race).
	const compactNow = (ctx: ExtensionContext): void => {
		midRunCompactionArmed = true;
		const usage = ctx.getContextUsage();
		const tokens = usage?.tokens ?? null;
		ctx.ui.notify(
			`Context at ${tokens !== null ? tokens.toLocaleString() : "unknown"} tokens, compacting mid-goal...`,
			"info",
		);

		// The abort that precedes compaction pauses the goal. Resume on a fresh
		// tick, after pi has cleared _compactionAbortController. The pause lands on a
		// later async chain (pi-goal pauses from onAgentEnd), so poll for a while
		// instead of giving up after a few hundred milliseconds.
		const resumeGoalIfPaused = (attempt: number) => {
			const delay = [0, 250, 500, 1000, 2000, 3000][attempt] ?? 3000;
			setTimeout(() => {
				if (readGoal(ctx)?.status === "paused") {
					pi.sendUserMessage("/goal resume", { expandPromptTemplates: true });
				} else if (attempt < 5) {
					resumeGoalIfPaused(attempt + 1);
				}
			}, delay);
		};

		ctx.compact({
			onComplete: () => {
				midRunCompactionArmed = false;
				awaitingAssistantTurn = false;
				clearCompactionLatch();
				resumeGoalIfPaused(0);
			},
			onError: (error) => {
				midRunCompactionArmed = false;

				// ctx.compact() aborts the run first; that abort produces an error
				// turn at the run boundary, which triggers pi's own auto-compaction.
				// When the native pass wins the race, our compact() reports
				// "Already compacted" - the session was compacted, so resume the
				// (now paused) goal the same way onComplete would.
				if (error.message.includes(ALREADY_COMPACTED)) {
					awaitingAssistantTurn = false;
					clearCompactionLatch();
					resumeGoalIfPaused(0);
					return;
				}

				if (error.message.includes(NOTHING_TO_COMPACT)) {
					const refusedAt = ctx.getContextUsage()?.tokens ?? null;

					// A turn boundary only helps when the branch really ends at a tool
					// result, and only once per epoch. pi refuses the compaction for any
					// other reason because the history itself is not summarizable at this
					// size, and no assistant turn changes that.
					const canNudge = !turnBoundaryRequested && lastEntryIsToolResult(ctx);

					if (canNudge) {
						awaitingAssistantTurn = true;
						turnBoundaryRequested = true;
						ctx.ui.notify(
							"Compaction found no cut point; asking the model for a short turn boundary.",
							"info",
						);
						try {
							pi.sendUserMessage(buildTurnBoundaryPrompt(readGoal(ctx)), { deliverAs: "steer" });
						} catch (nudgeError) {
							awaitingAssistantTurn = false;
							turnBoundaryRequested = false;
							const text = nudgeError instanceof Error ? nudgeError.message : String(nudgeError);
							ctx.ui.notify(`Turn-boundary request failed: ${text}`, "warning");
						}
						return;
					}

					// Terminal. pi refused this compaction, so stop attempting one at
					// this context level instead of nudging forever. The captured
					// checkpoint stays pending, so a later successful compaction (pi's own
					// post-run check, a manual /compact, or a trigger once the context
					// falls back) still carries it verbatim. Without a token count there is
					// nothing to compare against later, so leave the latch off and simply
					// stop here.
					awaitingAssistantTurn = false;
					if (refusedAt !== null) {
						compactionBlocked = true;
						blockedAtTokens = refusedAt;
						ctx.ui.notify(
							"Compaction has nothing to summarize at this context level; not retrying until the context changes.",
							"warning",
						);
					}
					// The abort that precedes ctx.compact() pauses the goal; leaving it paused
					// would stop the run silently, so hand it back.
					resumeGoalIfPaused(0);
					return;
				}

				// Any other failure: no compaction happened, so surface the error and
				// stop attempting one at this context level rather than looping.
				const failedAt = ctx.getContextUsage()?.tokens ?? null;
				awaitingAssistantTurn = false;
				if (failedAt !== null) {
					compactionBlocked = true;
					blockedAtTokens = failedAt;
				}
				ctx.ui.notify(`Mid-goal compaction failed: ${error.message}`, "warning");
				resumeGoalIfPaused(0);
			},
		});
	};

	pi.on("session_start", async (_event, ctx) => {
		midRunCompactionArmed = false;
		lastTriggerTokens = null;
		prepInjected = false;
		pendingCheckpoint = null;
		awaitingAssistantTurn = false;
		clearCompactionLatch();
		settings = compactionSettings(ctx.cwd);
	});

	// Mid-run trigger. pi's own mid-run check can silently no-op (upstream #9740),
	// so this hook drives compaction from the turn boundary instead: once the context
	// crosses the threshold, or once a requested turn boundary has been produced. A
	// captured checkpoint never compacts on its own; it rides along with the next
	// threshold compaction.
	pi.on("turn_end", async (event, ctx) => {
		if (midRunCompactionArmed) {
			return;
		}

		const message = event.message as { role?: string; stopReason?: string; content?: unknown };
		if (message.role !== "assistant") {
			return;
		}

		const config = settings ?? (settings = compactionSettings(ctx.cwd));
		if (!config.enabled) {
			return;
		}

		const goal = readGoal(ctx);
		if (goal?.status !== "active") {
			return;
		}

		const usage = ctx.getContextUsage();
		const tokens = usage?.tokens ?? null;
		if (compactionBlocked) {
			if (tokens === null || blockedAtTokens === null || tokens >= blockedAtTokens) {
				// pi already refused a compaction at this context level, and the refusal is
				// not retryable. Stay quiet until the context drops below where it was
				// refused instead of spending a model turn per attempt.
				return;
			}
			clearCompactionLatch();
		}
		const threshold = usage ? usage.contextWindow - config.reserveTokens : null;
		const prepThreshold = threshold === null ? null : threshold - prepLeadTokens();

		// 0) The requested turn boundary arrived: the branch now ends with an
		//    assistant message, so pi has a cut point and compaction can proceed. Only
		//    if the context is still over the threshold, since the point of the nudge
		//    was the compaction that was due.
		if (awaitingAssistantTurn && message.stopReason !== "toolUse") {
			awaitingAssistantTurn = false;
			if (tokens !== null && threshold !== null && tokens > threshold) {
				compactNow(ctx);
			}
			return;
		}

		// 1) Plain-text fallback: the model replied with a `## Checkpoint` message
		//    instead of calling the tool. Capture it; it is folded into the next
		//    threshold compaction below, exactly like the tool path.
		if (prepInjected) {
			const text = extractCheckpointText(message);
			if (text !== null) {
				pendingCheckpoint = capCheckpoint(text);
				pi.appendEntry(CHECKPOINT_ENTRY_TYPE, {
					text: pendingCheckpoint,
					goalId: goal.goalId ?? null,
					at: Date.now(),
				});
				ctx.ui.notify("Captured plain-text checkpoint for the next compaction.", "info");
			}
		}

		// Only tool-use turns continue the same run. A natural "stop" is handled by
		// pi's own check at the run end.
		if (message.stopReason !== "toolUse") {
			return;
		}

		if (tokens === null || threshold === null || prepThreshold === null) {
			return;
		}

		if (tokens > threshold) {
			if (lastTriggerTokens !== null && tokens >= lastTriggerTokens) {
				// A compaction already ran at this context level (or higher) and did
				// not reduce the context below it. Retrying would loop; leave it to
				// pi or the user.
				return;
			}
			lastTriggerTokens = tokens;
			compactNow(ctx);
			return;
		}

		// Back under the compaction threshold: re-arm the loop guards for the next epoch.
		lastTriggerTokens = null;
		awaitingAssistantTurn = false;
		turnBoundaryRequested = false;

		if (tokens <= prepThreshold) {
			// Dropped back below the prep threshold (e.g. after a compaction):
			// re-arm the checkpoint prompt for the next cycle.
			prepInjected = false;
			return;
		}
		if (prepInjected) {
			return;
		}

		if (!pi.getActiveTools().includes(CHECKPOINT_TOOL_NAME)) {
			prepInjected = true;
			ctx.ui.notify(
				`Pre-compaction checkpoint skipped: ${CHECKPOINT_TOOL_NAME} tool is not active.`,
				"warning",
			);
			return;
		}

		prepInjected = true;
		ctx.ui.notify(
			`Context at ${tokens.toLocaleString()} tokens, asking the model to record a pre-compaction checkpoint...`,
			"info",
		);
		try {
			pi.sendUserMessage(buildPrepPrompt(goal), { deliverAs: "steer" });
		} catch (error) {
			prepInjected = false;
			const text = error instanceof Error ? error.message : String(error);
			ctx.ui.notify(`Pre-compaction checkpoint prompt failed: ${text}`, "warning");
		}
	});

	pi.on("session_before_compact", async (event, ctx) => {
		const goal = readGoal(ctx);

		// Read the checkpoint without consuming it: this hook also runs for compactions
		// that pi cancels or that fail afterwards, and a captured checkpoint must survive
		// those. It is cleared in session_compact, which only fires on a successful
		// compaction, so a failed attempt still carries it into the next one.
		const checkpoint = pendingCheckpoint;

		const { preparation, signal } = event;
		const { messagesToSummarize, turnPrefixMessages, tokensBefore, firstKeptEntryId, previousSummary } =
			preparation;

		const model = ctx.model;
		if (!model) {
			// No model to summarize with, but a captured checkpoint alone is
			// enough to carry the state forward.
			if (checkpoint) {
				return {
					compaction: {
						summary: `## Pre-compaction checkpoint (verbatim)\n${checkpoint}`,
						firstKeptEntryId,
						tokensBefore,
					},
				};
			}
			return;
		}

		// Include split-turn prefix messages when the cut lands mid-turn.
		const allMessages = [...messagesToSummarize, ...turnPrefixMessages];
		const conversationText = serializeConversation(convertToLlm(allMessages));
		const previousContext = previousSummary
			? `\n\nPrevious session summary for context:\n${previousSummary}`
			: "";

		const summaryMessages = [
			{
				role: "user" as const,
				content: [
					{
						type: "text" as const,
						text: buildSummaryPrompt(goal, conversationText, previousContext, checkpoint),
					},
				],
				timestamp: Date.now(),
			},
		];

		try {
			const response = await ctx.modelRegistry.complete(
				model,
				{ messages: summaryMessages },
				{
					maxTokens: SUMMARY_MAX_TOKENS,
					signal,
					cacheRetention: "none",
					sessionId: crypto.randomUUID(),
				},
			);

			const summary = response.content
				.filter((c): c is { type: "text"; text: string } => c.type === "text")
				.map((c) => c.text)
				.join("\n");

			if (!summary.trim()) {
				// A captured checkpoint alone is enough to carry the state forward.
				if (checkpoint) {
					return {
						compaction: {
							summary: `## Pre-compaction checkpoint (verbatim)\n${checkpoint}`,
							firstKeptEntryId,
							tokensBefore,
							usage: response.usage,
						},
					};
				}
				if (!signal.aborted) {
					ctx.ui.notify("Compaction summary was empty, using default compaction", "warning");
				}
				return;
			}

			// Carry the captured checkpoint verbatim, ahead of the summarized
			// conversation, so exact paths/errors/next steps survive byte-for-byte.
			const finalSummary = checkpoint
				? `## Pre-compaction checkpoint (verbatim)\n${checkpoint}\n\n---\n\n${summary}`
				: summary;

			return {
				compaction: {
					summary: finalSummary,
					firstKeptEntryId,
					tokensBefore,
					usage: response.usage,
				},
			};
		} catch (error) {
			// Summarization failed, but a captured checkpoint is still enough to
			// carry the state forward; use it alone rather than losing it.
			if (checkpoint) {
				return {
					compaction: {
						summary: `## Pre-compaction checkpoint (verbatim)\n${checkpoint}`,
						firstKeptEntryId,
						tokensBefore,
					},
				};
			}
			const message = error instanceof Error ? error.message : String(error);
			if (!signal.aborted) {
				ctx.ui.notify(`Compaction summary failed, using default compaction: ${message}`, "warning");
			}
			// Fall back to pi's default compaction.
			return;
		}
	});

	// Re-arm the checkpoint prompt after every compaction, so the next context climb
	// requests a fresh checkpoint even if the compacted context never dropped below
	// the prep threshold.
	pi.on("session_compact", async () => {
		prepInjected = false;
		awaitingAssistantTurn = false;
		// Only a compaction that actually happened consumes the captured checkpoint.
		pendingCheckpoint = null;
		clearCompactionLatch();
	});
}
