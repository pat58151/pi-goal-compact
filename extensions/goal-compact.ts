/**
 * Goal-aware compaction
 *
 * Three mechanisms keep context bounded:
 *
 * 1. A mid-run trigger (see the `turn_end` handler) that compacts once the
 *    context crosses `contextWindow - reserveTokens`. pi's native threshold
 *    compaction only runs at agent-run boundaries, but a /goal keeps the
 *    agent inside one long tool-call run, so those boundaries are rare and
 *    context would otherwise overshoot the threshold. This hook fires on every
 *    tool-use turn, aborts + compacts, then resumes the goal via `/goal resume`.
 *
 * 2. pi's native threshold compaction, which still applies to ordinary
 *    sessions and fires between runs. Configure `compaction.reserveTokens` in
 *    settings so the trigger fires at the desired context size. For deepseek's
 *    1M-token window, `reserveTokens: 616000` triggers at ~384k.
 *
 * 3. For active /goal sessions, a pre-compaction checkpoint (also in
 *    `turn_end`) that, a fixed amount of tokens *before* the compaction
 *    trigger, prompts the model to hand off its working state through the
 *    `save_checkpoint` tool (or a plain-text `## Checkpoint` message if the
 *    tool is unavailable). The text is captured out-of-band (as a durable
 *    custom entry) and prepended verbatim to the compaction summary, so
 *    in-progress work, touched files, decisions, and next steps survive
 *    byte-for-byte instead of being re-summarized.
 *
 * When a /goal (pi-goal) is active, compaction summaries are generated
 * with the goal in mind: the goal objective, status, and budget are injected
 * into the summarization prompt so the summary preserves goal-relevant
 * decisions, progress, and next steps. Falls back to pi's default compaction
 * if the goal-aware pass fails.
 *
 * Threshold auto-compaction applies to all sessions (goal or not). Every
 * compaction — threshold, manual /compact, or overflow — is summarized here
 * with a bounded output budget, so pi's reserveTokens-derived default summary
 * budget is never used.
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
 * Tokens of headroom between the pre-compaction checkpoint prompt and the
 * compaction trigger. With deepseek's ~384k trigger this fires at ~368k.
 * Override with `PI_GOAL_PREP_LEAD_TOKENS`.
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

function readJson(path: string): Record<string, unknown> | null {
	try {
		return JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>;
	} catch {
		return null;
	}
}

/**
 * Read `compaction.reserveTokens` the same way pi does: global settings first,
 * then project `.pi/settings.json` overrides, matching pi's precedence. Honors
 * `PI_GOAL_MIDRUN_RESERVE_TOKENS` as an explicit override for the mid-run trigger.
 */
function compactionSettings(cwd: string): CompactionConfig {
	const agentDir = process.env.PI_CODING_AGENT_DIR ?? join(homedir(), ".pi", "agent");
	const global = (readJson(join(agentDir, "settings.json"))?.compaction ?? {}) as Record<string, unknown>;
	const project = (readJson(join(cwd, ".pi", "settings.json"))?.compaction ?? {}) as Record<string, unknown>;
	const merged = { ...global, ...project };
	const override = Number(process.env.PI_GOAL_MIDRUN_RESERVE_TOKENS);
	return {
		enabled: merged.enabled !== false,
		reserveTokens:
			Number.isFinite(override) && override > 0
				? override
				: typeof merged.reserveTokens === "number"
					? merged.reserveTokens
					: DEFAULT_RESERVE_TOKENS,
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
// Compaction summary (goal-aware when a goal is active)
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
 * Prompt injected shortly before compaction, asking the live model (which still
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
	let settings: CompactionConfig | null = null;

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
		// tick, after pi has cleared _compactionAbortController. Re-check a few
		// times because the pause can land on a slightly later async chain.
		const resumeGoalIfPaused = (attempt: number) => {
			setTimeout(() => {
				if (readGoal(ctx)?.status === "paused") {
					pi.sendUserMessage("/goal resume", { expandPromptTemplates: true });
				} else if (attempt < 3) {
					resumeGoalIfPaused(attempt + 1);
				}
			}, attempt === 0 ? 0 : 100);
		};

		ctx.compact({
			onComplete: () => {
				midRunCompactionArmed = false;
				resumeGoalIfPaused(0);
			},
			onError: (error) => {
				midRunCompactionArmed = false;
				// ctx.compact() aborts the run first; that abort produces an error
				// turn at the run boundary, which triggers pi's own native
				// auto-compaction. When the native pass wins the race, our compact()
				// reports "Already compacted" — the session was compacted, so just
				// resume the (now paused) goal the same way onComplete would.
				if (error.message === "Already compacted") {
					resumeGoalIfPaused(0);
				} else {
					// "Nothing to compact (session too small)" or any other failure: no
					// compaction happened, so resuming would retrigger the same trigger
					// and loop. Drop any pending checkpoint so a captured-but-unconsumed
					// checkpoint does not retrigger the same failure in a loop, leave
					// the goal paused, and surface the error.
					pendingCheckpoint = null;
					ctx.ui.notify(`Mid-goal compaction failed: ${error.message}`, "warning");
				}
			},
		});
	};

	pi.on("session_start", async (_event, ctx) => {
		midRunCompactionArmed = false;
		lastTriggerTokens = null;
		prepInjected = false;
		pendingCheckpoint = null;
		settings = compactionSettings(ctx.cwd);
	});

	// Mid-run trigger. A /goal keeps the agent inside one long tool-call run, so
	// pi's native threshold check (which only runs at run boundaries) never fires.
	// Compact here on tool-use turns: either as soon as a checkpoint is captured,
	// or once context crosses the hard threshold (contextWindow - reserveTokens).
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

		// 1) Compact immediately once a checkpoint has been captured — but only
		//    when that checkpoint was requested via the pre-compaction prompt
		//    (prepInjected). The model also calls save_checkpoint voluntarily at
		//    progress milestones; those must NOT trigger an immediate compaction
		//    at whatever the context happens to be (e.g. 115k/133k/180k tokens,
		//    far below the 384k prep threshold). A voluntary checkpoint is left
		//    stashed and is folded verbatim into the next threshold compaction.
		if (pendingCheckpoint !== null && prepInjected) {
			compactNow(ctx);
			return;
		}

		// 2) Plain-text fallback: if a prep prompt was injected and the model
		//    replied with a text checkpoint instead of calling the tool, capture
		//    it verbatim and compact now, exactly like the tool path.
		if (prepInjected) {
			const text = extractCheckpointText(message);
			if (text !== null) {
				pendingCheckpoint = capCheckpoint(text);
				pi.appendEntry(CHECKPOINT_ENTRY_TYPE, {
					text: pendingCheckpoint,
					goalId: goal.goalId ?? null,
					at: Date.now(),
				});
				ctx.ui.notify("Captured plain-text checkpoint, compacting...", "info");
				compactNow(ctx);
				return;
			}
		}

		// Only tool-use turns continue the same run. A natural "stop" is handled
		// by pi's own threshold check at run end.
		if (message.stopReason !== "toolUse") {
			return;
		}

		const usage = ctx.getContextUsage();
		if (!usage || usage.tokens === null) {
			return;
		}
		const threshold = usage.contextWindow - config.reserveTokens;
		const prepThreshold = threshold - prepLeadTokens();

		if (usage.tokens <= threshold) {
			// Back under the compaction threshold: re-arm the loop guard for next time.
			lastTriggerTokens = null;

			// Once we cross the prep threshold (but are still below the compaction
			// trigger), ask the model to record its working state via the checkpoint
			// tool, so the upcoming compaction carries it verbatim.
			if (prepThreshold > 0 && usage.tokens > prepThreshold && !prepInjected) {
				prepInjected = true;
				// Only prompt if the checkpoint tool is actually available; otherwise
				// fall through to the hard-threshold compaction below.
				if (pi.getActiveTools().includes(CHECKPOINT_TOOL_NAME)) {
					ctx.ui.notify(
						`Context at ${usage.tokens.toLocaleString()} tokens, asking the model to record a pre-compaction checkpoint...`,
						"info",
					);
					try {
						pi.sendUserMessage(buildPrepPrompt(goal), { deliverAs: "steer" });
					} catch (error) {
						prepInjected = false;
						const message = error instanceof Error ? error.message : String(error);
						ctx.ui.notify(`Pre-compaction checkpoint prompt failed: ${message}`, "warning");
					}
				} else {
					ctx.ui.notify(
						`Pre-compaction checkpoint skipped: ${CHECKPOINT_TOOL_NAME} tool is not active (relying on hard-threshold compaction).`,
						"warning",
					);
				}
			} else if (usage.tokens <= prepThreshold) {
				// Dropped back below the prep threshold (e.g. after a compaction):
				// re-arm the checkpoint prompt for the next cycle.
				prepInjected = false;
			}

			return;
		}

		// 2) Hard backstop: context crossed the compaction threshold.
		if (lastTriggerTokens !== null && usage.tokens >= lastTriggerTokens) {
			// A compaction already ran at this context level (or higher) and did not
			// reduce the context below it. Re-compacting would just loop; leave it to
			// pi's native threshold compaction or the user.
			return;
		}
		lastTriggerTokens = usage.tokens;
		compactNow(ctx);
	});

	pi.on("session_before_compact", async (event, ctx) => {
		const goal = readGoal(ctx);

		// Consume any pending checkpoint captured by the save_checkpoint tool, so
		// it is folded verbatim into this summary and never reused later.
		const checkpoint = pendingCheckpoint;
		pendingCheckpoint = null;

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

	// Re-arm the checkpoint prompt after every compaction, so the next context
	// climb requests a fresh checkpoint even if the compacted context never
	// dropped below the prep threshold.
	pi.on("session_compact", async () => {
		prepInjected = false;
	});
}
