/**
 * Goal-aware compaction
 *
 * pi already compacts on its own: threshold compaction runs before each next
 * assistant response inside a run (`prepareNextTurnWithContext` ->
 * `_compactBeforeNextAssistantResponse`), again after a run ends
 * (`_handlePostAgentRun` -> `_checkCompaction`), on context overflow, and on
 * manual `/compact`. This extension deliberately adds no trigger of its own and
 * never calls `ctx.compact()`. It makes pi's existing compactions behave better:
 *
 * 1. Pre-compaction checkpoint. A fixed amount of tokens *before* pi's threshold
 *    (`contextWindow - compaction.reserveTokens`), the live model is asked to hand
 *    off its working state through the `save_checkpoint` tool (or a plain-text
 *    `## Checkpoint` message as a fallback). The text is captured out-of-band as a
 *    durable custom entry and prepended **verbatim** to the next compaction summary,
 *    so in-progress work, touched files, decisions and next steps survive
 *    byte-for-byte instead of being re-summarized.
 *
 * 2. Goal-aware summaries. While a /goal (pi-goal) is active, the summary prompt
 *    carries the goal objective, status and budget, and the summary is generated
 *    with a bounded output budget (SUMMARY_MAX_TOKENS) instead of pi's
 *    reserveTokens-derived budget, which can be as large as the model's max output.
 *
 * Hooks used:
 *   session_start          reset per-session state and read compaction settings
 *   turn_end               request a checkpoint short of the threshold; capture a
 *                          plain-text checkpoint reply
 *   session_before_compact supply the goal-aware summary and fold in the checkpoint
 *   session_compact        re-arm the checkpoint prompt for the next context climb
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
 * Tokens of headroom between the checkpoint prompt and pi's compaction trigger.
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
 * then project `.pi/settings.json` overrides, matching pi's precedence. Only used
 * to place the checkpoint prompt relative to pi's own threshold.
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
 * Prompt injected shortly before pi compacts, asking the live model (which still
 * holds its full working state) to hand off a checkpoint via the
 * `save_checkpoint` tool. The tool captures the text out-of-band so it can be
 * carried verbatim through compaction instead of being re-summarized. pi's own
 * threshold compaction then does the actual compaction.
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

	pi.on("session_start", async (_event, ctx) => {
		prepInjected = false;
		pendingCheckpoint = null;
		settings = compactionSettings(ctx.cwd);
	});

	// Decide when to ask for a checkpoint. The prompt is injected a fixed amount of
	// tokens *before* pi's threshold (contextWindow - reserveTokens) so the captured
	// state is folded into the compaction that follows. Compaction itself is pi's
	// job: this hook never triggers it.
	pi.on("turn_end", async (event, ctx) => {
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

		// Plain-text fallback: the model replied to the prep prompt with a
		// `## Checkpoint` message instead of calling the tool. Capture it verbatim;
		// the next compaction folds it into the summary.
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
				return;
			}
		}

		// Only tool-use turns are actively working. A natural stop ends the goal's
		// turn; pi still checks its own threshold at run end.
		if (message.stopReason !== "toolUse") {
			return;
		}

		const usage = ctx.getContextUsage();
		if (!usage || usage.tokens === null) {
			return;
		}
		const threshold = usage.contextWindow - config.reserveTokens;
		const prepThreshold = threshold - prepLeadTokens();

		if (usage.tokens <= prepThreshold) {
			// Dropped back below the prep threshold (e.g. after a compaction):
			// re-arm the checkpoint prompt for the next cycle.
			prepInjected = false;
			return;
		}
		if (usage.tokens > threshold) {
			// pi's threshold compaction is already due. Asking for a checkpoint now
			// would land after it, so leave the cycle alone.
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
			`Context at ${usage.tokens.toLocaleString()} tokens, asking the model to record a pre-compaction checkpoint...`,
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

	// Re-arm the checkpoint prompt after every compaction, so the next context climb
	// requests a fresh checkpoint even if the compacted context never dropped below
	// the prep threshold.
	pi.on("session_compact", async () => {
		prepInjected = false;
	});
}
