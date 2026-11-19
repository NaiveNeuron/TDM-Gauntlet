import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { after, describe, it } from "node:test";
import {
	UsageLedger,
	createLmCompleteHandler,
	modelSpecForRole,
	parseOptimizerModels,
	promptToContext,
	thinkingOptions,
} from "./lm.ts";
import type { OptimizerModels } from "./lm.ts";
import type { AssistantMessage, Context, Model, SimpleStreamOptions } from "./omp-types.ts";

const CURRENT: Model = { provider: "anthropic", id: "claude-current", api: "anthropic-messages" };
const SLOW: Model = { provider: "openai", id: "gpt-slow", api: "openai-responses" };
const EXACT: Model = { provider: "google", id: "gemini-exact", api: "google" };

interface Call {
	model: Model;
	context: Context;
	options: SimpleStreamOptions | undefined;
}

function reply(overrides: Partial<AssistantMessage> = {}): AssistantMessage {
	return {
		role: "assistant",
		content: [
			{ type: "thinking" },
			{ type: "text", text: "new " },
			{ type: "text", text: "instructions" },
		],
		provider: "openai",
		model: "gpt-slow-2026",
		usage: { input: 1200, output: 300, cacheRead: 10, cacheWrite: 0, totalTokens: 1510, cost: { total: 0.0125 } },
		stopReason: "stop",
		...overrides,
	};
}

function harness(optimizer: OptimizerModels, message: AssistantMessage = reply()) {
	const calls: Call[] = [];
	const entries: Array<{ customType: string; data: unknown }> = [];
	const statuses: Array<string | undefined> = [];
	const resolverCalls: Array<{ model: Model; sessionId: string | undefined }> = [];
	const optimizerLookups: Array<string | undefined> = [];
	const ledger = new UsageLedger({
		appendEntry: (customType, data) => entries.push({ customType, data }),
		setStatus: text => statuses.push(text),
	});
	const handler = createLmCompleteHandler({
		completeSimple: async (model, context, options) => {
			calls.push({ model, context, options });
			return message;
		},
		ctx: {
			models: {
				current: () => CURRENT,
				resolve: spec => ({ "@slow": SLOW, "google/gemini-exact": EXACT })[spec],
			},
			modelRegistry: {
				resolver: (model, sessionId) => {
					resolverCalls.push({ model, sessionId });
					return `resolver:${model.id}`;
				},
			},
			sessionManager: { getSessionId: () => "session-1" },
		},
		optimizerFor: runDir => {
			optimizerLookups.push(runDir);
			return optimizer;
		},
		ledger,
	});
	return { handler, calls, entries, statuses, resolverCalls, optimizerLookups, ledger };
}

const tmpRoots: string[] = [];
after(async () => {
	await Promise.all(tmpRoots.map(dir => rm(dir, { recursive: true, force: true })));
});

describe("role → model spec", () => {
	it("prefers the role model, then optimizer.model, then current", () => {
		const optimizer = { model: "@slow", reflection_model: "google/gemini-exact" };
		assert.equal(modelSpecForRole("reflection", optimizer), "google/gemini-exact");
		assert.equal(modelSpecForRole("structural", optimizer), "@slow");
		assert.equal(modelSpecForRole("structural", {}), "current");
		assert.throws(() => modelSpecForRole("toString", optimizer), /unknown lm.complete role/);
	});

	it("drops null/blank/non-string config values so defaults apply", () => {
		assert.deepEqual(
			parseOptimizerModels({ model: "current", reflection_model: null, structural_model: "  ", thinking: 3 }),
			{ model: "current" },
		);
		assert.deepEqual(parseOptimizerModels(undefined), {});
	});
});

describe("thinking mapping", () => {
	it("maps off/auto/effort levels to pi-ai options", () => {
		assert.deepEqual(thinkingOptions("off"), { disableReasoning: true });
		assert.deepEqual(thinkingOptions("auto"), {});
		assert.deepEqual(thinkingOptions(undefined), {});
		assert.deepEqual(thinkingOptions("high"), { reasoning: "high" });
		assert.deepEqual(thinkingOptions("minimal"), { reasoning: "minimal" });
		assert.throws(() => thinkingOptions("extreme"), /invalid optimizer.thinking "extreme"/);
		assert.throws(() => thinkingOptions("constructor"), /invalid optimizer.thinking/);
	});
});

describe("prompt conversion", () => {
	it("wraps a string prompt as one user message", () => {
		const context = promptToContext("rewrite the instructions", CURRENT);
		assert.equal(context.systemPrompt, undefined);
		assert.equal(context.messages.length, 1);
		assert.deepEqual(context.messages[0].content, [{ type: "text", text: "rewrite the instructions" }]);
	});

	it("splits chat lists into systemPrompt and messages", () => {
		const context = promptToContext(
			[
				{ role: "system", content: "You improve Jev programs." },
				{ role: "user", content: [{ type: "text", text: "first" }] },
				{ role: "assistant", content: "draft" },
				{ role: "user", content: "again" },
			],
			SLOW,
		);
		assert.deepEqual(context.systemPrompt, ["You improve Jev programs."]);
		assert.deepEqual(
			context.messages.map(message => [message.role, message.content]),
			[
				["user", [{ type: "text", text: "first" }]],
				["assistant", [{ type: "text", text: "draft" }]],
				["user", [{ type: "text", text: "again" }]],
			],
		);
		const assistant = context.messages[1];
		assert.equal(assistant.role === "assistant" && `${assistant.provider}/${assistant.model}`, "openai/gpt-slow");
	});

	it("rejects prompts without a user turn or with unknown roles", () => {
		assert.throws(() => promptToContext([{ role: "system", content: "x" }], CURRENT), /no user message/);
		assert.throws(() => promptToContext([{ role: "tool", content: "x" }], CURRENT), /unsupported role "tool"/);
		assert.throws(() => promptToContext(42, CURRENT), /must be a string/);
	});
});

describe("lm.complete handler", () => {
	it("resolves @slow through omp, forwards thinking/max_tokens/credentials and returns text + usage", async () => {
		const runDir = await mkdtemp(path.join(tmpdir(), "jev-opt-lm-"));
		tmpRoots.push(runDir);
		const h = harness({ model: "@slow", thinking: "high" });
		const signal = new AbortController().signal;
		const result = await h.handler({ role: "structural", prompt: "propose", max_tokens: 2048, run_dir: runDir }, signal);

		assert.deepEqual(result, {
			text: "new instructions",
			usage: { input: 1200, output: 300, cost_total: 0.0125 },
			provider: "openai",
			model: "gpt-slow-2026",
		});
		assert.deepEqual(h.optimizerLookups, [runDir]);
		assert.equal(h.calls.length, 1);
		const [call] = h.calls;
		assert.equal(call.model, SLOW);
		assert.deepEqual(h.resolverCalls, [{ model: SLOW, sessionId: "session-1" }]);
		assert.deepEqual(call.options, { apiKey: "resolver:gpt-slow", signal, maxTokens: 2048, reasoning: "high" });

		const rows = (await readFile(path.join(runDir, "usage.jsonl"), "utf8")).trim().split("\n").map(line => JSON.parse(line));
		assert.equal(rows.length, 1);
		assert.deepEqual(
			{ ...rows[0], at: "<ts>" },
			{
				source: "omp",
				role: "structural",
				spec: "@slow",
				provider: "openai",
				model: "gpt-slow-2026",
				input: 1200,
				output: 300,
				cache_read: 10,
				cache_write: 0,
				cost_total: 0.0125,
				stop_reason: "stop",
				at: "<ts>",
			},
		);
		assert.deepEqual(
			h.entries.map(entry => entry.customType),
			["jev-opt.usage"],
		);
		assert.deepEqual(h.entries[0].data, { ...rows[0], run_dir: runDir });
		assert.equal(h.statuses.at(-1), "jev-opt · LLM 1 call 1.5k tok $0.0125");
	});

	it("uses the live session model for `current` and disables reasoning for thinking: off", async () => {
		const h = harness({ thinking: "off" });
		await h.handler({ role: "reflection", prompt: "reflect" }, new AbortController().signal);
		assert.equal(h.calls[0].model, CURRENT);
		assert.equal(h.calls[0].options?.disableReasoning, true);
		assert.equal(h.calls[0].options?.reasoning, undefined);
		assert.equal("maxTokens" in (h.calls[0].options ?? {}), false);
	});

	it("fails before any provider call when the model spec does not resolve", async () => {
		const h = harness({ reflection_model: "nope/unknown" });
		await assert.rejects(
			h.handler({ role: "reflection", prompt: "x" }, new AbortController().signal),
			/no authenticated omp model matches optimizer model spec "nope\/unknown"/,
		);
		assert.equal(h.calls.length, 0);
		assert.equal(h.ledger.totals.calls, 0);
	});

	it("surfaces stopReason error/aborted as failures but still books the usage", async () => {
		const failed = harness(
			{},
			reply({ stopReason: "error", errorMessage: "429 rate limited", usage: { input: 50, output: 0, cost: { total: 0.001 } } }),
		);
		await assert.rejects(
			failed.handler({ role: "reflection", prompt: "x" }, new AbortController().signal),
			/openai\/gpt-slow-2026: 429 rate limited/,
		);
		assert.deepEqual(failed.ledger.totals, { calls: 1, input: 50, output: 0, cost_total: 0.001 });

		const aborted = harness({}, reply({ stopReason: "aborted" }));
		await assert.rejects(aborted.handler({ role: "reflection", prompt: "x" }, new AbortController().signal), /aborted/);
	});

	it("validates role and max_tokens from the worker", async () => {
		const h = harness({});
		const signal = new AbortController().signal;
		await assert.rejects(h.handler({ prompt: "x" }, signal), /string `role`/);
		await assert.rejects(h.handler({ role: "reflection", prompt: "x", max_tokens: 0 }, signal), /positive integer/);
		assert.equal(h.calls.length, 0);
	});
});

describe("usage ledger status line", () => {
	it("combines run progress with running totals and clears when idle", async () => {
		const statuses: Array<string | undefined> = [];
		const ledger = new UsageLedger({ setStatus: text => statuses.push(text) });
		ledger.setProgress("run-1 12s jev-text-opt it=3");
		assert.equal(statuses.at(-1), "jev-opt · run-1 12s jev-text-opt it=3");
		const row = {
			source: "omp" as const,
			role: "reflection",
			spec: "current",
			provider: "p",
			model: "m",
			input: 400,
			output: 100,
			cache_read: 0,
			cache_write: 0,
			cost_total: 0.002,
			stop_reason: "stop",
			at: "t",
		};
		await ledger.record(row, undefined);
		await ledger.record(row, undefined);
		assert.equal(statuses.at(-1), "jev-opt · run-1 12s jev-text-opt it=3 · LLM 2 calls 1.0k tok $0.0040");
		ledger.setProgress(undefined);
		assert.equal(statuses.at(-1), "jev-opt · LLM 2 calls 1.0k tok $0.0040");
		assert.equal(new UsageLedger({}).statusText(), undefined);
	});
});
