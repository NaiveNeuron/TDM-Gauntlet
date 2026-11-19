// `lm.complete` (worker → host): every optimizer LLM call goes through omp model resolution,
// omp credentials and pi-ai `completeSimple` (architecture §1, §3). No provider code here.
import { appendFile } from "node:fs/promises";
import path from "node:path";
import type { JsonObject } from "./bridge.ts";
import type {
	AssistantMessage,
	CompleteSimple,
	Context,
	Effort,
	ExtensionContext,
	Model,
	SimpleStreamOptions,
	UserMessage,
} from "./omp-types.ts";

/** `optimizer:` section of the run config, as reported by the worker's `run_started` event. */
export interface OptimizerModels {
	model?: string;
	reflection_model?: string;
	structural_model?: string;
	thinking?: string;
}

export interface LmResult {
	text: string;
	usage: { input: number; output: number; cost_total: number };
	provider: string;
	model: string;
}

const ROLE_MODEL_KEY: Record<string, "reflection_model" | "structural_model"> = {
	reflection: "reflection_model",
	structural: "structural_model",
};

const EFFORTS: Record<Effort, true> = { minimal: true, low: true, medium: true, high: true, xhigh: true, max: true };

function isObject(value: unknown): value is JsonObject {
	return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** Keep the string-valued model/thinking fields of an untrusted `optimizer` object; null/empty mean unset. */
export function parseOptimizerModels(value: unknown): OptimizerModels {
	const parsed: OptimizerModels = {};
	if (!isObject(value)) return parsed;
	for (const key of ["model", "reflection_model", "structural_model", "thinking"] as const) {
		const field = value[key];
		if (typeof field === "string" && field.trim()) parsed[key] = field.trim();
	}
	return parsed;
}

/** role → spec: `optimizer.<role>_model`, else `optimizer.model`, else `current`. */
export function modelSpecForRole(role: string, optimizer: OptimizerModels): string {
	const key = Object.hasOwn(ROLE_MODEL_KEY, role) ? ROLE_MODEL_KEY[role] : undefined;
	if (!key) throw new Error(`unknown lm.complete role ${JSON.stringify(role)}; expected reflection or structural`);
	return optimizer[key] ?? optimizer.model ?? "current";
}

/** `off` → disableReasoning, `auto`/unset → provider default, effort level → reasoning. */
export function thinkingOptions(thinking: string | undefined): Pick<SimpleStreamOptions, "reasoning" | "disableReasoning"> {
	if (thinking === undefined || thinking === "auto") return {};
	if (thinking === "off") return { disableReasoning: true };
	if (Object.hasOwn(EFFORTS, thinking)) return { reasoning: thinking as Effort }; // membership checked above
	throw new Error(
		`invalid optimizer.thinking ${JSON.stringify(thinking)}; expected off, auto, ${Object.keys(EFFORTS).join(", ")}`,
	);
}

/** `current` is the live session model; everything else (`@slow`, `provider/id`, bare id) is omp's resolver. */
export function resolveModel(models: ExtensionContext["models"], spec: string): Model {
	const model = spec === "current" ? models.current() : models.resolve(spec);
	if (model) return model;
	throw new Error(
		spec === "current"
			? "optimizer model is `current` but the omp session has no model selected"
			: `no authenticated omp model matches optimizer model spec ${JSON.stringify(spec)}`,
	);
}

function contentText(content: unknown): string {
	if (typeof content === "string") return content;
	if (Array.isArray(content)) {
		return content
			.map(part => (typeof part === "string" ? part : isObject(part) && typeof part.text === "string" ? part.text : ""))
			.join("");
	}
	throw new Error("lm.complete message content must be a string or a list of text parts");
}

function userMessage(text: string): UserMessage {
	return { role: "user", content: [{ type: "text", text }], timestamp: Date.now() };
}

/** Worker prompt (string, or chat list with system/user/assistant roles) → pi-ai Context. */
export function promptToContext(prompt: unknown, model: Model): Context {
	if (typeof prompt === "string") return { messages: [userMessage(prompt)] };
	if (!Array.isArray(prompt)) throw new Error("lm.complete prompt must be a string or a [{role, content}] list");
	const systemPrompt: string[] = [];
	const messages: Context["messages"] = [];
	for (const entry of prompt) {
		if (!isObject(entry)) throw new Error("lm.complete prompt entries must be {role, content} objects");
		const text = contentText(entry.content);
		if (entry.role === "system" || entry.role === "developer") systemPrompt.push(text);
		else if (entry.role === "user") messages.push(userMessage(text));
		else if (entry.role === "assistant") {
			messages.push({
				role: "assistant",
				content: [{ type: "text", text }],
				api: model.api,
				provider: model.provider,
				model: model.id,
				usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { total: 0 } },
				stopReason: "stop",
				timestamp: Date.now(),
			});
		} else throw new Error(`lm.complete prompt has unsupported role ${JSON.stringify(entry.role)}`);
	}
	if (!messages.some(message => message.role === "user")) throw new Error("lm.complete prompt has no user message");
	return systemPrompt.length ? { systemPrompt, messages } : { messages };
}

/** One host-side optimizer LLM call, as appended to `<run_dir>/usage.jsonl`. */
export interface UsageRow {
	source: "omp";
	role: string;
	spec: string;
	provider: string;
	model: string;
	input: number;
	output: number;
	cache_read: number;
	cache_write: number;
	cost_total: number;
	stop_reason: string;
	at: string;
}

export interface LedgerSinks {
	appendEntry?: (customType: string, data: unknown) => void;
	setStatus?: (text: string | undefined) => void;
	warn?: (message: string) => void;
}

/**
 * Optimizer-LLM usage ledger (architecture §12.4): direct pi-ai calls do not enter omp session
 * stats, so usage is kept per run (`usage.jsonl`), in the session (`jev-opt.usage` entries) and
 * on the `jev-opt` status line together with the current run progress.
 */
export class UsageLedger {
	readonly totals = { calls: 0, input: 0, output: 0, cost_total: 0 };
	readonly #sinks: LedgerSinks;
	#progress: string | undefined;

	constructor(sinks: LedgerSinks) {
		this.#sinks = sinks;
	}

	async record(row: UsageRow, runDir: string | undefined): Promise<void> {
		this.totals.calls += 1;
		this.totals.input += row.input;
		this.totals.output += row.output;
		this.totals.cost_total += row.cost_total;
		this.#sinks.appendEntry?.("jev-opt.usage", runDir ? { ...row, run_dir: runDir } : row);
		this.#render();
		if (!runDir) return;
		try {
			await appendFile(path.join(runDir, "usage.jsonl"), `${JSON.stringify(row)}\n`);
		} catch (error) {
			this.#sinks.warn?.(`jev-opt: could not append usage to ${runDir}: ${String(error)}`);
		}
	}

	setProgress(text: string | undefined): void {
		this.#progress = text;
		this.#render();
	}

	statusText(): string | undefined {
		const { calls, input, output, cost_total } = this.totals;
		const parts = this.#progress ? [this.#progress] : [];
		if (calls) {
			const tokens = input + output;
			const shown = tokens >= 1000 ? `${(tokens / 1000).toFixed(1)}k` : String(tokens);
			parts.push(`LLM ${calls} call${calls === 1 ? "" : "s"} ${shown} tok $${cost_total.toFixed(4)}`);
		}
		return parts.length ? `jev-opt · ${parts.join(" · ")}` : undefined;
	}

	#render(): void {
		this.#sinks.setStatus?.(this.statusText());
	}
}

export interface LmDeps {
	completeSimple: CompleteSimple;
	ctx: Pick<ExtensionContext, "models" | "modelRegistry" | "sessionManager">;
	/** Optimizer config of the run a request belongs to (`run_dir` from the request, if any). */
	optimizerFor: (runDir: string | undefined) => OptimizerModels;
	ledger: UsageLedger;
}

function textOf(message: AssistantMessage): string {
	let text = "";
	for (const block of message.content) if (block.type === "text" && "text" in block) text += block.text;
	return text;
}

/** Build the `lm.complete` handler; assignable to the bridge's `RequestHandler`. */
export function createLmCompleteHandler(deps: LmDeps): (params: JsonObject, signal: AbortSignal) => Promise<LmResult> {
	return async (params, signal) => {
		const { role, max_tokens: maxTokens } = params;
		if (typeof role !== "string") throw new Error("lm.complete requires a string `role`");
		if (maxTokens !== undefined && maxTokens !== null && !(Number.isInteger(maxTokens) && Number(maxTokens) > 0)) {
			throw new Error("lm.complete `max_tokens` must be a positive integer");
		}
		const runDir = typeof params.run_dir === "string" && params.run_dir ? params.run_dir : undefined;
		const optimizer = deps.optimizerFor(runDir);
		const spec = modelSpecForRole(role, optimizer);
		const reasoning = thinkingOptions(optimizer.thinking);
		const model = resolveModel(deps.ctx.models, spec);
		const context = promptToContext(params.prompt, model);
		const message = await deps.completeSimple(model, context, {
			apiKey: deps.ctx.modelRegistry.resolver(model, deps.ctx.sessionManager.getSessionId()),
			signal,
			...(typeof maxTokens === "number" ? { maxTokens } : {}),
			...reasoning,
		});
		const usage = message.usage;
		const cost = usage?.cost?.total ?? 0;
		await deps.ledger.record(
			{
				source: "omp",
				role,
				spec,
				provider: message.provider,
				model: message.model,
				input: usage?.input ?? 0,
				output: usage?.output ?? 0,
				cache_read: usage?.cacheRead ?? 0,
				cache_write: usage?.cacheWrite ?? 0,
				cost_total: cost,
				stop_reason: message.stopReason,
				at: new Date().toISOString(),
			},
			runDir,
		);
		if (message.stopReason === "error") {
			throw new Error(`${message.provider}/${message.model}: ${message.errorMessage ?? "completion failed"}`);
		}
		if (message.stopReason === "aborted") throw new Error(`${message.provider}/${message.model}: completion aborted`);
		return {
			text: textOf(message),
			usage: { input: usage?.input ?? 0, output: usage?.output ?? 0, cost_total: cost },
			provider: message.provider,
			model: message.model,
		};
	};
}
