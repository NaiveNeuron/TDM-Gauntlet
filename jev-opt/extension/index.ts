// jev-opt omp extension: commands, run/worker lifecycle and the LLM bridge. No optimization
// logic lives here — handlers parse arguments, manage worker processes, forward `lm.complete`
// to omp and render worker results (architecture §1–§3).
import { createWriteStream, existsSync, mkdirSync, readFileSync } from "node:fs";
import type { WriteStream } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { completeSimple } from "@oh-my-pi/pi-ai";
import { WorkerBridge } from "./bridge.ts";
import type { JsonObject } from "./bridge.ts";
import { formatProgress, formatValue, parseArgs, renderSection } from "./cli.ts";
import type { FlagSpec } from "./cli.ts";
import { UsageLedger, createLmCompleteHandler, parseOptimizerModels } from "./lm.ts";
import type { OptimizerModels } from "./lm.ts";
import type { CompleteSimple, ExtensionAPI, ExtensionContext } from "./omp-types.ts";

const PLUGIN_ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const WORKER_SCRIPT = path.join(PLUGIN_ROOT, "python", "worker.py");
const PROTOCOL_VERSION = 1;
const KILL_TIMEOUT_MS = 3000;
const HEARTBEAT_MS = 5000;
/** Progress events worth a TUI toast; the rest only update the status line. */
const NOTIFY_EVENTS: Record<string, "info" | "error"> = {
	run_started: "info",
	condition_finished: "info",
	run_finished: "info",
	error: "error",
};

interface Job {
	label: string;
	runId?: string;
	startedAt: number;
	controller: AbortController;
	progress?: string;
}

interface RunRef {
	runsRoot: string;
	runId: string;
	runDir: string;
}

function isObject(value: unknown): value is JsonObject {
	return value !== null && typeof value === "object" && !Array.isArray(value);
}

function errorText(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}

function defaultRunsRoot(ctx: ExtensionContext): string {
	return path.join(ctx.cwd, "runs", "jev-opt");
}

/** `<run>` is a run id under the default runs root, or a path to a run directory. */
function resolveRun(ctx: ExtensionContext, run: string): RunRef {
	if (run.includes("/") || run.includes(path.sep) || run.startsWith(".")) {
		const runDir = path.resolve(ctx.cwd, run);
		return { runsRoot: path.dirname(runDir), runId: path.basename(runDir), runDir };
	}
	const runsRoot = defaultRunsRoot(ctx);
	return { runsRoot, runId: run, runDir: path.join(runsRoot, run) };
}

function elapsed(ms: number): string {
	const s = Math.round(ms / 1000);
	return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m${String(s % 60).padStart(2, "0")}s`;
}

function modelLabel(ctx: ExtensionContext, spec: string): string {
	const model = spec === "current" ? ctx.models.current() : ctx.models.resolve(spec);
	return model ? `${model.provider}/${model.id}` : "(unresolved)";
}

function renderOptimize(result: JsonObject): string {
	return `jev-opt run ${String(result.run_id)} finished → ${String(result.run_dir)}\n${renderSection("conditions", result.conditions)}`;
}

function renderShow(run: string, candidate: string, result: JsonObject): string {
	const yaml = typeof result.program_yaml === "string" ? result.program_yaml.trimEnd() : formatValue(result.program_yaml);
	return [
		`# ${run} · ${candidate}`,
		yaml,
		renderSection("scores", result.scores),
		renderSection("rationale", result.rationale ?? null),
	].join("\n");
}

export function createJevOptExtension(complete: CompleteSimple) {
	return function jevOpt(pi: ExtensionAPI): void {
		const jobs = new Set<Job>();
		const workers = new Set<WorkerBridge>();
		let uiCtx: ExtensionContext | undefined;
		let heartbeat: { ctx: ExtensionContext; timer: unknown } | undefined;
		let disposed = false;

		// Late worker events can arrive while the session tears down; never touch the session after that.
		const ledger = new UsageLedger({
			appendEntry: (customType, data) => {
				if (!disposed) pi.appendEntry(customType, data);
			},
			setStatus: text => {
				if (!disposed) uiCtx?.ui.setStatus("jev-opt", text);
			},
			warn: message => (uiCtx && !disposed ? uiCtx.ui.notify(message, "warning") : process.stderr.write(`${message}\n`)),
		});

		function noteCtx(ctx: ExtensionContext): void {
			if (ctx.hasUI) uiCtx = ctx;
		}

		function emit(ctx: ExtensionContext, text: string, details?: unknown): void {
			if (disposed) return;
			if (!ctx.hasUI) {
				process.stdout.write(`${text.trimEnd()}\n`);
				return;
			}
			pi.sendMessage(
				{ customType: "jev-opt", content: text, display: true, details },
				ctx.isIdle() ? { triggerTurn: false } : { triggerTurn: false, deliverAs: "followUp" },
			);
		}

		function fail(ctx: ExtensionContext, error: unknown): void {
			if (disposed) return;
			const text = `jev-opt: ${errorText(error)}`;
			if (!ctx.hasUI) {
				process.stderr.write(`${text}\n`);
				return;
			}
			ctx.ui.notify(text.split("\n")[0], "error");
			if (text.includes("\n")) emit(ctx, text);
		}

		function refreshStatus(): void {
			if (disposed) return;
			const now = Date.now();
			const active = [...jobs].map(job => {
				const head = `${job.runId ?? job.label} ${elapsed(now - job.startedAt)}`;
				const tail = job.progress && job.progress.length > 80 ? `${job.progress.slice(0, 79)}…` : job.progress;
				return tail ? `${head} ${tail}` : head;
			});
			ledger.setProgress(active.length ? active.join(" | ") : undefined);
			if (!jobs.size && heartbeat) {
				heartbeat.ctx.clearTimer(heartbeat.timer);
				heartbeat = undefined;
			}
		}

		function startWorker(ctx: ExtensionContext, job?: Job): WorkerBridge {
			const optimizers = new Map<string, OptimizerModels>();
			let latest: OptimizerModels = {};
			let log: WriteStream | undefined;
			if (ctx.hasUI) {
				// Worker stderr would corrupt the TUI; keep it in a log next to the runs.
				const runsRoot = defaultRunsRoot(ctx);
				mkdirSync(runsRoot, { recursive: true });
				log = createWriteStream(path.join(runsRoot, "worker.log"), { flags: "a" });
				log.on("error", () => {});
			}
			const bridge = new WorkerBridge({
				command: "uv",
				args: ["run", WORKER_SCRIPT],
				cwd: ctx.cwd,
				killTimeoutMs: KILL_TIMEOUT_MS,
				handlers: {
					"lm.complete": createLmCompleteHandler({
						completeSimple: complete,
						ctx,
						ledger,
						optimizerFor: runDir => (runDir && optimizers.get(path.resolve(ctx.cwd, runDir))) || latest,
					}),
				},
				onProgress: params => {
					if (params.event === "run_started") {
						latest = parseOptimizerModels(params.optimizer);
						if (typeof params.run_dir === "string") optimizers.set(path.resolve(ctx.cwd, params.run_dir), latest);
					}
					if (!job || disposed) return;
					if (typeof params.run_id === "string") job.runId = params.run_id;
					const line = formatProgress(params);
					job.progress = typeof params.message === "string" && params.message ? params.message : line;
					if (!ctx.hasUI) {
						process.stderr.write(`[jev-opt] ${line}\n`);
						return;
					}
					refreshStatus();
					const event = typeof params.event === "string" ? params.event : "";
					if (Object.hasOwn(NOTIFY_EVENTS, event)) ctx.ui.notify(`jev-opt: ${line}`, NOTIFY_EVENTS[event]);
				},
				onStderr: text => (log ? log.write(text) : process.stderr.write(text)),
				onExit: () => {
					workers.delete(bridge);
					log?.end();
				},
			});
			workers.add(bridge);
			return bridge;
		}

		async function hello(bridge: WorkerBridge): Promise<JsonObject> {
			const reply = await bridge.request<JsonObject>("hello");
			if (!isObject(reply) || reply.protocol !== PROTOCOL_VERSION) {
				throw new Error(`worker speaks protocol ${JSON.stringify(reply?.protocol)}, expected ${PROTOCOL_VERSION}`);
			}
			return reply;
		}

		/** Short request on a throwaway worker (status/show/pareto/selftest). */
		async function withWorker<T>(ctx: ExtensionContext, fn: (bridge: WorkerBridge) => Promise<T>): Promise<T> {
			const bridge = startWorker(ctx);
			try {
				await hello(bridge);
				return await fn(bridge);
			} finally {
				await bridge.shutdown();
			}
		}

		/** Long request on a dedicated worker; awaited headless, backgrounded in the TUI. */
		async function runJob(
			ctx: ExtensionContext,
			label: string,
			method: string,
			params: JsonObject,
			render: (result: JsonObject) => string,
		): Promise<void> {
			const job: Job = { label, startedAt: Date.now(), controller: new AbortController() };
			if (typeof params.run_id === "string") job.runId = params.run_id;
			jobs.add(job);
			if (ctx.hasUI && !heartbeat) heartbeat = { ctx, timer: ctx.setInterval(refreshStatus, HEARTBEAT_MS) };
			refreshStatus();
			let bridge: WorkerBridge | undefined;
			try {
				bridge = startWorker(ctx, job);
				await hello(bridge);
				const result = await bridge.request<JsonObject>(method, params, { signal: job.controller.signal });
				emit(ctx, render(isObject(result) ? result : { result }), result);
			} catch (error) {
				fail(ctx, error);
			} finally {
				jobs.delete(job);
				refreshStatus();
				await bridge?.shutdown();
			}
		}

		function launch(ctx: ExtensionContext, job: Promise<void>, started: string): Promise<void> {
			if (!ctx.hasUI) return job;
			ctx.ui.notify(`jev-opt: ${started} in background (/jev-status, /jev-cancel)`, "info");
			return Promise.resolve();
		}

		function command(
			name: string,
			description: string,
			spec: FlagSpec,
			handler: (ctx: ExtensionContext, positional: string[], flags: Record<string, string | true>) => Promise<void>,
		): void {
			pi.registerCommand(name, {
				description,
				handler: async (args, ctx) => {
					noteCtx(ctx);
					try {
						const { positional, flags } = parseArgs(args ?? "", spec);
						await handler(ctx, positional, flags);
					} catch (error) {
						fail(ctx, error);
					}
				},
			});
		}

		function requireArgs(positional: string[], min: number, max: number, usage: string): void {
			if (positional.length < min || positional.length > max) throw new Error(`usage: ${usage}`);
		}

		pi.setLabel("Jev Optimizer");

		command(
			"jev-optimize",
			"Optimize a Jev decision program: /jev-optimize <task.yaml> [--conditions a,b] [--run-id x] [--seed n]",
			{ conditions: "string", "run-id": "string", seed: "string" },
			async (ctx, positional, flags) => {
				requireArgs(positional, 1, 1, "/jev-optimize <task.yaml> [--conditions a,b] [--run-id x] [--seed n]");
				const taskPath = path.resolve(ctx.cwd, positional[0]);
				if (!existsSync(taskPath)) throw new Error(`task file not found: ${taskPath}`);
				const params: JsonObject = { task_path: taskPath, runs_root: defaultRunsRoot(ctx) };
				if (typeof flags["run-id"] === "string") params.run_id = flags["run-id"];
				if (typeof flags.seed === "string") {
					if (!/^\d+$/.test(flags.seed)) throw new Error(`--seed must be a non-negative integer, got ${flags.seed}`);
					params.seed = Number(flags.seed);
				}
				if (typeof flags.conditions === "string") {
					params.conditions = flags.conditions
						.split(",")
						.map(item => item.trim())
						.filter(Boolean);
				}
				const job = runJob(ctx, path.basename(taskPath), "optimize", params, renderOptimize);
				await launch(ctx, job, `optimizing ${positional[0]}`);
			},
		);

		command("jev-resume", "Resume an unfinished jev-opt run: /jev-resume <run>", {}, async (ctx, positional) => {
			requireArgs(positional, 1, 1, "/jev-resume <run>");
			const run = resolveRun(ctx, positional[0]);
			const params = { runs_root: run.runsRoot, run_id: run.runId, resume: true };
			await launch(ctx, runJob(ctx, run.runId, "optimize", params, renderOptimize), `resuming ${run.runId}`);
		});

		command("jev-status", "List jev-opt runs and in-session jobs: /jev-status [run]", {}, async (ctx, positional) => {
			requireArgs(positional, 0, 1, "/jev-status [run]");
			const run = positional[0] ? resolveRun(ctx, positional[0]) : undefined;
			const params: JsonObject = { runs_root: run?.runsRoot ?? defaultRunsRoot(ctx) };
			if (run) params.run_id = run.runId;
			const result = await withWorker(ctx, bridge => bridge.request<JsonObject>("status", params));
			const lines = [renderSection(`jev-opt runs (${String(params.runs_root)})`, result.runs)];
			const now = Date.now();
			for (const job of jobs) {
				lines.push(`active: ${job.runId ?? job.label} ${elapsed(now - job.startedAt)}${job.progress ? ` — ${job.progress}` : ""}`);
			}
			if (ledger.totals.calls) lines.push(ledger.statusText() ?? "");
			emit(ctx, lines.join("\n"), result);
		});

		command(
			"jev-show",
			"Show a candidate program: /jev-show <run> <candidate|representative (best-quality, top-quality, top-<condition>, ensemble-<k>, …)>",
			{},
			async (ctx, positional) => {
				requireArgs(positional, 2, 2, "/jev-show <run> <candidate|representative (best-quality, top-quality, top-<condition>, ensemble-<k>, …)>");
				const run = resolveRun(ctx, positional[0]);
				const candidate = positional[1];
				const result = await withWorker(ctx, bridge =>
					bridge.request<JsonObject>("show", { run_dir: run.runDir, candidate }),
				);
				emit(ctx, renderShow(run.runId, candidate, result), result);
			},
		);

		command("jev-pareto", "Show representatives and the Pareto front: /jev-pareto <run>", {}, async (ctx, positional) => {
			requireArgs(positional, 1, 1, "/jev-pareto <run>");
			const run = resolveRun(ctx, positional[0]);
			const result = await withWorker(ctx, bridge => bridge.request<JsonObject>("pareto", { run_dir: run.runDir }));
			emit(
				ctx,
				[
					`# ${run.runId} · pareto`,
					renderSection("representatives", result.representatives),
					renderSection("front", result.front),
				].join("\n"),
				result,
			);
		});

		command(
			"jev-evaluate-final",
			"Evaluate a representative once on the test split (irreversible): /jev-evaluate-final <run> [representative] [--yes]",
			{ yes: "boolean" },
			async (ctx, positional, flags) => {
				requireArgs(positional, 1, 2, "/jev-evaluate-final <run> [representative] [--yes]");
				const run = resolveRun(ctx, positional[0]);
				const representative = positional[1];
				const what = `${run.runId}${representative ? ` (${representative})` : ""}`;
				if (!flags.yes) {
					if (!ctx.hasUI) throw new Error("evaluating on the test split is irreversible; re-run with --yes");
					const ok = await ctx.ui.confirm(
						"Consume the test split?",
						`Evaluate ${what} on the held-out test split. This is recorded in test_consumed.json and the run can no longer be resumed.`,
					);
					if (!ok) {
						ctx.ui.notify("jev-opt: final evaluation cancelled", "info");
						return;
					}
				}
				const params: JsonObject = { run_dir: run.runDir };
				if (representative) params.representative = representative;
				const render = (result: JsonObject) =>
					`${renderSection(`test metrics · ${what}`, result.metrics)}\nconsumed_at: ${String(result.consumed_at)}`;
				await launch(ctx, runJob(ctx, run.runId, "evaluate_final", params, render), `evaluating ${what} on test`);
			},
		);

		command(
			"jev-seeds",
			"Write Jev-native seed programs from the task definition only (no Jev call, no data): /jev-seeds <task.yaml> --out <dir> [--kinds jev-native,decomposed] [--n 3]",
			{ out: "string", kinds: "string", n: "string" },
			async (ctx, positional, flags) => {
				const usage = "/jev-seeds <task.yaml> --out <dir> [--kinds jev-native,decomposed] [--n 3]";
				requireArgs(positional, 1, 1, usage);
				if (typeof flags.out !== "string") throw new Error(`usage: ${usage}`);
				const taskPath = path.resolve(ctx.cwd, positional[0]);
				if (!existsSync(taskPath)) throw new Error(`task file not found: ${taskPath}`);
				const params: JsonObject = { task_path: taskPath, out_dir: path.resolve(ctx.cwd, flags.out) };
				if (typeof flags.kinds === "string") params.kinds = flags.kinds.split(",").map(item => item.trim()).filter(Boolean);
				if (typeof flags.n === "string") {
					if (!/^[1-9]\d*$/.test(flags.n)) throw new Error(`--n must be a positive integer, got ${flags.n}`);
					params.n = Number(flags.n);
				}
				const render = (result: JsonObject) => `jev-opt seeds → ${String(params.out_dir)}\n${renderSection("kinds", result.kinds)}`;
				await launch(ctx, runJob(ctx, path.basename(taskPath), "generate_seeds", params, render), `writing seeds for ${positional[0]}`);
			},
		);

		command("jev-cancel", "Stop running jev-opt jobs gracefully: /jev-cancel [run]", {}, async (ctx, positional) => {
			requireArgs(positional, 0, 1, "/jev-cancel [run]");
			const target = positional[0];
			const matched = [...jobs].filter(job => !target || job.runId === target || job.label === target);
			if (!matched.length) throw new Error(target ? `no running jev-opt job for ${target}` : "no running jev-opt jobs");
			for (const job of matched) job.controller.abort();
			const names = matched.map(job => job.runId ?? job.label).join(", ");
			if (ctx.hasUI) ctx.ui.notify(`jev-opt: cancel requested for ${names}`, "info");
			else process.stdout.write(`jev-opt: cancel requested for ${names}\n`);
		});

		command(
			"jev-opt-selftest",
			"Show jev-opt load status and resolved models (no LLM call): /jev-opt-selftest [--worker]",
			{ worker: "boolean" },
			async (ctx, _positional, flags) => {
				const pkg: unknown = JSON.parse(readFileSync(path.join(PLUGIN_ROOT, "package.json"), "utf8"));
				const version = isObject(pkg) && typeof pkg.version === "string" ? pkg.version : "?";
				const lines = [
					`jev-opt loaded v${version} · current=${modelLabel(ctx, "current")} · @slow=${modelLabel(ctx, "@slow")} · @smol=${modelLabel(ctx, "@smol")}`,
					`worker=${WORKER_SCRIPT} (${existsSync(WORKER_SCRIPT) ? "present" : "missing"}) · runs_root=${defaultRunsRoot(ctx)}`,
				];
				if (flags.worker) {
					const reply = await withWorker(ctx, hello);
					lines.push(`worker ok: protocol ${String(reply.protocol)}`, renderSection("versions", reply.versions));
				}
				emit(ctx, lines.join("\n"));
			},
		);

		pi.on("session_start", (_event, ctx) => noteCtx(ctx));

		pi.on("session_shutdown", async () => {
			disposed = true;
			for (const job of jobs) job.controller.abort();
			await Promise.all([...workers].map(worker => worker.shutdown()));
		});
	};
}

export default createJevOptExtension(completeSimple);
