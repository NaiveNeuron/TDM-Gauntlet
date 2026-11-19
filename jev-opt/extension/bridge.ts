// JSONL RPC bridge to the Python worker (architecture §3). Pure module: no omp imports,
// runs under omp's Bun and under node (tests) via node:child_process.
import { spawn } from "node:child_process";
import type { ChildProcess } from "node:child_process";

export type JsonObject = Record<string, unknown>;

/** Worker → host request handler. `params` is untrusted worker input; validate it. */
export type RequestHandler = (params: JsonObject, signal: AbortSignal) => unknown;

export interface ExitInfo {
	code: number | null;
	signal: string | null;
}

export interface WorkerBridgeOptions {
	command: string;
	args?: string[];
	cwd?: string;
	env?: Record<string, string | undefined>;
	/** Worker → host request handlers, e.g. `lm.complete`. */
	handlers?: Record<string, RequestHandler>;
	/** `progress` notifications. */
	onProgress?: (params: JsonObject) => void;
	/** Any other worker → host notification. */
	onNotification?: (method: string, params: JsonObject) => void;
	onStderr?: (text: string) => void;
	/** `error` is set when the worker failed to start or broke the protocol. */
	onExit?: (info: ExitInfo, error: Error | undefined) => void;
	/** Maximum bytes of one stdout line before the worker is treated as broken. */
	maxLineBytes?: number;
	/** Grace period of each shutdown stage (shutdown request, stdin EOF, SIGTERM) before escalating. */
	killTimeoutMs?: number;
}

export interface RequestOptions {
	/** Abort sends a cooperative `cancel` to the worker; the request still settles with the worker's reply. */
	signal?: AbortSignal;
}

/** Error reply from the worker (`{"id", "error": {"message"}}`). */
export class WorkerRemoteError extends Error {
	override name = "WorkerRemoteError";
}

/** Worker process ended (or failed to start) while requests were pending. */
export class WorkerExitedError extends Error {
	override name = "WorkerExitedError";
	readonly info: ExitInfo;
	constructor(message: string, info: ExitInfo) {
		super(message);
		this.info = info;
	}
}

/** Malformed or oversized frame on the worker's stdout. */
export class WorkerProtocolError extends Error {
	override name = "WorkerProtocolError";
}

interface Pending {
	resolve: (value: unknown) => void;
	reject: (error: Error) => void;
	dispose: () => void;
}

const NEWLINE = 0x0a;
const STDERR_TAIL_CHARS = 4000;

function isObject(value: unknown): value is JsonObject {
	return value !== null && typeof value === "object" && !Array.isArray(value);
}

export class WorkerBridge {
	readonly exited: Promise<ExitInfo>;
	readonly #child: ChildProcess;
	readonly #options: WorkerBridgeOptions;
	readonly #handlers = new Map<string, RequestHandler>();
	readonly #pending = new Map<string, Pending>();
	/** Aborted when the worker goes away so in-flight host handlers (LLM calls) stop. */
	readonly #incoming = new AbortController();
	readonly #maxLineBytes: number;
	readonly #killTimeoutMs: number;
	readonly #resolveExited: (info: ExitInfo) => void;
	#nextId = 0;
	#writeTail: Promise<void> = Promise.resolve();
	#partial: Buffer[] = [];
	#partialBytes = 0;
	#stderrTail = "";
	#exitInfo: ExitInfo | undefined;
	#failure: Error | undefined;
	#stdinOpen = true;
	#shutdown: Promise<ExitInfo> | undefined;

	constructor(options: WorkerBridgeOptions) {
		this.#options = options;
		this.#maxLineBytes = options.maxLineBytes ?? 16 * 1024 * 1024;
		this.#killTimeoutMs = options.killTimeoutMs ?? 5000;
		for (const [method, handler] of Object.entries(options.handlers ?? {})) this.#handlers.set(method, handler);
		const exited = Promise.withResolvers<ExitInfo>();
		this.exited = exited.promise;
		this.#resolveExited = exited.resolve;
		const child = spawn(options.command, options.args ?? [], {
			cwd: options.cwd,
			env: options.env,
			stdio: ["pipe", "pipe", "pipe"],
			// Own process group: terminal ^C reaches omp only, and kill() also reaches uv's python child.
			detached: process.platform !== "win32",
		});
		this.#child = child;
		child.stdout?.on("data", (chunk: Buffer) => this.#onStdout(chunk));
		child.stderr?.on("data", (chunk: Buffer) => this.#onStderr(chunk.toString("utf8")));
		child.stdin?.on("error", error => {
			this.#stdinOpen = false;
			this.#fail(new WorkerProtocolError(`worker stdin failed: ${error.message}`), false);
		});
		child.on("error", error => {
			// Spawn failure (e.g. `uv` not on PATH) has no pid and no reliable 'close'.
			if (child.pid === undefined) this.#finish({ code: null, signal: null }, error);
			else this.#fail(error, true);
		});
		child.on("close", (code, signal) => this.#finish({ code, signal }, undefined));
	}

	get pid(): number | undefined {
		return this.#child.pid;
	}

	get running(): boolean {
		return this.#exitInfo === undefined;
	}

	get stderrTail(): string {
		return this.#stderrTail;
	}

	/** Register a worker → host request handler. */
	handle(method: string, handler: RequestHandler): void {
		this.#handlers.set(method, handler);
	}

	/** Host → worker request. `T` is the caller's expectation of the worker's result shape. */
	request<T = unknown>(method: string, params: JsonObject = {}, options: RequestOptions = {}): Promise<T> {
		const unusable = this.#unusableError();
		if (unusable) return Promise.reject(unusable);
		const { signal } = options;
		if (signal?.aborted) return Promise.reject(signal.reason ?? new Error("aborted"));
		const id = `h${++this.#nextId}`;
		const reply = Promise.withResolvers<T>();
		const onAbort = () => {
			if (this.#pending.has(id) && this.running) {
				this.#write({ id: `h${++this.#nextId}`, method: "cancel", params: {} }).catch(() => {});
			}
		};
		signal?.addEventListener("abort", onAbort, { once: true });
		this.#pending.set(id, {
			resolve: reply.resolve as (value: unknown) => void, // T is caller-declared; worker results are not validated here
			reject: reply.reject,
			dispose: () => signal?.removeEventListener("abort", onAbort),
		});
		this.#write({ id, method, params }).catch(error => this.#settle(id, undefined, error));
		return reply.promise;
	}

	/**
	 * Graceful stop: `shutdown` request, stdin EOF, SIGTERM, SIGKILL — each stage waits up to
	 * `killTimeoutMs`. Idempotent; resolves with the exit status.
	 */
	shutdown(): Promise<ExitInfo> {
		this.#shutdown ??= this.#runShutdown();
		return this.#shutdown;
	}

	/** Signal the worker's process group (uv and its python child). */
	kill(signal: NodeJS.Signals = "SIGKILL"): void {
		if (!this.running) return;
		const pid = this.#child.pid;
		try {
			if (pid !== undefined && process.platform !== "win32") process.kill(-pid, signal);
			else this.#child.kill(signal);
		} catch {
			try {
				this.#child.kill(signal);
			} catch {
				// already gone
			}
		}
	}

	async #runShutdown(): Promise<ExitInfo> {
		if (!this.running) return this.exited;
		if (!this.#unusableError()) {
			// Bypass request(): the shutdown flag set by our caller would refuse it.
			const id = `h${++this.#nextId}`;
			const ack = Promise.withResolvers<unknown>();
			this.#pending.set(id, { resolve: ack.resolve, reject: ack.reject, dispose: () => {} });
			this.#write({ id, method: "shutdown", params: {} }).catch(error => this.#settle(id, undefined, error));
			await this.#settledWithin(ack.promise, this.#killTimeoutMs);
		}
		this.#endStdin();
		if (await this.#settledWithin(this.exited, this.#killTimeoutMs)) return this.exited;
		this.kill("SIGTERM");
		if (await this.#settledWithin(this.exited, this.#killTimeoutMs)) return this.exited;
		this.kill("SIGKILL");
		return this.exited;
	}

	#settledWithin(promise: Promise<unknown>, ms: number): Promise<boolean> {
		const { promise: done, resolve } = Promise.withResolvers<boolean>();
		const timer = setTimeout(() => resolve(false), ms);
		timer.unref?.();
		const settle = () => {
			clearTimeout(timer);
			resolve(true);
		};
		promise.then(settle, settle);
		return done;
	}

	#unusableError(): Error | undefined {
		if (this.#failure) return this.#failure;
		if (this.#exitInfo) return this.#exitError(this.#exitInfo);
		if (this.#shutdown) return new Error("jev-opt worker is shutting down");
		return undefined;
	}

	#endStdin(): void {
		if (!this.#stdinOpen) return;
		this.#stdinOpen = false;
		try {
			this.#child.stdin?.end();
		} catch {
			// process already gone
		}
	}

	/** Serialized, backpressure-aware line write; lines never interleave. */
	#write(message: JsonObject): Promise<void> {
		const line = `${JSON.stringify(message)}\n`;
		const run = () => {
			const { promise, resolve, reject } = Promise.withResolvers<void>();
			const stdin = this.#child.stdin;
			if (!stdin || !this.#stdinOpen || !this.running) reject(new Error("jev-opt worker stdin is closed"));
			else stdin.write(line, error => (error ? reject(error) : resolve()));
			return promise;
		};
		const written = this.#writeTail.then(run, run);
		this.#writeTail = written.catch(() => {});
		return written;
	}

	#onStderr(text: string): void {
		this.#stderrTail = (this.#stderrTail + text).slice(-STDERR_TAIL_CHARS);
		this.#options.onStderr?.(text);
	}

	#onStdout(chunk: Buffer): void {
		if (this.#failure) return;
		const overflow = () =>
			this.#fail(new WorkerProtocolError(`worker stdout line exceeds ${this.#maxLineBytes} bytes`), true);
		let start = 0;
		for (let nl = chunk.indexOf(NEWLINE); nl !== -1; nl = chunk.indexOf(NEWLINE, start)) {
			const piece = chunk.subarray(start, nl);
			start = nl + 1;
			if (this.#partialBytes + piece.length > this.#maxLineBytes) return overflow();
			const line = this.#partial.length ? Buffer.concat([...this.#partial, piece]) : piece;
			this.#partial = [];
			this.#partialBytes = 0;
			this.#onLine(line.toString("utf8"));
			if (this.#failure) return;
		}
		const rest = chunk.length - start;
		if (rest === 0) return;
		if (this.#partialBytes + rest > this.#maxLineBytes) return overflow();
		this.#partialBytes += rest;
		this.#partial.push(chunk.subarray(start)); // stream chunks are never reused; a view is safe
	}

	#onLine(line: string): void {
		const text = line.trim();
		if (!text) return;
		let message: unknown;
		try {
			message = JSON.parse(text);
		} catch {
			this.#fail(new WorkerProtocolError(`worker wrote a non-JSON stdout line: ${text.slice(0, 200)}`), true);
			return;
		}
		if (!isObject(message)) {
			this.#fail(new WorkerProtocolError(`worker frame is not an object: ${text.slice(0, 200)}`), true);
			return;
		}
		const { id, method } = message;
		const hasId = typeof id === "string" || typeof id === "number";
		const params = isObject(message.params) ? message.params : {};
		if (typeof method === "string") {
			if (hasId) this.#onRequest(String(id), method, params);
			else this.#onNotification(method, params);
		} else if (hasId && ("result" in message || "error" in message)) {
			const { error } = message;
			const remote =
				error === undefined || error === null
					? undefined
					: new WorkerRemoteError(
							isObject(error) && typeof error.message === "string" ? error.message : JSON.stringify(error),
						);
			this.#settle(String(id), message.result, remote);
		} else {
			this.#fail(new WorkerProtocolError(`unrecognised worker frame: ${text.slice(0, 200)}`), true);
		}
	}

	#onNotification(method: string, params: JsonObject): void {
		try {
			if (method === "progress") this.#options.onProgress?.(params);
			else this.#options.onNotification?.(method, params);
		} catch (error) {
			this.#options.onStderr?.(`[jev-opt] ${method} handler failed: ${String(error)}\n`);
		}
	}

	#onRequest(id: string, method: string, params: JsonObject): void {
		const handler = this.#handlers.get(method);
		if (!handler) {
			this.#reply({ id, error: { message: `unknown host method: ${method}` } });
			return;
		}
		Promise.resolve()
			.then(() => handler(params, this.#incoming.signal))
			.then(
				result => this.#reply({ id, result: result ?? {} }),
				error => this.#reply({ id, error: { message: error instanceof Error ? error.message : String(error) } }),
			);
	}

	#reply(message: JsonObject): void {
		if (this.running && !this.#failure) this.#write(message).catch(() => {});
	}

	#settle(id: string, result: unknown, error: Error | undefined): void {
		const pending = this.#pending.get(id);
		if (!pending) return; // late reply to a request that already settled
		this.#pending.delete(id);
		pending.dispose();
		if (error) pending.reject(error);
		else pending.resolve(result);
	}

	#rejectAll(error: Error): void {
		const pending = [...this.#pending.values()];
		this.#pending.clear();
		for (const entry of pending) {
			entry.dispose();
			entry.reject(error);
		}
	}

	#fail(error: Error, kill: boolean): void {
		if (this.#failure || this.#exitInfo) return;
		this.#failure = error;
		this.#rejectAll(error);
		this.#incoming.abort(error);
		if (!kill) return;
		this.#endStdin();
		this.kill("SIGTERM");
		const timer = setTimeout(() => this.kill("SIGKILL"), this.#killTimeoutMs);
		timer.unref?.();
		this.exited.then(() => clearTimeout(timer));
	}

	#exitError(info: ExitInfo, cause?: Error): Error {
		const how = cause ? `failed to start (${cause.message})` : `exited (code ${info.code}, signal ${info.signal})`;
		const tail = this.#stderrTail.trim();
		return new WorkerExitedError(`jev-opt worker ${how}${tail ? `\n--- worker stderr (tail) ---\n${tail}` : ""}`, info);
	}

	#finish(info: ExitInfo, cause: Error | undefined): void {
		if (this.#exitInfo) return;
		this.#exitInfo = info;
		this.#stdinOpen = false;
		const error = this.#failure ?? this.#exitError(info, cause);
		this.#rejectAll(error);
		this.#incoming.abort(error);
		try {
			this.#options.onExit?.(info, this.#failure ?? (cause ? error : undefined));
		} finally {
			this.#resolveExited(info);
		}
	}
}
