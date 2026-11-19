import assert from "node:assert/strict";
import { after, describe, it } from "node:test";
import { WorkerBridge, WorkerExitedError, WorkerProtocolError, WorkerRemoteError } from "./bridge.ts";
import type { JsonObject, WorkerBridgeOptions } from "./bridge.ts";

// Fake worker speaking protocol v1 over stdin/stdout; behaviour is chosen by the request method.
const FAKE_WORKER = String.raw`
import readline from "node:readline";
const send = message => process.stdout.write(JSON.stringify(message) + "\n");
const waiting = new Map();
const echoes = [];
let seq = 0;
let held;
const call = (method, params) => {
	const id = "w" + ++seq;
	send({ id, method, params });
	return new Promise(resolve => waiting.set(id, resolve));
};
const rl = readline.createInterface({ input: process.stdin });
rl.on("line", async line => {
	const msg = JSON.parse(line);
	if (!msg.method) {
		waiting.get(msg.id)?.(msg);
		return;
	}
	switch (msg.method) {
		case "hello":
			send({ id: msg.id, result: { protocol: 1 } });
			break;
		case "optimize": {
			send({ method: "progress", params: { run_id: "r1", event: "run_started", message: "go" } });
			const lm = await call("lm.complete", { role: "reflection", prompt: "improve this" });
			const unknown = await call("no.such.method", {});
			send({ id: msg.id, result: { lm, unknown } });
			break;
		}
		case "echo":
			// Hold every echo until all arrived, then answer in reverse order.
			echoes.push(msg);
			if (echoes.length === msg.params.total) for (const m of echoes.reverse()) send({ id: m.id, result: m.params });
			break;
		case "fail":
			send({ id: msg.id, error: { message: "boom" } });
			break;
		case "slow":
			held = msg.id;
			break;
		case "cancel":
			send({ id: msg.id, result: { ok: true } });
			if (held) send({ id: held, result: { cancelled: true } });
			break;
		case "lm-then-die":
			call("lm.complete", {});
			process.exit(9);
		case "crash":
			process.stderr.write("Traceback: worker exploded\n");
			process.exit(3);
		case "garbage":
			process.stdout.write("GEPA iteration 1 started\n");
			break;
		case "huge":
			process.stdout.write("x".repeat(4096));
			break;
		case "shutdown":
			send({ id: msg.id, result: { ok: true } });
			process.exit(0);
	}
});
rl.on("close", () => process.exit(0));
`;

const spawned: WorkerBridge[] = [];

function fakeWorker(options: Partial<WorkerBridgeOptions> = {}): WorkerBridge {
	const bridge = new WorkerBridge({
		command: process.execPath,
		args: ["--input-type=module", "-e", FAKE_WORKER],
		killTimeoutMs: 200,
		...options,
	});
	spawned.push(bridge);
	return bridge;
}

// A failed assertion must not leave a fake worker holding the test process open.
after(() => {
	for (const bridge of spawned) bridge.kill();
});

describe("WorkerBridge", { timeout: 10_000 }, () => {
	it("answers worker → host requests while a host request is pending", async () => {
		const progress: JsonObject[] = [];
		const bridge = fakeWorker({
			onProgress: params => progress.push(params),
			handlers: {
				"lm.complete": async params => ({ text: `ok:${String(params.prompt)}` }),
			},
		});
		assert.deepEqual(await bridge.request("hello"), { protocol: 1 });
		const result = await bridge.request("optimize", {});
		assert.deepEqual(result, {
			lm: { id: "w1", result: { text: "ok:improve this" } },
			unknown: { id: "w2", error: { message: "unknown host method: no.such.method" } },
		});
		assert.deepEqual(progress, [{ run_id: "r1", event: "run_started", message: "go" }]);
		assert.deepEqual(await bridge.shutdown(), { code: 0, signal: null });
	});

	it("matches out-of-order replies to their requests", async () => {
		const bridge = fakeWorker();
		const total = 6;
		const replies = await Promise.all(Array.from({ length: total }, (_, n) => bridge.request("echo", { n, total })));
		assert.deepEqual(
			replies,
			Array.from({ length: total }, (_, n) => ({ n, total })),
		);
		await bridge.shutdown();
	});

	it("rejects with the worker's error message and stays usable", async () => {
		const bridge = fakeWorker();
		await assert.rejects(bridge.request("fail"), (error: Error) => {
			assert.ok(error instanceof WorkerRemoteError);
			assert.equal(error.message, "boom");
			return true;
		});
		assert.deepEqual(await bridge.request("hello"), { protocol: 1 });
		await bridge.shutdown();
	});

	it("turns abort into a cooperative cancel and still delivers the worker's reply", async () => {
		const bridge = fakeWorker();
		const controller = new AbortController();
		const pending = bridge.request("slow", {}, { signal: controller.signal });
		controller.abort();
		assert.deepEqual(await pending, { cancelled: true });
		await assert.rejects(bridge.request("hello", {}, { signal: controller.signal }), /abort/i);
		await bridge.shutdown();
	});

	it("rejects pending requests with the stderr tail when the worker crashes", async () => {
		const bridge = fakeWorker();
		const slow = bridge.request("slow");
		await assert.rejects(bridge.request("crash"), (error: Error) => {
			assert.ok(error instanceof WorkerExitedError);
			assert.equal(error.info.code, 3);
			assert.match(error.message, /Traceback: worker exploded/);
			return true;
		});
		await assert.rejects(slow, WorkerExitedError);
		await assert.rejects(bridge.request("hello"), WorkerExitedError);
		assert.equal(bridge.running, false);
	});

	it("aborts in-flight host handlers when the worker dies", async () => {
		const { promise: handlerSignal, resolve } = Promise.withResolvers<AbortSignal>();
		const bridge = fakeWorker({
			handlers: {
				"lm.complete": (_params, signal) => {
					resolve(signal);
					return new Promise(() => {});
				},
			},
		});
		bridge.request("lm-then-die").catch(() => {});
		assert.deepEqual(await bridge.exited, { code: 9, signal: null });
		assert.equal((await handlerSignal).aborted, true);
	});

	it("treats a non-JSON stdout line as a fatal protocol error", async () => {
		const bridge = fakeWorker();
		await assert.rejects(bridge.request("garbage"), (error: Error) => {
			assert.ok(error instanceof WorkerProtocolError);
			assert.match(error.message, /GEPA iteration 1 started/);
			return true;
		});
		await bridge.exited;
		await assert.rejects(bridge.request("hello"), WorkerProtocolError);
	});

	it("enforces the line-size limit", async () => {
		const bridge = fakeWorker({ maxLineBytes: 1024 });
		await assert.rejects(bridge.request("huge"), /exceeds 1024 bytes/);
		await bridge.exited;
	});

	it("escalates shutdown to SIGKILL when the worker ignores shutdown, EOF and SIGTERM", async () => {
		// Real signals and grace timers are the behaviour under test, so this uses the wall clock.
		const stubborn = new WorkerBridge({
			command: process.execPath,
			args: ["-e", "process.on('SIGTERM', () => {}); process.stdin.resume(); setInterval(() => {}, 1000);"],
			killTimeoutMs: 100,
		});
		assert.deepEqual(await stubborn.shutdown(), { code: null, signal: "SIGKILL" });
		await assert.rejects(stubborn.request("hello"), WorkerExitedError);
	});

	it("reports a missing executable as a start failure", async () => {
		const bridge = new WorkerBridge({ command: "/nonexistent/jev-opt-worker" });
		await assert.rejects(bridge.request("hello"), /failed to start/);
		assert.equal(bridge.running, false);
	});
});
