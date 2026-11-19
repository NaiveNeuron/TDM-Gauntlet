import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { formatProgress, formatValue, parseArgs, splitArgs } from "./cli.ts";

describe("command arguments", () => {
	it("splits quoted paths and escapes like a shell", () => {
		assert.deepEqual(splitArgs(`"tasks/my task.yaml"  --run-id 'a b' c\\ d ""`), [
			"tasks/my task.yaml",
			"--run-id",
			"a b",
			"c d",
			"",
		]);
		assert.throws(() => splitArgs(`"open`), /unterminated " quote/);
	});

	it("parses value and boolean flags anywhere, keeping positionals in order", () => {
		const spec = { conditions: "string", "run-id": "string", yes: "boolean" } as const;
		assert.deepEqual(parseArgs("task.yaml --conditions jev-zero,jev-text-opt --run-id=r7 --yes extra", spec), {
			positional: ["task.yaml", "extra"],
			flags: { conditions: "jev-zero,jev-text-opt", "run-id": "r7", yes: true },
		});
	});

	it("rejects unknown flags, missing values and values on boolean flags", () => {
		const spec = { "run-id": "string", yes: "boolean" } as const;
		assert.throws(() => parseArgs("task.yaml --resume", spec), /unknown option --resume/);
		assert.throws(() => parseArgs("task.yaml --run-id", spec), /--run-id needs a value/);
		assert.throws(() => parseArgs("task.yaml --run-id --yes", spec), /--run-id needs a value/);
		assert.throws(() => parseArgs("run --yes=1", spec), /--yes takes no value/);
		assert.throws(() => parseArgs("--constructor x", spec), /unknown option --constructor/);
	});
});

describe("rendering", () => {
	it("renders nested worker results as indented YAML-like text", () => {
		const text = formatValue({ "jev-text-opt": { val: { macro_f1: 0.812345, n: 128 }, candidates: 7 }, front: [1, 2] });
		assert.equal(text, "\njev-text-opt:\n  val:\n    macro_f1: 0.8123\n    n: 128\n  candidates: 7\nfront: [1, 2]");
		assert.equal(formatValue([{ id: 1, q: 0.5 }, { id: 2 }]), "\n- id: 1\n  q: 0.5\n- id: 2");
	});

	it("formats progress notifications with optional fields", () => {
		assert.equal(
			formatProgress({ run_id: "r1", condition: "jev-text-opt", event: "new_best", iteration: 4, best: 0.8, message: "val 0.80" }),
			"r1/jev-text-opt new_best it=4 best=0.8: val 0.80",
		);
		assert.equal(formatProgress({ run_id: "r1", condition: null, event: "run_started" }), "r1 run_started");
	});
});
