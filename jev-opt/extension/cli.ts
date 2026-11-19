// Pure helpers for command handlers: argument parsing and plain-text rendering of worker results.
import type { JsonObject } from "./bridge.ts";

export type FlagSpec = Record<string, "string" | "boolean">;

export interface ParsedArgs {
	positional: string[];
	flags: Record<string, string | true>;
}

/** Shell-like split: whitespace separated, '…' literal, "…" with backslash escapes. */
export function splitArgs(input: string): string[] {
	const words: string[] = [];
	let word = "";
	let inWord = false;
	let quote: '"' | "'" | undefined;
	for (let i = 0; i < input.length; i++) {
		const ch = input[i];
		if (quote) {
			if (ch === quote) quote = undefined;
			else if (ch === "\\" && quote === '"' && i + 1 < input.length) word += input[++i];
			else word += ch;
		} else if (ch === '"' || ch === "'") {
			quote = ch;
			inWord = true;
		} else if (ch === "\\" && i + 1 < input.length) {
			word += input[++i];
			inWord = true;
		} else if (/\s/.test(ch)) {
			if (inWord) words.push(word);
			word = "";
			inWord = false;
		} else {
			word += ch;
			inWord = true;
		}
	}
	if (quote) throw new Error(`unterminated ${quote} quote in arguments`);
	if (inWord) words.push(word);
	return words;
}

/** Parse `--flag value`, `--flag=value` and boolean `--flag` against `spec`; rejects unknown flags. */
export function parseArgs(input: string, spec: FlagSpec): ParsedArgs {
	const parsed: ParsedArgs = { positional: [], flags: {} };
	const words = splitArgs(input);
	for (let i = 0; i < words.length; i++) {
		const word = words[i];
		if (!word.startsWith("--") || word === "--") {
			parsed.positional.push(word);
			continue;
		}
		const eq = word.indexOf("=");
		const name = word.slice(2, eq === -1 ? undefined : eq);
		const kind = Object.hasOwn(spec, name) ? spec[name] : undefined;
		if (!kind) throw new Error(`unknown option --${name}`);
		if (kind === "boolean") {
			if (eq !== -1) throw new Error(`option --${name} takes no value`);
			parsed.flags[name] = true;
		} else if (eq !== -1) parsed.flags[name] = word.slice(eq + 1);
		else if (i + 1 < words.length && !words[i + 1].startsWith("--")) parsed.flags[name] = words[++i];
		else throw new Error(`option --${name} needs a value`);
	}
	return parsed;
}

function isObject(value: unknown): value is JsonObject {
	return value !== null && typeof value === "object" && !Array.isArray(value);
}

function scalar(value: unknown): string {
	if (typeof value === "number") return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(4)));
	if (typeof value === "string") return value;
	return JSON.stringify(value) ?? "null";
}

/** `key: value` for scalars, `key:` followed by the indented block for nested values. */
function keyed(key: string, rendered: string): string {
	return rendered.startsWith("\n") ? `${key}:${rendered}` : `${key}: ${rendered}`;
}

/** YAML-like indented rendering of a JSON value (worker results are rendered, not interpreted). */
export function formatValue(value: unknown, indent = ""): string {
	if (Array.isArray(value)) {
		if (value.every(item => !isObject(item) && !Array.isArray(item))) return `[${value.map(scalar).join(", ")}]`;
		return value.map(item => `\n${indent}- ${formatValue(item, `${indent}  `).trimStart()}`).join("");
	}
	if (isObject(value)) {
		const entries = Object.entries(value);
		if (!entries.length) return "{}";
		return entries.map(([key, item]) => `\n${indent}${keyed(key, formatValue(item, `${indent}  `))}`).join("");
	}
	return scalar(value);
}

/** One line per `progress` notification: `run/condition event it=N best=B: message`. */
export function formatProgress(params: JsonObject): string {
	const where = [params.run_id, params.condition].filter(part => typeof part === "string" && part).join("/");
	const bits = [where, typeof params.event === "string" ? params.event : "progress"];
	if (params.iteration !== undefined && params.iteration !== null) bits.push(`it=${scalar(params.iteration)}`);
	if (params.best !== undefined && params.best !== null) bits.push(`best=${scalar(params.best)}`);
	const head = bits.filter(Boolean).join(" ");
	return typeof params.message === "string" && params.message ? `${head}: ${params.message}` : head;
}

export function renderSection(title: string, value: unknown): string {
	return keyed(title, formatValue(value, "  "));
}
