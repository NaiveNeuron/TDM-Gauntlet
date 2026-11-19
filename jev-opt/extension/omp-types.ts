// Minimal structural subset of the omp extension and pi-ai surfaces jev-opt uses.
// The host provides the real implementations at runtime (@oh-my-pi/pi-coding-agent,
// @oh-my-pi/pi-ai 18.3.x); declaring only what we touch keeps tests free of omp.

export interface Model {
	provider: string;
	id: string;
	api?: string;
}

export interface Usage {
	input: number;
	output: number;
	cacheRead?: number;
	cacheWrite?: number;
	totalTokens?: number;
	cost?: { input?: number; output?: number; cacheRead?: number; cacheWrite?: number; total?: number };
}

export interface TextContent {
	type: "text";
	text: string;
}

export interface AssistantMessage {
	role: "assistant";
	content: Array<TextContent | { type: string }>;
	api?: string;
	provider: string;
	model: string;
	usage: Usage;
	stopReason: string;
	errorMessage?: string;
	timestamp?: number;
}

export interface UserMessage {
	role: "user";
	content: TextContent[];
	timestamp: number;
}

export interface Context {
	systemPrompt?: string[];
	messages: Array<UserMessage | AssistantMessage>;
}

/** Effort levels accepted by pi-ai `SimpleStreamOptions.reasoning`. */
export type Effort = "minimal" | "low" | "medium" | "high" | "xhigh" | "max";

export interface SimpleStreamOptions {
	apiKey?: unknown;
	maxTokens?: number;
	reasoning?: Effort;
	disableReasoning?: boolean;
	signal?: AbortSignal;
}

export type CompleteSimple = (model: Model, context: Context, options?: SimpleStreamOptions) => Promise<AssistantMessage>;

export interface ExtensionUI {
	notify(message: string, type?: "info" | "warning" | "error"): void;
	setStatus(key: string, text: string | undefined): void;
	confirm(title: string, message: string): Promise<boolean>;
}

export interface ModelQuery {
	current(): Model | undefined;
	resolve(spec: string): Model | undefined;
}

export interface ExtensionContext {
	cwd: string;
	hasUI: boolean;
	ui: ExtensionUI;
	models: ModelQuery;
	modelRegistry: { resolver(model: Model, sessionId?: string): unknown };
	sessionManager: { getSessionId(): string | undefined };
	isIdle(): boolean;
	setInterval(fn: () => void, ms: number): unknown;
	clearTimer(timer: unknown): void;
}

export interface CustomMessage {
	customType: string;
	content: string;
	display: boolean;
	details?: unknown;
}

export interface ExtensionAPI {
	setLabel(label: string): void;
	registerCommand(
		name: string,
		command: { description?: string; handler: (args: string, ctx: ExtensionContext) => Promise<void> | void },
	): void;
	on(event: "session_start" | "session_shutdown", handler: (event: unknown, ctx: ExtensionContext) => unknown): void;
	appendEntry(customType: string, data: unknown): void;
	sendMessage(message: CustomMessage, options?: { triggerTurn?: boolean; deliverAs?: "steer" | "followUp" }): void;
}
