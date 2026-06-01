/**
 * Tool-level error handling helpers.
 *
 * The contract for this server: a tool NEVER throws an unstructured error to
 * the client. Bad input, missing files, division by zero — all of it comes
 * back as a structured MCP tool result with `isError: true`. The process
 * stays alive and keeps serving the next request.
 *
 * `ToolError` carries a machine-readable `code` so clients (and our smoke
 * test) can branch on the failure category instead of string-matching.
 */
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";

export type ToolErrorCode =
  | "INVALID_INPUT"
  | "FORBIDDEN"
  | "NOT_FOUND"
  | "EXECUTION_ERROR";

export class ToolError extends Error {
  readonly code: ToolErrorCode;
  readonly details?: Record<string, unknown>;

  constructor(
    code: ToolErrorCode,
    message: string,
    details?: Record<string, unknown>,
  ) {
    super(message);
    this.name = "ToolError";
    this.code = code;
    if (details !== undefined) this.details = details;
  }
}

/** Build a structured error result. Output-schema validation is skipped by the
 * SDK whenever `isError` is true, so no `structuredContent` is required here. */
export function toErrorResult(err: unknown): CallToolResult {
  const isToolError = err instanceof ToolError;
  const code: ToolErrorCode = isToolError ? err.code : "EXECUTION_ERROR";
  const message =
    err instanceof Error ? err.message : typeof err === "string" ? err : String(err);

  const payload = {
    ok: false as const,
    error: {
      code,
      message,
      ...(isToolError && err.details ? { details: err.details } : {}),
    },
  };

  return {
    isError: true,
    content: [{ type: "text", text: JSON.stringify(payload, null, 2) }],
  };
}

/**
 * Wraps a tool handler so any thrown value becomes a structured error result.
 *
 * Defense in depth: the SDK already wraps handlers in try/catch, but doing it
 * here lets us emit our typed `{ ok, error: { code, message } }` envelope
 * instead of a bare error string.
 */
export function safeHandler<Args>(
  fn: (args: Args) => Promise<CallToolResult> | CallToolResult,
): (args: Args) => Promise<CallToolResult> {
  return async (args: Args): Promise<CallToolResult> => {
    try {
      return await fn(args);
    } catch (err) {
      return toErrorResult(err);
    }
  };
}
