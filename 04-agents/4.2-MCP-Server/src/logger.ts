/**
 * stderr-only structured logger.
 *
 * CRITICAL (stdio transport): in an MCP `stdio` server, `stdout` is the
 * JSON-RPC channel. A single stray `console.log` corrupts the message stream
 * and instantly crashes the client connection. ALL diagnostics must go to
 * `stderr`. Never use `console.log` anywhere in this server.
 *
 * See: "Transport Layer — stdio vs HTTP+SSE" — "Logging to stdout in a stdio
 * server ... will corrupt the JSON-RPC stream and immediately crash the client."
 */

type Level = "debug" | "info" | "warn" | "error";

function format(level: Level, args: unknown[]): string {
  const ts = new Date().toISOString();
  const body = args
    .map((a) => {
      if (a instanceof Error) return a.stack ?? `${a.name}: ${a.message}`;
      if (typeof a === "string") return a;
      try {
        return JSON.stringify(a);
      } catch {
        return String(a);
      }
    })
    .join(" ");
  return `${ts} [mcp-ts-server] ${level.toUpperCase()} ${body}\n`;
}

function emit(level: Level, args: unknown[]): void {
  // Always stderr — never stdout.
  process.stderr.write(format(level, args));
}

export const log = {
  debug: (...args: unknown[]): void => emit("debug", args),
  info: (...args: unknown[]): void => emit("info", args),
  warn: (...args: unknown[]): void => emit("warn", args),
  error: (...args: unknown[]): void => emit("error", args),
};
