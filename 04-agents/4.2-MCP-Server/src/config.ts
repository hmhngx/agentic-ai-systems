/**
 * Centralised, environment-overridable configuration.
 *
 * Security note (least privilege): both the database location and the
 * file-read sandbox root are resolved here once, so every tool shares the
 * same trust boundary. `FILE_ROOT` is the hard ceiling for `file_read` — the
 * tool refuses to resolve any path outside it (see tools/file-read.ts).
 */
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));

/** Project root = the `ts-server/` directory (parent of `src/`). */
export const PROJECT_ROOT = path.resolve(HERE, "..");

/** SQLite database path. Override with MCP_DB_PATH. */
export const DB_PATH = process.env.MCP_DB_PATH
  ? path.resolve(process.env.MCP_DB_PATH)
  : path.join(PROJECT_ROOT, "data", "agent.db");

/**
 * Sandbox root for `file_read`. Override with MCP_FILE_ROOT.
 * Defaults to the project root so the server can only read files it ships
 * with — never the wider filesystem (no ~/.ssh, no /etc, etc.).
 */
export const FILE_ROOT = process.env.MCP_FILE_ROOT
  ? path.resolve(process.env.MCP_FILE_ROOT)
  : PROJECT_ROOT;

export const SERVER_INFO = {
  name: "mcp-ts-server",
  version: "1.0.0",
} as const;
