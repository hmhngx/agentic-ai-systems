/**
 * Tool: sqlite_query — run a single READ-ONLY SQL query against the seeded
 * agent database and return typed rows.
 *
 * Three independent layers keep this safe:
 *   1. Zod input schema  — shape/length/limit constraints before execution.
 *   2. Statement allow-list — must be a single SELECT / WITH / PRAGMA / EXPLAIN.
 *   3. Read-only connection — better-sqlite3 + `query_only` pragma physically
 *      reject any write that somehow gets past layers 1–2.
 */
import { z } from "zod";
import type BetterSqlite3 from "better-sqlite3";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { getDb } from "../db.js";
import { ToolError, safeHandler } from "../errors.js";

/** Values better-sqlite3 can bind to a `?` placeholder. */
type BindValue = string | number | null;

const ScalarParam = z.union([z.string(), z.number(), z.boolean(), z.null()]);

const inputSchema = {
  sql: z
    .string()
    .min(1)
    .max(2000)
    .describe(
      "A single read-only SQL statement. Only SELECT, WITH, PRAGMA and EXPLAIN are permitted. " +
        "Use ? placeholders for values and pass them via `params`.",
    ),
  params: z
    .array(ScalarParam)
    .max(64)
    .optional()
    .describe("Positional values bound to the query's ? placeholders, in order."),
  limit: z
    .number()
    .int()
    .min(1)
    .max(1000)
    .default(100)
    .describe("Maximum number of rows to return (default 100, max 1000)."),
} as const;

const outputSchema = {
  rowCount: z.number().int().describe("Number of rows returned (after the limit)."),
  columns: z.array(z.string()).describe("Column names in result order."),
  rows: z
    .array(z.record(z.string(), z.unknown()))
    .describe("Result rows as objects keyed by column name."),
  truncated: z
    .boolean()
    .describe("True if more rows existed than `limit` and were dropped."),
} as const;

/** Allow-list gate: a single statement that only reads. */
function assertReadOnly(rawSql: string): string {
  // Strip a single trailing semicolon, then reject any remaining one — this
  // blocks statement stacking like "SELECT 1; DROP TABLE x".
  const sql = rawSql.trim().replace(/;\s*$/, "");
  if (sql.length === 0) {
    throw new ToolError("INVALID_INPUT", "SQL statement is empty.");
  }
  if (sql.includes(";")) {
    throw new ToolError(
      "INVALID_INPUT",
      "Only a single SQL statement is allowed (remove the ';').",
    );
  }
  if (!/^(select|with|pragma|explain)\b/i.test(sql)) {
    throw new ToolError(
      "FORBIDDEN",
      "Only read-only queries are permitted (SELECT, WITH, PRAGMA, EXPLAIN).",
      { received: sql.slice(0, 40) },
    );
  }
  return sql;
}

export function registerSqliteQueryTool(server: McpServer): void {
  server.registerTool(
    "sqlite_query",
    {
      title: "SQLite Query (read-only)",
      description:
        "Execute a single read-only SQL query (SELECT/WITH/PRAGMA/EXPLAIN) against the local " +
        "agent-analytics database and return typed rows. Schema: agent_runs(id, agent_name, task, " +
        "status, iterations, started_at, finished_at); tool_calls(id, run_id, tool_name, " +
        "arguments_json, latency_ms, success, created_at); documents(id, title, source, tokens, " +
        "indexed_at). Writes are rejected.",
      inputSchema,
      outputSchema,
      annotations: {
        title: "SQLite Query (read-only)",
        readOnlyHint: true,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    safeHandler(async ({ sql, params, limit }) => {
      const safeSql = assertReadOnly(sql);
      const db = getDb();

      let stmt: BetterSqlite3.Statement;
      try {
        stmt = db.prepare(safeSql);
      } catch (err) {
        throw new ToolError(
          "INVALID_INPUT",
          `SQL could not be prepared: ${(err as Error).message}`,
        );
      }

      // SQLite has no boolean type; coerce true/false to 1/0 before binding.
      const bound: BindValue[] = (params ?? []).map((p) =>
        typeof p === "boolean" ? (p ? 1 : 0) : p,
      );
      let columns: string[] = [];
      let rows: Array<Record<string, unknown>> = [];

      if (stmt.reader) {
        // Statement returns rows (SELECT/WITH/PRAGMA-with-output/EXPLAIN).
        columns = stmt.columns().map((c) => c.name);
        rows = stmt.all(...bound) as Array<Record<string, unknown>>;
      } else {
        // No result set (e.g. a `PRAGMA ... = ...`). On a read-only connection
        // any write attempt throws here and surfaces as a structured error.
        stmt.run(...bound);
      }

      const truncated = rows.length > limit;
      const limited = truncated ? rows.slice(0, limit) : rows;

      const structuredContent = {
        rowCount: limited.length,
        columns,
        rows: limited,
        truncated,
      };

      return {
        content: [{ type: "text", text: JSON.stringify(structuredContent, null, 2) }],
        structuredContent,
      };
    }),
  );
}
