#!/usr/bin/env node
/**
 * MCP stdio server entrypoint.
 *
 * Lifecycle (see "Server Lifecycle — Initialize to Shutdown"):
 *   1. Construct McpServer and register the three tools.
 *   2. Connect a StdioServerTransport — the SDK handles the initialize /
 *      initialized / tools.list / tools.call JSON-RPC flow.
 *   3. On SIGINT / SIGTERM / SIGHUP or stdin close, shut down gracefully:
 *      close the transport, release the SQLite handle, exit(0).
 *
 * Robustness contract: the process must never die on bad tool input. Each
 * tool wraps its body in `safeHandler`, and the SDK additionally converts any
 * thrown error into a structured `isError` result. The global handlers below
 * are a last-resort net for anything truly unexpected.
 */
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { SERVER_INFO } from "./config.js";
import { closeDb } from "./db.js";
import { log } from "./logger.js";
import { registerSqliteQueryTool } from "./tools/sqlite-query.js";
import { registerFileReadTool } from "./tools/file-read.js";
import { registerCalcTool } from "./tools/calc.js";

async function main(): Promise<void> {
  const server = new McpServer(SERVER_INFO);

  registerSqliteQueryTool(server);
  registerFileReadTool(server);
  registerCalcTool(server);

  const transport = new StdioServerTransport();
  await server.connect(transport);
  log.info("connected on stdio — registered tools: sqlite_query, file_read, calc");

  // ---- Graceful shutdown -------------------------------------------------
  let shuttingDown = false;
  const shutdown = async (reason: string): Promise<void> => {
    if (shuttingDown) return; // idempotent: ignore repeat signals
    shuttingDown = true;
    log.info(`shutting down (${reason})`);
    try {
      await server.close();
    } catch (err) {
      log.error("error closing server", err);
    }
    try {
      closeDb();
    } catch (err) {
      log.error("error closing database", err);
    }
    log.info("shutdown complete");
    process.exit(0);
  };

  for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"] as const) {
    process.on(signal, () => {
      void shutdown(signal);
    });
  }

  // When the MCP client disconnects, stdin closes — treat as shutdown.
  process.stdin.on("close", () => {
    void shutdown("stdin closed");
  });
}

// Last-resort safety net: log and keep serving rather than crashing.
process.on("uncaughtException", (err) => {
  log.error("uncaughtException (kept alive)", err);
});
process.on("unhandledRejection", (reason) => {
  log.error("unhandledRejection (kept alive)", reason);
});

main().catch((err) => {
  log.error("fatal error during startup", err);
  process.exit(1);
});
