/**
 * End-to-end smoke test.
 *
 * Spawns the real server as a subprocess and talks to it over genuine stdio
 * JSON-RPC using the MCP SDK client. Verifies:
 *   - all three tools are discoverable (tools/list)
 *   - each tool executes correctly on good input (typed structuredContent)
 *   - every flavour of bad input returns a structured { isError: true } result
 *     and the server stays alive for the next call
 *
 *   npm run smoke
 */
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SERVER_ENTRY = path.join(HERE, "index.ts");

let passed = 0;
let failed = 0;

function check(name: string, condition: boolean, detail = ""): void {
  if (condition) {
    passed++;
    process.stdout.write(`  ✓ ${name}\n`);
  } else {
    failed++;
    process.stdout.write(`  ✗ ${name}${detail ? ` — ${detail}` : ""}\n`);
  }
}

function structured<T = Record<string, unknown>>(res: CallToolResult): T {
  return res.structuredContent as T;
}

async function main(): Promise<void> {
  const transport = new StdioClientTransport({
    command: process.execPath, // node
    args: ["--import", "tsx", SERVER_ENTRY],
    stderr: "inherit",
  });
  const client = new Client({ name: "smoke-test", version: "1.0.0" });

  await client.connect(transport);
  process.stdout.write("Connected to server.\n\n");

  // --- Discovery ----------------------------------------------------------
  const { tools } = await client.listTools();
  const names = tools.map((t) => t.name).sort();
  process.stdout.write("tools/list:\n");
  check("3 tools discovered", tools.length === 3, `got ${tools.length}`);
  check(
    "tool names == [calc, file_read, sqlite_query]",
    JSON.stringify(names) === JSON.stringify(["calc", "file_read", "sqlite_query"]),
    names.join(","),
  );
  check(
    "every tool exposes an inputSchema",
    tools.every((t) => t.inputSchema && t.inputSchema.type === "object"),
  );

  // --- calc ---------------------------------------------------------------
  process.stdout.write("\ncalc:\n");
  {
    const r = (await client.callTool({ name: "calc", arguments: { expression: "2 * -3" } })) as CallToolResult;
    check("2 * -3 === -6", structured<{ result: number }>(r).result === -6);
  }
  {
    const r = (await client.callTool({ name: "calc", arguments: { expression: "-2^2" } })) as CallToolResult;
    check("-2^2 === -4", structured<{ result: number }>(r).result === -4);
  }
  {
    const r = (await client.callTool({ name: "calc", arguments: { expression: "(2^16)/1024" } })) as CallToolResult;
    check("(2^16)/1024 === 64", structured<{ result: number }>(r).result === 64);
  }
  {
    const r = (await client.callTool({ name: "calc", arguments: { expression: "10 / 0" } })) as CallToolResult;
    check("division by zero -> isError", r.isError === true);
  }
  {
    const r = (await client.callTool({ name: "calc", arguments: { expression: "2 +" } })) as CallToolResult;
    check("malformed '2 +' -> isError", r.isError === true);
  }
  {
    const r = (await client.callTool({ name: "calc", arguments: { expression: "2 + abc" } })) as CallToolResult;
    check("non-numeric '2 + abc' -> isError", r.isError === true);
  }
  {
    // Zod input violation (wrong type) must also be a structured error.
    const r = (await client.callTool({ name: "calc", arguments: { expression: 42 } })) as CallToolResult;
    check("bad input type (number) -> isError", r.isError === true);
  }

  // --- sqlite_query -------------------------------------------------------
  process.stdout.write("\nsqlite_query:\n");
  {
    const r = (await client.callTool({
      name: "sqlite_query",
      arguments: { sql: "SELECT count(*) AS n FROM documents" },
    })) as CallToolResult;
    const s = structured<{ rows: Array<{ n: number }>; rowCount: number }>(r);
    check("count documents -> rows[0].n === 6", s.rows[0]?.n === 6, JSON.stringify(s.rows));
  }
  {
    const r = (await client.callTool({
      name: "sqlite_query",
      arguments: {
        sql: "SELECT agent_name FROM agent_runs WHERE status = ? ORDER BY id",
        params: ["success"],
      },
    })) as CallToolResult;
    const s = structured<{ rowCount: number; columns: string[] }>(r);
    check("parameterised query returns rows", s.rowCount === 3, `rowCount=${s.rowCount}`);
    check("columns == [agent_name]", JSON.stringify(s.columns) === JSON.stringify(["agent_name"]));
  }
  {
    const r = (await client.callTool({
      name: "sqlite_query",
      arguments: { sql: "DROP TABLE agent_runs" },
    })) as CallToolResult;
    check("DROP TABLE -> isError (forbidden)", r.isError === true);
  }
  {
    const r = (await client.callTool({
      name: "sqlite_query",
      arguments: { sql: "SELECT 1; SELECT 2" },
    })) as CallToolResult;
    check("stacked statements -> isError", r.isError === true);
  }
  {
    const r = (await client.callTool({
      name: "sqlite_query",
      arguments: { sql: "SELECT * FROM no_such_table" },
    })) as CallToolResult;
    check("unknown table -> isError (not crash)", r.isError === true);
  }
  {
    // Verify the connection is still alive after all those errors.
    const r = (await client.callTool({
      name: "sqlite_query",
      arguments: { sql: "SELECT 1 AS alive" },
    })) as CallToolResult;
    check("server still serving after errors", structured<{ rows: Array<{ alive: number }> }>(r).rows[0]?.alive === 1);
  }

  // --- file_read ----------------------------------------------------------
  process.stdout.write("\nfile_read:\n");
  {
    const r = (await client.callTool({
      name: "file_read",
      arguments: { path: "package.json" },
    })) as CallToolResult;
    const s = structured<{ content: string }>(r);
    check("read package.json contains server name", s.content.includes("mcp-ts-server"));
  }
  {
    const r = (await client.callTool({
      name: "file_read",
      arguments: { path: "../../../../../../etc/passwd" },
    })) as CallToolResult;
    check("path traversal -> isError (forbidden)", r.isError === true);
  }
  {
    const r = (await client.callTool({
      name: "file_read",
      arguments: { path: "does-not-exist.txt" },
    })) as CallToolResult;
    check("missing file -> isError (not found)", r.isError === true);
  }

  await client.close();

  process.stdout.write(`\n${passed} passed, ${failed} failed.\n`);
  process.exit(failed === 0 ? 0 : 1);
}

main().catch((err) => {
  process.stderr.write(`smoke test crashed: ${err instanceof Error ? err.stack : String(err)}\n`);
  process.exit(1);
});
