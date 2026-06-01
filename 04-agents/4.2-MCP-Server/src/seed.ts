/**
 * Deterministic database seeder.
 *
 * Creates `data/agent.db` with a small, realistic schema themed to this repo:
 * agent runs, the tool calls each run made, and indexed documents. Running it
 * repeatedly is safe — it drops and recreates every table.
 *
 *   npm run seed
 */
import fs from "node:fs";
import path from "node:path";
import Database from "better-sqlite3";
import { DB_PATH } from "./config.js";
import { log } from "./logger.js";

function seed(): void {
  fs.mkdirSync(path.dirname(DB_PATH), { recursive: true });

  // Open read-write here (the server itself only ever opens read-only).
  const db = new Database(DB_PATH);
  db.pragma("journal_mode = WAL");
  db.pragma("foreign_keys = ON");

  db.exec(`
    DROP TABLE IF EXISTS tool_calls;
    DROP TABLE IF EXISTS agent_runs;
    DROP TABLE IF EXISTS documents;

    CREATE TABLE agent_runs (
      id          INTEGER PRIMARY KEY,
      agent_name  TEXT    NOT NULL,
      task        TEXT    NOT NULL,
      status      TEXT    NOT NULL CHECK (status IN ('success', 'failed', 'running')),
      iterations  INTEGER NOT NULL,
      started_at  TEXT    NOT NULL,
      finished_at TEXT
    );

    CREATE TABLE tool_calls (
      id             INTEGER PRIMARY KEY,
      run_id         INTEGER NOT NULL REFERENCES agent_runs(id),
      tool_name      TEXT    NOT NULL,
      arguments_json TEXT    NOT NULL,
      latency_ms     INTEGER NOT NULL,
      success        INTEGER NOT NULL CHECK (success IN (0, 1)),
      created_at     TEXT    NOT NULL
    );

    CREATE TABLE documents (
      id         INTEGER PRIMARY KEY,
      title      TEXT    NOT NULL,
      source     TEXT    NOT NULL,
      tokens     INTEGER NOT NULL,
      indexed_at TEXT    NOT NULL
    );

    CREATE INDEX idx_tool_calls_run ON tool_calls(run_id);
    CREATE INDEX idx_agent_runs_status ON agent_runs(status);
  `);

  const runs: Array<[number, string, string, string, number, string, string | null]> = [
    [1, "react-agent", "Summarize the Q3 incident report", "success", 4, "2026-05-28T09:01:00Z", "2026-05-28T09:01:12Z"],
    [2, "react-agent", "Find customers with overdue invoices", "success", 6, "2026-05-28T10:14:00Z", "2026-05-28T10:14:31Z"],
    [3, "planner-agent", "Draft migration plan for auth service", "failed", 10, "2026-05-29T15:42:00Z", "2026-05-29T15:43:05Z"],
    [4, "react-agent", "Compute total tokens indexed this week", "success", 3, "2026-05-30T08:20:00Z", "2026-05-30T08:20:07Z"],
    [5, "research-agent", "Compare RRF vs Cohere rerank latency", "running", 2, "2026-06-01T11:00:00Z", null],
  ];

  const calls: Array<[number, number, string, string, number, number, string]> = [
    [1, 1, "file_read", `{"path":"reports/q3.md"}`, 12, 1, "2026-05-28T09:01:03Z"],
    [2, 1, "calc", `{"expression":"3*4+2"}`, 1, 1, "2026-05-28T09:01:05Z"],
    [3, 1, "sqlite_query", `{"sql":"SELECT count(*) FROM documents"}`, 4, 1, "2026-05-28T09:01:08Z"],
    [4, 2, "sqlite_query", `{"sql":"SELECT * FROM agent_runs WHERE status='failed'"}`, 6, 1, "2026-05-28T10:14:10Z"],
    [5, 2, "sqlite_query", `{"sql":"DROP TABLE agent_runs"}`, 2, 0, "2026-05-28T10:14:15Z"],
    [6, 2, "file_read", `{"path":"../../etc/passwd"}`, 1, 0, "2026-05-28T10:14:20Z"],
    [7, 3, "calc", `{"expression":"10/0"}`, 1, 0, "2026-05-29T15:42:30Z"],
    [8, 3, "sqlite_query", `{"sql":"SELECT * FROM users"}`, 3, 0, "2026-05-29T15:42:45Z"],
    [9, 4, "sqlite_query", `{"sql":"SELECT sum(tokens) FROM documents"}`, 5, 1, "2026-05-30T08:20:03Z"],
    [10, 4, "calc", `{"expression":"(2^16)/1024"}`, 1, 1, "2026-05-30T08:20:05Z"],
    [11, 5, "sqlite_query", `{"sql":"SELECT tool_name, avg(latency_ms) FROM tool_calls GROUP BY tool_name"}`, 7, 1, "2026-06-01T11:00:30Z"],
  ];

  const docs: Array<[number, string, string, number, string]> = [
    [1, "Q3 Incident Report", "reports/q3.md", 1840, "2026-05-20T12:00:00Z"],
    [2, "ReAct Paper Notes", "notes/react.md", 920, "2026-05-21T12:00:00Z"],
    [3, "MCP Specification Summary", "notes/mcp.md", 1310, "2026-05-22T12:00:00Z"],
    [4, "Auth Service Runbook", "runbooks/auth.md", 2200, "2026-05-23T12:00:00Z"],
    [5, "Hybrid Search Benchmark", "benchmarks/rrf.md", 760, "2026-05-24T12:00:00Z"],
    [6, "Onboarding Guide", "docs/onboarding.md", 3050, "2026-05-25T12:00:00Z"],
  ];

  const insertRun = db.prepare(
    `INSERT INTO agent_runs (id, agent_name, task, status, iterations, started_at, finished_at)
     VALUES (?, ?, ?, ?, ?, ?, ?)`,
  );
  const insertCall = db.prepare(
    `INSERT INTO tool_calls (id, run_id, tool_name, arguments_json, latency_ms, success, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?)`,
  );
  const insertDoc = db.prepare(
    `INSERT INTO documents (id, title, source, tokens, indexed_at) VALUES (?, ?, ?, ?, ?)`,
  );

  const tx = db.transaction(() => {
    for (const r of runs) insertRun.run(...r);
    for (const c of calls) insertCall.run(...c);
    for (const d of docs) insertDoc.run(...d);
  });
  tx();

  const counts = {
    agent_runs: db.prepare("SELECT count(*) AS n FROM agent_runs").get() as { n: number },
    tool_calls: db.prepare("SELECT count(*) AS n FROM tool_calls").get() as { n: number },
    documents: db.prepare("SELECT count(*) AS n FROM documents").get() as { n: number },
  };

  db.close();
  log.info(
    `seeded ${DB_PATH} — agent_runs=${counts.agent_runs.n}, tool_calls=${counts.tool_calls.n}, documents=${counts.documents.n}`,
  );
}

seed();
