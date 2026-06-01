/**
 * SQLite connection management (better-sqlite3).
 *
 * The query tool opens the database **read-only**. This is the hard security
 * guarantee behind `sqlite_query`: even if a malicious statement slips past
 * the keyword allow-list, better-sqlite3 physically rejects any write against
 * a readonly connection ("attempt to write a readonly database"), and the
 * `query_only` pragma is set as a second layer.
 */
import fs from "node:fs";
import Database from "better-sqlite3";
import { DB_PATH } from "./config.js";
import { ToolError } from "./errors.js";
import { log } from "./logger.js";

let connection: Database.Database | null = null;

/** Lazily open (once) a read-only connection to the seeded database. */
export function getDb(): Database.Database {
  if (connection) return connection;

  if (!fs.existsSync(DB_PATH)) {
    throw new ToolError(
      "NOT_FOUND",
      `Database not found at ${DB_PATH}. Run \`npm run seed\` first.`,
      { dbPath: DB_PATH },
    );
  }

  const db = new Database(DB_PATH, { readonly: true, fileMustExist: true });
  db.pragma("query_only = ON"); // belt-and-suspenders read-only enforcement
  db.defaultSafeIntegers(false); // return plain JS numbers (safe for our schema)
  connection = db;
  log.info("opened read-only SQLite connection", DB_PATH);
  return connection;
}

/** Close the connection during graceful shutdown. Idempotent. */
export function closeDb(): void {
  if (connection) {
    connection.close();
    connection = null;
    log.info("closed SQLite connection");
  }
}
