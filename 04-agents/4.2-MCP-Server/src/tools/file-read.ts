/**
 * Tool: file_read — read a file, but only from inside the server's sandbox
 * root (`FILE_ROOT`).
 *
 * This is the canonical MCP least-privilege pattern: the LLM has no ambient
 * filesystem authority; it can only reach what we explicitly mount. Every
 * requested path is resolved against `FILE_ROOT` and rejected if it escapes —
 * including via `..` traversal or symlinks (checked with realpath).
 *
 * Note on MCP semantics: reading a file is genuinely a "Resource" in MCP's
 * model (idempotent, state-providing). It is implemented here as a Tool because
 * the task brief calls for three tools; the schema is annotated read-only and
 * idempotent to reflect that.
 */
import fs from "node:fs";
import path from "node:path";
import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { FILE_ROOT } from "../config.js";
import { ToolError, safeHandler } from "../errors.js";

const MAX_BYTES_CAP = 1_048_576; // 1 MiB hard ceiling

const inputSchema = {
  path: z
    .string()
    .min(1)
    .max(1024)
    .describe(
      "Path to read, relative to (or absolute within) the server sandbox root. " +
        "Paths that escape the sandbox via '..' or symlinks are rejected.",
    ),
  maxBytes: z
    .number()
    .int()
    .min(1)
    .max(MAX_BYTES_CAP)
    .default(65536)
    .describe("Maximum number of bytes to read (default 64 KiB, max 1 MiB)."),
  encoding: z
    .enum(["utf8", "base64"])
    .default("utf8")
    .describe("Return text as utf8, or binary-safe base64."),
} as const;

const outputSchema = {
  path: z.string().describe("The path as requested."),
  absolutePath: z.string().describe("The resolved absolute path that was read."),
  bytes: z.number().int().describe("Number of bytes actually returned."),
  truncated: z.boolean().describe("True if the file was larger than `maxBytes`."),
  encoding: z.string().describe("Encoding of the returned content."),
  content: z.string().describe("File content in the requested encoding."),
} as const;

/** Resolve `requested` under FILE_ROOT, rejecting any escape. Returns abs path. */
function resolveInsideSandbox(requested: string): string {
  const resolved = path.resolve(FILE_ROOT, requested);
  const rel = path.relative(FILE_ROOT, resolved);

  // `rel` starting with ".." or being absolute => outside the root.
  if (rel === ".." || rel.startsWith(".." + path.sep) || path.isAbsolute(rel)) {
    throw new ToolError("FORBIDDEN", `Path escapes the sandbox root (${FILE_ROOT}).`, {
      requested,
      sandboxRoot: FILE_ROOT,
    });
  }
  return resolved;
}

/** Second check after we know the file exists: defeat symlink escapes. */
function assertRealPathInside(absPath: string): void {
  const realFile = fs.realpathSync(absPath);
  const realRoot = fs.realpathSync(FILE_ROOT);
  const rel = path.relative(realRoot, realFile);
  if (rel === ".." || rel.startsWith(".." + path.sep) || path.isAbsolute(rel)) {
    throw new ToolError("FORBIDDEN", "Resolved real path escapes the sandbox root.", {
      sandboxRoot: realRoot,
    });
  }
}

export function registerFileReadTool(server: McpServer): void {
  server.registerTool(
    "file_read",
    {
      title: "File Read (sandboxed)",
      description:
        "Read a UTF-8 or binary file from within the server's sandbox root. Paths are resolved " +
        "relative to that root; attempts to escape it (via '..' or symlinks) are rejected. " +
        "Large files are truncated to `maxBytes`.",
      inputSchema,
      outputSchema,
      annotations: {
        title: "File Read (sandboxed)",
        readOnlyHint: true,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    safeHandler(async ({ path: requested, maxBytes, encoding }) => {
      const absPath = resolveInsideSandbox(requested);

      let stat: fs.Stats;
      try {
        stat = fs.statSync(absPath);
      } catch {
        throw new ToolError("NOT_FOUND", `File not found: ${requested}`, { requested });
      }
      if (stat.isDirectory()) {
        throw new ToolError("INVALID_INPUT", `Path is a directory, not a file: ${requested}`);
      }
      assertRealPathInside(absPath);

      const toRead = Math.min(stat.size, maxBytes);
      const buf = Buffer.alloc(toRead);
      const fd = fs.openSync(absPath, "r");
      try {
        fs.readSync(fd, buf, 0, toRead, 0);
      } finally {
        fs.closeSync(fd);
      }

      const structuredContent = {
        path: requested,
        absolutePath: absPath,
        bytes: toRead,
        truncated: stat.size > maxBytes,
        encoding,
        content: encoding === "base64" ? buf.toString("base64") : buf.toString("utf8"),
      };

      return {
        content: [
          {
            type: "text",
            text:
              encoding === "utf8"
                ? structuredContent.content
                : JSON.stringify(
                    { ...structuredContent, content: "<base64 omitted from text view>" },
                    null,
                    2,
                  ),
          },
        ],
        structuredContent,
      };
    }),
  );
}
