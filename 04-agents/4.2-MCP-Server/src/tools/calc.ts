/**
 * Tool: calc — evaluate an arithmetic expression safely.
 *
 * Security: this deliberately does NOT use `eval` or `new Function`. Doing so
 * would let an attacker run arbitrary JavaScript inside the server process.
 * Instead we tokenize the input and evaluate it with a small recursive-descent
 * parser. The only things that can ever execute are `+ - * / % ^` on numbers.
 *
 * Grammar (precedence low -> high; '^' is right-associative and binds
 * tighter than unary minus, matching Python/maths):
 *   expr    := term   (('+' | '-') term)*
 *   term    := factor (('*' | '/' | '%') factor)*
 *   factor  := ('+' | '-') factor | power
 *   power   := primary ('^' factor)?
 *   primary := number | '(' expr ')'
 *
 * This grammar gives the conventional readings: '2 * -3' = -6 and '-2^2' = -4.
 */
import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { ToolError, safeHandler } from "../errors.js";

const inputSchema = {
  expression: z
    .string()
    .min(1)
    .max(500)
    .describe(
      "Arithmetic expression using numbers, + - * / % ^, parentheses and unary minus. " +
        "Example: '(2 ^ 16) / 1024 - 3'.",
    ),
} as const;

const outputSchema = {
  expression: z.string().describe("The expression that was evaluated."),
  result: z.number().describe("The numeric result."),
} as const;

type Token =
  | { kind: "num"; value: number }
  | { kind: "op"; value: "+" | "-" | "*" | "/" | "%" | "^" }
  | { kind: "lparen" }
  | { kind: "rparen" };

const OP_CHARS = new Set(["+", "-", "*", "/", "%", "^"]);
type OpChar = "+" | "-" | "*" | "/" | "%" | "^";

function tokenize(expr: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;

  const isDigit = (c: string): boolean => c >= "0" && c <= "9";

  while (i < expr.length) {
    const ch = expr[i]!;

    if (ch === " " || ch === "\t" || ch === "\n" || ch === "\r") {
      i++;
      continue;
    }

    if (isDigit(ch) || ch === ".") {
      let j = i;
      while (j < expr.length && (isDigit(expr[j]!) || expr[j] === ".")) j++;
      // Optional scientific notation: 1e3, 2.5E-4
      if (j < expr.length && (expr[j] === "e" || expr[j] === "E")) {
        j++;
        if (j < expr.length && (expr[j] === "+" || expr[j] === "-")) j++;
        while (j < expr.length && isDigit(expr[j]!)) j++;
      }
      const raw = expr.slice(i, j);
      const value = Number(raw);
      if (!Number.isFinite(value)) {
        throw new ToolError("INVALID_INPUT", `Invalid number: '${raw}'.`);
      }
      tokens.push({ kind: "num", value });
      i = j;
      continue;
    }

    if (ch === "(") {
      tokens.push({ kind: "lparen" });
      i++;
      continue;
    }
    if (ch === ")") {
      tokens.push({ kind: "rparen" });
      i++;
      continue;
    }
    if (OP_CHARS.has(ch)) {
      tokens.push({ kind: "op", value: ch as OpChar });
      i++;
      continue;
    }

    throw new ToolError("INVALID_INPUT", `Unexpected character '${ch}' at position ${i}.`);
  }

  return tokens;
}

function applyBinary(op: OpChar, a: number, b: number): number {
  switch (op) {
    case "+":
      return a + b;
    case "-":
      return a - b;
    case "*":
      return a * b;
    case "/":
      if (b === 0) throw new ToolError("INVALID_INPUT", "Division by zero.");
      return a / b;
    case "%":
      if (b === 0) throw new ToolError("INVALID_INPUT", "Modulo by zero.");
      return a % b;
    case "^":
      return a ** b;
  }
}

/** Recursive-descent parser/evaluator over the token stream. */
class Parser {
  private pos = 0;
  constructor(private readonly tokens: Token[]) {}

  evaluate(): number {
    const value = this.parseExpr();
    if (this.pos < this.tokens.length) {
      throw new ToolError("INVALID_INPUT", "Unexpected trailing input in expression.");
    }
    return value;
  }

  private peek(): Token | undefined {
    return this.tokens[this.pos];
  }

  private isOp(...ops: OpChar[]): boolean {
    const t = this.peek();
    return t !== undefined && t.kind === "op" && ops.includes(t.value);
  }

  // expr := term (('+' | '-') term)*
  private parseExpr(): number {
    let value = this.parseTerm();
    while (this.isOp("+", "-")) {
      const op = (this.tokens[this.pos] as { value: OpChar }).value;
      this.pos++;
      value = applyBinary(op, value, this.parseTerm());
    }
    return value;
  }

  // term := factor (('*' | '/' | '%') factor)*
  private parseTerm(): number {
    let value = this.parseFactor();
    while (this.isOp("*", "/", "%")) {
      const op = (this.tokens[this.pos] as { value: OpChar }).value;
      this.pos++;
      value = applyBinary(op, value, this.parseFactor());
    }
    return value;
  }

  // factor := ('+' | '-') factor | power   -- unary binds looser than '^'
  private parseFactor(): number {
    if (this.isOp("+", "-")) {
      const op = (this.tokens[this.pos] as { value: OpChar }).value;
      this.pos++;
      const operand = this.parseFactor();
      return op === "-" ? -operand : operand;
    }
    return this.parsePower();
  }

  // power := primary ('^' factor)?   -- right-associative
  private parsePower(): number {
    const base = this.parsePrimary();
    if (this.isOp("^")) {
      this.pos++;
      return applyBinary("^", base, this.parseFactor());
    }
    return base;
  }

  // primary := number | '(' expr ')'
  private parsePrimary(): number {
    const tok = this.peek();
    if (tok === undefined) {
      throw new ToolError("INVALID_INPUT", "Unexpected end of expression.");
    }
    if (tok.kind === "num") {
      this.pos++;
      return tok.value;
    }
    if (tok.kind === "lparen") {
      this.pos++;
      const value = this.parseExpr();
      const close = this.peek();
      if (!close || close.kind !== "rparen") {
        throw new ToolError("INVALID_INPUT", "Mismatched parentheses (missing ')').");
      }
      this.pos++;
      return value;
    }
    throw new ToolError("INVALID_INPUT", "Expected a number or '('.");
  }
}

export function evaluate(expression: string): number {
  const tokens = tokenize(expression);
  if (tokens.length === 0) {
    throw new ToolError("INVALID_INPUT", "Empty expression.");
  }
  const result = new Parser(tokens).evaluate();
  if (!Number.isFinite(result)) {
    throw new ToolError("INVALID_INPUT", "Result is not a finite number.");
  }
  return result;
}

export function registerCalcTool(server: McpServer): void {
  server.registerTool(
    "calc",
    {
      title: "Calculator",
      description:
        "Evaluate an arithmetic expression with + - * / % ^, parentheses and unary minus. " +
        "Implemented with a safe recursive-descent evaluator (no eval). Returns a numeric result.",
      inputSchema,
      outputSchema,
      annotations: {
        title: "Calculator",
        readOnlyHint: true,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    safeHandler(async ({ expression }) => {
      const result = evaluate(expression);
      const structuredContent = { expression, result };
      return {
        content: [{ type: "text", text: JSON.stringify(structuredContent, null, 2) }],
        structuredContent,
      };
    }),
  );
}
