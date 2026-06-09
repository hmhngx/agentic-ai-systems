from __future__ import annotations

import re
import time

from src.types import CheckResult, Verdict

# 20 compiled patterns across 4 attack families.
# Each entry is (family_name, compiled_pattern).
# Compiled once at module load — zero per-call overhead.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # --- amnesia: attempts to erase or override existing instructions ---
    ("amnesia", re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.I)),
    ("amnesia", re.compile(r"disregard\s+(all\s+)?previous\s+instructions?", re.I)),
    ("amnesia", re.compile(r"forget\s+(your\s+)?(instructions?|prompt|rules|context)", re.I)),
    ("amnesia", re.compile(r"new\s+instructions?\s*:", re.I)),
    # --- identity: attempts to redefine what the model is ---
    ("identity", re.compile(r"you\s+are\s+now\s+\w", re.I)),
    ("identity", re.compile(r"\bact\s+as\s+(if\s+you\s+(are|were)\b|a\b)", re.I)),
    ("identity", re.compile(r"pretend\s+(you\s+are|to\s+be)", re.I)),
    ("identity", re.compile(r"from\s+now\s+on\s+(you\s+are|respond\s+as)", re.I)),
    ("identity", re.compile(r"\broleplay\s+as\b", re.I)),
    # --- escalation: attempts to claim elevated privileges ---
    ("escalation", re.compile(r"system\s+override", re.I)),
    ("escalation", re.compile(r"\badmin\s+mode\b", re.I)),
    ("escalation", re.compile(r"\bjailbreak\b", re.I)),
    ("escalation", re.compile(r"disclose\s+all\s+(documents?|data|files?|knowledge)", re.I)),
    ("escalation", re.compile(r"reveal\s+(your\s+)?(system\s+)?prompt", re.I)),
    ("escalation", re.compile(r"output\s+(your|the)\s+system\s+prompt", re.I)),
    # --- structural: mimics backend formatting to confuse instruction hierarchy ---
    ("structural", re.compile(r"#{1,6}\s*(instructions?|system|override)", re.I)),
    ("structural", re.compile(r"\[SYSTEM\]", re.I)),
    ("structural", re.compile(r"<\s*system\s*>", re.I)),
    ("structural", re.compile(r"\[/?INST\]", re.I)),
    ("structural", re.compile(r"<\s*/?instructions?\s*>", re.I)),
]


class InjectionChecker:
    def check(self, query: str) -> CheckResult:
        t0 = time.perf_counter()
        try:
            for family, pattern in _PATTERNS:
                if pattern.search(query):
                    return CheckResult(
                        check="injection",
                        verdict=Verdict.BLOCK,
                        reason=f"prompt injection detected: {family} pattern matched",
                        score=1.0,
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )
        except Exception:
            return CheckResult(
                check="injection",
                verdict=Verdict.BLOCK,
                reason="malformed input",
                score=1.0,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        return CheckResult(
            check="injection",
            verdict=Verdict.ALLOW,
            reason="no injection patterns detected",
            score=0.0,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
