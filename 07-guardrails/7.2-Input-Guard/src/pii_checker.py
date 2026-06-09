from __future__ import annotations

import re
import time

from src.types import CheckResult, GuardConfig, Verdict

_EMAIL_RE = re.compile(r"[\w.+\-]+@[\w\-]+\.[\w.]{2,}", re.I)
_PHONE_RE = re.compile(r"\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# Matches 16-digit Visa/MC/Discover and 15-digit Amex in grouped or compact form
_CC_RE = re.compile(
    r"\b\d{4}[ \-]?\d{4}[ \-]?\d{4}[ \-]?\d{4}\b"  # 16-digit (4-4-4-4)
    r"|\b\d{4}[ \-]?\d{6}[ \-]?\d{5}\b",             # 15-digit Amex (4-6-5)
)
_NAME_RE = re.compile(r"\b[A-Z][a-z]{1,19}\s+[A-Z][a-z]{1,24}\b")

# ~80 common English/international surnames to reduce false positives
_COMMON_SURNAMES = frozenset({
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Doe", "Jensen", "Chen", "Kim", "Patel", "Singh",
    "Zhang", "Wang", "Li", "Sharma", "Kumar", "Cohen", "Levy", "Murphy",
    "Kelly", "Ryan", "McCarthy", "Fitzgerald", "Brennan", "Lynch", "Walsh",
    "Butler", "O'Brien", "Reed", "Cook", "Morgan", "Bell", "Bailey", "Cooper",
    "Richardson", "Cox", "Howard", "Ward", "Peterson", "Gray",
})


def _luhn_check(digits: str) -> bool:
    """Return True iff the digit string satisfies the Luhn algorithm."""
    clean = [int(c) for c in digits if c.isdigit()]
    if len(clean) < 13:
        return False
    clean.reverse()
    total = 0
    for i, d in enumerate(clean):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _find_pii(query: str) -> tuple[str, float]:
    """Return (pii_type_label, confidence) for the first detected PII, or ("", 0.0)."""
    if _SSN_RE.search(query):
        return "SSN", 0.95
    if _EMAIL_RE.search(query):
        return "email", 0.95
    if _PHONE_RE.search(query):
        return "phone number", 0.90
    for m in _CC_RE.finditer(query):
        digits = re.sub(r"[ \-]", "", m.group())
        if _luhn_check(digits):
            return "credit card number", 0.99
    for m in _NAME_RE.finditer(query):
        parts = m.group().split()
        if len(parts) == 2 and parts[1] in _COMMON_SURNAMES:
            return "full name", 0.60
    return "", 0.0


class PIIChecker:
    def __init__(self, config: GuardConfig) -> None:
        self._hard = config.pii_hard_threshold
        self._soft = config.pii_soft_threshold

    def check(self, query: str) -> CheckResult:
        t0 = time.perf_counter()
        try:
            pii_type, confidence = _find_pii(query)
        except Exception:
            return CheckResult(
                check="pii",
                verdict=Verdict.BLOCK,
                reason="malformed input",
                score=1.0,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        latency = (time.perf_counter() - t0) * 1000

        if confidence >= self._hard:
            return CheckResult(
                check="pii",
                verdict=Verdict.BLOCK,
                reason=f"PII detected: {pii_type} (confidence {confidence:.0%})",
                score=confidence,
                latency_ms=latency,
            )
        if confidence >= self._soft:
            return CheckResult(
                check="pii",
                verdict=Verdict.BLOCK,
                reason=f"possible PII detected: {pii_type} (confidence {confidence:.0%})",
                score=confidence,
                latency_ms=latency,
            )
        return CheckResult(
            check="pii",
            verdict=Verdict.ALLOW,
            reason="no PII detected",
            score=0.0,
            latency_ms=latency,
        )
