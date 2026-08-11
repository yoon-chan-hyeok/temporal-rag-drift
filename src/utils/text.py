"""Text normalization and fuzzy matching helpers."""

from __future__ import annotations

import re
import string
from collections import Counter

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(text: str | None) -> str:
    """Lowercase, strip punctuation, and normalize whitespace."""
    if text is None:
        return ""
    lowered = text.lower().translate(_PUNCT_TABLE)
    return re.sub(r"\s+", " ", lowered).strip()


def tokenize(text: str | None) -> list[str]:
    """Tokenize normalized text on whitespace."""
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def token_f1(prediction: str | None, reference: str | None) -> float:
    """Compute token-level F1 with a QA-style bag-of-words overlap."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def lcs_length(a_tokens: list[str], b_tokens: list[str]) -> int:
    """Return longest-common-subsequence length."""
    if not a_tokens or not b_tokens:
        return 0
    previous = [0] * (len(b_tokens) + 1)
    for a_token in a_tokens:
        current = [0]
        for index, b_token in enumerate(b_tokens, start=1):
            if a_token == b_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str | None, reference: str | None) -> float:
    """Compute a compact ROUGE-L F1 approximation."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def substring_match(prediction: str | None, reference: str | None) -> bool:
    """Return whether either normalized string contains the other."""
    pred = normalize_text(prediction)
    ref = normalize_text(reference)
    if not pred or not ref:
        return False
    return ref in pred or pred in ref


def compact_whitespace(text: str) -> str:
    """Collapse whitespace without lowercasing or punctuation stripping."""
    return re.sub(r"\s+", " ", text).strip()


def stable_text_hash(text: str) -> str:
    """Return a short stable hash for text payloads."""
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
