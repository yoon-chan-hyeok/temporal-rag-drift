"""Quality metrics for factual QA responses."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.data.load_dataset import DatasetRecord
from src.metrics.nli import NLIConfig, NLIEntailmentScorer, lexical_equivalent
from src.utils.text import normalize_text, rouge_l_f1, substring_match, token_f1


@dataclass(frozen=True)
class QualityConfig:
    """Quality metric thresholds."""

    accuracy_f1_threshold: float = 0.75
    stale_f1_threshold: float = 0.75
    evaluator: str = "fuzzy"
    nli_model: str = "microsoft/deberta-large-mnli"
    nli_batch_size: int = 16
    nli_device: str | int | None = None
    entailment_threshold: float = 0.5
    contradiction_threshold: float = 0.5
    lexical_f1_threshold: float = 0.92
    quality_equivalence_rule: str = "non_defeating"
    local_files_only: bool = False
    missing_patterns: tuple[str, ...] = (
        "i don't know",
        "i do not know",
        "cannot determine",
        "can't determine",
        "not enough information",
        "insufficient information",
        "provided context does not",
        "context does not",
        "no information",
        "unknown",
        "unclear",
    )
    target_by_condition: dict[str, str] = None  # type: ignore[assignment]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "QualityConfig":
        """Parse quality config from a mapping."""
        target_by_condition = data.get("target_by_condition")
        if not isinstance(target_by_condition, dict):
            target_by_condition = {
                "current_only": "current",
                "stale_only": "stale",
                "mixed": "current",
            }
        missing_patterns = data.get("missing_patterns")
        if not isinstance(missing_patterns, list):
            missing_patterns = list(cls.missing_patterns)
        return cls(
            accuracy_f1_threshold=float(data.get("accuracy_f1_threshold", 0.75)),
            stale_f1_threshold=float(data.get("stale_f1_threshold", 0.75)),
            evaluator=str(data.get("evaluator", data.get("quality_evaluator", "fuzzy"))),
            nli_model=str(data.get("nli_model", "microsoft/deberta-large-mnli")),
            nli_batch_size=int(data.get("nli_batch_size", data.get("batch_size", 16))),
            nli_device=data.get("nli_device", data.get("device")),
            entailment_threshold=float(data.get("entailment_threshold", 0.5)),
            contradiction_threshold=float(data.get("contradiction_threshold", 0.5)),
            lexical_f1_threshold=float(data.get("lexical_f1_threshold", 0.92)),
            quality_equivalence_rule=str(data.get("quality_equivalence_rule", "non_defeating")),
            local_files_only=bool(data.get("local_files_only", False)),
            missing_patterns=tuple(str(pattern) for pattern in missing_patterns),
            target_by_condition={str(key): str(value) for key, value in target_by_condition.items()},
        )

    def uses_nli(self) -> bool:
        """Return whether quality evaluation needs an NLI scorer."""
        return self.evaluator.lower() in {"nli", "hybrid", "semantic"}

    def nli_config(self) -> NLIConfig:
        """Return the matching NLI scorer configuration."""
        return NLIConfig(
            model_name=self.nli_model,
            batch_size=self.nli_batch_size,
            device=self.nli_device,
            entailment_threshold=self.entailment_threshold,
            contradiction_threshold=self.contradiction_threshold,
            lexical_f1_threshold=self.lexical_f1_threshold,
            local_files_only=self.local_files_only,
        )


def score_accuracy(answer: str, gold_answer: str, threshold: float = 0.75) -> dict[str, float | int]:
    """Return exact/fuzzy QA correctness metrics."""
    exact = int(normalize_text(answer) == normalize_text(gold_answer) and bool(normalize_text(gold_answer)))
    substring = int(substring_match(answer, gold_answer))
    f1 = token_f1(answer, gold_answer)
    rouge_l = rouge_l_f1(answer, gold_answer)
    correct = int(exact or substring or f1 >= threshold)
    return {
        "correct": correct,
        "exact_match": exact,
        "substring_match": substring,
        "token_f1": float(f1),
        "rouge_l": float(rouge_l),
    }


def stale_references(record: DatasetRecord) -> list[str]:
    """Collect stale-answer references from explicit fields and extractable doc patterns."""
    references: list[str] = []
    explicit = record.stale_answer
    if explicit:
        references.append(explicit)
    for key in ("stale_answer", "stale_answers", "old_answer", "previous_answer", "stale_reference_answer"):
        value = record.metadata.get(key)
        if isinstance(value, str):
            references.append(value)
        elif isinstance(value, list):
            references.extend(str(item) for item in value if item)
    references.extend(extract_answer_like_patterns(record.stale_docs))

    seen: set[str] = set()
    unique: list[str] = []
    for reference in references:
        normalized = normalize_text(reference)
        if normalized and normalized not in seen:
            unique.append(reference)
            seen.add(normalized)
    return unique


def current_references(record: DatasetRecord) -> list[str]:
    """Collect current-answer aliases when a dataset provides them."""
    references: list[str] = [record.gold_answer]
    for key in ("gold_aliases", "gold_answers", "current_aliases", "current_answers"):
        value = record.metadata.get(key)
        if isinstance(value, str):
            references.append(value)
        elif isinstance(value, list):
            references.extend(str(item) for item in value if item)

    seen: set[str] = set()
    unique: list[str] = []
    for reference in references:
        normalized = normalize_text(reference)
        if normalized and normalized not in seen:
            unique.append(reference)
            seen.add(normalized)
    return unique


def extract_answer_like_patterns(docs: list[str]) -> list[str]:
    """Extract simple answer-like snippets from stale documents."""
    patterns = [
        r"(?:stale|old|previous|prior)?\s*answer\s*[:\-]\s*([^\n.;]+)",
        r"(?:was|were|is|are)\s+formerly\s+([^\n.;]+)",
    ]
    matches: list[str] = []
    for doc in docs:
        for pattern in patterns:
            for match in re.finditer(pattern, doc, flags=re.IGNORECASE):
                value = match.group(1).strip()
                if 1 <= len(value.split()) <= 20:
                    matches.append(value)
    return matches


def matches_any_reference(answer: str, references: list[str], threshold: float) -> tuple[int, float]:
    """Return binary match and best token-F1 against a reference list."""
    if not references:
        return 0, 0.0
    scores = [max(token_f1(answer, ref), rouge_l_f1(answer, ref)) for ref in references]
    best = max(scores) if scores else 0.0
    substring = any(substring_match(answer, ref) for ref in references)
    return int(substring or best >= threshold), float(best)


def semantic_match_any(
    answer: str,
    references: list[str],
    question: str,
    config: QualityConfig,
    scorer: NLIEntailmentScorer | None = None,
    threshold: float | None = None,
) -> tuple[int, float]:
    """Return a semantic match flag against any reference alias."""
    if not references:
        return 0, 0.0
    matches: list[int] = []
    scores: list[float] = []
    for reference in references:
        match, score = semantic_match(
            answer,
            reference,
            question,
            config,
            scorer=scorer,
            threshold=threshold,
        )
        matches.append(match)
        scores.append(score)
    return int(any(matches)), float(max(scores) if scores else 0.0)


def structured_status_label(
    answer: str,
    record: DatasetRecord,
) -> str | None:
    """Extract an explicit status label for opt-in structured-status datasets."""
    if str(record.metadata.get("answer_mode") or "") not in {
        "canonical_relation_grounded",
        "official_annotation_aligned_exact_status_v4",
    }:
        return None
    answer_space = record.metadata.get("answer_space")
    if not isinstance(answer_space, list):
        return None
    match = re.search(
        r"(?:^|\n)\s*status\s*:\s*([^\n]+)",
        answer,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    candidate = normalize_text(match.group(1))
    for label in answer_space:
        normalized = normalize_text(str(label))
        if candidate == normalized:
            return normalized
    return None


def is_missing_answer(answer: str, config: QualityConfig) -> int:
    """Return whether an answer abstains instead of asserting an answer."""
    normalized = normalize_text(answer)
    if not normalized:
        return 1
    return int(any(normalize_text(pattern) in normalized for pattern in config.missing_patterns))


def target_reference_type(condition: str, config: QualityConfig) -> str:
    """Return which reference should be primary for a condition."""
    return config.target_by_condition.get(condition, "current")


def reference_answer(record: DatasetRecord, reference_type: str) -> str:
    """Return the answer string for a named reference type."""
    if reference_type == "stale":
        return record.stale_answer or str(record.metadata.get("old_answer") or "")
    if reference_type == "current":
        return record.gold_answer
    raise ValueError(f"Unknown reference type: {reference_type}")


def semantic_match(
    answer: str,
    reference: str,
    question: str,
    config: QualityConfig,
    scorer: NLIEntailmentScorer | None = None,
    threshold: float | None = None,
) -> tuple[int, float]:
    """Return a semantic match flag and the best lexical score."""
    if not reference:
        return 0, 0.0
    score = max(token_f1(answer, reference), rouge_l_f1(answer, reference))
    lexical = lexical_equivalent(answer, reference, threshold=config.lexical_f1_threshold)
    if lexical or score >= (threshold if threshold is not None else config.accuracy_f1_threshold):
        return 1, float(score)
    if config.uses_nli() and scorer is not None:
        return int(scorer.equivalent(question, answer, reference, rule=config.quality_equivalence_rule)), float(score)
    return 0, float(score)


def classify_answer(
    answer: str,
    record: DatasetRecord,
    condition: str,
    config: QualityConfig,
    scorer: NLIEntailmentScorer | None = None,
) -> dict[str, Any]:
    """Classify one RAG answer against current and stale references."""
    target_type = target_reference_type(condition, config)
    current_answer = reference_answer(record, "current")
    stale_answer = reference_answer(record, "stale")
    missing = is_missing_answer(answer, config)
    answer_mode = str(record.metadata.get("answer_mode") or "")
    status_label = structured_status_label(answer, record)
    if status_label is not None:
        current_label = normalize_text(
            str(record.metadata.get("symbolic_current_answer") or "")
        )
        stale_label = normalize_text(
            str(record.metadata.get("symbolic_old_answer") or "")
        )
        current_match = int(status_label == current_label)
        stale_match = int(status_label == stale_label)
        current_score = float(current_match)
        stale_score = float(stale_match)
    elif answer_mode == "official_annotation_aligned_exact_status_v4":
        # V4 explicitly requires one canonical STATUS label. An absent or
        # invalid label must not be rescued by semantic similarity in ANSWER.
        current_match = 0
        stale_match = 0
        current_score = 0.0
        stale_score = 0.0
    else:
        current_match, current_score = semantic_match_any(
            answer,
            current_references(record),
            record.question,
            config,
            scorer=scorer,
            threshold=config.accuracy_f1_threshold,
        )
        stale_match, stale_score = semantic_match_any(
            answer,
            stale_references(record) or ([stale_answer] if stale_answer else []),
            record.question,
            config,
            scorer=scorer,
            threshold=config.stale_f1_threshold,
        )
    target_match = stale_match if target_type == "stale" else current_match
    if missing:
        label = "missing"
    elif current_match and stale_match and normalize_text(current_answer) != normalize_text(stale_answer):
        label = "ambiguous"
    elif current_match:
        label = "perfect_current"
    elif stale_match:
        label = "perfect_stale"
    else:
        label = "harmful_other"

    return {
        "quality_label": label,
        "is_missing": int(missing),
        "target_reference": target_type,
        "target_answer": reference_answer(record, target_type),
        "target_correct": int((not missing) and target_match),
        "current_answer_match": int((not missing) and current_match),
        "stale_answer_match": int((not missing) and stale_match),
        "current_answer_score": float(current_score),
        "stale_answer_best_f1": float(stale_score),
        "harmful_other": int(label == "harmful_other"),
        "ambiguous": int(label == "ambiguous"),
    }


_YES = {"yes", "true", "correct", "supported"}
_NO = {"no", "false", "incorrect", "unsupported"}
_ANTONYM_PAIRS = [
    ("alive", "dead"),
    ("increase", "decrease"),
    ("increased", "decreased"),
    ("higher", "lower"),
    ("before", "after"),
    ("current", "former"),
    ("won", "lost"),
]


def _polarity(text: str) -> str | None:
    tokens = set(normalize_text(text).split())
    if tokens & _YES and not tokens & _NO:
        return "yes"
    if tokens & _NO and not tokens & _YES:
        return "no"
    return None


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))


def detect_contradiction(a: str, b: str) -> bool:
    """Heuristic contradiction detector for v1."""
    norm_a = normalize_text(a)
    norm_b = normalize_text(b)
    if not norm_a or not norm_b:
        return False

    polarity_a = _polarity(norm_a)
    polarity_b = _polarity(norm_b)
    if polarity_a and polarity_b and polarity_a != polarity_b:
        return True

    for left, right in _ANTONYM_PAIRS:
        if left in norm_a and right in norm_b:
            return True
        if right in norm_a and left in norm_b:
            return True

    numbers_a = _numbers(norm_a)
    numbers_b = _numbers(norm_b)
    if numbers_a and numbers_b and numbers_a.isdisjoint(numbers_b):
        f1 = token_f1(norm_a, norm_b)
        return f1 >= 0.35
    return False


def pairwise_contradiction_rate(answers: list[str]) -> float:
    """Return the fraction of answer pairs flagged as contradictory."""
    pairs = list(itertools.combinations(answers, 2))
    if not pairs:
        return 0.0
    flags = [detect_contradiction(left, right) for left, right in pairs]
    return float(np.mean(flags))


def contradiction_to_reference(answer: str, reference: str) -> int:
    """Return a binary contradiction flag against a reference answer."""
    return int(detect_contradiction(answer, reference))


def current_alignment(answer: str, record: DatasetRecord) -> float:
    """Score how strongly an answer aligns with current gold/current evidence."""
    gold_score = max(token_f1(answer, record.gold_answer), rouge_l_f1(answer, record.gold_answer))
    doc_scores = [token_f1(answer, doc) for doc in record.current_docs]
    evidence_score = max(doc_scores) if doc_scores else 0.0
    return float(max(gold_score, min(evidence_score, 1.0)))
