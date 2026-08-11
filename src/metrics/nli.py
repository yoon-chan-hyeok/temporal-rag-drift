"""Question-conditioned NLI helpers for semantic equivalence checks."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.utils.text import normalize_text, substring_match, token_f1

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NLIConfig:
    """Configuration for a local NLI entailment scorer."""

    model_name: str = "microsoft/deberta-large-mnli"
    batch_size: int = 16
    device: str | int | None = None
    entailment_threshold: float = 0.5
    contradiction_threshold: float = 0.5
    lexical_f1_threshold: float = 0.92
    local_files_only: bool = False

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "NLIConfig":
        """Parse NLI options from a config mapping."""
        device = data.get("device")
        return cls(
            model_name=str(data.get("nli_model", data.get("model_name", "microsoft/deberta-large-mnli"))),
            batch_size=int(data.get("nli_batch_size", data.get("batch_size", 16))),
            device=device if device is not None else None,
            entailment_threshold=float(data.get("entailment_threshold", 0.5)),
            contradiction_threshold=float(data.get("contradiction_threshold", 0.5)),
            lexical_f1_threshold=float(data.get("lexical_f1_threshold", 0.92)),
            local_files_only=bool(data.get("local_files_only", False)),
        )


def qa_text(question: str, answer: str) -> str:
    """Format a question-answer pair for NLI comparison."""
    return f"Question: {question}\nAnswer: {answer}"


def lexical_equivalent(left: str, right: str, threshold: float = 0.92) -> bool:
    """Return whether two answers are very likely equivalent by surface form."""
    if not normalize_text(left) or not normalize_text(right):
        return False
    return (
        normalize_text(left) == normalize_text(right)
        or substring_match(left, right)
        or token_f1(left, right) >= threshold
    )


class NLIEntailmentScorer:
    """Cached local NLI scorer.

    The special model names ``heuristic`` and ``fuzzy`` skip transformer loading
    and return lexical/fuzzy pseudo-probabilities. They are intended for smoke
    tests and offline pipeline validation, not for the main experiment.
    """

    def __init__(self, config: NLIConfig) -> None:
        self.config = config
        self._cache: dict[tuple[str, str], dict[str, float]] = {}
        self._model = None
        self._tokenizer = None
        self._label_map: dict[str, int] = {}
        self._heuristic = config.model_name.lower() in {"heuristic", "fuzzy", "none", "mock"}
        if not self._heuristic:
            self._load_model()

    def _load_model(self) -> None:
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        LOGGER.info("Loading NLI model: %s", self.config.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            local_files_only=self.config.local_files_only,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            use_safetensors=False,
            local_files_only=self.config.local_files_only,
        )
        device = _resolve_torch_device(self.config.device, torch)
        self._model.to(device)
        self._model.eval()
        raw_map = getattr(self._model.config, "label2id", {}) or {}
        self._label_map = {str(label).lower(): int(index) for label, index in raw_map.items()}

    def predict(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Return entailment, contradiction, and neutral probabilities."""
        key = (premise, hypothesis)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if self._heuristic:
            result = self._predict_heuristic(premise, hypothesis)
        else:
            result = self._predict_model(premise, hypothesis)
        self._cache[key] = result
        return result

    def _predict_heuristic(self, premise: str, hypothesis: str) -> dict[str, float]:
        score = token_f1(premise, hypothesis)
        if substring_match(premise, hypothesis):
            score = max(score, 0.95)
        contradiction = 0.05
        entailment = min(0.99, max(0.01, score))
        neutral = max(0.0, 1.0 - entailment - contradiction)
        return {"entailment": entailment, "contradiction": contradiction, "neutral": neutral}

    def _predict_model(self, premise: str, hypothesis: str) -> dict[str, float]:
        import torch

        if self._tokenizer is None or self._model is None:
            raise RuntimeError("NLI model has not been loaded")
        encoded = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        device = next(self._model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self._model(**encoded).logits[0].detach().cpu().numpy()
        probabilities = _softmax(logits)
        return {
            "entailment": float(probabilities[_label_index(self._label_map, "entailment", 2, len(probabilities))]),
            "contradiction": float(probabilities[_label_index(self._label_map, "contradiction", 0, len(probabilities))]),
            "neutral": float(probabilities[_label_index(self._label_map, "neutral", 1, len(probabilities))]),
        }

    def entails(self, premise: str, hypothesis: str) -> bool:
        """Return whether premise entails hypothesis under configured thresholds."""
        result = self.predict(premise, hypothesis)
        return (
            result["entailment"] >= self.config.entailment_threshold
            and result["contradiction"] < self.config.contradiction_threshold
        )

    def contradicts(self, premise: str, hypothesis: str) -> bool:
        """Return whether premise contradicts hypothesis."""
        result = self.predict(premise, hypothesis)
        return result["contradiction"] >= self.config.contradiction_threshold

    def equivalent(self, question: str, left_answer: str, right_answer: str, rule: str = "bidirectional") -> bool:
        """Return whether two answers are semantically equivalent for a question."""
        if lexical_equivalent(left_answer, right_answer, threshold=self.config.lexical_f1_threshold):
            return True
        left = qa_text(question, left_answer)
        right = qa_text(question, right_answer)
        left_entails_right = self.entails(left, right)
        right_entails_left = self.entails(right, left)
        if rule == "bidirectional":
            return left_entails_right and right_entails_left
        if rule in {"non_defeating", "non-defeating", "one_way_no_contradiction"}:
            contradiction = self.contradicts(left, right) or self.contradicts(right, left)
            return (left_entails_right or right_entails_left) and not contradiction
        raise ValueError(f"Unsupported NLI equivalence rule: {rule}")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _label_index(label_map: dict[str, int], label: str, fallback: int, n_labels: int) -> int:
    for key, index in label_map.items():
        if label in key:
            return int(index)
    return min(fallback, n_labels - 1)


def _resolve_torch_device(requested: str | int | None, torch_module: Any) -> str | int:
    """Return a usable torch device, falling back to CPU when CUDA is unavailable."""
    if requested is None:
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if isinstance(requested, int):
        if torch_module.cuda.is_available():
            return requested
        LOGGER.warning("Requested CUDA device %s but torch has no CUDA support; falling back to CPU.", requested)
        return "cpu"
    requested_text = str(requested)
    if requested_text.startswith("cuda") and not torch_module.cuda.is_available():
        LOGGER.warning("Requested device %s but torch has no CUDA support; falling back to CPU.", requested_text)
        return "cpu"
    return requested
