"""Sentence embedding backends with optional deterministic fallback."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as sk_normalize

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding model configuration."""

    primary_model: str
    secondary_model: str | None = None
    use_secondary_for_sensitivity: bool = False
    batch_size: int = 32
    normalize: bool = True
    pca_dim: int = 10
    allow_hashing_fallback: bool = False
    device: str | None = None
    local_files_only: bool = False

    @classmethod
    def from_mapping(cls, data: dict[str, object]) -> "EmbeddingConfig":
        """Parse embedding config from a mapping."""
        return cls(
            primary_model=str(data.get("primary_model", "BAAI/bge-large-en-v1.5")),
            secondary_model=str(data["secondary_model"]) if data.get("secondary_model") else None,
            use_secondary_for_sensitivity=bool(data.get("use_secondary_for_sensitivity", False)),
            batch_size=int(data.get("batch_size", 32)),
            normalize=bool(data.get("normalize", True)),
            pca_dim=int(data.get("pca_dim", 10)),
            allow_hashing_fallback=bool(data.get("allow_hashing_fallback", False)),
            device=str(data["device"]) if data.get("device") else None,
            local_files_only=bool(data.get("local_files_only", False)),
        )


class TextEmbedder:
    """Wrapper around sentence-transformers with a hashing fallback option."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        normalize: bool = True,
        allow_hashing_fallback: bool = False,
        device: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device
        self.local_files_only = local_files_only
        self._model = None
        self._hashing = None
        if allow_hashing_fallback and model_name in {"smoke-hashing", "hashing"}:
            LOGGER.info("Using HashingVectorizer embeddings for %s", model_name)
            self._hashing = HashingVectorizer(n_features=768, alternate_sign=False, norm=None)
            return
        resolved_device = _resolve_sentence_transformer_device(device)
        try:
            from sentence_transformers import SentenceTransformer

            LOGGER.info("Loading embedding model: %s", model_name)
            self._model = SentenceTransformer(model_name, device=resolved_device, local_files_only=local_files_only)
        except Exception as exc:
            if not allow_hashing_fallback:
                raise
            LOGGER.warning("Falling back to HashingVectorizer embeddings for %s: %s", model_name, exc)
            self._hashing = HashingVectorizer(n_features=768, alternate_sign=False, norm=None)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed texts into a dense matrix."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if self._model is not None:
            embeddings = self._model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
            return np.asarray(embeddings, dtype=np.float32)
        if self._hashing is None:
            raise RuntimeError("No embedding backend is available")
        matrix = self._hashing.transform(texts).astype(np.float32)
        embeddings = matrix.toarray()
        if self.normalize:
            embeddings = sk_normalize(embeddings, norm="l2")
        return embeddings.astype(np.float32)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embedding rows, leaving zero rows stable."""
    if embeddings.size == 0:
        return embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


def embedding_fingerprint(model_name: str, texts: list[str]) -> str:
    """Return a short fingerprint for a model/text batch pairing."""
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    for text in texts:
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()[:16]


def _resolve_sentence_transformer_device(requested: str | None) -> str | None:
    """Return a usable device string for sentence-transformers."""
    if requested is None:
        return None
    requested_text = str(requested)
    if not requested_text.startswith("cuda"):
        return requested_text
    try:
        import torch

        if torch.cuda.is_available():
            return requested_text
        LOGGER.warning("Requested device %s but torch has no CUDA support; falling back to CPU.", requested_text)
        return "cpu"
    except Exception:
        return requested_text
