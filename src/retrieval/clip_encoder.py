"""CLIP encoder singleton using sentence-transformers clip-ViT-B-32.

Provides a shared image+text embedding space (512-d) for zero-shot visual
retrieval: uploaded images are compared against CLIP-text embeddings of
catalogue item ``search_text`` fields.

Privacy contract
----------------
This module never persists image bytes.  ``encode_image`` operates entirely
on an in-memory ``PIL.Image`` object.  Callers are responsible for clearing
the reference once embedding is complete; the endpoint layer drops bytes
explicitly after calling this function.

Usage
-----
    from src.retrieval.clip_encoder import get_clip_encoder

    encoder = get_clip_encoder(model_id="clip-ViT-B-32")
    vec = encoder.encode_image(pil_img)        # (512,) float32, L2-normalised
    vecs = encoder.encode_texts(["a red dress"])  # (N, 512) float32
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton cache — one encoder per model_id.
# Protected by a threading.Lock so concurrent requests don't double-load.
# ---------------------------------------------------------------------------

_encoders: dict[str, "CLIPEncoder"] = {}
_encoder_lock = threading.Lock()

# Dimension of clip-ViT-B-32 shared text/image space.
CLIP_DIM: int = 512


class CLIPEncoder:
    """Thin wrapper around a sentence-transformers CLIP model.

    Both ``encode_image`` and ``encode_texts`` return L2-normalised float32
    vectors in the shared 512-d CLIP space, ready for IndexFlatIP (cosine)
    search.

    Load errors (missing model files, network unavailable) propagate as
    ``RuntimeError`` so the endpoint layer can return 503 / feature-off.

    Args:
        model_id: sentence-transformers model name, e.g. ``"clip-ViT-B-32"``.
    """

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        try:
            self._model = SentenceTransformer(model_id, device="cpu")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load CLIP model '{model_id}': {exc}. "
                "Ensure the model is available (sentence-transformers cache or offline copy)."
            ) from exc
        # Patch CLIPModel.tokenize to add truncation=True so texts longer than
        # CLIP's 77-token limit are silently truncated rather than raising a
        # ValueError from the HuggingFace CLIP model.  The sentence-transformers
        # CLIPModel.tokenize does not pass truncation=True to the processor
        # tokenizer, which causes an error with newer transformers versions.
        self._patch_clip_truncation()
        logger.info("CLIPEncoder loaded: model=%s dim=%d", model_id, CLIP_DIM)

    def _patch_clip_truncation(self) -> None:
        """Monkey-patch CLIPModel.tokenize to enable text truncation at 77 tokens.

        sentence-transformers' CLIPModel.tokenize calls
        ``self.processor.tokenizer(texts, return_tensors='pt', padding=padding)``
        without ``truncation=True``.  Newer HuggingFace transformers raise a
        ``ValueError`` when any text exceeds 77 tokens (CLIP's hard limit).
        This patch wraps the tokenize method to inject ``truncation=True`` and
        ``max_length=77``.
        """
        import types

        clip_mod = self._model[0]  # type: ignore[index]

        def _tokenize_with_truncation(self_inner: object, texts: object, padding: object = True) -> dict:  # noqa: ANN001
            from PIL import Image as _PILImage  # type: ignore[import-untyped]

            images = []
            texts_values = []
            image_text_info = []
            for data in texts:  # type: ignore[union-attr]
                if isinstance(data, _PILImage.Image):
                    images.append(data)
                    image_text_info.append(0)
                else:
                    texts_values.append(data)
                    image_text_info.append(1)
            encoding: dict = {}
            if texts_values:
                # Add truncation=True + max_length=77 to prevent ValueError on long texts.
                encoding = self_inner.processor.tokenizer(  # type: ignore[union-attr]
                    texts_values,
                    return_tensors="pt",
                    padding=padding,
                    truncation=True,
                    max_length=77,
                )
            if images:
                image_features = self_inner.processor.image_processor(images, return_tensors="pt")  # type: ignore[union-attr]
                encoding["pixel_values"] = image_features.pixel_values
            encoding["image_text_info"] = image_text_info
            return encoding

        # Bind the patched method to this specific CLIPModel instance only
        # (not the class) so other code that might use the class unpatched is unaffected.
        clip_mod.tokenize = types.MethodType(_tokenize_with_truncation, clip_mod)  # type: ignore[method-assign]
        logger.debug("CLIPModel.tokenize patched for truncation at 77 tokens")

    def encode_image(self, img: "PILImage") -> np.ndarray:
        """Embed a PIL image into the shared CLIP space.

        Args:
            img: An in-memory PIL Image (any mode; converted to RGB internally).

        Returns:
            L2-normalised float32 array of shape (512,).

        Note:
            The caller should drop all references to the image bytes after this
            call.  This function does not cache or persist the image.
        """
        rgb = img.convert("RGB")
        # SentenceTransformer.encode accepts PIL images directly when the model
        # has a vision modality (CLIP models do).
        vec = self._model.encode(
            rgb,  # type: ignore[arg-type]
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        arr = np.array(vec, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[0]
        return arr

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of text strings into the shared CLIP space.

        CLIP's text encoder has a hard 77-token limit.  Texts longer than this
        are silently truncated by the monkey-patched ``CLIPModel.tokenize``
        (applied in ``_patch_clip_truncation`` during ``__init__``), so callers
        do not need to pre-process texts.

        Args:
            texts: Non-empty list of text strings.

        Returns:
            L2-normalised float32 array of shape (N, 512).
        """
        vecs = self._model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(vecs, dtype=np.float32)


def get_clip_encoder(model_id: str = "clip-ViT-B-32") -> CLIPEncoder:
    """Return the cached ``CLIPEncoder`` for *model_id*, loading it on first call.

    Thread-safe: uses a module-level lock so concurrent startup requests
    don't trigger double-load.

    Args:
        model_id: sentence-transformers CLIP model identifier.

    Returns:
        A ready-to-use ``CLIPEncoder`` instance.

    Raises:
        RuntimeError: if the model cannot be loaded.
    """
    if model_id in _encoders:
        return _encoders[model_id]
    with _encoder_lock:
        # Double-checked locking
        if model_id not in _encoders:
            _encoders[model_id] = CLIPEncoder(model_id)
    return _encoders[model_id]
