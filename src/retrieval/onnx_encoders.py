"""ONNX Runtime CPU encoders: torch-free replacements for the sentence-transformers encoders.

Two encoders, both fp32, both reproducing what the sentence-transformers pipelines do today:

* ``OnnxTextEncoder``  -- ``sentence-transformers/all-MiniLM-L6-v2`` (BERT -> mean pooling over the
  attention mask -> optional L2 normalisation, ``max_seq_length`` 256, 384-d).
* ``OnnxCLIPEncoder``  -- ``sentence-transformers/clip-ViT-B-32`` (CLIP text tower and image tower,
  projected 512-d, L2-normalised; text truncated at 77 tokens).

Only ``onnxruntime``, ``tokenizers``, ``numpy`` and ``Pillow`` are needed at runtime. The ``.onnx``
files and ``tokenizer.json`` are produced at image build time by ``scripts/export_onnx.py`` (the
only place torch is used) and are looked up under ``ONNX_MODEL_DIR`` (default ``<repo>/onnx_models``).

The ``encode`` signature of ``OnnxTextEncoder`` is a subset-compatible stand-in for
``SentenceTransformer.encode`` so ``DenseRetriever`` and the hot-reload model reuse keep working.

Backend selection (``encoder_backend``): env ``ENCODER_BACKEND`` = ``onnx`` (default) or ``torch``
(local dev only; needs ``pip install -r requirements-export.txt``).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

_REPO_ROOT = Path(__file__).resolve().parents[2]

BACKENDS = ("onnx", "torch")

# CLIP's hard text limit; the sentence-transformers path was patched to truncate at this length.
CLIP_MAX_TOKENS = 77
# openai/clip-vit-base-patch32 preprocessor_config.json (checked by the export script's self-test).
CLIP_IMAGE_SIZE = 224
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def encoder_backend() -> str:
    """Return the configured encoder backend (``ENCODER_BACKEND``, default ``onnx``).

    Raises:
        ValueError: if the env var is set to something other than ``onnx`` / ``torch``.
    """
    value = os.environ.get("ENCODER_BACKEND", "onnx").strip().lower() or "onnx"
    if value not in BACKENDS:
        raise ValueError(f"ENCODER_BACKEND={value!r} is not one of {BACKENDS}")
    return value


def onnx_model_dir(model_name: str, root: str | os.PathLike[str] | None = None) -> Path:
    """Directory holding the exported files for ``model_name`` (last path segment is the key)."""
    base = Path(root or os.environ.get("ONNX_MODEL_DIR") or _REPO_ROOT / "onnx_models")
    return base / model_name.rsplit("/", 1)[-1]


def parse_cpu_max(text: str) -> int | None:
    """CPUs granted by a cgroup-v2 ``cpu.max`` line (``"<quota> <period>"`` or ``"max <period>"``).

    Returns ceil(quota/period), or None when unlimited / unparsable.
    """
    try:
        quota, period = text.split()
        if quota == "max":
            return None
        return max(1, -(-int(quota) // int(period)))
    except ValueError:
        return None


def cpu_budget(cpu_max_path: str = "/sys/fs/cgroup/cpu.max") -> int:
    """CPUs this process can really use: the container CPU quota if one is set, else the visible
    cores. ORT sizes its pool from visible cores and ignores the quota, so on a 1 vCPU Cloud Run
    instance an unclamped pool of many threads spins against the throttle."""
    visible = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        quota = parse_cpu_max(Path(cpu_max_path).read_text())
    except OSError:
        quota = None
    return min(visible, quota) if quota else visible


def _session(path: Path) -> Any:
    """Create a CPU InferenceSession; raises RuntimeError with the export hint if missing."""
    if not path.is_file():
        raise RuntimeError(
            f"ONNX model file not found: {path}. Run `python scripts/export_onnx.py` "
            "(needs requirements-export.txt) or set ONNX_MODEL_DIR; the Docker image does this "
            "at build time."
        )
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # errors only; ORT warns on every initializer it re-folds
    # ORT_INTRA_OP_THREADS overrides; default = the container's real CPU budget (see cpu_budget)
    opts.intra_op_num_threads = int(os.environ.get("ORT_INTRA_OP_THREADS") or cpu_budget())
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _tokenizer(path: Path, max_length: int) -> Any:
    from tokenizers import Tokenizer

    if not path.is_file():
        raise RuntimeError(f"tokenizer file not found: {path}. See scripts/export_onnx.py.")
    tok = Tokenizer.from_file(str(path))
    tok.enable_truncation(max_length=max_length)
    tok.no_padding()  # a shipped tokenizer.json may carry a padding config; we pad per batch
    return tok


def _out_dim(session: Any) -> int:
    """Embedding width from the graph's first output (last axis is static in our exports)."""
    dim = session.get_outputs()[0].shape[-1]
    if not isinstance(dim, int):
        raise RuntimeError(f"ONNX output has a dynamic embedding axis ({dim!r}); re-export.")
    return dim


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def _pad_batch(encodings: list[Any], pad_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-pad to the longest sequence in the batch (what ``padding=True`` does)."""
    longest = max(len(e.ids) for e in encodings)
    ids = np.full((len(encodings), longest), pad_id, dtype=np.int64)
    mask = np.zeros((len(encodings), longest), dtype=np.int64)
    type_ids = np.zeros((len(encodings), longest), dtype=np.int64)
    for i, e in enumerate(encodings):
        n = len(e.ids)
        ids[i, :n] = e.ids
        mask[i, :n] = e.attention_mask
        type_ids[i, :n] = e.type_ids
    return ids, mask, type_ids


class OnnxTextEncoder:
    """all-MiniLM-L6-v2 on ONNX Runtime; ``encode`` mirrors ``SentenceTransformer.encode``."""

    def __init__(self, model_name: str, root: str | os.PathLike[str] | None = None) -> None:
        d = onnx_model_dir(model_name, root)
        self.max_seq_length = 256  # sentence-transformers' sentence_bert_config.json value
        self._session = _session(d / "model.onnx")
        self._tok = _tokenizer(d / "tokenizer.json", self.max_seq_length)
        self._pad_id = self._tok.token_to_id("[PAD]") or 0
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._dim = _out_dim(self._session)

    def encode(
        self,
        sentences: str | list[str],
        batch_size: int = 32,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,  # noqa: ARG002 - signature compatibility
        **_ignored: Any,
    ) -> np.ndarray:
        """Embed texts; returns float32 ``(n, 384)`` (``(384,)`` for a single str)."""
        single = isinstance(sentences, str)
        texts = [sentences] if single else list(sentences)
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        # Length-sorted batches like sentence-transformers: less padding, identical results.
        order = np.argsort([-len(t) for t in texts], kind="stable")
        for start in range(0, len(texts), batch_size):
            idx = order[start : start + batch_size]
            enc = self._tok.encode_batch([texts[i] for i in idx])
            ids, mask, type_ids = _pad_batch(enc, self._pad_id)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = type_ids
            hidden = self._session.run(None, feeds)[0]  # (b, seq, 384)
            m = mask[:, :, None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)
            out[idx] = pooled
        if normalize_embeddings:
            out = _l2_normalise(out)
        return out[0] if single else out


def preprocess_clip_image(img: PILImage) -> np.ndarray:
    """CLIPImageProcessor on PIL: resize short side to 224 (bicubic), centre crop, normalise.

    Returns float32 ``(3, 224, 224)``.
    """
    from PIL import Image

    rgb = img.convert("RGB")
    w, h = rgb.size
    short, long = (w, h) if w <= h else (h, w)
    new_short, new_long = CLIP_IMAGE_SIZE, int(CLIP_IMAGE_SIZE * long / short)
    new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    resized = rgb.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
    top = (new_h - CLIP_IMAGE_SIZE) // 2
    left = (new_w - CLIP_IMAGE_SIZE) // 2
    crop = resized.crop((left, top, left + CLIP_IMAGE_SIZE, top + CLIP_IMAGE_SIZE))
    arr = np.asarray(crop, dtype=np.float32) * np.float32(1 / 255)
    arr = (arr - CLIP_MEAN) / CLIP_STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32)


class OnnxCLIPEncoder:
    """clip-ViT-B-32 on ONNX Runtime; same public interface as ``clip_encoder.CLIPEncoder``.

    Load errors surface as ``RuntimeError`` (the endpoint layer maps that to feature-off / 503).
    """

    def __init__(self, model_id: str, root: str | os.PathLike[str] | None = None) -> None:
        self._model_id = model_id
        d = onnx_model_dir(model_id, root)
        # The text tower is only needed to BUILD the CLIP text index (scripts/build_clip_index.py);
        # the request path encodes images and searches the prebuilt index. It is loaded on first
        # use: eagerly it costs ~250 MB of resident memory in a 4 GiB service for nothing.
        self._text_path = d / "text.onnx"
        self._text: Any = None
        self._text_lock = threading.Lock()
        try:
            if not self._text_path.is_file():
                raise RuntimeError(f"ONNX model file not found: {self._text_path}. See scripts/export_onnx.py.")
            self._vision = _session(d / "vision.onnx")
            self._tok = _tokenizer(d / "tokenizer.json", CLIP_MAX_TOKENS)
        except RuntimeError as exc:
            raise RuntimeError(f"Failed to load CLIP model '{model_id}': {exc}") from exc
        # CLIP pads with <|endoftext|>; pooling picks the first EOS (argmax), so pad id is inert.
        self._pad_id = self._tok.token_to_id("<|endoftext|>") or 49407

    def _text_session(self) -> Any:
        """The text-tower session, created on first use (thread-safe)."""
        if self._text is None:
            with self._text_lock:
                if self._text is None:
                    self._text = _session(self._text_path)
        return self._text

    def encode_image(self, img: PILImage) -> np.ndarray:
        """Embed one in-memory PIL image; returns L2-normalised float32 ``(512,)``.

        The image is never persisted (privacy contract of ``clip_encoder``).
        """
        pixels = preprocess_clip_image(img)[None]
        vec = self._vision.run(None, {"pixel_values": pixels})[0]
        return _l2_normalise(np.asarray(vec, dtype=np.float32))[0]

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Embed texts (truncated at 77 tokens); returns L2-normalised float32 ``(n, 512)``."""
        text = self._text_session()
        out = np.zeros((len(texts), _out_dim(text)), dtype=np.float32)
        for start in range(0, len(texts), 64):
            enc = self._tok.encode_batch(texts[start : start + 64])
            ids, mask, _ = _pad_batch(enc, self._pad_id)
            emb = text.run(None, {"input_ids": ids, "attention_mask": mask})[0]
            out[start : start + len(enc)] = emb
        return _l2_normalise(out)
