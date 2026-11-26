"""Index state file: what the incremental builder carries from one run to the next.

One ``.npz`` file (for a GitHub Release asset on an ``index-state-*`` pre-release tag). Holds, per
kept row: store, article id, content hash, the catalogue row as JSON, the text vector, the image
vector (zeros plus ``has_image=False`` when absent) and ``image_pending``. A JSON header carries the
schema version, build time, vector dims, per-store status and a sha256 checksum over the payload.

Reading refuses an unknown schema version, wrong dims or a checksum mismatch (``StateError``);
callers must treat that as a hard failure, never as an empty state. See
docs/ops/incremental-build-rules.md section 4.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
TEXT_DIM = 384  # MiniLM, same as src/retrieval/build_manifest.DENSE_DIM
IMAGE_DIM = 512  # CLIP ViT-B/32, same as build_manifest.CLIP_DIM

_ARRAYS = (
    "store",
    "article_id",
    "content_hash",
    "row_json",
    "text_vec",
    "image_vec",
    "has_image",
    "image_pending",
)


class StateError(Exception):
    """The state file is unreadable, corrupt or from an incompatible schema."""


@dataclass
class StateRow:
    """One kept index row with its vectors. ``image_vec`` is None when it is absent or pending."""

    store: str
    article_id: str
    content_hash: str
    row: dict
    text_vec: np.ndarray
    image_vec: np.ndarray | None = None
    image_pending: bool = False


@dataclass
class IndexState:
    """All rows plus the last known fetch status per store."""

    rows: list[StateRow] = field(default_factory=list)
    store_status: dict[str, str] = field(default_factory=dict)
    built_at: str = ""

    def by_store(self) -> dict[str, dict[str, StateRow]]:
        """Rows grouped ``store -> article_id -> row``."""
        out: dict[str, dict[str, StateRow]] = {}
        for r in self.rows:
            out.setdefault(r.store, {})[r.article_id] = r
        return out


def _checksum(arrays: dict[str, np.ndarray]) -> str:
    """sha256 over names, dtypes, shapes and bytes of the payload arrays, sorted by name."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(f"{name}|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _to_arrays(state: IndexState) -> dict[str, np.ndarray]:
    n = len(state.rows)
    text = np.zeros((n, TEXT_DIM), dtype=np.float32)
    image = np.zeros((n, IMAGE_DIM), dtype=np.float32)
    has_image = np.zeros(n, dtype=bool)
    for i, r in enumerate(state.rows):
        text[i] = r.text_vec
        if r.image_vec is not None:
            image[i] = r.image_vec
            has_image[i] = True
    return {
        "store": np.array([r.store for r in state.rows], dtype=np.str_),
        "article_id": np.array([r.article_id for r in state.rows], dtype=np.str_),
        "content_hash": np.array([r.content_hash for r in state.rows], dtype=np.str_),
        "row_json": np.array(
            [json.dumps(r.row, sort_keys=True, ensure_ascii=False) for r in state.rows],
            dtype=np.str_,
        ),
        "text_vec": text,
        "image_vec": image,
        "has_image": has_image,
        "image_pending": np.array([r.image_pending for r in state.rows], dtype=bool),
    }


def write_state(state: IndexState, path: Path) -> str:
    """Write ``state`` to ``path`` atomically (temp file then rename); return the checksum."""
    arrays = _to_arrays(state)
    checksum = _checksum(arrays)
    header = {
        "schema_version": SCHEMA_VERSION,
        "built_at": state.built_at,
        "rows": len(state.rows),
        "dims": {"text": TEXT_DIM, "image": IMAGE_DIM},
        "store_status": state.store_status,
        "checksum": checksum,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, header=np.array(json.dumps(header, sort_keys=True)), **arrays)
    tmp.replace(path)
    return checksum


def read_state(path: Path) -> IndexState:
    """Read and verify a state file. Raises ``StateError`` on any problem."""
    try:
        with np.load(path, allow_pickle=False) as z:
            header = json.loads(str(z["header"]))
            arrays = {name: z[name] for name in _ARRAYS}
    except (OSError, ValueError, KeyError) as exc:
        raise StateError(f"cannot read state file {path}: {exc}") from exc
    if header.get("schema_version") != SCHEMA_VERSION:
        raise StateError(
            f"unsupported state schema_version {header.get('schema_version')!r}; "
            f"this reader handles {SCHEMA_VERSION}"
        )
    dims = header.get("dims") or {}
    if dims.get("text") != TEXT_DIM or dims.get("image") != IMAGE_DIM:
        raise StateError(f"state dims {dims} do not match {TEXT_DIM}/{IMAGE_DIM}")
    if _checksum(arrays) != header.get("checksum"):
        raise StateError("state checksum mismatch: file is corrupt or was modified")
    rows = [
        StateRow(
            store=str(arrays["store"][i]),
            article_id=str(arrays["article_id"][i]),
            content_hash=str(arrays["content_hash"][i]),
            row=json.loads(str(arrays["row_json"][i])),
            text_vec=arrays["text_vec"][i].copy(),
            image_vec=arrays["image_vec"][i].copy() if arrays["has_image"][i] else None,
            image_pending=bool(arrays["image_pending"][i]),
        )
        for i in range(len(arrays["article_id"]))
    ]
    return IndexState(
        rows=rows,
        store_status=dict(header.get("store_status") or {}),
        built_at=str(header.get("built_at", "")),
    )
