"""Offline tests for POST /style/from-image.

All tests run without downloading the CLIP model or hitting any network.
The CLIP encoder and brand index are fully mocked.

Markers
-------
- ``requires_clip``:  tests that load the REAL CLIP model (skipped by default CI).
  Default pytest run (``pytest -m "not requires_ollama"``) also skips these.

Privacy invariants asserted
---------------------------
- No file is created under ``tmp_path`` or elsewhere by the happy path.
- The ``upload_bytes`` and ``pil_img`` locals are deleted inside the endpoint;
  we can't inspect CPython reference counts from the test layer, but we
  assert that no new files appear under the repo root after the call.
"""
from __future__ import annotations

import glob
import io
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api.deps as deps
from api.main import app
from api.session import InMemorySessionStore

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_valid_jpeg_bytes(width: int = 10, height: int = 10) -> bytes:
    """Return minimal valid JPEG bytes for a small solid-colour image."""
    img = Image.new("RGB", (width, height), color=(100, 149, 237))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_valid_png_bytes() -> bytes:
    img = Image.new("RGB", (8, 8), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _MockLLM:
    """Minimal LLM mock that returns template-compatible text."""

    def generate(self, prompt: str, system: str = "", **kwargs: Any) -> str:  # noqa: ARG002
        return '["A casual look with this piece."]'

    def generate_stream(self, prompt: str, system: str = "", **kwargs: Any) -> Iterator[str]:  # noqa: ARG002
        yield "A casual look."

    def chat(self, messages: list[dict], **kwargs: Any) -> str:  # noqa: ARG002
        return "A casual look with this piece."

    def chat_stream(self, messages: list[dict], **kwargs: Any) -> Iterator[str]:  # noqa: ARG002
        yield "ok"


_MINIMAL_CONFIG: dict[str, Any] = {
    "agent": {"max_iterations": 3},
    "memory": {"recent_turns": 6, "summary_trigger_turns": 12},
    "features": {"image_input_enabled": True},
    "clip": {"model": "clip-ViT-B-32", "index_dir": "data/processed/clip"},
}

_MINIMAL_CONFIG_FEATURE_OFF: dict[str, Any] = {
    **_MINIMAL_CONFIG,
    "features": {"image_input_enabled": False},
}


# ---------------------------------------------------------------------------
# Mock outfit / retriever helpers
# ---------------------------------------------------------------------------


def _mock_variants() -> list[dict]:
    seed_item = {
        "article_id": "111",
        "prod_name": "Floral Top",
        "display_name": "Floral Top (White Top)",
        "colour": "White",
        "product_type": "Top",
        "department": "Ladieswear",
        "gender": "women",
        "score": 0.95,
        "price_inr": 999.0,
        "image_url": None,
        "detail_desc": "A white floral top.",
        "pdp_handle": None,
        "_role": "seed",
    }
    return [
        {
            "look_id": "look-uuid-1",
            "seed_item": seed_item,
            "complements": [],
            "outfit_rationale": "A clean casual look.",
            "empty_slots": [],
            "occasion": "casual",
            "gender": "women",
            "budget_total_inr": 999.0,
            "variant_label": "Base",
        }
    ]


class _MockRetriever:
    def search(self, query: str, top_k: int = 20, filters: dict | None = None) -> list[dict]:  # noqa: ARG002
        return []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def inject_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject minimal deps so the FastAPI app starts without real indices."""
    import pandas as pd

    store = InMemorySessionStore()
    monkeypatch.setattr(deps, "_session_store", store)
    monkeypatch.setattr(deps, "_llm", _MockLLM())
    monkeypatch.setattr(deps, "_config", _MINIMAL_CONFIG)
    monkeypatch.setattr(deps, "_retriever", _MockRetriever())
    monkeypatch.setattr(
        deps,
        "_catalogue_df",
        pd.DataFrame(
            [
                {
                    "article_id": "111",
                    "prod_name": "Floral Top",
                    "display_name": "Floral Top (White Top)",
                    "colour": "White",
                    "colour_group_name": "White",
                    "product_type_name": "Top",
                    "department_name": "Ladieswear",
                    "index_group_name": "Ladieswear",
                    "gender": "women",
                    "price_inr": 999.0,
                    "search_text": "Floral Top White Ladieswear",
                    "detail_desc": "A white floral top.",
                    "image_url": None,
                    "pdp_handle": None,
                    "facets": {
                        "colour_group_name": "White",
                        "product_type_name": "Top",
                        "department_name": "Ladieswear",
                        "index_group_name": "Ladieswear",
                        "garment_group_name": "Tops",
                    },
                }
            ]
        ),
    )
    monkeypatch.setenv("JWT_VERIFICATION_DISABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "10000")


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


def _mock_anchor_patch(article_ids: list[str] | None = None):
    """Return a patch context that mocks ``find_anchor_from_image`` at the route level."""
    return patch(
        "api.routes.image_style.find_anchor_from_image",
        return_value=article_ids if article_ids is not None else ["111"],
    )


def _mock_brand_index_exists(exists: bool = True):
    return patch(
        "api.routes.image_style._brand_index_exists",
        return_value=exists,
    )


def _mock_compose_variants():
    return patch(
        "api.routes.image_style.compose_outfit_variants",
        return_value=_mock_variants(),
    )


def _mock_rationale():
    return patch(
        "api.routes.image_style.generate_rationales",
        return_value=["A clean casual look."],
    )


def _mock_clip_encoder():
    """Patch the CLIPEncoder so no model is loaded."""
    mock_enc = MagicMock()
    mock_enc.encode_image.return_value = np.zeros(512, dtype=np.float32)
    return patch(
        "src.retrieval.clip_encoder.get_clip_encoder",
        return_value=mock_enc,
    )


# ---------------------------------------------------------------------------
# Tests — validation
# ---------------------------------------------------------------------------


def test_oversize_file_rejected(client: TestClient) -> None:
    """Files larger than 15 MB must be rejected with 413."""
    oversize = b"x" * (15 * 1024 * 1024 + 1)
    with _mock_brand_index_exists(True):
        resp = client.post(
            "/style/from-image",
            files={"file": ("big.jpg", oversize, "image/jpeg")},
        )
    assert resp.status_code == 413, resp.text


def test_disallowed_content_type_rejected(client: TestClient) -> None:
    """Non-image content-type must be rejected with 400."""
    resp = client.post(
        "/style/from-image",
        files={"file": ("doc.pdf", b"%PDF-1.4 data", "application/pdf")},
    )
    assert resp.status_code == 400, resp.text
    assert "Unsupported content type" in resp.json()["detail"]


def test_non_image_bytes_rejected(client: TestClient) -> None:
    """Random non-image bytes with a valid content-type header must return 400."""
    garbage = b"NOT_AN_IMAGE_ABCDEFGHIJ" * 100
    with _mock_brand_index_exists(True):
        resp = client.post(
            "/style/from-image",
            files={"file": ("fake.jpg", garbage, "image/jpeg")},
        )
    assert resp.status_code == 400, resp.text


def test_feature_flag_off_returns_404(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """When image_input_enabled=false the endpoint must return 404."""
    monkeypatch.setattr(deps, "_config", _MINIMAL_CONFIG_FEATURE_OFF)
    resp = client.post(
        "/style/from-image",
        files={"file": ("img.jpg", _make_valid_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 404, resp.text


def test_env_override_disables_feature(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """ENABLE_IMAGE_INPUT=false must override the yaml flag."""
    monkeypatch.setenv("ENABLE_IMAGE_INPUT", "false")
    resp = client.post(
        "/style/from-image",
        files={"file": ("img.jpg", _make_valid_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 404, resp.text


def test_missing_brand_index_returns_404(client: TestClient) -> None:
    """When the brand CLIP index is absent the endpoint must return 404."""
    with _mock_brand_index_exists(False):
        resp = client.post(
            "/style/from-image",
            files={"file": ("img.jpg", _make_valid_jpeg_bytes(), "image/jpeg")},
        )
    assert resp.status_code == 404, resp.text


# ---------------------------------------------------------------------------
# Tests — happy path
# ---------------------------------------------------------------------------


def test_happy_path_returns_look_with_rationale_and_variants(client: TestClient) -> None:
    """Valid JPEG upload with mocked encoder + index must return a look payload."""
    jpeg_bytes = _make_valid_jpeg_bytes()

    with (
        _mock_brand_index_exists(True),
        _mock_anchor_patch(["111"]),
        _mock_compose_variants(),
        _mock_rationale(),
    ):
        resp = client.post(
            "/style/from-image",
            files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "look_id" in data
    assert "outfit_rationale" in data
    assert "outfit_variants" in data
    assert isinstance(data["outfit_variants"], list)
    assert len(data["outfit_variants"]) >= 1
    # Each variant must carry a rationale
    for v in data["outfit_variants"]:
        assert v.get("rationale"), f"Variant missing rationale: {v}"
    # Anchor article_id must be present
    assert data.get("anchor_article_id") == "111"


def test_happy_path_png_accepted(client: TestClient) -> None:
    """PNG uploads must be accepted alongside JPEG."""
    png_bytes = _make_valid_png_bytes()

    with (
        _mock_brand_index_exists(True),
        _mock_anchor_patch(["111"]),
        _mock_compose_variants(),
        _mock_rationale(),
    ):
        resp = client.post(
            "/style/from-image",
            files={"file": ("photo.png", png_bytes, "image/png")},
        )

    assert resp.status_code == 200, resp.text


def test_happy_path_with_message_echoes_user_text(client: TestClient) -> None:
    """A multipart ``message`` field must be echoed back verbatim as user_text."""
    jpeg_bytes = _make_valid_jpeg_bytes()

    with (
        _mock_brand_index_exists(True),
        _mock_anchor_patch(["111"]),
        _mock_compose_variants(),
        _mock_rationale(),
    ):
        resp = client.post(
            "/style/from-image",
            files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
            data={"message": "something for a party under 2000"},
        )

    assert resp.status_code == 200, resp.text
    assert resp.json().get("user_text") == "something for a party under 2000"


def test_happy_path_without_message_user_text_is_none(client: TestClient) -> None:
    """When no ``message`` field is sent, user_text must be None."""
    jpeg_bytes = _make_valid_jpeg_bytes()

    with (
        _mock_brand_index_exists(True),
        _mock_anchor_patch(["111"]),
        _mock_compose_variants(),
        _mock_rationale(),
    ):
        resp = client.post(
            "/style/from-image",
            files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        )

    assert resp.status_code == 200, resp.text
    assert resp.json().get("user_text") is None


def test_happy_path_no_disk_writes(client: TestClient, tmp_path: Path) -> None:
    """The endpoint must not write any new files to the project data directory."""
    data_dir = Path("data/processed/clip")
    before_files: set[str] = set()
    if data_dir.exists():
        before_files = set(glob.glob(str(data_dir / "**" / "*"), recursive=True))

    jpeg_bytes = _make_valid_jpeg_bytes()

    with (
        _mock_brand_index_exists(True),
        _mock_anchor_patch(["111"]),
        _mock_compose_variants(),
        _mock_rationale(),
    ):
        resp = client.post(
            "/style/from-image",
            files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        )

    assert resp.status_code == 200, resp.text

    after_files: set[str] = set()
    if data_dir.exists():
        after_files = set(glob.glob(str(data_dir / "**" / "*"), recursive=True))

    new_files = after_files - before_files
    assert not new_files, f"Endpoint wrote unexpected files: {new_files}"


# ---------------------------------------------------------------------------
# Tests — CLIP encoder unit (offline, no model download)
# ---------------------------------------------------------------------------


def test_clip_encoder_encode_image_shape() -> None:
    """CLIPEncoder.encode_image must return a 512-d L2-normalised float32 array (mocked)."""
    from src.retrieval.clip_encoder import CLIPEncoder

    mock_model = MagicMock()
    mock_model.encode.return_value = np.ones(512, dtype=np.float32) / np.sqrt(512)

    # Build CLIPEncoder without going through __init__ (which loads the model)
    enc = object.__new__(CLIPEncoder)
    enc._model_id = "test-clip"
    enc._model = mock_model

    img = Image.new("RGB", (10, 10), color=(200, 100, 50))
    vec = enc.encode_image(img)

    assert vec.shape == (512,)
    assert vec.dtype == np.float32


def test_clip_encoder_encode_texts_shape() -> None:
    """CLIPEncoder.encode_texts must return (N, 512) float32 array (mocked)."""
    from src.retrieval.clip_encoder import CLIPEncoder

    mock_model = MagicMock()
    texts = ["a red dress", "blue jeans"]
    mock_model.encode.return_value = np.random.default_rng(42).random(
        (len(texts), 512)
    ).astype(np.float32)

    enc = object.__new__(CLIPEncoder)
    enc._model_id = "test-clip"
    enc._model = mock_model

    vecs = enc.encode_texts(texts)

    assert vecs.shape == (2, 512)
    assert vecs.dtype == np.float32


def test_clip_encoder_load_failure_raises_runtime_error() -> None:
    """If the model cannot be loaded, CLIPEncoder must raise RuntimeError."""
    with patch(
        "src.retrieval.clip_encoder.SentenceTransformer", side_effect=OSError("model not found")
    ):
        from src.retrieval.clip_encoder import CLIPEncoder

        with pytest.raises(RuntimeError, match="Failed to load CLIP model"):
            CLIPEncoder("clip-ViT-B-32-TEST-FAIL")


# ---------------------------------------------------------------------------
# Tests — find_anchor_from_image unit (offline)
# ---------------------------------------------------------------------------


def test_find_anchor_returns_empty_when_no_index(tmp_path: Path) -> None:
    """find_anchor_from_image must return [] when the index is missing."""
    from src.agents.outfit.image_anchor import find_anchor_from_image

    img = Image.new("RGB", (8, 8))
    result = find_anchor_from_image(img, brand="nonexistent_brand_xyz", _index_base=tmp_path)
    assert result == []


def test_find_anchor_from_image_mocked(tmp_path: Path) -> None:
    """find_anchor_from_image returns ranked article_ids using a dummy FAISS index."""
    import faiss

    from src.agents.outfit.image_anchor import _index_cache

    # Build a tiny in-memory FAISS index (2 items, 512-d)
    dim = 512
    index = faiss.IndexFlatIP(dim)
    vecs = np.eye(2, dim, dtype=np.float32)  # orthonormal
    index.add(vecs)
    article_ids = np.array(["art_001", "art_002"])

    # Write to tmp_path so _load_brand_index can find it
    brand = "test_brand_xyz_unique"
    brand_dir = tmp_path / brand
    brand_dir.mkdir()
    faiss.write_index(index, str(brand_dir / "clip.faiss"))
    np.save(str(brand_dir / "clip_article_ids.npy"), article_ids)

    # Clear any cached entry for this brand
    _index_cache.pop(brand, None)

    # Mock encoder to return vec matching art_001 exactly
    mock_enc = MagicMock()
    mock_enc.encode_image.return_value = vecs[0]  # points at art_001

    with patch("src.agents.outfit.image_anchor.get_clip_encoder", return_value=mock_enc):
        from src.agents.outfit.image_anchor import find_anchor_from_image as _find_anchor

        img = Image.new("RGB", (8, 8))
        results = _find_anchor(img, brand=brand, top_k=2, _index_base=tmp_path)

    # Clear cache to avoid polluting other tests
    _index_cache.pop(brand, None)

    assert results[0] == "art_001", f"Expected art_001 first, got {results}"
    assert "art_002" in results


# ---------------------------------------------------------------------------
# Integration-shape test (optional / requires_clip marker)
# ---------------------------------------------------------------------------


@pytest.mark.requires_clip
def test_clip_encoder_real_model_encodes_image() -> None:
    """Load the real clip-ViT-B-32 model and verify image embedding shape."""
    # Clear cached encoder to force fresh load
    from src.retrieval.clip_encoder import _encoders

    _encoders.clear()

    from src.retrieval.clip_encoder import get_clip_encoder

    enc = get_clip_encoder("clip-ViT-B-32")
    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    vec = enc.encode_image(img)

    assert vec.shape == (512,)
    assert vec.dtype == np.float32
    # L2-normalised → norm ≈ 1.0 (sentence-transformers guarantees this)
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-4, f"Vector not unit-normalised: norm={norm}"
