"""Unit tests for src.retrieval.index_store.download_supplementary_assets.

All tests are fully offline — the GCS client is monkeypatched so no network
calls are made and no google-cloud-storage credentials are required.

Patching strategy
-----------------
``download_supplementary_assets`` does a lazy ``from google.cloud import storage``
inside its body (mirroring ``_download_from_gcs``).  Because ``storage`` is NOT
bound as a module-level name we cannot patch ``src.retrieval.index_store.storage``.
Instead we insert a fake ``google.cloud.storage`` module into ``sys.modules``
before calling the function, which satisfies the ``from google.cloud import storage``
import without touching the network.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

from src.retrieval.index_store import download_supplementary_assets

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_storage_module(
    mock_client: MagicMock,
) -> types.ModuleType:
    """Return a synthetic ``google.cloud.storage`` module whose ``Client()``
    returns *mock_client*.
    """
    mod = types.ModuleType("google.cloud.storage")
    mod.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    return mod


def _make_gcs_mock(
    clip_faiss_missing: bool = False,
    clip_ids_missing: bool = False,
    variants_missing: bool = False,
) -> MagicMock:
    """Return a mock ``google.cloud.storage.Client`` instance.

    Blobs listed in the *_missing flags raise ``Exception`` on
    ``download_to_filename`` — the production code catches broad ``Exception``
    and logs a warning so the exact exception type is irrelevant here.
    """
    missing_filenames: set[str] = set()
    if clip_faiss_missing:
        missing_filenames.add("clip.faiss")
    if clip_ids_missing:
        missing_filenames.add("clip_article_ids.npy")
    if variants_missing:
        # brand.json suffix — we check the filename portion only
        missing_filenames.add(".json")

    def _make_blob(name: str) -> MagicMock:
        blob = MagicMock()
        blob.name = name
        filename = name.split("/")[-1]
        should_fail = filename in missing_filenames or (
            # handle generic .json suffix check
            filename.endswith(".json") and ".json" in missing_filenames
        )
        if should_fail:
            blob.download_to_filename.side_effect = Exception(f"404 {name} not found")
        else:
            blob.download_to_filename.return_value = None
        return blob

    bucket = MagicMock()
    bucket.blob.side_effect = _make_blob

    client = MagicMock()
    client.bucket.return_value = bucket
    return client


class _GCSPatch:
    """Context manager that injects a fake ``google.cloud.storage`` module into
    ``sys.modules`` for the duration of the block and restores the original on exit.

    This satisfies the lazy ``from google.cloud import storage`` inside
    ``download_supplementary_assets`` without patching a non-existent module-level name.
    """

    def __init__(self, mock_client: MagicMock) -> None:
        self._mock_client = mock_client
        self._prev: dict[str, types.ModuleType | None] = {}

    def __enter__(self) -> MagicMock:
        fake_storage = _make_fake_storage_module(self._mock_client)
        # Ensure the parent packages exist in sys.modules so the import resolves.
        for key in ("google", "google.cloud", "google.cloud.storage"):
            self._prev[key] = sys.modules.get(key)
        if "google" not in sys.modules:
            sys.modules["google"] = types.ModuleType("google")
        if "google.cloud" not in sys.modules:
            sys.modules["google.cloud"] = types.ModuleType("google.cloud")
        sys.modules["google.cloud.storage"] = fake_storage
        # Also set the attribute on google.cloud so `from google.cloud import storage` works.
        sys.modules["google.cloud"].storage = fake_storage  # type: ignore[attr-defined]
        return self._mock_client

    def __exit__(self, *_: object) -> None:
        for key, orig in self._prev.items():
            if orig is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = orig


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDownloadSupplementaryAssets:
    """Happy-path and resilience tests for download_supplementary_assets."""

    def test_clip_files_downloaded_to_correct_paths(self, tmp_path: Path) -> None:
        """clip.faiss and clip_article_ids.npy land under data/processed/clip/<brand>/."""
        mock_client = _make_gcs_mock()

        with _GCSPatch(mock_client):
            download_supplementary_assets(
                uri="gs://asa-demo-indices/",
                brand="snitch",
                repo_root=tmp_path,
            )

        bucket = mock_client.bucket.return_value
        called_blob_names = [call.args[0] for call in bucket.blob.call_args_list]

        # Both CLIP blobs must have been requested
        assert any("clip/snitch/clip.faiss" in n for n in called_blob_names)
        assert any("clip/snitch/clip_article_ids.npy" in n for n in called_blob_names)

        # Destination dir must have been created
        clip_dir = tmp_path / "data" / "processed" / "clip" / "snitch"
        assert clip_dir.is_dir()

    def test_variant_map_downloaded_to_correct_path(self, tmp_path: Path) -> None:
        """snitch.json lands under data/processed/shopify_variants/snitch.json."""
        mock_client = _make_gcs_mock()

        with _GCSPatch(mock_client):
            download_supplementary_assets(
                uri="gs://asa-demo-indices/",
                brand="snitch",
                repo_root=tmp_path,
            )

        bucket = mock_client.bucket.return_value
        called_blob_names = [call.args[0] for call in bucket.blob.call_args_list]

        assert any("shopify_variants/snitch.json" in n for n in called_blob_names)

        variants_dir = tmp_path / "data" / "processed" / "shopify_variants"
        assert variants_dir.is_dir()

    def test_download_to_filename_called_for_each_asset(self, tmp_path: Path) -> None:
        """download_to_filename must be called once per expected blob (3 total)."""
        mock_client = _make_gcs_mock()

        with _GCSPatch(mock_client):
            download_supplementary_assets(
                uri="gs://asa-demo-indices/",
                brand="hm",
                repo_root=tmp_path,
            )

        bucket = mock_client.bucket.return_value
        # bucket.blob() is a side_effect function, so each call returns a new MagicMock.
        # The side_effect results are tracked via bucket.blob.side_effect, not return_value.
        # Instead, count the number of blob() invocations — each must trigger one download.
        total_blob_calls = bucket.blob.call_count
        # 2 CLIP files + 1 variant file = 3 blob() calls
        assert total_blob_calls == 3

    def test_missing_clip_faiss_does_not_raise(self, tmp_path: Path) -> None:
        """A missing clip.faiss blob must log a warning and NOT raise an exception."""
        mock_client = _make_gcs_mock(clip_faiss_missing=True)

        with _GCSPatch(mock_client):
            # Must not raise
            download_supplementary_assets(
                uri="gs://asa-demo-indices/",
                brand="myntra",
                repo_root=tmp_path,
            )

    def test_missing_clip_ids_does_not_raise(self, tmp_path: Path) -> None:
        """A missing clip_article_ids.npy blob must not raise an exception."""
        mock_client = _make_gcs_mock(clip_ids_missing=True)

        with _GCSPatch(mock_client):
            download_supplementary_assets(
                uri="gs://asa-demo-indices/",
                brand="myntra",
                repo_root=tmp_path,
            )

    def test_missing_variant_map_does_not_raise(self, tmp_path: Path) -> None:
        """A missing shopify_variants blob must not raise an exception."""
        mock_client = _make_gcs_mock(variants_missing=True)

        with _GCSPatch(mock_client):
            download_supplementary_assets(
                uri="gs://asa-demo-indices/",
                brand="snitch",
                repo_root=tmp_path,
            )

    def test_all_blobs_missing_does_not_raise(self, tmp_path: Path) -> None:
        """When every supplementary blob is absent, startup must still complete."""
        mock_client = _make_gcs_mock(
            clip_faiss_missing=True,
            clip_ids_missing=True,
            variants_missing=True,
        )

        with _GCSPatch(mock_client):
            download_supplementary_assets(
                uri="gs://asa-demo-indices/",
                brand="snitch",
                repo_root=tmp_path,
            )

    def test_uri_with_path_prefix_builds_correct_gcs_paths(self, tmp_path: Path) -> None:
        """When the URI has a non-empty path (e.g. gs://bucket/prod/), the prefix
        is correctly prepended to each blob path.
        """
        mock_client = _make_gcs_mock()

        with _GCSPatch(mock_client):
            download_supplementary_assets(
                uri="gs://asa-demo-indices/prod/",
                brand="hm",
                repo_root=tmp_path,
            )

        bucket = mock_client.bucket.return_value
        called_blob_names = [call.args[0] for call in bucket.blob.call_args_list]

        # prefix "prod" must appear in each blob path
        assert all("prod/" in n for n in called_blob_names), (
            f"Expected 'prod/' prefix in all blob names, got: {called_blob_names}"
        )
        assert any("prod/clip/hm/clip.faiss" in n for n in called_blob_names)
        assert any("prod/shopify_variants/hm.json" in n for n in called_blob_names)

    def test_correct_bucket_name_used(self, tmp_path: Path) -> None:
        """The bucket name extracted from the URI must match the one passed to client.bucket()."""
        mock_client = _make_gcs_mock()

        with _GCSPatch(mock_client):
            download_supplementary_assets(
                uri="gs://my-custom-bucket/",
                brand="hm",
                repo_root=tmp_path,
            )

        mock_client.bucket.assert_called_with("my-custom-bucket")
