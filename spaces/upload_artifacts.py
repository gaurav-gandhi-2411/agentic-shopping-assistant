"""
Upload pre-built retrieval artifacts to a HuggingFace Space repo.

Usage:
    # Upload original 20k artifacts (1:1 filenames):
    python spaces/upload_artifacts.py --repo <user>/<space>

    # Upload Space-optimised 1800-item subset + images:
    python spaces/upload_artifacts.py --repo <user>/<space> --space

The --space flag maps local _space-suffixed files to the plain names the
app expects (dense.faiss, bm25.pkl, catalogue.parquet) and also uploads
the resized images/ folder.

Requires:
    pip install huggingface-hub
    huggingface-cli login   (or set HF_TOKEN env var)
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


_REPO_ROOT = Path(__file__).parent.parent
_LOCAL_DIR = _REPO_ROOT / "data" / "processed"
_REMOTE_PREFIX = "data/processed"
_DB_MODEL_DIR = _REPO_ROOT / "models" / "distilbert_router"

# Default (20k): local filename -> remote path (same name)
_DEFAULT_ARTIFACTS = [
    ("dense.faiss",           f"{_REMOTE_PREFIX}/dense.faiss"),
    ("dense_article_ids.npy", f"{_REMOTE_PREFIX}/dense_article_ids.npy"),
    ("bm25.pkl",              f"{_REMOTE_PREFIX}/bm25.pkl"),
    ("bm25_article_ids.npy",  f"{_REMOTE_PREFIX}/bm25_article_ids.npy"),
    ("catalogue.parquet",     f"{_REMOTE_PREFIX}/catalogue.parquet"),
]

# Space mode (1800-item subset): local _space files -> plain remote names
_SPACE_ARTIFACTS = [
    ("dense_space.faiss",           f"{_REMOTE_PREFIX}/dense.faiss"),
    ("dense_space_article_ids.npy", f"{_REMOTE_PREFIX}/dense_article_ids.npy"),
    ("bm25_space.pkl",              f"{_REMOTE_PREFIX}/bm25.pkl"),
    ("bm25_space_article_ids.npy",  f"{_REMOTE_PREFIX}/bm25_article_ids.npy"),
    ("catalogue_space.parquet",     f"{_REMOTE_PREFIX}/catalogue.parquet"),
]


_DB_SMALL_FILES = [
    "config.json", "tokenizer.json", "tokenizer_config.json", "training_log.json"
]


def upload_model(repo_id: str, token: str | None = None, small_only: bool = False) -> None:
    """Upload DistilBERT router model to the Space.

    --model            : full upload (model.safetensors + small files, requires LFS quota)
    --model --small    : config + tokenizer only (no binary, for storage-constrained Spaces)
    """
    if not _DB_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"{_DB_MODEL_DIR} not found — run scripts/train_router_distilbert.py first"
        )
    api = HfApi(token=token)

    if small_only:
        print("Uploading DistilBERT config + tokenizer files only (no model binary) ...")
        for fname in _DB_SMALL_FILES:
            local = _DB_MODEL_DIR / fname
            if not local.exists():
                print(f"  SKIP {fname} (not found)")
                continue
            remote = f"models/distilbert_router/{fname}"
            print(f"  {fname} -> {remote} ...", end=" ", flush=True)
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=remote,
                repo_id=repo_id,
                repo_type="space",
            )
            print("done")
    else:
        print(f"Uploading full DistilBERT model to {repo_id}/models/distilbert_router/ ...")
        api.upload_folder(
            folder_path=str(_DB_MODEL_DIR),
            path_in_repo="models/distilbert_router",
            repo_id=repo_id,
            repo_type="space",
        )
    print(f"\nModel files uploaded to https://huggingface.co/spaces/{repo_id}")
    if small_only:
        print("NOTE: model.safetensors was NOT uploaded. DistilBERT router will be unavailable")
        print("      in the Space until the full model file is present. Local dev works normally.")


def upload(repo_id: str, space_mode: bool = False, token: str | None = None) -> None:
    api = HfApi(token=token)
    artifacts = _SPACE_ARTIFACTS if space_mode else _DEFAULT_ARTIFACTS

    for local_name, remote_path in artifacts:
        local_path = _LOCAL_DIR / local_name
        if not local_path.exists():
            raise FileNotFoundError(
                f"{local_path} not found — run "
                f"{'scripts/03_build_image_subset.py' if space_mode else 'scripts/01_build_retrieval.py'} first"
            )
        print(f"Uploading {local_name} -> {repo_id}/{remote_path} ...", end=" ", flush=True)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="space",
        )
        print("done")

    if space_mode:
        images_dir = _LOCAL_DIR / "images"
        if not images_dir.exists():
            raise FileNotFoundError(
                f"{images_dir} not found — run scripts/03_build_image_subset.py first"
            )
        n_images = sum(1 for _ in images_dir.rglob("*.jpg"))
        print(f"Uploading images/ folder ({n_images} files) -> {repo_id}/{_REMOTE_PREFIX}/images/ ...")
        api.upload_folder(
            folder_path=str(images_dir),
            path_in_repo=f"{_REMOTE_PREFIX}/images",
            repo_id=repo_id,
            repo_type="space",
        )
        print("done")

    print(f"\nAll artifacts uploaded to https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload retrieval artifacts to HF Space")
    parser.add_argument(
        "--repo", required=True,
        help="HuggingFace Space repo ID, e.g. your-username/agentic-shopping-assistant",
    )
    parser.add_argument(
        "--space", action="store_true",
        help="Upload Space-optimised 1800-item subset + images (uses _space-suffixed local files)",
    )
    parser.add_argument(
        "--model", action="store_true",
        help="Upload DistilBERT router model (models/distilbert_router/) to the Space",
    )
    parser.add_argument(
        "--small", action="store_true",
        help="With --model: upload config/tokenizer only, skip model.safetensors (for storage-limited Spaces)",
    )
    parser.add_argument(
        "--token", default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token (defaults to HF_TOKEN env var or cached login)",
    )
    args = parser.parse_args()
    if args.model:
        upload_model(args.repo, token=args.token, small_only=args.small)
    else:
        upload(args.repo, space_mode=args.space, token=args.token)
