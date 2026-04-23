"""
Upload pre-built retrieval artifacts to a HuggingFace Space repo.

Usage:
    python spaces/upload_artifacts.py --repo <your-hf-username>/<space-name>

The script uploads these five files from data/processed/ to the Space:
    dense.faiss
    dense_article_ids.npy
    bm25.pkl
    bm25_article_ids.npy
    catalogue.parquet

They land at data/processed/<filename> inside the Space repo so that
spaces/app.py can load them with the same relative path it uses locally.

Requires:
    pip install huggingface-hub
    huggingface-cli login   (or set HF_TOKEN env var)
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi


_ARTIFACTS = [
    "dense.faiss",
    "dense_article_ids.npy",
    "bm25.pkl",
    "bm25_article_ids.npy",
    "catalogue.parquet",
]

_REPO_ROOT = Path(__file__).parent.parent
_LOCAL_DIR = _REPO_ROOT / "data" / "processed"
_SPACE_PATH_PREFIX = "data/processed"


def upload(repo_id: str, token: str | None = None) -> None:
    api = HfApi(token=token)
    for filename in _ARTIFACTS:
        local_path = _LOCAL_DIR / filename
        if not local_path.exists():
            raise FileNotFoundError(
                f"{local_path} not found — run scripts/01_build_retrieval.py first"
            )
        remote_path = f"{_SPACE_PATH_PREFIX}/{filename}"
        print(f"Uploading {filename} → {repo_id}/{remote_path} …", end=" ", flush=True)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="space",
        )
        print("done")
    print(f"\nAll artifacts uploaded to https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(description="Upload retrieval artifacts to HF Space")
    parser.add_argument(
        "--repo",
        required=True,
        help="HuggingFace Space repo ID, e.g. your-username/agentic-shopping-assistant",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token (defaults to HF_TOKEN env var or cached login)",
    )
    args = parser.parse_args()
    upload(args.repo, args.token)
