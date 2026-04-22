import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from rank_bm25 import BM25Okapi


class SparseRetriever:
    def __init__(self, config: dict):
        self.config = config
        self.bm25: BM25Okapi | None = None
        self.article_ids: np.ndarray | None = None

    def build_index(self, catalogue_df: pd.DataFrame, save_dir: Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Building BM25 index over {len(catalogue_df):,} articles...")
        tokenized = [self._tokenize(t) for t in catalogue_df["search_text"].tolist()]
        self.bm25 = BM25Okapi(tokenized)
        self.article_ids = catalogue_df["article_id"].values.astype(str)

        with open(save_dir / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        np.save(str(save_dir / "bm25_article_ids.npy"), self.article_ids)
        print("BM25 index saved.")

    @classmethod
    def load(cls, config: dict, save_dir: Path) -> "SparseRetriever":
        save_dir = Path(save_dir)
        instance = cls.__new__(cls)
        instance.config = config
        with open(save_dir / "bm25.pkl", "rb") as f:
            instance.bm25 = pickle.load(f)
        instance.article_ids = np.load(
            str(save_dir / "bm25_article_ids.npy"), allow_pickle=True
        )
        return instance

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (str(self.article_ids[i]), float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return [t for t in tokens if len(t) >= 2]
