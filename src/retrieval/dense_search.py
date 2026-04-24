import numpy as np
import faiss
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(self, config: dict):
        self.config = config
        model_name = config["retrieval"]["dense_model"]
        self.model = SentenceTransformer(model_name, device="cpu")
        self.index: faiss.IndexFlatIP | None = None
        self.article_ids: np.ndarray | None = None

    def build_index(self, catalogue_df: pd.DataFrame, save_dir: Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        texts = catalogue_df["search_text"].tolist()
        batch_size = self.config["retrieval"]["dense_batch_size"]

        print(f"Encoding {len(texts):,} articles with {self.config['retrieval']['dense_model']}...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        embeddings = embeddings.astype(np.float32)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.article_ids = catalogue_df["article_id"].values.astype(str)

        faiss.write_index(self.index, str(save_dir / "dense.faiss"))
        np.save(str(save_dir / "dense_article_ids.npy"), self.article_ids)
        print(f"Dense index saved: {self.index.ntotal:,} vectors, dim={dim}")

    @classmethod
    def load(cls, config: dict, save_dir: Path) -> "DenseRetriever":
        save_dir = Path(save_dir)
        instance = cls.__new__(cls)
        instance.config = config
        model_name = config["retrieval"]["dense_model"]
        instance.model = SentenceTransformer(model_name, device="cpu")
        instance.index = faiss.read_index(str(save_dir / "dense.faiss"))
        instance.article_ids = np.load(str(save_dir / "dense_article_ids.npy"), allow_pickle=True)
        return instance

    def search_by_id(self, article_id: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Return items similar to article_id using its stored FAISS embedding."""
        pos_arr = np.where(self.article_ids == article_id)[0]
        if len(pos_arr) == 0:
            return []
        pos = int(pos_arr[0])
        vec = self.index.reconstruct(pos).reshape(1, -1)
        scores, indices = self.index.search(vec, top_k + 1)  # +1 to skip seed itself
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                aid = str(self.article_ids[idx])
                if aid != article_id:
                    results.append((aid, float(score)))
        return results[:top_k]

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        query_vec = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((str(self.article_ids[idx]), float(score)))
        return results
