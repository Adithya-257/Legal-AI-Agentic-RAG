"""
retriever.py

Retrieves relevant chunks from ChromaDB vector databases.
"""

from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer


class DatabaseRetriever:
    """Retrieve relevant chunks from multiple vector databases"""

    def __init__(self, vector_dbs_path: str = "Vector_DBs"):

        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.vector_dbs_path = vector_dbs_path

        self.db_paths = {
            "db_a_defs": f"{vector_dbs_path}/db_a_defs",
            "db_b_risks": f"{vector_dbs_path}/db_b_risks",
            "db_c_standards": f"{vector_dbs_path}/db_c_bprac",
            "db_d_summary": f"{vector_dbs_path}/db_d_summary"
        }

    def retrieve(
        self,
        query: str,
        databases: List[str],
        top_k: int = 5
    ) -> Dict[str, List[Dict]]:
        """Retrieve top-k relevant chunks from selected databases."""

        results = {}

        query_embedding = self.model.encode(query).tolist()

        for db_id in databases:

            db_path = self.db_paths.get(db_id)

            if not db_path:
                continue

            try:
                chunks = self._query_single_db(
                    db_path,
                    query_embedding,
                    top_k
                )

                results[db_id] = chunks

            except Exception:
                results[db_id] = []

        return results

    def _query_single_db(
        self,
        db_path: str,
        query_embedding: List[float],
        top_k: int
    ) -> List[Dict]:
        """Query a single ChromaDB collection."""

        client = chromadb.PersistentClient(path=db_path)

        collection = client.get_collection("legal_docs")

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        chunks = []

        for i in range(len(docs)):
            chunks.append({
                "text": docs[i],
                "metadata": metas[i],
                "similarity": 1 - dists[i]
            })

        return chunks
