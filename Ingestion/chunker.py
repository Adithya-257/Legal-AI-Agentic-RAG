"""
chunker.py

Splits documents into overlapping text chunks for embedding.
"""

from typing import List, Dict


class DocumentChunker:

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Chunk a list of loaded documents."""
        chunks = []

        for doc in documents:
            text = doc["content"]
            parts = self._split_text(text)

            for i, part in enumerate(parts):
                chunks.append({
                    "text": part,
                    "metadata": {
                        "source": doc["filename"],
                        "chunk_id": i,
                        "total_chunks": len(parts),
                        "num_pages": doc["num_pages"]
                    }
                })

        return chunks

    def _split_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            start = end - self.overlap

        return chunks
