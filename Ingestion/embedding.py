"""
embedding.py
Creates embeddings for document chunks and stores them in ChromaDB.
"""

from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm


class Embedder:
    """Embed text chunks and store them in a ChromaDB vector database"""

    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("Model loaded")

    def embed_chunks(self, chunks: List[Dict], db_path: str, batch_size: int = 64):
        """
        Generate embeddings and store them in ChromaDB

        Args:
            chunks: Output from DocumentChunker
            db_path: Path to vector database folder
            batch_size: Batch size for embedding generation
        """

        print(f"\nCreating vector database at: {db_path}")
        os.makedirs(db_path, exist_ok=True)

        client = chromadb.PersistentClient(path=db_path)

        collection_name = "legal_docs"

        existing = [c.name for c in client.list_collections()]
        if collection_name in existing:
            client.delete_collection(collection_name)

        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        texts = [chunk["text"] for chunk in chunks]

        print("Generating embeddings...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        print("Storing embeddings in database...")

        for i in tqdm(range(0, len(chunks), batch_size)):

            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            ids = []
            documents = []
            metadatas = []
            embeds = []

            for j, chunk in enumerate(batch_chunks):
                chunk_id = f"{chunk['metadata']['source']}_{chunk['metadata']['chunk_id']}"

                ids.append(chunk_id)
                documents.append(chunk["text"])
                metadatas.append(chunk["metadata"])
                embeds.append(batch_embeddings[j].tolist())

            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeds
            )

        print(f"\nDatabase created with {len(chunks)} embeddings")
        print(f"Saved at: {db_path}")


# Example pipeline
if __name__ == "__main__":

    from data_loader import DocumentLoader
    from chunker import DocumentChunker

    # Step 1: Load documents
    loader = DocumentLoader(folder_path="DATA/DB-A")
    documents = loader.load_documents()

    # Step 2: Chunk documents
    chunker = DocumentChunker(chunk_size=500, overlap=100)
    chunks = chunker.chunk_documents(documents)

    # Step 3: Create vector database
    embedder = Embedder()

    embedder.embed_chunks(
        chunks,
        db_path="Vector_DBs/db_a_defs",
        batch_size=64
    )

    print("\nVector database ready.")
