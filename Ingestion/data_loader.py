"""
data_loader.py

Loads PDF and TXT documents from a folder.
"""

from pathlib import Path
from typing import List, Dict
import PyPDF2


class DocumentLoader:
    def __init__(self, folder_path: str):
        self.folder = Path(folder_path)

        if not self.folder.exists():
            raise ValueError(f"Folder not found: {folder_path}")

    def load_documents(self) -> List[Dict]:
        """Load all PDF and TXT files in the folder."""
        documents = []

        for file in self.folder.iterdir():

            if file.suffix.lower() == ".pdf":
                documents.append(self._load_pdf(file))

            elif file.suffix.lower() == ".txt":
                text = file.read_text(encoding="utf-8", errors="ignore")

                if text.strip():
                    documents.append({
                        "filename": file.name,
                        "content": text,
                        "num_pages": 1,
                        "file_type": "txt"
                    })

        return documents

    def _load_pdf(self, file: Path) -> Dict:
        """Extract text from a PDF."""
        with open(file, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)

        return {
            "filename": file.name,
            "content": text,
            "num_pages": len(reader.pages),
            "file_type": "pdf"
        }
