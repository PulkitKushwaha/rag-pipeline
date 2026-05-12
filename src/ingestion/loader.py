import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
 
 
@dataclass
class Document:
    """
    Represents a loaded document before chunking.
 
    Attributes:
        content  : Raw text content of the document
        metadata : Source, file type, page number, and custom tags
        doc_id   : Unique identifier for this document
    """
    content: str
    metadata: Dict[str, Any]
    doc_id: str
 
 
class DocumentLoader:
    """
    Loads documents from disk into Document objects.
 
    Supports PDF, TXT, and DOCX formats. Extracts metadata
    (filename, file type, page count) alongside content.
 
    Design decision: We separate loading from chunking deliberately.
    Loading produces full-document text. Chunking is a separate
    concern that operates on that text. This makes it easy to swap
    chunking strategies without touching the loading logic.
 
    Usage:
        loader = DocumentLoader()
        docs = loader.load_directory("data/sample_docs/")
        for doc in docs:
            print(doc.metadata["source"], len(doc.content))
    """
 
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
 
    def load_file(self, file_path: str) -> Optional[Document]:
        """
        Load a single file into a Document object.
 
        Args:
            file_path: Path to the file to load.
 
        Returns:
            Document if successful, None if unsupported format.
 
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(file_path)
 
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
 
        extension = path.suffix.lower()
 
        if extension not in self.SUPPORTED_EXTENSIONS:
            print(f"Unsupported file type: {extension}. Skipping {path.name}")
            return None
 
        if extension == ".pdf":
            return self._load_pdf(path)
        elif extension in {".txt", ".md"}:
            return self._load_text(path)
 
    def load_directory(
        self,
        directory_path: str,
        recursive: bool = False
    ) -> List[Document]:
        """
        Load all supported files from a directory.
 
        Args:
            directory_path : Path to the directory.
            recursive      : If True, also loads files in subdirectories.
 
        Returns:
            List of Document objects, one per loaded file.
        """
        dir_path = Path(directory_path)
 
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
 
        pattern = "**/*" if recursive else "*"
        files = [
            f for f in dir_path.glob(pattern)
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]
 
        documents = []
        for file_path in files:
            doc = self.load_file(str(file_path))
            if doc:
                documents.append(doc)
 
        print(f"Loaded {len(documents)} document(s) from {directory_path}")
        return documents
 
    def _load_pdf(self, path: Path) -> Document:
        """
        Load a PDF file using pypdf.
 
        Extracts text from all pages and joins them.
        Preserves page-level metadata for later filtering.
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf is required for PDF loading. "
                "Install it with: pip install pypdf"
            )
 
        reader = PdfReader(str(path))
        pages_content = []
 
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages_content.append(text)
 
        full_content = "\n\n".join(pages_content)
 
        return Document(
            content=full_content,
            metadata={
                "source": str(path),
                "filename": path.name,
                "file_type": "pdf",
                "page_count": len(reader.pages),
                "file_size_bytes": path.stat().st_size,
            },
            doc_id=f"doc_{path.stem}"
        )
 
    def _load_text(self, path: Path) -> Document:
        """
        Load a plain text or markdown file.
        """
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
 
        return Document(
            content=content,
            metadata={
                "source": str(path),
                "filename": path.name,
                "file_type": path.suffix.lower().strip("."),
                "file_size_bytes": path.stat().st_size,
            },
            doc_id=f"doc_{path.stem}"
        )
