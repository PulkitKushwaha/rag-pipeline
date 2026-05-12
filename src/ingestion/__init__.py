# Ingestion module
# Handles document loading, parsing, and chunking strategies
# Supports: PDF, TXT, DOCX, HTML
# Chunking strategies: fixed, recursive, semantic, sentence-window
 
from src.ingestion.loader import DocumentLoader, Document
from src.ingestion.metadata import DocumentMetadata, enrich_metadata
 
__all__ = ["DocumentLoader", "Document", "DocumentMetadata", "enrich_metadata"]
