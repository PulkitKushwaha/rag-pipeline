from typing import Dict, Any, List
from dataclasses import dataclass
 
 
@dataclass
class DocumentMetadata:
    """
    Structured metadata for a document.
 
    Used for metadata-based filtering at retrieval time.
    Every chunk inherits its parent document's metadata.
 
    Design decision: Metadata is attached at ingestion time,
    not retrieval time. This means filtering is fast. We
    filter before vector search, not after.
    """
    source: str
    filename: str
    file_type: str
    access_level: str = "internal"      # public / internal / confidential / restricted
    department: str = "general"          # HR / Legal / Finance / Engineering / General
    doc_type: str = "document"           # policy / contract / report / guide / other
    language: str = "en"
    tags: List[str] = None
 
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "filename": self.filename,
            "file_type": self.file_type,
            "access_level": self.access_level,
            "department": self.department,
            "doc_type": self.doc_type,
            "language": self.language,
            "tags": self.tags or [],
        }
 
 
def enrich_metadata(
    base_metadata: Dict[str, Any],
    access_level: str = "internal",
    department: str = "general",
    doc_type: str = "document",
    tags: List[str] = None
) -> Dict[str, Any]:
    """
    Enrich base metadata from DocumentLoader with access control
    and classification fields.
 
    This is called after loading, before chunking, so all chunks
    from this document inherit the enriched metadata.
 
    Args:
        base_metadata : Metadata dict from DocumentLoader
        access_level  : Access control level (public/internal/confidential/restricted)
        department    : Owning department
        doc_type      : Type of document
        tags          : Optional list of custom tags
 
    Returns:
        Enriched metadata dict ready for vector store indexing.
    """
    return {
        **base_metadata,
        "access_level": access_level,
        "department": department,
        "doc_type": doc_type,
        "tags": tags or [],
    }
