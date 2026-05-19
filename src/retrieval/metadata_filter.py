 
"""
Metadata-Based Retrieval Filtering
 
Why metadata filtering matters:
    Vector similarity retrieves semantically similar chunks
    but similarity alone doesn't enforce access control, source
    restrictions, or temporal relevance.
 
    Example problems without metadata filtering:
        - User A's private documents retrieved for User B's query
        - Outdated policy documents retrieved alongside current ones
        - Internal documents surfaced in public-facing chatbot
        - All departments' data mixed in a department-specific bot
 
    Metadata filtering solves these by pre-filtering the candidate
    pool before or during vector search: only chunks matching
    the filter criteria are considered.
 
Two filtering approaches:
    Pre-filtering  — Filter at the vector store level (faster but
                     requires vector store support for metadata)
    Post-filtering — Filter results after retrieval (always works,
                     slightly less efficient due to over-fetching)
 
    This implementation uses post-filtering via FAISSVectorStore's
    built-in metadata_filter parameter. It is clean and works with any
    vector store that exposes metadata.
"""
 
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from src.ingestion.chunker import Chunk
 
 
@dataclass
class FilterCondition:
    """
    A single metadata filter condition.
 
    Supports exact match, list membership, range comparisons,
    and existence checks.
 
    Examples:
        FilterCondition("access_level", "eq", "public")
        FilterCondition("department", "in", ["HR", "Legal"])
        FilterCondition("year", "gte", 2023)
        FilterCondition("tags", "contains", "policy")
    """
    field: str
    operator: str  # eq, neq, in, not_in, gte, lte, gt, lt, contains, exists
    value: Any
 
    def matches(self, metadata: Dict[str, Any]) -> bool:
        """Check if metadata satisfies this condition."""
        field_value = metadata.get(self.field)
 
        if self.operator == "exists":
            return field_value is not None
 
        if field_value is None:
            return False
 
        if self.operator == "eq":
            return field_value == self.value
        elif self.operator == "neq":
            return field_value != self.value
        elif self.operator == "in":
            return field_value in self.value
        elif self.operator == "not_in":
            return field_value not in self.value
        elif self.operator == "gte":
            return field_value >= self.value
        elif self.operator == "lte":
            return field_value <= self.value
        elif self.operator == "gt":
            return field_value > self.value
        elif self.operator == "lt":
            return field_value < self.value
        elif self.operator == "contains":
            if isinstance(field_value, list):
                return self.value in field_value
            return self.value in str(field_value)
        else:
            raise ValueError(f"Unknown operator: {self.operator}")
 
 
class MetadataFilter:
    """
    Composable metadata filter for retrieval.
 
    Combines multiple FilterConditions with AND/OR logic.
    Applied post-retrieval to filter chunks by metadata.
 
    Usage:
        # Simple exact match
        filter = MetadataFilter.from_dict({
            "access_level": "public",
            "department": "HR"
        })
 
        # Complex conditions
        filter = MetadataFilter(conditions=[
            FilterCondition("access_level", "in", ["public", "internal"]),
            FilterCondition("year", "gte", 2023),
        ], logic="AND")
 
        # Apply to chunks
        filtered = filter.apply(chunks)
    """
 
    def __init__(
        self,
        conditions: List[FilterCondition],
        logic: str = "AND"
    ):
        """
        Args:
            conditions : List of FilterCondition objects
            logic      : "AND" (all must match) or "OR" (any must match)
        """
        if logic not in ("AND", "OR"):
            raise ValueError(f"Logic must be 'AND' or 'OR', got '{logic}'")
        self.conditions = conditions
        self.logic = logic
 
    @classmethod
    def from_dict(
        cls,
        filter_dict: Dict[str, Any],
        logic: str = "AND"
    ) -> "MetadataFilter":
        """
        Create a MetadataFilter from a simple key-value dict.
 
        All conditions use exact match (eq) operator.
        Values that are lists use 'in' operator automatically.
 
        Args:
            filter_dict : Dict of {field: value} pairs
            logic       : "AND" or "OR"
 
        Example:
            MetadataFilter.from_dict({
                "access_level": "public",
                "department": ["HR", "Legal"]
            })
        """
        conditions = []
        for field, value in filter_dict.items():
            if isinstance(value, list):
                conditions.append(FilterCondition(field, "in", value))
            else:
                conditions.append(FilterCondition(field, "eq", value))
        return cls(conditions=conditions, logic=logic)
 
    def apply(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Filter a list of chunks by this filter's conditions.
 
        Args:
            chunks: List of Chunk objects with metadata
 
        Returns:
            Filtered list of chunks that satisfy the conditions
        """
        if not self.conditions:
            return chunks
 
        filtered = []
        for chunk in chunks:
            if self._matches(chunk.metadata):
                filtered.append(chunk)
 
        return filtered
 
    def _matches(self, metadata: Dict[str, Any]) -> bool:
        """Check if metadata satisfies all/any conditions."""
        if self.logic == "AND":
            return all(c.matches(metadata) for c in self.conditions)
        else:  # OR
            return any(c.matches(metadata) for c in self.conditions)
 
    def to_dict(self) -> Dict[str, Any]:
        """Convert to simple dict for FAISSVectorStore metadata_filter."""
        if self.logic != "AND":
            raise ValueError(
                "FAISSVectorStore only supports AND logic for metadata_filter dict. "
                "Use apply() directly for OR logic."
            )
        result = {}
        for condition in self.conditions:
            if condition.operator == "eq":
                result[condition.field] = condition.value
            elif condition.operator == "in":
                result[condition.field] = condition.value
        return result
 
    def __repr__(self) -> str:
        conditions_str = f" {self.logic} ".join(
            f"{c.field} {c.operator} {c.value}"
            for c in self.conditions
        )
        return f"MetadataFilter({conditions_str})"
 
 
# Common pre-built filters for enterprise RAG use cases
 
class AccessLevelFilter:
    """Pre-built filter for RBAC-based access control."""
 
    ACCESS_HIERARCHY = {
        "public": 0,
        "internal": 1,
        "confidential": 2,
        "restricted": 3
    }
 
    @classmethod
    def for_user(cls, user_access_level: str) -> MetadataFilter:
        """
        Create a filter that allows access to documents at or below
        the user's access level.
 
        Args:
            user_access_level: User's maximum access level
 
        Returns:
            MetadataFilter allowing appropriate document access
        """
        max_level = cls.ACCESS_HIERARCHY.get(user_access_level, 0)
        allowed_levels = [
            level for level, value in cls.ACCESS_HIERARCHY.items()
            if value <= max_level
        ]
        return MetadataFilter.from_dict({"access_level": allowed_levels})
 
 
class DepartmentFilter:
    """Pre-built filter for department-specific retrieval."""
 
    @classmethod
    def for_departments(cls, departments: List[str]) -> MetadataFilter:
        return MetadataFilter.from_dict({"department": departments})
 
    @classmethod
    def all_except(cls, excluded_departments: List[str]) -> MetadataFilter:
        return MetadataFilter(
            conditions=[FilterCondition("department", "not_in", excluded_departments)]
        )
 
 
class RecencyFilter:
    """Pre-built filter for time-based document filtering."""
 
    @classmethod
    def since_year(cls, year: int) -> MetadataFilter:
        return MetadataFilter(
            conditions=[FilterCondition("year", "gte", year)]
        )
 
    @classmethod
    def current_version_only(cls) -> MetadataFilter:
        return MetadataFilter.from_dict({"is_current": True})
