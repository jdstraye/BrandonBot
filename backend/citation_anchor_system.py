"""
Citation Anchor System for BrandonBot

Implements Citation Verification per specification:

1. Preprocessing (Anchor Injection):
   - Assign rich metadata to every chunk
   - Store: document_id, page_number, paragraph_id, unique citation anchor
   - Inject anchor directly into chunk text before embedding

2. LLM Prompting:
   - Strict instruction requiring citation anchors for factual claims
   - Format: [CITE-BP-xxx], [CITE-QA-xxx], [CITE-WEB-xxx]

3. Verification:
   - Extract all citation anchors using regex
   - Lookup each anchor in metadata store
   - Flag missing or invalid anchors
"""

import logging
import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CitationMetadata:
    """Metadata for a citation anchor."""
    anchor: str
    document_id: str
    page_number: int
    paragraph_id: int
    line_number: Optional[int] = None
    content_snippet: str = ""
    source_type: str = "BP"
    source_file: str = ""
    created_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "anchor": self.anchor,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "paragraph_id": self.paragraph_id,
            "line_number": self.line_number,
            "content_snippet": self.content_snippet,
            "source_type": self.source_type,
            "source_file": self.source_file,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CitationMetadata":
        return cls(**data)


@dataclass
class CitationVerificationResult:
    """Result from citation verification."""
    valid: bool
    anchors_found: List[str] = field(default_factory=list)
    anchors_resolved: List[str] = field(default_factory=list)
    anchors_unresolved: List[str] = field(default_factory=list)
    format_issues: List[str] = field(default_factory=list)
    missing_citations: List[str] = field(default_factory=list)
    score: int = 0
    explanation: str = ""


class CitationAnchorStore:
    """
    In-memory store for citation anchors and metadata.
    
    Maps anchor IDs to their source metadata for resolution.
    """
    
    def __init__(self, persistence_path: Optional[str] = None):
        self._store: Dict[str, CitationMetadata] = {}
        self._persistence_path = persistence_path
        self._counter = {"BP": 0, "QA": 0, "WEB": 0, "HISTORY": 0}
        
        if persistence_path:
            self._load_from_disk()
    
    def _load_from_disk(self):
        """Load citation store from disk."""
        try:
            path = Path(self._persistence_path)
            if path.exists():
                with open(path, "r") as f:
                    data = json.load(f)
                    for anchor, meta in data.get("citations", {}).items():
                        self._store[anchor] = CitationMetadata.from_dict(meta)
                    self._counter = data.get("counter", self._counter)
                logger.info(f"Loaded {len(self._store)} citations from {path}")
        except Exception as e:
            logger.error(f"Failed to load citation store: {e}")
    
    def _save_to_disk(self):
        """Save citation store to disk."""
        if not self._persistence_path:
            return
        
        try:
            path = Path(self._persistence_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "citations": {k: v.to_dict() for k, v in self._store.items()},
                "counter": self._counter
            }
            
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save citation store: {e}")
    
    def generate_anchor(
        self,
        source_type: str,
        document_id: str,
        page_number: int = 1,
        paragraph_id: int = 1,
        line_number: Optional[int] = None,
        content_snippet: str = "",
        source_file: str = ""
    ) -> str:
        """
        Generate a new citation anchor and store its metadata.
        
        Args:
            source_type: BP (Brandon Platform), QA, WEB, or HISTORY
            document_id: Unique document identifier
            page_number: Page number in source document
            paragraph_id: Paragraph ID within page
            line_number: Optional line number
            content_snippet: First ~100 chars of content
            source_file: Original source file path
        
        Returns:
            Citation anchor string (e.g., "CITE-BP-001")
        """
        source_type = source_type.upper()
        if source_type not in self._counter:
            source_type = "BP"
        
        self._counter[source_type] += 1
        anchor_id = f"{source_type}-{self._counter[source_type]:03d}"
        anchor = f"CITE-{anchor_id}"
        
        from datetime import datetime
        
        metadata = CitationMetadata(
            anchor=anchor,
            document_id=document_id,
            page_number=page_number,
            paragraph_id=paragraph_id,
            line_number=line_number,
            content_snippet=content_snippet[:200] if content_snippet else "",
            source_type=source_type,
            source_file=source_file,
            created_at=datetime.now().isoformat()
        )
        
        self._store[anchor] = metadata
        self._save_to_disk()
        
        return anchor
    
    def inject_anchor(self, content: str, anchor: str) -> str:
        """
        Inject a citation anchor into content text.
        
        Per specification: Add anchor at the end of the chunk.
        
        Args:
            content: Original content text
            anchor: Citation anchor to inject
        
        Returns:
            Content with anchor appended
        """
        return f"{content.rstrip()} [{anchor}]"
    
    def resolve_anchor(self, anchor: str) -> Optional[CitationMetadata]:
        """
        Resolve a citation anchor to its metadata.
        
        Args:
            anchor: Citation anchor (with or without brackets)
        
        Returns:
            CitationMetadata if found, None otherwise
        """
        clean_anchor = anchor.strip("[]")
        
        if clean_anchor in self._store:
            return self._store[clean_anchor]
        
        for key in self._store.keys():
            if clean_anchor in key or key in clean_anchor:
                return self._store[key]
        
        return None
    
    def get_all_anchors(self) -> List[str]:
        """Get all anchor IDs in the store."""
        return list(self._store.keys())
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about stored citations."""
        stats = {
            "total": len(self._store),
            "by_type": {}
        }
        
        for meta in self._store.values():
            source_type = meta.source_type
            stats["by_type"][source_type] = stats["by_type"].get(source_type, 0) + 1
        
        return stats


class CitationVerifier:
    """
    Verifies citations in LLM responses.
    
    Implements the Output Validation stage:
    1. Extract all citation anchors from response
    2. Lookup each anchor in metadata store
    3. Check for sentences lacking required anchors
    4. Return validation result
    """
    
    ANCHOR_PATTERN = re.compile(r'\[CITE-(?:BP|QA|WEB|HISTORY)-([A-Z0-9]+)\]')
    
    INCOMPLETE_PATTERNS = [
        (re.compile(r'\[CITE\]'), "Incomplete [CITE] without reference"),
        (re.compile(r'\[CITE:\s*[\w-]+\]'), "Non-standard format [CITE: xxx]"),
        (re.compile(r'\[CITE-\d+\]'), "Invalid format [CITE-nnn] (placeholder)"),
        (re.compile(r'\[WEBCITE:\s*[\w\s]+\]'), "Non-standard WEBCITE format"),
    ]
    
    FACTUAL_CLAIM_PATTERNS = [
        re.compile(r'\b(?:approximately|about|around|estimated)?\s*\$?[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion|percent|%))\b'),
        re.compile(r'\bpopulation\b.*\b[\d.,]+\s*(?:billion|million)\b', re.I),
        re.compile(r'\bstudies show\b', re.I),
        re.compile(r'\baccording to\b', re.I),
        re.compile(r'\bresearch indicates\b', re.I),
        re.compile(r'\bdata shows\b', re.I),
    ]
    
    def __init__(self, anchor_store: Optional[CitationAnchorStore] = None):
        self._store = anchor_store or CitationAnchorStore()
    
    def set_store(self, anchor_store: CitationAnchorStore):
        """Set the citation anchor store."""
        self._store = anchor_store
    
    def verify(self, response: str) -> CitationVerificationResult:
        """
        Verify all citations in a response.
        
        Args:
            response: LLM response text to verify
        
        Returns:
            CitationVerificationResult with validation details
        """
        anchors_found = []
        anchors_resolved = []
        anchors_unresolved = []
        format_issues = []
        missing_citations = []
        
        for match in self.ANCHOR_PATTERN.finditer(response):
            full_anchor = match.group(0)
            anchor_id = match.group(1)
            anchors_found.append(full_anchor.strip("[]"))
            
            metadata = self._store.resolve_anchor(full_anchor)
            if metadata:
                anchors_resolved.append(full_anchor.strip("[]"))
            else:
                anchors_unresolved.append(full_anchor.strip("[]"))
        
        for pattern, issue_msg in self.INCOMPLETE_PATTERNS:
            if pattern.search(response):
                format_issues.append(issue_msg)
        
        sentences = re.split(r'[.!?]+', response)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            has_factual_claim = any(p.search(sentence) for p in self.FACTUAL_CLAIM_PATTERNS)
            has_citation = self.ANCHOR_PATTERN.search(sentence) is not None
            
            if has_factual_claim and not has_citation:
                missing_citations.append(sentence[:50] + "...")
        
        score, explanation = self._calculate_score(
            anchors_found, anchors_resolved, anchors_unresolved,
            format_issues, missing_citations
        )
        
        valid = score <= 2
        
        return CitationVerificationResult(
            valid=valid,
            anchors_found=anchors_found,
            anchors_resolved=anchors_resolved,
            anchors_unresolved=anchors_unresolved,
            format_issues=format_issues,
            missing_citations=missing_citations,
            score=score,
            explanation=explanation
        )
    
    def _calculate_score(
        self,
        anchors_found: List[str],
        anchors_resolved: List[str],
        anchors_unresolved: List[str],
        format_issues: List[str],
        missing_citations: List[str]
    ) -> Tuple[int, str]:
        """Calculate violation score based on citation issues."""
        
        if format_issues:
            if any("placeholder" in issue.lower() for issue in format_issues):
                return 5, f"Invalid placeholder citation: {format_issues[0]}"
            else:
                return 3, f"Format issue: {format_issues[0]}"
        
        if anchors_unresolved:
            return 4, f"Unresolved anchors: {', '.join(anchors_unresolved[:3])}"
        
        if missing_citations:
            count = len(missing_citations)
            if count >= 3:
                return 5, f"Multiple factual claims without citations ({count})"
            elif count >= 1:
                return 4, f"Factual claim without citation: {missing_citations[0]}"
        
        if anchors_resolved:
            return 0, f"All {len(anchors_resolved)} citations resolved"
        
        return 0, "No citation-requiring claims detected"


class CitationPromptBuilder:
    """
    Builds LLM prompts with citation requirements.
    
    Per specification: The prompt must include strict instruction
    requiring citation anchors for every factual sentence.
    """
    
    CITATION_INSTRUCTION = """
CITATION REQUIREMENTS:
You are REQUIRED to provide a citation anchor from the provided context for every single sentence of factual information you generate.

Citation Format: [CITE-{TYPE}-{ID}]
- Use the exact anchor provided in the context
- Do NOT generate information that lacks an anchor
- If you cannot cite a claim, use hedging language instead

Example:
- Correct: "Brandon supports a 10% tax reduction [CITE-BP-003]"
- Incorrect: "Brandon supports a 10% tax reduction" (missing citation)
"""
    
    @classmethod
    def build_prompt(
        cls,
        query: str,
        context_chunks: List[Dict[str, Any]],
        base_instruction: str = ""
    ) -> str:
        """
        Build a prompt with citation requirements and context.
        
        Args:
            query: User query
            context_chunks: RAG context chunks with anchors
            base_instruction: Base system instruction
        
        Returns:
            Complete prompt with citation requirements
        """
        context_text = ""
        for chunk in context_chunks:
            content = chunk.get("content", "")
            anchor = chunk.get("anchor", "")
            source = chunk.get("source", "")
            
            if anchor:
                context_text += f"\n[{anchor}] {content}\nSource: {source}\n"
            else:
                context_text += f"\n{content}\nSource: {source}\n"
        
        prompt = f"""{base_instruction}

{cls.CITATION_INSTRUCTION}

CONTEXT (use these anchors in your response):
{context_text}

USER QUESTION: {query}

RESPONSE (include citation anchors for all factual claims):"""
        
        return prompt


citation_store = CitationAnchorStore()
citation_verifier = CitationVerifier(citation_store)
