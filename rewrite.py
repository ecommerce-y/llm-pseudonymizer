"""
Rewrite Module - Text Sanitization with Entity Replacement

This module transforms raw text into sanitized text by replacing detected sensitive
entities with opaque placeholders. It serves as the critical bridge between entity
detection and LLM communication, ensuring no sensitive information is transmitted
to external services.

Key features:
- Right-to-left replacement strategy to maintain character indices
- Overlap detection and validation
- Leak detection with configurable strict mode
- Comprehensive redaction reporting
- Fail-closed error handling for security
"""

import re
from typing import Dict, List, Any, Tuple, Union, Optional


# Compiled patterns for leak detection - exclude placeholder patterns
# Use case-insensitive matching and proper TLD pattern
EMAIL_LEAK_PATTERN = re.compile(
    r'\b(?!(?:EMAIL|URL|PERSON|ORG)_\d+\b)'  # Negative lookahead for placeholders
    r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
    re.IGNORECASE
)

# URL pattern that excludes trailing punctuation
URL_LEAK_PATTERN = re.compile(
    r'\b(?!(?:EMAIL|URL|PERSON|ORG)_\d+\b)'  # Negative lookahead for placeholders
    r'(?:https?://[^\s<>"{}|\\^`\[\]]+|'  # HTTP(S) URLs
    r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,})'  # Domain-only
    r'(?=[^a-zA-Z0-9-._~:/?#[\]@!$&\'()*+,;=]|$)',  # Exclude trailing punctuation
    re.IGNORECASE
)

# Valid entity types
VALID_ENTITY_TYPES = {"EMAIL", "URL", "PERSON", "ORG"}


def rewrite_text(text: str, detected_entities: List[Dict[str, Any]],
                alias_manager: Any, *, strict_mode: bool = False,
                with_report: bool = False) -> Tuple[str, Dict[str, str], Optional[Dict[str, Any]], Optional[List[str]]]:
    """
    Replace detected entities with placeholders.
    
    Args:
        text: Original text
        detected_entities: Non-overlapping entities from detection pipeline
        alias_manager: AliasManager for placeholder generation
        strict_mode: If True, raise on leak detection
        with_report: If True, include redaction report in return
        
    Returns:
        Tuple based on with_report flag:
        - If with_report=False: (sanitized_text, alias_snapshot)
        - If with_report=True: (sanitized_text, alias_snapshot, redaction_report, leaks)
        
    Raises:
        ValueError: If inputs are invalid or leaks detected in strict mode
    """
    # Validate inputs
    if text is None:
        raise ValueError("Text cannot be None")
    
    if detected_entities is None:
        detected_entities = []
    
    if alias_manager is None:
        raise ValueError("AliasManager cannot be None")
    
    # Handle empty cases
    if not text:
        empty_snapshot = {}
        empty_report = {"total": 0, "by_type": {}, "examples": {}} if with_report else None
        empty_leaks = [] if with_report else None
        return text, empty_snapshot, empty_report, empty_leaks
    
    if not detected_entities:
        empty_snapshot = {}
        empty_report = {"total": 0, "by_type": {}, "examples": {}} if with_report else None
        empty_leaks = [] if with_report else None
        return text, empty_snapshot, empty_report, empty_leaks
    
    # Validate entities
    for entity in detected_entities:
        validate_entity_indices(entity, text)
    
    # Check for overlaps (should not happen if merge.py works correctly)
    check_no_overlaps(detected_entities)
    
    # Sort entities by end position (right to left processing)
    sorted_entities = sort_entities_by_position(detected_entities)
    
    # Apply replacements and collect alias snapshot
    result, alias_snapshot = apply_replacements(text, sorted_entities, alias_manager)
    
    # Perform leak check
    leaks = leak_check(result)
    if leaks and strict_mode:
        raise ValueError(f"Leak detection failed. Found sensitive data: {leaks}")
    
    # Generate report if requested
    report = generate_redaction_report(detected_entities, alias_snapshot) if with_report else None
    leaks_result = leaks if with_report else None
    
    return result, alias_snapshot, report, leaks_result


def check_no_overlaps(entities: List[Dict[str, Any]]) -> None:
    """
    Verify no entities overlap.
    
    Args:
        entities: List of entities to check
        
    Raises:
        ValueError: If overlapping entities detected
    """
    if len(entities) <= 1:
        return
    
    # Sort by start position
    sorted_entities = sorted(entities, key=lambda e: e['start'])
    
    for i in range(1, len(sorted_entities)):
        prev = sorted_entities[i-1]
        curr = sorted_entities[i]
        
        if prev['end'] > curr['start']:
            raise ValueError(
                f"Overlapping entities detected: "
                f"'{prev['text']}' [{prev['start']}:{prev['end']}] overlaps "
                f"'{curr['text']}' [{curr['start']}:{curr['end']}]"
            )


def sort_entities_by_position(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort entities by end position in descending order.
    
    This ensures right-to-left processing to maintain valid indices.
    
    Args:
        entities: List of entities to sort
        
    Returns:
        Sorted entities (rightmost first)
    """
    return sorted(entities, key=lambda e: e['end'], reverse=True)


def apply_replacements(text: str, sorted_entities: List[Dict[str, Any]], 
                      alias_manager: Any) -> Tuple[str, Dict[str, str]]:
    """
    Apply entity replacements from right to left.
    
    Args:
        text: Original text
        sorted_entities: Entities sorted by end position (descending)
        alias_manager: AliasManager for placeholders
        
    Returns:
        Tuple of (text with replacements, alias snapshot)
    """
    result = text
    alias_snapshot = {}
    
    for entity in sorted_entities:
        # Get or create placeholder using AliasManager API
        placeholder = alias_manager.get_or_create_alias(entity['text'], entity['type'])
        
        # Store in snapshot for rehydration
        alias_snapshot[placeholder] = entity['text']
        
        # Replace entity with placeholder
        result = (result[:entity['start']] + 
                 placeholder + 
                 result[entity['end']:])
    
    return result, alias_snapshot


def leak_check(sanitized_text: str) -> List[str]:
    """
    Check for potential data leaks in sanitized text.
    
    Args:
        sanitized_text: Text that should be clean of sensitive data
        
    Returns:
        List of potential leaks found
    """
    leaks = []
    
    # Check for email leaks
    email_matches = EMAIL_LEAK_PATTERN.findall(sanitized_text)
    leaks.extend(email_matches)
    
    # Check for URL leaks
    url_matches = URL_LEAK_PATTERN.findall(sanitized_text)
    leaks.extend(url_matches)
    
    return leaks


def generate_redaction_report(entities: List[Dict[str, Any]], 
                            alias_snapshot: Dict[str, str]) -> Dict[str, Any]:
    """
    Generate a report of redacted entities.
    
    Args:
        entities: List of entities that were redacted
        alias_snapshot: Placeholder to original text mapping
        
    Returns:
        Dictionary with redaction statistics and examples
    """
    if not entities:
        return {"total": 0, "by_type": {}, "examples": {}}
    
    # Count by type and collect examples
    type_counts = {}
    examples = {}
    
    # Reverse the alias snapshot for lookup
    text_to_placeholder = {v: k for k, v in alias_snapshot.items()}
    
    for entity in entities:
        entity_type = entity['type']
        entity_text = entity['text']
        
        # Count
        type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        
        # Store examples with placeholders
        if entity_type not in examples:
            examples[entity_type] = []
        
        if len(examples[entity_type]) < 3:  # Limit examples
            placeholder = text_to_placeholder.get(entity_text, "UNKNOWN")
            examples[entity_type].append({
                "placeholder": placeholder,
                "original": entity_text[:20] + "..." if len(entity_text) > 20 else entity_text
            })
    
    return {
        "total": len(entities),
        "by_type": type_counts,
        "examples": examples
    }


def validate_entity_indices(entity: Dict[str, Any], text: str) -> None:
    """
    Validate entity indices against the text.
    
    Args:
        entity: Entity with start/end indices
        text: Original text
        
    Raises:
        ValueError: If indices are invalid
    """
    # Validate entity type
    if entity.get('type') not in VALID_ENTITY_TYPES:
        raise ValueError(f"Invalid entity type: {entity.get('type')}. "
                        f"Must be one of {VALID_ENTITY_TYPES}")
    
    start, end = entity['start'], entity['end']
    
    if start < 0 or end < 0:
        raise ValueError(f"Negative indices: start={start}, end={end}")
    
    if start >= end:
        raise ValueError(f"Invalid span: start={start} >= end={end}")
    
    if end > len(text):
        raise ValueError(f"End index {end} exceeds text length {len(text)}")
    
    # Verify the entity text matches the span
    expected_text = text[start:end]
    if entity['text'] != expected_text:
        raise ValueError(f"Entity text '{entity['text']}' doesn't match "
                        f"span '{expected_text}' at [{start}:{end}]")


# Utility functions for testing and debugging

def validate_rewrite(original: str, rewritten: str, 
                    mappings: Dict[str, str]) -> bool:
    """
    Validate that rewrite is reversible.
    
    Args:
        original: Original text
        rewritten: Sanitized text
        mappings: Placeholder mappings
        
    Returns:
        True if rewrite is valid
    """
    # Attempt to reverse the rewrite
    restored = rewritten
    
    # Sort placeholders by length (descending) to avoid partial matches
    sorted_placeholders = sorted(mappings.keys(), key=len, reverse=True)
    
    for placeholder in sorted_placeholders:
        entity_text = mappings[placeholder]
        restored = restored.replace(placeholder, entity_text)
    
    return restored == original


def rehydrate_response(response: str, alias_snapshot: Dict[str, str]) -> str:
    """
    Restore placeholders in LLM response with original entities.
    
    Args:
        response: LLM response containing placeholders
        alias_snapshot: Mapping from placeholders to original text
        
    Returns:
        Response with placeholders replaced by original entities
    """
    result = response
    
    # Sort placeholders by length (descending) to avoid partial matches
    # e.g., replace PERSON_10 before PERSON_1
    sorted_placeholders = sorted(alias_snapshot.keys(), key=len, reverse=True)
    
    for placeholder in sorted_placeholders:
        if placeholder in result:
            original_text = alias_snapshot[placeholder]
            result = result.replace(placeholder, original_text)
    
    return result


def get_session_stats(alias_manager: Any) -> Dict[str, Any]:
    """
    Get statistics about the current session.
    
    Args:
        alias_manager: AliasManager instance
        
    Returns:
        Dictionary with session statistics
    """
    entity_counts = alias_manager.get_entity_count()
    total_entities = sum(entity_counts.values())
    
    return {
        "total_entities": total_entities,
        "by_type": entity_counts,
        "active_placeholders": len(alias_manager.get_all_placeholders())
    }