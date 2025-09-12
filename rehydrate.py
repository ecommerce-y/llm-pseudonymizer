"""
Rehydrate Module - Placeholder Restoration

This module restores original sensitive entities from their placeholder representations
in LLM responses. It serves as the final step in the pseudonymization pipeline,
transforming sanitized responses back into meaningful text while tracking any unknown
placeholders that may have been introduced by the LLM.

Key features:
- Length-based placeholder sorting to prevent partial replacements
- Unknown placeholder tracking and reporting
- Word boundary enforcement for accurate matching
- Comprehensive validation and reporting capabilities
- Unicode support for international text
"""

import re
from typing import Dict, List, Tuple, Any, Optional


# Compiled pattern for placeholder detection
# Matches: PERSON_1, ORG_2, EMAIL_10, URL_999, etc.
PLACEHOLDER_PATTERN = re.compile(r'\b(?:PERSON|ORG|EMAIL|URL)_[1-9]\d*\b')


def rehydrate_response(response_text: str, alias_snapshot: Dict[str, str]) -> Tuple[str, List[str]]:
    """
    Restore placeholders in LLM response to original entities.
    
    Args:
        response_text: LLM response containing placeholders
        alias_snapshot: Mapping from placeholders to original text
        
    Returns:
        Tuple of (restored text, list of unknown placeholders)
        
    Raises:
        ValueError: If response_text is None
    """
    # Validate inputs
    if response_text is None:
        raise ValueError("Response text cannot be None")
    
    if alias_snapshot is None:
        alias_snapshot = {}  # Treat as empty mapping
    
    # Validate alias snapshot format
    for placeholder, original in alias_snapshot.items():
        if not isinstance(placeholder, str) or not isinstance(original, str):
            raise ValueError(f"Invalid mapping: {placeholder} -> {original}")
    
    if not response_text:
        return response_text, []
    
    if not alias_snapshot:
        # No mappings, check for any placeholders
        found_placeholders = find_placeholders(response_text)
        return response_text, found_placeholders
    
    # Find all placeholders in the response
    placeholders = find_placeholders(response_text)
    
    # Sort by length descending to avoid partial replacements
    sorted_placeholders = sort_placeholders_by_length(placeholders)
    
    # Apply replacements and track unknowns
    result, unknown = apply_replacements(response_text, sorted_placeholders, alias_snapshot)
    
    return result, unknown


def find_placeholders(text: str) -> List[str]:
    """
    Find all valid placeholders in text.
    
    Args:
        text: Text to search
        
    Returns:
        List of unique placeholders
    """
    if not text:
        return []
    
    # Find all matches
    matches = PLACEHOLDER_PATTERN.findall(text)
    
    # Return unique placeholders maintaining order
    seen = set()
    unique = []
    for placeholder in matches:
        if placeholder not in seen:
            seen.add(placeholder)
            unique.append(placeholder)
    
    return unique


def sort_placeholders_by_length(placeholders: List[str]) -> List[str]:
    """
    Sort placeholders by length descending.
    
    This prevents partial replacements (e.g., PERSON_1 in PERSON_10).
    
    Args:
        placeholders: List of placeholders
        
    Returns:
        Sorted placeholders (longest first)
    """
    return sorted(placeholders, key=len, reverse=True)


def apply_replacements(text: str, sorted_placeholders: List[str], 
                      alias_snapshot: Dict[str, str]) -> Tuple[str, List[str]]:
    """
    Apply placeholder replacements to text.
    
    Args:
        text: Text containing placeholders
        sorted_placeholders: Placeholders sorted by length (descending)
        alias_snapshot: Placeholder to original text mapping
        
    Returns:
        Tuple of (restored text, unknown placeholders)
    """
    result = text
    unknown = []
    
    for placeholder in sorted_placeholders:
        if placeholder in alias_snapshot:
            # Replace all occurrences
            original_text = alias_snapshot[placeholder]
            result = result.replace(placeholder, original_text)
        else:
            # Track unknown placeholder
            if placeholder not in unknown:  # Avoid duplicates
                unknown.append(placeholder)
    
    return result, unknown


def validate_restoration(original: str, sanitized: str, restored: str, 
                        alias_snapshot: Dict[str, str]) -> bool:
    """
    Validate that restoration is complete and accurate.
    
    Args:
        original: Original text before sanitization
        sanitized: Sanitized text with placeholders
        restored: Restored text after rehydration
        alias_snapshot: Placeholder mappings
        
    Returns:
        True if restoration is valid
    """
    # First check: no placeholders should remain
    remaining_placeholders = find_placeholders(restored)
    if remaining_placeholders:
        # Unless they were unknown (not in snapshot)
        for placeholder in remaining_placeholders:
            if placeholder in alias_snapshot:
                return False  # Known placeholder wasn't replaced
    
    # Second check: if we rehydrate the sanitized text, should match restored
    test_restored, _ = rehydrate_response(sanitized, alias_snapshot)
    
    return test_restored == restored


def count_placeholder_occurrences(text: str) -> Dict[str, int]:
    """
    Count occurrences of each placeholder.
    
    Args:
        text: Text containing placeholders
        
    Returns:
        Dictionary mapping placeholder to count
    """
    placeholders = PLACEHOLDER_PATTERN.findall(text)
    
    counts = {}
    for placeholder in placeholders:
        counts[placeholder] = counts.get(placeholder, 0) + 1
    
    return counts


def extract_placeholder_types(placeholders: List[str]) -> Dict[str, List[str]]:
    """
    Group placeholders by entity type.
    
    Args:
        placeholders: List of placeholders
        
    Returns:
        Dictionary mapping type to list of placeholders
    """
    types = {}
    
    for placeholder in placeholders:
        # Extract type from placeholder (e.g., "PERSON" from "PERSON_1")
        entity_type = placeholder.split('_')[0]
        
        if entity_type not in types:
            types[entity_type] = []
        types[entity_type].append(placeholder)
    
    return types


def generate_restoration_report(response_text: str, restored_text: str, 
                              unknown: List[str]) -> Dict[str, Any]:
    """
    Generate a report of the restoration process.
    
    Args:
        response_text: Original LLM response
        restored_text: Text after restoration
        unknown: Unknown placeholders found
        
    Returns:
        Dictionary with restoration statistics
    """
    # Count placeholders before and after
    before_counts = count_placeholder_occurrences(response_text)
    after_counts = count_placeholder_occurrences(restored_text)
    
    # Calculate restoration stats
    total_before = sum(before_counts.values())
    total_after = sum(after_counts.values())
    restored_count = total_before - total_after
    
    # Group unknown by type
    unknown_by_type = extract_placeholder_types(unknown) if unknown else {}
    
    return {
        "total_placeholders": total_before,
        "restored_count": restored_count,
        "remaining_count": total_after,
        "unknown_count": len(unknown),
        "unknown_by_type": unknown_by_type,
        "restoration_rate": (restored_count / total_before * 100) if total_before > 0 else 100.0
    }


# Utility functions for integration and debugging

def get_placeholder_stats(text: str) -> Dict[str, Any]:
    """
    Get statistics about placeholders in text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with placeholder statistics
    """
    placeholders = find_placeholders(text)
    counts = count_placeholder_occurrences(text)
    types = extract_placeholder_types(placeholders)
    
    return {
        "unique_placeholders": len(placeholders),
        "total_occurrences": sum(counts.values()),
        "by_type": {entity_type: len(phs) for entity_type, phs in types.items()},
        "occurrence_counts": counts
    }


def preview_restoration(response_text: str, alias_snapshot: Dict[str, str], 
                       max_length: int = 100) -> str:
    """
    Preview what restoration would look like (for debugging).
    
    Args:
        response_text: Text to restore
        alias_snapshot: Placeholder mappings
        max_length: Maximum length of preview
        
    Returns:
        Preview string showing before/after
    """
    restored, unknown = rehydrate_response(response_text, alias_snapshot)
    
    # Truncate if too long
    original_preview = response_text[:max_length] + "..." if len(response_text) > max_length else response_text
    restored_preview = restored[:max_length] + "..." if len(restored) > max_length else restored
    
    preview = f"Before: {original_preview}\nAfter:  {restored_preview}"
    if unknown:
        preview += f"\nUnknown: {unknown}"
    
    return preview


def batch_rehydrate(responses: List[str], alias_snapshot: Dict[str, str]) -> List[Tuple[str, List[str]]]:
    """
    Rehydrate multiple responses efficiently.
    
    Args:
        responses: List of response texts
        alias_snapshot: Placeholder mappings
        
    Returns:
        List of (restored_text, unknown_placeholders) tuples
    """
    results = []
    for response in responses:
        restored, unknown = rehydrate_response(response, alias_snapshot)
        results.append((restored, unknown))
    
    return results


def find_modified_placeholders(response_text: str, alias_snapshot: Dict[str, str]) -> List[str]:
    """
    Find placeholders that may have been modified by the LLM.
    
    This looks for text that resembles placeholders but doesn't match exactly.
    
    Args:
        response_text: LLM response text
        alias_snapshot: Known placeholder mappings
        
    Returns:
        List of potentially modified placeholder-like strings
    """
    # Pattern for placeholder-like strings (more permissive)
    loose_pattern = re.compile(r'\b(?:PERSON|ORG|EMAIL|URL)_?\d*\b', re.IGNORECASE)
    
    # Find all placeholder-like matches
    loose_matches = loose_pattern.findall(response_text)
    
    # Find strict matches
    strict_matches = find_placeholders(response_text)
    
    # Find differences (potential modifications)
    modified = []
    for match in loose_matches:
        # Normalize to check if it should be a valid placeholder
        normalized = match.upper()
        if normalized not in strict_matches and normalized not in alias_snapshot:
            # Check if it looks like a modified placeholder
            if '_' in normalized or any(normalized.startswith(prefix) for prefix in ['PERSON', 'ORG', 'EMAIL', 'URL']):
                modified.append(match)
    
    return list(set(modified))  # Remove duplicates


def create_reverse_snapshot(alias_snapshot: Dict[str, str]) -> Dict[str, str]:
    """
    Create reverse mapping from original text to placeholders.
    
    Args:
        alias_snapshot: Placeholder to original text mapping
        
    Returns:
        Original text to placeholder mapping
    """
    return {original: placeholder for placeholder, original in alias_snapshot.items()}


def validate_alias_snapshot(alias_snapshot: Dict[str, str]) -> List[str]:
    """
    Validate alias snapshot for common issues.
    
    Args:
        alias_snapshot: Placeholder mappings to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not isinstance(alias_snapshot, dict):
        errors.append("Alias snapshot must be a dictionary")
        return errors
    
    placeholder_pattern = re.compile(r'^(?:PERSON|ORG|EMAIL|URL)_[1-9]\d*$')
    
    for placeholder, original in alias_snapshot.items():
        # Check placeholder format
        if not isinstance(placeholder, str):
            errors.append(f"Placeholder key must be string: {type(placeholder)}")
            continue
            
        if not placeholder_pattern.match(placeholder):
            errors.append(f"Invalid placeholder format: {placeholder}")
        
        # Check original text
        if not isinstance(original, str):
            errors.append(f"Original text must be string for {placeholder}: {type(original)}")
            continue  # Skip further checks for non-string values
        
        if not original.strip():
            errors.append(f"Empty original text for placeholder: {placeholder}")
    
    # Check for duplicate original texts (potential issue)
    originals = list(alias_snapshot.values())
    duplicates = [orig for orig in set(originals) if originals.count(orig) > 1]
    if duplicates:
        errors.append(f"Duplicate original texts found: {duplicates}")
    
    return errors