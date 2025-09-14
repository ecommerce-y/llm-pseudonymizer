"""
Regex-based entity detection for emails and URLs.

This module provides deterministic pattern matching for known entity types
using regular expressions. It serves as the foundation of the detection layer,
identifying emails and URLs with high precision before any ML-based detection occurs.
"""

import re
from typing import List, Dict, Any


# Compile patterns at module import time for performance
EMAIL_PATTERN = re.compile(
    r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    re.IGNORECASE
)

# Complex URL pattern handling multiple formats
URL_PATTERN = re.compile(
    r'''
    (?:
        # Protocol URLs: http://, https://, ftp://
        (?:https?|ftp)://
        (?:
            # Domain name or IP address
            (?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}  # Domain
            |
            (?:\d{1,3}\.){3}\d{1,3}  # IP address
        )
        (?::\d{1,5})?  # Optional port
        (?:/[^\s]*)?   # Optional path
    )
    |
    (?:
        # Bare domains: example.com, sub.example.com
        (?<![a-zA-Z0-9.-])  # Negative lookbehind to avoid partial matches
        (?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}
        (?::\d{1,5})?  # Optional port
        (?:/[^\s]*)?   # Optional path
        (?![a-zA-Z0-9.-])  # Negative lookahead to avoid partial matches
    )
    ''',
    re.IGNORECASE | re.VERBOSE
)

# Phone number pattern - supports US/Canada formats
PHONE_PATTERN = re.compile(
    r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
)

# Pattern to clean trailing punctuation from URLs
TRAILING_PUNCT_PATTERN = re.compile(r'[.,;:!?…)\]}]+$')


def create_entity(text: str, start: int, end: int, entity_type: str) -> Dict[str, Any]:
    """
    Create a standardized entity dictionary.
    
    Args:
        text: The matched entity text
        start: Starting position in original text
        end: Ending position in original text
        entity_type: Type of entity (EMAIL or URL)
        
    Returns:
        Standardized entity dictionary
    """
    return {
        "text": text,
        "start": start,
        "end": end,
        "type": entity_type,
        "source": "regex"
    }


def detect_emails(text: str) -> List[Dict[str, Any]]:
    """
    Detect email addresses in text using regex pattern.
    
    Args:
        text: Input text to search
        
    Returns:
        List of entity dictionaries with email information
    """
    if not text:
        return []
    
    try:
        entities = []
        for match in EMAIL_PATTERN.finditer(text):
            entities.append(create_entity(
                match.group(),
                match.start(),
                match.end(),
                "EMAIL"
            ))
        return entities
    except Exception as e:
        # Log error but don't crash
        print(f"Warning: Email detection error: {e}")
        return []


def _clean_url_boundaries(url: str) -> str:
    """
    Clean URL boundaries by removing trailing punctuation.
    
    Special handling for parentheses - only remove closing paren
    if there's no matching opening paren in the URL.
    
    Args:
        url: Raw URL match
        
    Returns:
        Cleaned URL
    """
    # Handle parentheses specially
    if url.endswith(')'):
        # Count parentheses in the URL
        open_count = url.count('(')
        close_count = url.count(')')
        
        # If there are unmatched closing parens, remove them
        while close_count > open_count and url.endswith(')'):
            url = url[:-1]
            close_count -= 1
    
    # Remove other trailing punctuation
    url = TRAILING_PUNCT_PATTERN.sub('', url)
    
    return url


def detect_urls(text: str) -> List[Dict[str, Any]]:
    """
    Detect URLs and domain names in text using regex patterns.
    
    Handles:
    - Full URLs with protocol (http://, https://, ftp://)
    - Bare domains (example.com)
    - URLs with paths, query strings, and fragments
    - Special punctuation handling
    
    Args:
        text: Input text to search
        
    Returns:
        List of entity dictionaries with URL information
    """
    if not text:
        return []
    
    try:
        entities = []
        
        for match in URL_PATTERN.finditer(text):
            raw_url = match.group()
            cleaned_url = _clean_url_boundaries(raw_url)
            
            # Skip if cleaning removed everything or made it too short
            if not cleaned_url or len(cleaned_url) < 4:
                continue
            
            # Skip if it's just a TLD (like ".com")
            if cleaned_url.startswith('.'):
                continue
            
            # Calculate adjusted end position after cleaning
            chars_removed = len(raw_url) - len(cleaned_url)
            adjusted_end = match.end() - chars_removed
            
            entities.append(create_entity(
                cleaned_url,
                match.start(),
                adjusted_end,
                "URL"
            ))
        
        return entities
    except Exception as e:
        # Log error but don't crash
        print(f"Warning: URL detection error: {e}")
        return []


def detect_phones(text: str) -> List[Dict[str, Any]]:
    """
    Detect phone numbers in text using regex pattern.
    
    Supports US/Canada phone formats:
    - (555) 123-4567
    - 555-123-4567
    - 555.123.4567
    - +1-555-123-4567
    - 1 555 123 4567
    
    Args:
        text: Input text to search
        
    Returns:
        List of entity dictionaries with phone information
    """
    if not text:
        return []
    
    try:
        entities = []
        for match in PHONE_PATTERN.finditer(text):
            entities.append(create_entity(
                match.group(),
                match.start(),
                match.end(),
                "PHONE"
            ))
        return entities
    except Exception as e:
        # Log error but don't crash
        print(f"Warning: Phone detection error: {e}")
        return []


def detect_patterns(text: str) -> List[Dict[str, Any]]:
    """
    Run all regex pattern detections on the input text.
    
    This is the main entry point for the rules module,
    combining results from all pattern detectors.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Combined list of all detected entities, sorted by position
    """
    if not text:
        return []
    
    entities = []
    
    # Run email detection
    entities.extend(detect_emails(text))
    
    # Run URL detection
    entities.extend(detect_urls(text))
    
    # Run phone detection
    entities.extend(detect_phones(text))
    
    # Sort by position for consistent output
    entities.sort(key=lambda e: (e['start'], e['end']))
    
    return entities


# For backward compatibility and testing
def detect_patterns_with_config(text: str, config=None) -> List[Dict[str, Any]]:
    """
    Run regex pattern detections with optional config filtering.
    
    This version respects configuration settings for which entity types
    to detect. If no config is provided, detects all types.
    
    Args:
        text: Input text to analyze
        config: Optional Config object to filter entity types
        
    Returns:
        Combined list of detected entities based on config
    """
    if not text:
        return []
    
    entities = []
    
    # If no config provided, detect everything
    if config is None:
        return detect_patterns(text)
    
    # Check config for enabled entity types
    if config.is_entity_enabled("EMAIL"):
        entities.extend(detect_emails(text))
    
    if config.is_entity_enabled("URL"):
        entities.extend(detect_urls(text))
    
    if config.is_entity_enabled("PHONE"):
        entities.extend(detect_phones(text))
    
    # Sort by position for consistent output
    entities.sort(key=lambda e: (e['start'], e['end']))
    
    return entities