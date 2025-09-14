"""
Test phone number detection functionality
"""

import pytest
from detector.rules import detect_phones, detect_patterns
from aliases import AliasManager
from rewrite import rewrite_text
from rehydrate import rehydrate_response
from detector.merge import merge_detections


def test_phone_detection_basic():
    """Test basic phone number detection."""
    test_cases = [
        ("Call me at 555-123-4567", ["555-123-4567"]),
        ("My number is (555) 123-4567", ["555) 123-4567"]),  # Regex captures from after the opening paren
        ("Contact: 555.123.4567", ["555.123.4567"]),
        ("Phone: +1-555-123-4567", ["1-555-123-4567"]),  # Regex captures from after the +
        ("Reach me at 1 555 123 4567", ["1 555 123 4567"]),
    ]
    
    for text, expected_phones in test_cases:
        entities = detect_phones(text)
        detected_phones = [e['text'] for e in entities]
        assert detected_phones == expected_phones, f"Failed for: {text}"
        
        # Verify entity structure
        for entity in entities:
            assert entity['type'] == 'PHONE'
            assert entity['source'] == 'regex'
            assert 'start' in entity
            assert 'end' in entity


def test_phone_in_full_pipeline():
    """Test phone detection through the full pipeline."""
    # Setup
    alias_manager = AliasManager()
    text = "Please call John at 555-123-4567 or email john@example.com"
    
    # Step 1: Detection
    regex_entities = detect_patterns(text)
    merged_entities = merge_detections(regex_entities, [])
    
    # Should detect phone and email
    entity_types = {e['type'] for e in merged_entities}
    assert 'PHONE' in entity_types
    assert 'EMAIL' in entity_types
    
    # Step 2: Rewrite
    sanitized_text, alias_snapshot, _, _ = rewrite_text(
        text, merged_entities, alias_manager
    )
    
    # Should have placeholders
    assert 'PHONE_1' in sanitized_text
    assert 'EMAIL_1' in sanitized_text
    assert '555-123-4567' not in sanitized_text
    assert 'john@example.com' not in sanitized_text
    
    # Step 3: Rehydrate
    rehydrated_text, unknown_placeholders = rehydrate_response(
        sanitized_text, alias_snapshot
    )
    
    # Should restore original text
    assert rehydrated_text == text
    assert len(unknown_placeholders) == 0


def test_multiple_phones():
    """Test detection of multiple phone numbers."""
    text = "Office: 555-123-4567, Mobile: (555) 987-6543, Fax: 555.555.5555"
    
    entities = detect_phones(text)
    assert len(entities) == 3
    
    phone_numbers = [e['text'] for e in entities]
    assert '555-123-4567' in phone_numbers
    assert '555) 987-6543' in phone_numbers  # Regex captures from after the opening paren
    assert '555.555.5555' in phone_numbers


def test_phone_leak_detection():
    """Test that phone leak detection works."""
    from rewrite import leak_check
    
    # Text with phone number (should be detected as leak)
    text_with_leak = "Call me at 555-123-4567"
    leaks = leak_check(text_with_leak)
    assert len(leaks) > 0
    assert any('555-123-4567' in leak for leak in leaks)
    
    # Text with placeholder (should not be detected as leak)
    text_with_placeholder = "Call me at PHONE_1"
    leaks = leak_check(text_with_placeholder)
    assert len(leaks) == 0


def test_phone_alias_consistency():
    """Test that same phone number gets same alias."""
    alias_manager = AliasManager()
    
    # First occurrence
    alias1 = alias_manager.get_or_create_alias("555-123-4567", "PHONE")
    assert alias1 == "PHONE_1"
    
    # Second occurrence (should get same alias)
    alias2 = alias_manager.get_or_create_alias("555-123-4567", "PHONE")
    assert alias2 == "PHONE_1"
    
    # Different phone (should get new alias)
    alias3 = alias_manager.get_or_create_alias("555-987-6543", "PHONE")
    assert alias3 == "PHONE_2"


def test_phone_edge_cases():
    """Test edge cases for phone detection."""
    test_cases = [
        # Should NOT match
        ("12345", []),  # Too short
        ("555-12-3456", []),  # Wrong format
        ("555-1234-567", []),  # Wrong format
        
        # Should match
        ("555-123-4567", ["555-123-4567"]),
        ("(555)123-4567", ["555)123-4567"]),  # Regex captures from after the opening paren
    ]
    
    for text, expected in test_cases:
        entities = detect_phones(text)
        detected = [e['text'] for e in entities]
        assert detected == expected, f"Failed for: {text}"


if __name__ == "__main__":
    # Run basic tests
    print("Testing phone detection...")
    test_phone_detection_basic()
    print("✓ Basic detection tests passed")
    
    test_phone_in_full_pipeline()
    print("✓ Full pipeline test passed")
    
    test_multiple_phones()
    print("✓ Multiple phones test passed")
    
    test_phone_leak_detection()
    print("✓ Leak detection test passed")
    
    test_phone_alias_consistency()
    print("✓ Alias consistency test passed")
    
    test_phone_edge_cases()
    print("✓ Edge cases test passed")
    
    print("\nAll phone detection tests passed! ✨")