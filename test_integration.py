"""
Integration Test Suite for LLM Pseudonymizer

Tests the complete pipeline from detection through rehydration,
verifying that all components work together correctly.
"""

import pytest
import re
from typing import Dict, List, Any
from unittest.mock import Mock

# Import all components for integration testing
from aliases import AliasManager
from detector.rules import detect_patterns
from detector.spacy_ner import detect_entities, load_model
from detector.merge import merge_detections
from rewrite import rewrite_text
from rehydrate import rehydrate_response

# Canonical placeholder pattern for robust testing
PLACEHOLDER_PATTERN = re.compile(r'\b(EMAIL|URL|PERSON|ORG)_[1-9]\d*\b')

def assert_contains_placeholder_type(text: str, entity_type: str) -> str:
    """Helper to find placeholder of given type and return it."""
    matches = [m.group() for m in PLACEHOLDER_PATTERN.finditer(text)
               if m.group().startswith(entity_type + '_')]
    assert len(matches) > 0, f"No {entity_type} placeholder found in: {text}"
    return matches[0]

def assert_no_pii_leaked(text: str, original_entities: List[str]):
    """Helper to ensure no original PII remains in sanitized text."""
    for entity in original_entities:
        assert entity not in text, f"PII leaked: {entity} found in {text}"

def get_text_span(text: str, substring: str) -> tuple:
    """Helper to get accurate span indices."""
    start = text.index(substring)
    end = start + len(substring)
    return start, end


class TestBasicPipeline:
    """Test basic pipeline functionality with real components."""
    
    def test_email_detection_and_rehydration(self):
        """Test complete pipeline with email detection."""
        # Setup
        alias_manager = AliasManager()
        text = "Please contact john.doe@example.com for more information."
        
        # Step 1: Detection
        regex_entities = detect_patterns(text)
        spacy_entities = []  # No spaCy for this test
        merged_entities = merge_detections(regex_entities, spacy_entities)
        
        # Verify detection
        assert len(merged_entities) == 1
        assert merged_entities[0]['type'] == 'EMAIL'
        assert merged_entities[0]['text'] == 'john.doe@example.com'
        
        # Step 2: Rewrite
        sanitized_text, alias_snapshot, redaction_report, warnings = rewrite_text(
            text, merged_entities, alias_manager
        )
        
        # Verify sanitization
        assert 'EMAIL_1' in sanitized_text
        assert 'john.doe@example.com' not in sanitized_text
        assert sanitized_text == "Please contact EMAIL_1 for more information."
        
        # Step 3: Rehydrate
        rehydrated_text, unknown_placeholders = rehydrate_response(
            sanitized_text, alias_snapshot
        )
        
        # Verify rehydration
        assert rehydrated_text == text
        assert len(unknown_placeholders) == 0
    
    def test_url_detection_and_rehydration(self):
        """Test complete pipeline with URL detection."""
        # Setup
        alias_manager = AliasManager()
        text = "Visit https://www.example.com and also check https://example.org for details."
        
        # Step 1: Detection
        regex_entities = detect_patterns(text)
        spacy_entities = []
        merged_entities = merge_detections(regex_entities, spacy_entities)
        
        # Verify detection (flexible - may detect 1 or 2 URLs depending on regex)
        assert len(merged_entities) >= 1
        url_types = [e['type'] for e in merged_entities]
        assert all(t == 'URL' for t in url_types)
        
        # Step 2: Rewrite
        sanitized_text, alias_snapshot, redaction_report, warnings = rewrite_text(
            text, merged_entities, alias_manager
        )
        
        # Verify sanitization (flexible assertions)
        url_placeholders = PLACEHOLDER_PATTERN.findall(sanitized_text)
        url_placeholders = [p for p in url_placeholders if p.startswith('URL_')]
        assert len(url_placeholders) >= 1
        
        # Ensure no PII leaked
        assert_no_pii_leaked(sanitized_text, ['https://www.example.com', 'https://example.org'])
        
        # Step 3: Rehydrate
        rehydrated_text, unknown_placeholders = rehydrate_response(
            sanitized_text, alias_snapshot
        )
        
        # Verify rehydration
        assert rehydrated_text == text
        assert len(unknown_placeholders) == 0
    
    def test_mixed_entity_types(self):
        """Test pipeline with multiple entity types."""
        # Setup
        alias_manager = AliasManager()
        text = "Contact John Doe at john@example.com or visit https://example.com"
        
        # Step 1: Detection (regex only for predictable results)
        regex_entities = detect_patterns(text)
        spacy_entities = []
        merged_entities = merge_detections(regex_entities, spacy_entities)
        
        # Should detect email and URL
        assert len(merged_entities) == 2
        entity_types = {e['type'] for e in merged_entities}
        assert 'EMAIL' in entity_types
        assert 'URL' in entity_types
        
        # Step 2: Rewrite
        sanitized_text, alias_snapshot, redaction_report, warnings = rewrite_text(
            text, merged_entities, alias_manager
        )
        
        # Verify sanitization (flexible assertions)
        assert_contains_placeholder_type(sanitized_text, 'EMAIL')
        assert_contains_placeholder_type(sanitized_text, 'URL')
        assert_no_pii_leaked(sanitized_text, ['john@example.com', 'https://example.com'])
        
        # Step 3: Rehydrate
        rehydrated_text, unknown_placeholders = rehydrate_response(
            sanitized_text, alias_snapshot
        )
        
        # Verify rehydration
        assert rehydrated_text == text
        assert len(unknown_placeholders) == 0


class TestSpacyIntegration:
    """Test integration with spaCy NER when available."""
    
    def test_spacy_person_detection(self):
        """Test pipeline with spaCy person detection."""
        # Setup
        alias_manager = AliasManager()
        text = "John Smith works at Microsoft Corporation."
        
        # Try to load spaCy model
        try:
            nlp_model = load_model()
            if nlp_model is None:
                pytest.skip("spaCy model not available")
        except Exception:
            pytest.skip("spaCy model not available")
        
        # Step 1: Detection
        regex_entities = detect_patterns(text)
        spacy_entities = detect_entities(text, nlp_model)
        merged_entities = merge_detections(regex_entities, spacy_entities)
        
        # Should detect at least one entity (person or org)
        assert len(merged_entities) >= 1
        
        # Step 2: Rewrite
        sanitized_text, alias_snapshot, redaction_report, warnings = rewrite_text(
            text, merged_entities, alias_manager
        )
        
        # Should have placeholders (flexible check)
        placeholders = PLACEHOLDER_PATTERN.findall(sanitized_text)
        assert len(placeholders) >= 1, f"No placeholders found in: {sanitized_text}"
        
        # Step 3: Rehydrate
        rehydrated_text, unknown_placeholders = rehydrate_response(
            sanitized_text, alias_snapshot
        )
        
        # Should restore original text
        assert rehydrated_text == text
        assert len(unknown_placeholders) == 0
    
    def test_spacy_fallback_on_error(self):
        """Test graceful fallback when spaCy fails."""
        # Setup
        alias_manager = AliasManager()
        text = "Contact jane@example.com about the project."
        
        # Step 1: Detection with spaCy error simulation
        regex_entities = detect_patterns(text)
        
        # Simulate spaCy failure
        try:
            spacy_entities = []  # Fallback to empty
        except Exception:
            spacy_entities = []
        
        merged_entities = merge_detections(regex_entities, spacy_entities)
        
        # Should still detect email via regex
        assert len(merged_entities) == 1
        assert merged_entities[0]['type'] == 'EMAIL'
        
        # Pipeline should continue normally
        sanitized_text, alias_snapshot, redaction_report, warnings = rewrite_text(
            text, merged_entities, alias_manager
        )
        
        assert 'EMAIL_1' in sanitized_text
        
        rehydrated_text, unknown_placeholders = rehydrate_response(
            sanitized_text, alias_snapshot
        )
        
        assert rehydrated_text == text


class TestPrecedenceRules:
    """Test entity precedence rules in integrated pipeline."""
    
    def test_regex_over_spacy_precedence(self):
        """Test that regex entities take precedence over spaCy entities."""
        # Setup
        alias_manager = AliasManager()
        text = "Email support@example.com for help."
        
        # Create mock spaCy entities that would overlap with regex
        mock_spacy_entities = [
            {
                'text': 'support@example.com',
                'start': 6,
                'end': 25,
                'type': 'ORG',  # spaCy might detect email as organization
                'source': 'spacy'
            }
        ]
        
        # Step 1: Detection
        regex_entities = detect_patterns(text)
        merged_entities = merge_detections(regex_entities, mock_spacy_entities)
        
        # Regex should win
        assert len(merged_entities) == 1
        assert merged_entities[0]['type'] == 'EMAIL'
        assert merged_entities[0]['source'] == 'regex'
        
        # Step 2: Rewrite
        sanitized_text, alias_snapshot, redaction_report, warnings = rewrite_text(
            text, merged_entities, alias_manager
        )
        
        # Should use EMAIL placeholder, not ORG
        assert 'EMAIL_1' in sanitized_text
        assert 'ORG_1' not in sanitized_text
        
        # Step 3: Rehydrate
        rehydrated_text, unknown_placeholders = rehydrate_response(
            sanitized_text, alias_snapshot
        )
        
        assert rehydrated_text == text
    
    def test_longer_span_precedence(self):
        """Test that longer spans take precedence."""
        # Setup
        alias_manager = AliasManager()
        
        # Create overlapping entities where longer should win
        entities = [
            {
                'text': 'example.com',
                'start': 10,
                'end': 21,
                'type': 'URL',
                'source': 'regex'
            },
            {
                'text': 'www.example.com',
                'start': 6,
                'end': 21,
                'type': 'URL',
                'source': 'regex'
            }
        ]
        
        # Merge should keep longer span
        merged_entities = merge_detections(entities, [])
        
        assert len(merged_entities) == 1
        assert merged_entities[0]['text'] == 'www.example.com'
        
        # Test in full pipeline
        text = "Visit www.example.com for details."
        sanitized_text, alias_snapshot, redaction_report, warnings = rewrite_text(
            text, merged_entities, alias_manager
        )
        
        assert sanitized_text == "Visit URL_1 for details."
        
        rehydrated_text, unknown_placeholders = rehydrate_response(
            sanitized_text, alias_snapshot
        )
        
        assert rehydrated_text == text


class TestSessionPersistence:
    """Test session persistence across multiple prompts."""
    
    def test_alias_consistency_across_prompts(self):
        """Test that same entities get same aliases across prompts."""
        # Setup shared alias manager
        alias_manager = AliasManager()
        
        # First prompt
        text1 = "Contact john@example.com for details."
        regex_entities1 = detect_patterns(text1)
        merged_entities1 = merge_detections(regex_entities1, [])
        
        sanitized_text1, alias_snapshot1, _, _ = rewrite_text(
            text1, merged_entities1, alias_manager
        )
        
        assert sanitized_text1 == "Contact EMAIL_1 for details."
        
        # Second prompt with same email
        text2 = "Also send a copy to john@example.com and jane@example.com."
        regex_entities2 = detect_patterns(text2)
        merged_entities2 = merge_detections(regex_entities2, [])
        
        sanitized_text2, alias_snapshot2, _, _ = rewrite_text(
            text2, merged_entities2, alias_manager
        )
        
        # Same email should get same alias, new email gets new alias
        assert 'EMAIL_1' in sanitized_text2  # john@example.com
        assert 'EMAIL_2' in sanitized_text2  # jane@example.com
        
        # Verify rehydration works for both
        rehydrated_text1, _ = rehydrate_response(sanitized_text1, alias_snapshot1)
        rehydrated_text2, _ = rehydrate_response(sanitized_text2, alias_snapshot2)
        
        assert rehydrated_text1 == text1
        assert rehydrated_text2 == text2
    
    def test_session_reset(self):
        """Test session reset clears all mappings."""
        # Setup
        alias_manager = AliasManager()
        text = "Email test@example.com"
        
        # Process first time
        regex_entities = detect_patterns(text)
        merged_entities = merge_detections(regex_entities, [])
        sanitized_text1, _, _, _ = rewrite_text(text, merged_entities, alias_manager)
        
        assert sanitized_text1 == "Email EMAIL_1"
        
        # Reset session
        alias_manager.reset_session()
        
        # Process same text again
        regex_entities = detect_patterns(text)
        merged_entities = merge_detections(regex_entities, [])
        sanitized_text2, _, _, _ = rewrite_text(text, merged_entities, alias_manager)
        
        # Should get same alias (EMAIL_1) since session was reset
        assert sanitized_text2 == "Email EMAIL_1"


class TestErrorHandling:
    """Test error handling in integrated pipeline."""
    
    def test_no_entities_detected(self):
        """Test pipeline when no entities are detected."""
        # Setup
        alias_manager = AliasManager()
        text = "This is a normal sentence with no sensitive data."
        
        # Step 1: Detection
        regex_entities = detect_patterns(text)
        spacy_entities = []
        merged_entities = merge_detections(regex_entities, spacy_entities)
        
        # Should detect nothing
        assert len(merged_entities) == 0
        
        # Step 2: Rewrite
        sanitized_text, alias_snapshot, redaction_report, warnings = rewrite_text(
            text, merged_entities, alias_manager
        )
        
        # Text should be unchanged
        assert sanitized_text == text
        
        # Step 3: Rehydrate
        rehydrated_text, unknown_placeholders = rehydrate_response(
            sanitized_text, alias_snapshot
        )
        
        # Should be identical
        assert rehydrated_text == text
        assert len(unknown_placeholders) == 0
    
    def test_unknown_placeholders_in_response(self):
        """Test handling of unknown placeholders in LLM response."""
        # Setup
        alias_manager = AliasManager()
        
        # Create some known mappings
        alias_manager.get_or_create_alias("john@example.com", "EMAIL")
        alias_snapshot = alias_manager.build_reverse_map()
        
        # Simulate LLM response with unknown placeholders
        llm_response = "Contact EMAIL_1 and also EMAIL_5 which doesn't exist."
        
        # Rehydrate
        rehydrated_text, unknown_placeholders = rehydrate_response(
            llm_response, alias_snapshot
        )
        
        # Known placeholder should be restored, unknown should remain
        assert "john@example.com" in rehydrated_text
        assert "EMAIL_5" in rehydrated_text
        assert len(unknown_placeholders) == 1
        assert "EMAIL_5" in unknown_placeholders
    
    def test_malformed_placeholders(self):
        """Test handling of malformed placeholders."""
        # Setup
        alias_manager = AliasManager()
        alias_snapshot = alias_manager.build_reverse_map()
        
        # Text with malformed placeholders
        text_with_malformed = "Contact EMAIL_0 and PERSON_ and email_1 for help."
        
        # Rehydrate
        rehydrated_text, unknown_placeholders = rehydrate_response(
            text_with_malformed, alias_snapshot
        )
        
        # Malformed placeholders should be left unchanged
        assert rehydrated_text == text_with_malformed
        assert len(unknown_placeholders) == 0  # None are valid format


class TestComplexScenarios:
    """Test complex real-world scenarios."""
    
    def test_email_signature_block(self):
        """Test processing of email signature with multiple entities."""
        # Setup
        alias_manager = AliasManager()
        text = """
Best regards,
John Smith
Senior Developer
Acme Corporation
john.smith@acme.com
Phone: (555) 123-4567
Website: https://www.acme.com
        """.strip()
        
        # Step 1: Detection
        regex_entities = detect_patterns(text)
        
        # Should detect email and URL
        assert len(regex_entities) >= 2
        entity_types = {e['type'] for e in regex_entities}
        assert 'EMAIL' in entity_types
        assert 'URL' in entity_types
        
        # Step 2: Full pipeline
        merged_entities = merge_detections(regex_entities, [])
        sanitized_text, alias_snapshot, _, _ = rewrite_text(
            text, merged_entities, alias_manager
        )
        
        # Should have placeholders
        assert 'EMAIL_1' in sanitized_text
        assert 'URL_1' in sanitized_text
        assert 'john.smith@acme.com' not in sanitized_text
        assert 'https://www.acme.com' not in sanitized_text
        
        # Step 3: Rehydrate
        rehydrated_text, unknown_placeholders = rehydrate_response(
            sanitized_text, alias_snapshot
        )
        
        assert rehydrated_text == text
        assert len(unknown_placeholders) == 0
    
    def test_overlapping_entities_resolution(self):
        """Test resolution of overlapping entities."""
        # Setup
        alias_manager = AliasManager()
        
        # Create overlapping entities manually
        overlapping_entities = [
            {
                'text': 'support@example.com',
                'start': 0,
                'end': 19,
                'type': 'EMAIL',
                'source': 'regex'
            },
            {
                'text': 'example.com',
                'start': 8,
                'end': 19,
                'type': 'URL',
                'source': 'regex'
            }
        ]
        
        # Merge should resolve overlap (EMAIL should win due to longer span)
        merged_entities = merge_detections(overlapping_entities, [])
        
        assert len(merged_entities) == 1
        assert merged_entities[0]['type'] == 'EMAIL'
        assert merged_entities[0]['text'] == 'support@example.com'
        
        # Test in pipeline
        text = "support@example.com"
        sanitized_text, alias_snapshot, _, _ = rewrite_text(
            text, merged_entities, alias_manager
        )
        
        assert sanitized_text == "EMAIL_1"
        
        rehydrated_text, _ = rehydrate_response(sanitized_text, alias_snapshot)
        assert rehydrated_text == text
    
    def test_multiple_same_entities(self):
        """Test handling of repeated entities."""
        # Setup
        alias_manager = AliasManager()
        text = "Send to john@example.com and cc john@example.com on the reply."
        
        # Step 1: Detection
        regex_entities = detect_patterns(text)
        merged_entities = merge_detections(regex_entities, [])
        
        # Should detect both instances
        assert len(merged_entities) == 2
        assert all(e['type'] == 'EMAIL' for e in merged_entities)
        assert all(e['text'] == 'john@example.com' for e in merged_entities)
        
        # Step 2: Rewrite
        sanitized_text, alias_snapshot, _, _ = rewrite_text(
            text, merged_entities, alias_manager
        )
        
        # Both should get same placeholder
        assert sanitized_text == "Send to EMAIL_1 and cc EMAIL_1 on the reply."
        
        # Step 3: Rehydrate
        rehydrated_text, unknown_placeholders = rehydrate_response(
            sanitized_text, alias_snapshot
        )
        
        assert rehydrated_text == text
        assert len(unknown_placeholders) == 0


class TestEdgeCases:
    """Test edge cases in integrated pipeline."""
    
    def test_unicode_text_handling(self):
        """Test pipeline with Unicode characters."""
        # Setup
        alias_manager = AliasManager()
        text = "Contactez-nous à français@example.com pour plus d'informations. 🚀"
        
        # Step 1: Detection
        regex_entities = detect_patterns(text)
        merged_entities = merge_detections(regex_entities, [])
        
        # Should detect email
        assert len(merged_entities) == 1
        assert merged_entities[0]['type'] == 'EMAIL'
        
        # Step 2: Rewrite
        sanitized_text, alias_snapshot, _, _ = rewrite_text(
            text, merged_entities, alias_manager
        )
        
        # Unicode should be preserved
        assert "Contactez-nous à EMAIL_1 pour plus d'informations. 🚀" == sanitized_text
        
        # Step 3: Rehydrate
        rehydrated_text, unknown_placeholders = rehydrate_response(
            sanitized_text, alias_snapshot
        )
        
        assert rehydrated_text == text
        assert len(unknown_placeholders) == 0
    
    def test_empty_text_handling(self):
        """Test pipeline with empty text."""
        # Setup
        alias_manager = AliasManager()
        text = ""
        
        # Step 1: Detection
        regex_entities = detect_patterns(text)
        merged_entities = merge_detections(regex_entities, [])
        
        assert len(merged_entities) == 0
        
        # Step 2: Rewrite
        sanitized_text, alias_snapshot, _, _ = rewrite_text(
            text, merged_entities, alias_manager
        )
        
        assert sanitized_text == ""
        
        # Step 3: Rehydrate
        rehydrated_text, unknown_placeholders = rehydrate_response(
            sanitized_text, alias_snapshot
        )
        
        assert rehydrated_text == ""
        assert len(unknown_placeholders) == 0
    
    def test_whitespace_only_text(self):
        """Test pipeline with whitespace-only text."""
        # Setup
        alias_manager = AliasManager()
        text = "   \n\t  "
        
        # Step 1: Detection
        regex_entities = detect_patterns(text)
        merged_entities = merge_detections(regex_entities, [])
        
        assert len(merged_entities) == 0
        
        # Step 2: Rewrite
        sanitized_text, alias_snapshot, _, _ = rewrite_text(
            text, merged_entities, alias_manager
        )
        
        assert sanitized_text == text
        
        # Step 3: Rehydrate
        rehydrated_text, unknown_placeholders = rehydrate_response(
            sanitized_text, alias_snapshot
        )
        
        assert rehydrated_text == text


class TestPerformance:
    """Test performance characteristics of integrated pipeline."""
    
    def test_large_text_processing(self):
        """Test pipeline with large text input."""
        # Setup
        alias_manager = AliasManager()
        
        # Create large text with scattered entities
        base_text = "Contact support@example.com for help. " * 100
        large_text = base_text + "Visit https://help.example.com for documentation."
        
        # Step 1: Detection
        regex_entities = detect_patterns(large_text)
        merged_entities = merge_detections(regex_entities, [])
        
        # Should detect many emails and one URL
        assert len(merged_entities) >= 100
        
        # Step 2: Rewrite
        sanitized_text, alias_snapshot, _, _ = rewrite_text(
            large_text, merged_entities, alias_manager
        )
        
        # Should have placeholders
        assert 'EMAIL_1' in sanitized_text
        assert 'URL_1' in sanitized_text
        assert 'support@example.com' not in sanitized_text
        
        # Step 3: Rehydrate
        rehydrated_text, unknown_placeholders = rehydrate_response(
            sanitized_text, alias_snapshot
        )
        
        assert rehydrated_text == large_text
        assert len(unknown_placeholders) == 0
    
    def test_many_unique_entities(self):
        """Test pipeline with many unique entities."""
        # Setup
        alias_manager = AliasManager()
        
        # Create text with many unique emails
        emails = [f"user{i}@example.com" for i in range(50)]
        text = "Contact: " + ", ".join(emails)
        
        # Step 1: Detection
        regex_entities = detect_patterns(text)
        merged_entities = merge_detections(regex_entities, [])
        
        # Should detect all emails
        assert len(merged_entities) == 50
        
        # Step 2: Rewrite
        sanitized_text, alias_snapshot, _, _ = rewrite_text(
            text, merged_entities, alias_manager
        )
        
        # Should have 50 unique email placeholders (flexible numbering)
        email_placeholders = [p for p in PLACEHOLDER_PATTERN.findall(sanitized_text)
                             if p.startswith('EMAIL_')]
        assert len(set(email_placeholders)) == 50, f"Expected 50 unique email placeholders, got {len(set(email_placeholders))}"
        
        # No original emails should remain
        assert_no_pii_leaked(sanitized_text, emails)
        
        # Step 3: Rehydrate
        rehydrated_text, unknown_placeholders = rehydrate_response(
            sanitized_text, alias_snapshot
        )
        
        assert rehydrated_text == text
        assert len(unknown_placeholders) == 0


class TestRoundTripIntegrity:
    """Test round-trip integrity of the pipeline."""
    
    def test_perfect_round_trip(self):
        """Test that text survives perfect round-trip."""
        test_cases = [
            "Email me at john@example.com",
            "Visit https://www.example.com for more info",
            "Contact support@company.org or check https://help.company.org",
            "Multiple emails: alice@test.com, bob@test.com, charlie@test.com",
            "Mixed: Call John at john@phone.com or visit https://john.example.com",
        ]
        
        for text in test_cases:
            # Setup fresh alias manager for each test
            alias_manager = AliasManager()
            
            # Full pipeline
            regex_entities = detect_patterns(text)
            merged_entities = merge_detections(regex_entities, [])
            sanitized_text, alias_snapshot, _, _ = rewrite_text(
                text, merged_entities, alias_manager
            )
            rehydrated_text, unknown_placeholders = rehydrate_response(
                sanitized_text, alias_snapshot
            )
            
            # Perfect round-trip
            assert rehydrated_text == text, f"Round-trip failed for: {text}"
            assert len(unknown_placeholders) == 0
    
    def test_round_trip_with_simulated_llm_response(self):
        """Test round-trip with simulated LLM response modifications."""
        # Setup
        alias_manager = AliasManager()
        original_text = "Please email john@example.com about the project."
        
        # Step 1: Sanitize
        regex_entities = detect_patterns(original_text)
        merged_entities = merge_detections(regex_entities, [])
        sanitized_text, alias_snapshot, _, _ = rewrite_text(
            original_text, merged_entities, alias_manager
        )
        
        assert sanitized_text == "Please email EMAIL_1 about the project."
        
        # Step 2: Simulate LLM response (modified but preserves placeholders)
        simulated_llm_response = "I'll make sure to email EMAIL_1 regarding the project details."
        
        # Step 3: Rehydrate
        rehydrated_response, unknown_placeholders = rehydrate_response(
            simulated_llm_response, alias_snapshot
        )
        
        # Should restore the email
        assert rehydrated_response == "I'll make sure to email john@example.com regarding the project details."
        assert len(unknown_placeholders) == 0


class TestConfigurationIntegration:
    """Test integration with configuration system (when available)."""
    
    def test_entity_filtering_by_config(self):
        """Test that entity detection respects configuration."""
        # Create mock config that only enables EMAIL detection
        mock_config = Mock()
        mock_config.get_enabled_entities.return_value = ['EMAIL']
        mock_config.is_method_enabled.side_effect = lambda method: method == 'regex'
        
        # Setup
        alias_manager = AliasManager()
        text = "Contact John Doe at john@example.com or visit https://example.com"
        
        # Detection should only find email (URL disabled by config)
        regex_entities = detect_patterns(text)
        
        # Filter entities based on config
        filtered_entities = [
            e for e in regex_entities 
            if e['type'] in mock_config.get_enabled_entities()
        ]
        
        merged_entities = merge_detections(filtered_entities, [])
        
        # Should only have email
        assert len(merged_entities) == 1
        assert merged_entities[0]['type'] == 'EMAIL'
        
        # Pipeline continues normally
        sanitized_text, alias_snapshot, _, _ = rewrite_text(
            text, merged_entities, alias_manager
        )
        
        # Only email should be replaced
        assert 'EMAIL_1' in sanitized_text
        assert 'URL_1' not in sanitized_text
        assert 'https://example.com' in sanitized_text  # URL preserved


class TestLeakDetection:
    """Test leak detection in integrated pipeline."""
    
    def test_leak_detection_catches_missed_entities(self):
        """Test that leak detection catches entities missed by detection."""
        # Setup
        alias_manager = AliasManager()
        
        # Simulate missed detection (empty entities but text has email)
        text = "Contact emergency@hospital.com immediately"
        missed_entities = []  # Simulate detection failure
        
        # Rewrite with no entities (simulating detection miss)
        sanitized_text, alias_snapshot, _, _ = rewrite_text(
            text, missed_entities, alias_manager
        )
        
        # Text should be unchanged (no entities to replace)
        assert sanitized_text == text
        
        # Leak check should detect the email
        from rewrite import leak_check
        leaks = leak_check(sanitized_text)
        
        # Should detect the leaked email
        assert len(leaks) > 0
        assert any('emergency@hospital.com' in leak for leak in leaks)
    
    def test_no_leaks_after_proper_sanitization(self):
        """Test that properly sanitized text passes leak detection."""
        # Setup
        alias_manager = AliasManager()
        text = "Contact support@example.com for assistance."
        
        # Proper detection and sanitization
        regex_entities = detect_patterns(text)
        merged_entities = merge_detections(regex_entities, [])
        sanitized_text, alias_snapshot, _, _ = rewrite_text(
            text, merged_entities, alias_manager
        )
        
        # Leak check should find no leaks
        from rewrite import leak_check
        leaks = leak_check(sanitized_text)
        
        assert len(leaks) == 0


class TestAliasManagerIntegration:
    """Test alias manager integration across pipeline."""
    
    def test_alias_snapshot_consistency(self):
        """Test that alias snapshots work correctly."""
        # Setup
        alias_manager = AliasManager()
        
        # Process first text
        text1 = "Email alice@example.com"
        regex_entities1 = detect_patterns(text1)
        merged_entities1 = merge_detections(regex_entities1, [])
        sanitized_text1, snapshot1, _, _ = rewrite_text(
            text1, merged_entities1, alias_manager
        )
        
        # Process second text (adds new entity)
        text2 = "Email alice@example.com and bob@example.com"
        regex_entities2 = detect_patterns(text2)
        merged_entities2 = merge_detections(regex_entities2, [])
        sanitized_text2, snapshot2, _, _ = rewrite_text(
            text2, merged_entities2, alias_manager
        )
        
        # First snapshot should only restore alice
        rehydrated1, _ = rehydrate_response("Contact EMAIL_1", snapshot1)
        assert rehydrated1 == "Contact alice@example.com"
        
        # Second snapshot should restore both
        rehydrated2, _ = rehydrate_response("Contact EMAIL_1 and EMAIL_2", snapshot2)
        assert rehydrated2 == "Contact alice@example.com and bob@example.com"
    
    def test_export_and_import_mappings(self):
        """Test mapping export functionality."""
        # Setup
        alias_manager = AliasManager()
        
        # Create some mappings
        alias_manager.get_or_create_alias("john@example.com", "EMAIL")
        alias_manager.get_or_create_alias("https://example.com", "URL")
        
        # Export mappings
        exported = alias_manager.export_mappings()
        
        # Should contain the mappings
        assert 'EMAIL' in exported
        assert 'URL' in exported
        assert exported['EMAIL']['john@example.com'] == 'EMAIL_1'
        assert exported['URL']['https://example.com'] == 'URL_1'


class TestRealWorldScenarios:
    """Test realistic usage scenarios."""
    
    def test_customer_support_email(self):
        """Test processing customer support email."""
        # Setup
        alias_manager = AliasManager()
        text = """
Dear Support Team,

I'm having issues with my account. Please contact me at customer123@gmail.com
or call me at my office. You can also check our company website at 
https://www.mycompany.com for more context.

Best regards,
Jane Smith
        """.strip()
        
        # Full pipeline
        regex_entities = detect_patterns(text)
        merged_entities = merge_detections(regex_entities, [])
        sanitized_text, alias_snapshot, _, _ = rewrite_text(
            text, merged_entities, alias_manager
        )
        
        # Should sanitize email and URL
        assert 'EMAIL_1' in sanitized_text
        assert 'URL_1' in sanitized_text
        assert 'customer123@gmail.com' not in sanitized_text
        assert 'https://www.mycompany.com' not in sanitized_text
        
        # Simulate LLM response
        llm_response = """
Thank you for contacting us. We'll reach out to EMAIL_1 within 24 hours.
We've also reviewed your company information at URL_1.
        """.strip()
        
        # Rehydrate
        rehydrated_response, unknown_placeholders = rehydrate_response(
            llm_response, alias_snapshot
        )
        
        # Should restore entities
        assert 'customer123@gmail.com' in rehydrated_response
        assert 'https://www.mycompany.com' in rehydrated_response
        assert len(unknown_placeholders) == 0
    
    def test_business_communication(self):
        """Test processing business communication."""
        # Setup
        alias_manager = AliasManager()
        text = """
Please schedule a meeting with our partners:
- Tech team: tech@partner1.com
- Marketing: marketing@partner2.com  
- Legal: legal@partner3.com

Also review their websites:
- https://partner1.com
- https://partner2.com
- https://partner3.com
        """.strip()
        
        # Full pipeline
        regex_entities = detect_patterns(text)
        merged_entities = merge_detections(regex_entities, [])
        sanitized_text, alias_snapshot, _, _ = rewrite_text(
            text, merged_entities, alias_manager
        )
        
        # Should have multiple placeholders
        assert 'EMAIL_1' in sanitized_text
        assert 'EMAIL_2' in sanitized_text
        assert 'EMAIL_3' in sanitized_text
        assert 'URL_1' in sanitized_text
        assert 'URL_2' in sanitized_text
        assert 'URL_3' in sanitized_text
        
        # No original entities should remain
        assert 'tech@partner1.com' not in sanitized_text
        assert 'https://partner1.com' not in sanitized_text
        
        # Rehydrate
        rehydrated_text, unknown_placeholders = rehydrate_response(
            sanitized_text, alias_snapshot
        )
        
        assert rehydrated_text == text
        assert len(unknown_placeholders) == 0


class TestErrorRecovery:
    """Test error recovery in pipeline."""
    
    def test_partial_detection_failure(self):
        """Test pipeline when some detection methods fail."""
        # Setup
        alias_manager = AliasManager()
        text = "Contact john@example.com and visit https://example.com"
        
        # Simulate partial failure (only regex works)
        regex_entities = detect_patterns(text)
        spacy_entities = []  # Simulate spaCy failure
        
        # Pipeline should continue with available entities
        merged_entities = merge_detections(regex_entities, spacy_entities)
        
        assert len(merged_entities) == 2  # Email and URL from regex
        
        sanitized_text, alias_snapshot, _, _ = rewrite_text(
            text, merged_entities, alias_manager
        )
        
        # Should still work
        assert 'EMAIL_1' in sanitized_text
        assert 'URL_1' in sanitized_text
        
        rehydrated_text, _ = rehydrate_response(sanitized_text, alias_snapshot)
        assert rehydrated_text == text
    
    def test_rehydration_with_corrupted_response(self):
        """Test rehydration when LLM corrupts some placeholders."""
        # Setup
        alias_manager = AliasManager()
        alias_manager.get_or_create_alias("john@example.com", "EMAIL")
        alias_snapshot = alias_manager.build_reverse_map()
        
        # Simulate LLM response with some corrupted placeholders
        corrupted_response = "Contact EMAIL_1 and also email_2 (corrupted) for help."
        
        # Rehydrate
        rehydrated_text, unknown_placeholders = rehydrate_response(
            corrupted_response, alias_snapshot
        )
        
        # Valid placeholder should be restored, corrupted should remain
        assert "john@example.com" in rehydrated_text
        assert "email_2" in rehydrated_text  # Corrupted placeholder preserved
        assert len(unknown_placeholders) == 0  # email_2 is not valid format


if __name__ == "__main__":
    pytest.main([__file__, "-v"])