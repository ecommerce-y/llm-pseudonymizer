"""
Aliases Module - Entity-to-Placeholder Mapping Management

This module provides the AliasManager class for managing bidirectional mappings
between sensitive entities and their pseudonymized placeholders during a session.

The module maintains consistency across a session, ensuring that the same entity
always receives the same placeholder, and provides utilities for session management
and rehydration support.
"""

from typing import Dict, List, Optional
import copy


class AliasManager:
    """
    Manages entity-to-placeholder mappings for pseudonymization sessions.
    
    Maintains bidirectional mappings between sensitive entities (names, emails, etc.)
    and their placeholder representations (PERSON_1, EMAIL_1, etc.). Ensures
    consistency within a session and provides utilities for rehydration.
    """
    
    def __init__(self):
        """
        Initialize a new AliasManager with empty mappings.
        
        Creates forward maps for each supported entity type and initializes
        counters for sequential placeholder generation.
        """
        # Forward maps: entity_text -> placeholder
        # Organized by entity type for efficient management
        self.entity_maps = {
            "PERSON": {},    # e.g., {"John Doe": "PERSON_1"}
            "ORG": {},       # e.g., {"Acme Corp": "ORG_1"}
            "EMAIL": {},     # e.g., {"john@example.com": "EMAIL_1"}
            "URL": {}        # e.g., {"https://example.com": "URL_1"}
        }
        
        # Counters track the next available number for each type
        self.counters = {
            "PERSON": 1,
            "ORG": 1,
            "EMAIL": 1,
            "URL": 1
        }
    
    def get_or_create_alias(self, entity_text: str, entity_type: str) -> str:
        """
        Returns existing placeholder or creates a new one for the given entity.
        
        This is the main interface for obtaining placeholders during text sanitization.
        If the entity has been seen before, returns the existing placeholder.
        Otherwise, creates a new placeholder with the next sequential number.
        
        Args:
            entity_text: The sensitive text to be replaced (e.g., "John Doe")
            entity_type: One of "PERSON", "ORG", "EMAIL", "URL"
            
        Returns:
            Placeholder string (e.g., "PERSON_1")
            
        Raises:
            ValueError: If entity_type is not supported
        """
        # Validate entity type
        if entity_type not in self.entity_maps:
            raise ValueError(f"Unknown entity type: {entity_type}. "
                           f"Supported types: {list(self.entity_maps.keys())}")
        
        # Check if alias already exists
        if entity_text in self.entity_maps[entity_type]:
            return self.entity_maps[entity_type][entity_text]
        
        # Create new alias
        placeholder = f"{entity_type}_{self.counters[entity_type]}"
        
        # Store mapping
        self.entity_maps[entity_type][entity_text] = placeholder
        
        # Increment counter
        self.counters[entity_type] += 1
        
        return placeholder
    
    def build_reverse_map(self) -> Dict[str, str]:
        """
        Constructs and returns a reverse mapping (placeholder -> entity_text).
        
        Used by the rehydration module to restore original text from placeholders.
        Builds the mapping on-demand rather than maintaining it continuously.
        
        Returns:
            Dictionary mapping placeholders to original entities
            e.g., {"PERSON_1": "John Doe", "EMAIL_1": "john@example.com"}
        """
        reverse_map = {}
        
        # Iterate through all entity types
        for entity_type, entities in self.entity_maps.items():
            # Add each mapping in reverse
            for entity_text, placeholder in entities.items():
                reverse_map[placeholder] = entity_text
        
        return reverse_map
    
    def reset_session(self) -> None:
        """
        Clears all mappings and resets counters for a new session.
        
        Privacy-preserving session management that ensures no data persists
        between sessions. Clears all entity mappings and resets counters to 1.
        """
        # Clear all entity mappings
        for entity_type in self.entity_maps:
            self.entity_maps[entity_type].clear()
        
        # Reset all counters to 1
        for entity_type in self.counters:
            self.counters[entity_type] = 1
    
    def get_next_counter(self, entity_type: str) -> int:
        """
        Returns the next available counter value for a given entity type.
        
        Utility method for debugging and testing to inspect the current
        state of counter progression.
        
        Args:
            entity_type: One of "PERSON", "ORG", "EMAIL", "URL"
            
        Returns:
            Next counter value that would be used for new entities
            
        Raises:
            ValueError: If entity_type is not supported
        """
        if entity_type not in self.counters:
            raise ValueError(f"Unknown entity type: {entity_type}. "
                           f"Supported types: {list(self.counters.keys())}")
        
        return self.counters[entity_type]
    
    def export_mappings(self) -> Dict[str, Dict[str, str]]:
        """
        Returns current mappings for debugging purposes.
        
        WARNING: This method exposes sensitive entity mappings and should only
        be used for debugging. The returned data contains the original sensitive
        information that the system is designed to protect.
        
        Returns:
            Deep copy of current entity mappings organized by type
        """
        # Log security warning
        print("WARNING: Exporting sensitive entity mappings. Handle with care!")
        
        # Deep copy to prevent external modification
        return copy.deepcopy(self.entity_maps)
    
    def get_entity_count(self) -> Dict[str, int]:
        """
        Returns count of unique entities per type.
        
        Useful for statistics and monitoring session activity without
        exposing the actual sensitive data.
        
        Returns:
            Dictionary mapping entity types to their counts
            e.g., {"PERSON": 3, "ORG": 1, "EMAIL": 2, "URL": 0}
        """
        return {
            entity_type: len(entities)
            for entity_type, entities in self.entity_maps.items()
        }
    
    def has_entity(self, entity_text: str, entity_type: str) -> bool:
        """
        Checks if an entity already has a mapping.
        
        Utility method to test whether an entity has been encountered
        before without creating a new mapping.
        
        Args:
            entity_text: The entity text to check
            entity_type: One of "PERSON", "ORG", "EMAIL", "URL"
            
        Returns:
            True if the entity already has a placeholder, False otherwise
        """
        return (entity_type in self.entity_maps and 
                entity_text in self.entity_maps[entity_type])
    
    def get_all_placeholders(self) -> List[str]:
        """
        Returns all placeholders currently in use.
        
        Useful for debugging and validation to see all active placeholders
        across all entity types.
        
        Returns:
            List of all placeholder strings currently mapped
            e.g., ["PERSON_1", "PERSON_2", "EMAIL_1", "ORG_1"]
        """
        placeholders = []
        for entities in self.entity_maps.values():
            placeholders.extend(entities.values())
        return sorted(placeholders)  # Sort for consistent ordering
    
    def get_placeholder_for_entity(self, entity_text: str, entity_type: str) -> Optional[str]:
        """
        Returns the placeholder for an entity if it exists, None otherwise.
        
        Unlike get_or_create_alias(), this method does not create new mappings.
        Used when you want to check for existing mappings without side effects.
        
        Args:
            entity_text: The entity text to look up
            entity_type: One of "PERSON", "ORG", "EMAIL", "URL"
            
        Returns:
            Existing placeholder string, or None if not found
            
        Raises:
            ValueError: If entity_type is not supported
        """
        if entity_type not in self.entity_maps:
            raise ValueError(f"Unknown entity type: {entity_type}. "
                           f"Supported types: {list(self.entity_maps.keys())}")
        
        return self.entity_maps[entity_type].get(entity_text)