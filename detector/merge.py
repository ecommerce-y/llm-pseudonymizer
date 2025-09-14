"""
Entity merge and conflict resolution module.

This module combines detection results from multiple sources (regex and spaCy NER)
and resolves conflicts between overlapping entity spans according to the system's
precedence rules.
"""

from typing import List, Dict, Any, Optional


def merge_detections(regex_entities: List[Dict[str, Any]], 
                    spacy_entities: List[Dict[str, Any]], 
                    ml_entities: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Merge entity detections from multiple sources.
    
    Implements the global precedence rules:
    1. All regex > all NER/ML
    2. Within same source: longer span > shorter span
    3. Same length: earlier position wins
    
    Args:
        regex_entities: Entities from regex detection
        spacy_entities: Entities from spaCy NER
        ml_entities: Future ML detections (optional)
        
    Returns:
        List of non-overlapping entities sorted by position
    """
    # Handle None inputs
    if regex_entities is None:
        regex_entities = []
    if spacy_entities is None:
        spacy_entities = []
    if ml_entities is None:
        ml_entities = []
    
    # Validate all entities
    all_input_entities = regex_entities + spacy_entities + ml_entities
    for entity in all_input_entities:
        validate_entity_format(entity)
    
    # Combine all entities
    all_entities = []
    all_entities.extend(regex_entities)
    all_entities.extend(spacy_entities)
    all_entities.extend(ml_entities)
    
    if not all_entities:
        return []
    
    # Remove exact duplicates first
    unique_entities = deduplicate_entities(all_entities)
    
    # Resolve overlaps according to precedence rules
    non_overlapping = resolve_overlaps(unique_entities)
    
    # Sort by position for consistent output
    non_overlapping.sort(key=lambda e: (e['start'], e['end']))
    
    return non_overlapping


def resolve_overlaps(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Resolve overlapping entities according to precedence rules.
    
    Args:
        entities: List of entities that may overlap
        
    Returns:
        List of non-overlapping entities
    """
    if not entities:
        return []
    
    # Sort entities by start position (ascending) and end position (descending)
    # This ensures we process entities in a consistent order
    sorted_entities = sorted(entities, key=lambda e: (e['start'], -e['end']))
    
    non_overlapping = []
    
    for entity in sorted_entities:
        # Check if this entity overlaps with any already selected
        conflicts = []
        for i, selected in enumerate(non_overlapping):
            if overlaps(entity, selected):
                conflicts.append((i, selected))
        
        if not conflicts:
            # No conflicts, add the entity
            non_overlapping.append(entity)
        else:
            # Handle conflicts - determine if new entity should replace any existing ones
            entities_to_remove = []
            should_add_new = True
            
            for i, conflicting_entity in conflicts:
                if should_keep_new_entity(entity, conflicting_entity):
                    # New entity wins, mark old one for removal
                    entities_to_remove.append(i)
                else:
                    # Existing entity wins, don't add new one
                    should_add_new = False
                    break
            
            if should_add_new:
                # Remove conflicting entities (in reverse order to maintain indices)
                for i in sorted(entities_to_remove, reverse=True):
                    non_overlapping.pop(i)
                # Add the new entity
                non_overlapping.append(entity)
    
    return non_overlapping


def overlaps(entity1: Dict[str, Any], entity2: Dict[str, Any]) -> bool:
    """
    Check if two entities have overlapping spans.
    
    Two entities overlap if they share any character positions.
    
    Args:
        entity1: First entity
        entity2: Second entity
        
    Returns:
        True if entities overlap
    """
    # Entities overlap if one starts before the other ends
    return not (entity1['end'] <= entity2['start'] or 
                entity2['end'] <= entity1['start'])


def should_keep_new_entity(new_entity: Dict[str, Any], 
                          existing_entity: Dict[str, Any]) -> bool:
    """
    Determine which entity to keep based on precedence rules.
    
    Precedence rules:
    1. regex > spacy/ml
    2. Longer span > shorter span
    3. Earlier position > later position
    
    Args:
        new_entity: Entity being considered
        existing_entity: Already selected entity
        
    Returns:
        True if new entity should replace existing
    """
    # Rule 1: Source precedence (regex > others)
    if new_entity['source'] == 'regex' and existing_entity['source'] != 'regex':
        return True
    if existing_entity['source'] == 'regex' and new_entity['source'] != 'regex':
        return False
    
    # Same source level, check span length
    new_length = new_entity['end'] - new_entity['start']
    existing_length = existing_entity['end'] - existing_entity['start']
    
    # Rule 2: Longer span wins
    if new_length > existing_length:
        return True
    if existing_length > new_length:
        return False
    
    # Rule 3: Same length, earlier position wins
    if new_entity['start'] < existing_entity['start']:
        return True
    
    return False


def deduplicate_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove exact duplicate entities.
    
    Entities are considered duplicates if they have the same
    text, start, end, and type (source may differ).
    
    Args:
        entities: List of entities
        
    Returns:
        List with duplicates removed
    """
    seen = set()
    unique = []
    
    for entity in entities:
        # Create a key from the entity's core properties
        key = (entity['text'], entity['start'], entity['end'], entity['type'])
        
        if key not in seen:
            seen.add(key)
            unique.append(entity)
    
    return unique


def prioritize_entities(overlapping_group: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Select the highest priority entity from overlapping entities.
    
    Args:
        overlapping_group: List of overlapping entities
        
    Returns:
        The entity with highest precedence
        
    Raises:
        ValueError: If group is empty
    """
    if not overlapping_group:
        raise ValueError("Cannot prioritize empty group")
    
    # Sort by precedence rules
    def precedence_key(entity):
        # Source priority: regex=0, others=1
        source_priority = 0 if entity['source'] == 'regex' else 1
        
        # Span length (negative for descending sort)
        span_length = -(entity['end'] - entity['start'])
        
        # Start position
        start_pos = entity['start']
        
        return (source_priority, span_length, start_pos)
    
    # Return the highest priority entity
    return min(overlapping_group, key=precedence_key)


def validate_entity_format(entity: Dict[str, Any]) -> None:
    """
    Validate that entity has required fields and valid values.
    
    Args:
        entity: Entity dictionary to validate
        
    Raises:
        ValueError: If entity format is invalid
    """
    required_fields = ['text', 'start', 'end', 'type', 'source']
    
    for field in required_fields:
        if field not in entity:
            raise ValueError(f"Entity missing required field: {field}")
    
    # Validate field types
    if not isinstance(entity['text'], str):
        raise ValueError("Entity 'text' must be a string")
    
    if not isinstance(entity['start'], int) or not isinstance(entity['end'], int):
        raise ValueError("Entity 'start' and 'end' must be integers")
    
    if not isinstance(entity['type'], str):
        raise ValueError("Entity 'type' must be a string")
    
    if not isinstance(entity['source'], str):
        raise ValueError("Entity 'source' must be a string")
    
    # Validate values
    if entity['start'] < 0 or entity['end'] < 0:
        raise ValueError("Entity indices must be non-negative")
    
    if entity['start'] >= entity['end']:
        raise ValueError("Entity start must be less than end")
    
    # Validate entity type
    valid_types = {'PERSON', 'ORG', 'EMAIL', 'URL', 'PHONE'}
    if entity['type'] not in valid_types:
        raise ValueError(f"Invalid entity type: {entity['type']}. Must be one of {valid_types}")
    
    # Validate source
    valid_sources = {'regex', 'spacy', 'ml'}
    if entity['source'] not in valid_sources:
        raise ValueError(f"Invalid entity source: {entity['source']}. Must be one of {valid_sources}")


def get_overlap_groups(entities: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Group entities by overlapping spans.
    
    This is a utility function that can be used for analysis or debugging.
    
    Args:
        entities: List of entities
        
    Returns:
        List of groups, where each group contains overlapping entities
    """
    if not entities:
        return []
    
    # Sort entities by start position
    sorted_entities = sorted(entities, key=lambda e: e['start'])
    
    groups = []
    current_group = [sorted_entities[0]]
    
    for entity in sorted_entities[1:]:
        # Check if this entity overlaps with any in the current group
        overlaps_with_group = any(overlaps(entity, group_entity) 
                                 for group_entity in current_group)
        
        if overlaps_with_group:
            current_group.append(entity)
        else:
            # Start a new group
            groups.append(current_group)
            current_group = [entity]
    
    # Add the last group
    if current_group:
        groups.append(current_group)
    
    return groups


def merge_statistics(regex_entities: List[Dict[str, Any]], 
                    spacy_entities: List[Dict[str, Any]], 
                    merged_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate statistics about the merge process.
    
    Args:
        regex_entities: Original regex entities
        spacy_entities: Original spacy entities
        merged_entities: Final merged entities
        
    Returns:
        Dictionary with merge statistics
    """
    total_input = len(regex_entities) + len(spacy_entities)
    total_output = len(merged_entities)
    
    # Count by source in output
    source_counts = {}
    for entity in merged_entities:
        source = entity['source']
        source_counts[source] = source_counts.get(source, 0) + 1
    
    # Count by type in output
    type_counts = {}
    for entity in merged_entities:
        entity_type = entity['type']
        type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
    
    return {
        'input_total': total_input,
        'input_regex': len(regex_entities),
        'input_spacy': len(spacy_entities),
        'output_total': total_output,
        'entities_removed': total_input - total_output,
        'source_distribution': source_counts,
        'type_distribution': type_counts
    }