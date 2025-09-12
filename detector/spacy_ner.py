"""
SpaCy NER-based entity detection for PERSON and ORG entities.

This module provides Named Entity Recognition capabilities using spaCy's
pre-trained models. It identifies person names and organization names in text,
complementing the regex-based detection with machine learning-powered recognition.
"""

from typing import List, Dict, Any, Optional


# Module-level cache for the loaded model
_nlp_model = None

# Flag to track if we've already tried loading
_load_attempted = False


def load_model() -> Optional[Any]:
    """
    Load spaCy model with caching and graceful fallback.
    
    The model is loaded lazily on first use and cached for subsequent calls.
    If spaCy or the model is not available, returns None and the system
    continues with regex-only detection.
    
    Returns:
        Loaded spaCy Language object or None if unavailable
    """
    global _nlp_model, _load_attempted
    
    # Return cached model if already loaded
    if _nlp_model is not None:
        return _nlp_model
    
    # Don't retry if we've already failed
    if _load_attempted:
        return None
    
    _load_attempted = True
    
    try:
        import spacy
        _nlp_model = spacy.load("en_core_web_sm")
        
        # Disable unnecessary pipeline components for speed
        # Note: We keep 'ner' and its dependencies
        pipes_to_disable = ["tagger", "parser", "attribute_ruler", "lemmatizer"]
        _nlp_model.disable_pipes(pipes_to_disable)
        
        return _nlp_model
        
    except ImportError:
        print("Warning: spaCy not installed. Install with: pip install spacy")
        return None
    except OSError:
        print("Warning: spaCy model 'en_core_web_sm' not found. "
              "Install with: python -m spacy download en_core_web_sm")
        return None
    except Exception as e:
        print(f"Warning: Failed to load spaCy model: {e}")
        return None


def detect_entities(text: str, nlp_model: Optional[Any] = None) -> List[Dict[str, Any]]:
    """
    Detect PERSON and ORG entities using spaCy NER.
    
    This function extracts person names and organization names from the input text
    using spaCy's named entity recognition. If the model is not available,
    returns an empty list.
    
    Args:
        text: Input text to analyze
        nlp_model: Optional pre-loaded spaCy model (loads automatically if None)
        
    Returns:
        List of entity dictionaries with NER results
    """
    if not text:
        return []
    
    # Load model if not provided
    if nlp_model is None:
        nlp_model = load_model()
    
    # Return empty if model unavailable
    if nlp_model is None:
        return []
    
    try:
        # Process text
        doc = nlp_model(text)
        
        entities = []
        for ent in doc.ents:
            # Only extract PERSON and ORG entities
            if ent.label_ in ["PERSON", "ORG"]:
                entities.append({
                    "text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "type": ent.label_,
                    "source": "spacy"
                })
        
        return entities
        
    except Exception as e:
        print(f"Warning: spaCy NER error: {e}")
        return []


def detect_entities_with_config(text: str, nlp_model: Optional[Any], 
                               config: Any) -> List[Dict[str, Any]]:
    """
    Detect entities with configuration filtering.
    
    This function respects the configuration settings to determine which
    entity types should be detected and whether spaCy detection is enabled.
    
    Args:
        text: Input text to analyze
        nlp_model: Pre-loaded spaCy model
        config: Configuration object with detection settings
        
    Returns:
        Filtered list of entities based on config
    """
    # Check if spaCy method is enabled
    if not config.is_method_enabled("spacy"):
        return []
    
    # Get all entities
    all_entities = detect_entities(text, nlp_model)
    
    # Filter based on enabled types
    enabled_types = config.get_enabled_entities()
    filtered_entities = [
        entity for entity in all_entities 
        if entity["type"] in enabled_types
    ]
    
    return filtered_entities


def filter_entities(entities: List[Dict[str, Any]], 
                   min_confidence: float = 0.0) -> List[Dict[str, Any]]:
    """
    Filter entities by confidence score.
    
    Note: Current implementation returns all entities as spaCy
    doesn't expose confidence scores in the standard API.
    This is a placeholder for future enhancement.
    
    Args:
        entities: List of detected entities
        min_confidence: Minimum confidence threshold (0-1)
        
    Returns:
        Filtered list of entities
    """
    # Future enhancement: Add confidence filtering when available
    # For now, return all entities as they meet the threshold
    return entities


def detect_entities_batch(texts: List[str], 
                         nlp_model: Optional[Any] = None) -> List[List[Dict[str, Any]]]:
    """
    Process multiple texts in a batch for efficiency.
    
    This function processes multiple texts at once, which can be more
    efficient than processing them individually.
    
    Args:
        texts: List of texts to process
        nlp_model: Pre-loaded model
        
    Returns:
        List of entity lists, one per input text
    """
    if nlp_model is None:
        nlp_model = load_model()
    
    if nlp_model is None:
        return [[] for _ in texts]
    
    try:
        # Process as batch
        docs = nlp_model.pipe(texts)
        results = []
        
        for doc in docs:
            entities = []
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG"]:
                    entities.append({
                        "text": ent.text,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "type": ent.label_,
                        "source": "spacy"
                    })
            results.append(entities)
        
        return results
        
    except Exception as e:
        print(f"Warning: spaCy batch processing error: {e}")
        return [[] for _ in texts]


def convert_to_standard_format(spacy_entities: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert spaCy entity objects to standard dictionary format.
    
    This is a utility function for converting spaCy's entity objects
    to our standardized dictionary format.
    
    Args:
        spacy_entities: List of spaCy entity objects
        
    Returns:
        List of standardized entity dictionaries
    """
    standard_entities = []
    
    for ent in spacy_entities:
        if ent.label_ in ["PERSON", "ORG"]:
            standard_entities.append({
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "type": ent.label_,
                "source": "spacy"
            })
    
    return standard_entities


# For backward compatibility with the implementation plan
def detect_patterns(text: str, nlp_model: Optional[Any] = None) -> List[Dict[str, Any]]:
    """
    Alias for detect_entities to match implementation plan naming.
    
    Args:
        text: Input text to analyze
        nlp_model: Optional pre-loaded spaCy model
        
    Returns:
        List of detected entities
    """
    return detect_entities(text, nlp_model)