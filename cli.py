#!/usr/bin/env python3
"""
LLM Pseudonymizer CLI

Command-line interface for the LLM Pseudonymizer system.
Orchestrates the complete pipeline from user input through detection,
sanitization, LLM interaction, and response rehydration.
"""

import argparse
import json
import os
import sys
import traceback
from typing import Dict, List, Any, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if available
except ImportError:
    # dotenv not installed, continue with system environment variables
    pass


class Session:
    """Maintains session state across multiple prompts."""
    
    def __init__(self):
        self.alias_manager = None
        self.config = None
        self.nlp_model = None
        self.provider = None
        self.stats = {
            'prompts_processed': 0,
            'entities_detected': 0,
            'tokens_sent': 0,
            'tokens_received': 0
        }


class ProcessResult:
    """Results from processing a single prompt."""
    
    def __init__(self):
        self.sanitized_text = ""
        self.alias_snapshot = {}
        self.detected_count = 0
        self.llm_response = ""
        self.final_response = ""
        self.unknown_placeholders = []
        self.validation_warnings = []


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="LLM Pseudonymizer - Protect sensitive data in LLM interactions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with config
  %(prog)s --config config.yaml
  
  # Process single prompt without sending
  %(prog)s --config config.yaml --no-send --echo-sanitized
  
  # Test rehydration
  %(prog)s --config config.yaml --rehydrate "Hello PERSON_1"
  
  # Batch processing
  %(prog)s --config config.yaml --batch input.txt --output results.txt
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--config', 
        required=True,
        type=str,
        help='Path to configuration file (YAML format)'
    )
    
    # Operational modes
    parser.add_argument(
        '--no-send',
        action='store_true',
        help='Process text without sending to LLM (sanitization only)'
    )
    
    parser.add_argument(
        '--echo-sanitized',
        action='store_true',
        help='Display sanitized text and redaction report'
    )
    
    parser.add_argument(
        '--rehydrate',
        type=str,
        metavar='TEXT',
        help='Test rehydration on provided text with placeholders'
    )
    
    parser.add_argument(
        '--batch',
        type=str,
        metavar='FILE',
        help='Process prompts from file (one per line)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        metavar='FILE',
        help='Output file for batch mode results'
    )
    
    # Provider configuration
    parser.add_argument(
        '--provider',
        type=str,
        help='LLM provider (overrides config)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Model name (overrides config)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='API key for provider (overrides environment)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        help='Request timeout in seconds (overrides config)'
    )
    
    # Detection options
    parser.add_argument(
        '--disable-spacy',
        action='store_true',
        help='Disable spaCy NER detection'
    )
    
    parser.add_argument(
        '--disable-regex',
        action='store_true',
        help='Disable regex pattern detection'
    )
    
    # Advanced options
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Fail on pre-send leak detection'
    )
    
    parser.add_argument(
        '--export-mappings',
        type=str,
        metavar='FILE',
        help='Export entity mappings to file (security warning!)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress non-essential output'
    )
    
    return parser.parse_args()


def load_configuration(config_path: str):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded Config object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        from config import Config
        return Config(config_path)
    except ImportError:
        raise ImportError("config.py module not found. Please implement config.py first.")
    except Exception as e:
        raise ValueError(f"Invalid configuration file: {e}")


def create_session(config) -> Session:
    """
    Create a new pseudonymization session.
    
    Args:
        config: System configuration
        
    Returns:
        Initialized session object
    """
    try:
        from aliases import AliasManager
    except ImportError:
        raise ImportError("aliases.py module not found. Please implement aliases.py first.")
    
    session = Session()
    session.alias_manager = AliasManager()
    session.config = config
    
    # Load spaCy model if enabled
    if config.is_method_enabled('spacy'):
        try:
            from detector.spacy_ner import load_model
            session.nlp_model = load_model()
        except ImportError:
            print("Warning: detector.spacy_ner module not found.")
            session.nlp_model = None
        except Exception as e:
            print(f"Warning: Failed to load spaCy model: {e}")
            print("Continuing with regex-only detection.")
            session.nlp_model = None
    else:
        session.nlp_model = None
    
    # Initialize provider if sending is enabled
    session.provider = None  # Lazy initialization
    
    return session


def detect_all_entities(text: str, config, session: Session, args: argparse.Namespace) -> List[Dict]:
    """
    Detect all entities using enabled methods.
    
    Args:
        text: Input text
        config: System configuration
        session: Current session
        args: Command-line arguments
        
    Returns:
        List of merged, non-overlapping entities
    """
    try:
        from detector.rules import detect_patterns
        from detector.spacy_ner import detect_entities
        from detector.merge import merge_detections
    except ImportError as e:
        raise ImportError(f"Detection modules not found: {e}")
    
    regex_entities = []
    spacy_entities = []
    
    # Regex detection
    if config.is_method_enabled('regex') and not args.disable_regex:
        try:
            regex_entities = detect_patterns(text)
        except Exception as e:
            print(f"Warning: Regex detection failed: {e}")
    
    # SpaCy NER detection
    if config.is_method_enabled('spacy') and not args.disable_spacy and session.nlp_model:
        try:
            spacy_entities = detect_entities(text, session.nlp_model)
        except Exception as e:
            print(f"Warning: SpaCy detection failed: {e}")
    
    # Merge detections
    try:
        merged_entities = merge_detections(regex_entities, spacy_entities)
    except Exception as e:
        print(f"Warning: Entity merging failed: {e}")
        # Fallback to regex entities only
        merged_entities = regex_entities
    
    return merged_entities


def process_prompt(prompt: str, args: argparse.Namespace, config, session: Session) -> ProcessResult:
    """
    Process a single prompt through the complete pipeline.
    
    Args:
        prompt: User input text
        args: Command-line arguments
        config: System configuration
        session: Current session
        
    Returns:
        ProcessResult containing sanitized text, response, and metadata
    """
    result = ProcessResult()
    
    # Step 1: Detection
    if args.verbose:
        print("Detecting entities...")
    
    entities = detect_all_entities(prompt, config, session, args)
    result.detected_count = len(entities)
    session.stats['entities_detected'] += len(entities)
    
    # Step 2: Sanitization
    if args.verbose:
        print(f"Found {len(entities)} entities. Sanitizing...")
    
    try:
        from rewrite import rewrite_text
        sanitized_text, alias_snapshot, redaction_report, warnings = rewrite_text(prompt, entities, session.alias_manager)
        result.sanitized_text = sanitized_text
        result.alias_snapshot = alias_snapshot
    except ImportError:
        raise ImportError("rewrite.py module not found. Please implement rewrite.py first.")
    
    # Step 3: Leak check
    try:
        from rewrite import leak_check
        leaks = leak_check(sanitized_text)
        if leaks:
            if args.strict:
                raise ValueError(f"Leak detected in sanitized text: {leaks}")
            else:
                print(f"Warning: Potential leaks detected: {leaks}")
    except ImportError:
        print("Warning: leak_check function not available in rewrite.py")
    
    # Step 4: Display sanitized text if requested
    if args.echo_sanitized:
        display_sanitized_report(sanitized_text, entities, alias_snapshot)
    
    # Step 5: Send to LLM (unless --no-send)
    if not args.no_send:
        if args.verbose:
            print("Sending to LLM...")
        
        response = send_to_llm(sanitized_text, args, config, session)
        result.llm_response = response
        
        # Step 6: Rehydrate response
        if args.verbose:
            print("Rehydrating response...")
        
        try:
            from rehydrate import rehydrate_response
            final_response, unknown_placeholders = rehydrate_response(response, alias_snapshot)
            result.final_response = final_response
            result.unknown_placeholders = unknown_placeholders
        except ImportError:
            raise ImportError("rehydrate.py module not found. Please implement rehydrate.py first.")
        
        # Validate response
        if session.provider and hasattr(session.provider, 'validate_response'):
            result.validation_warnings = session.provider.validate_response(response)
    
    session.stats['prompts_processed'] += 1
    return result


def send_to_llm(sanitized_text: str, args: argparse.Namespace, config, session: Session) -> str:
    """
    Send sanitized text to LLM provider.
    
    Args:
        sanitized_text: Text with placeholders
        args: Command-line arguments
        config: System configuration
        session: Current session
        
    Returns:
        LLM response text
    """
    # Initialize provider if needed
    if not session.provider:
        session.provider = initialize_provider(args, config)
    
    # Prepare parameters
    provider_config = config.get_provider_config(config.get_default_provider())
    model = args.model or provider_config.get('model')
    
    # Send prompt
    try:
        response = session.provider.send_prompt(
            sanitized_text,
            model=model
        )
        return response
    except Exception as e:
        raise RuntimeError(f"LLM provider error: {e}")


def initialize_provider(args: argparse.Namespace, config):
    """
    Initialize LLM provider.
    
    Args:
        args: Command-line arguments
        config: System configuration
        
    Returns:
        Initialized provider instance
    """
    provider_name = args.provider or config.get_default_provider()
    
    if provider_name == 'openai':
        try:
            from providers.openai_client import OpenAIProvider
        except ImportError:
            raise ImportError("providers.openai_client module not found. Please implement providers/openai_client.py first.")
        
        # Get API key
        api_key = args.api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or use --api-key parameter."
            )
        
        # Get provider config
        provider_config = config.get_provider_config('openai')
        
        return OpenAIProvider(api_key=api_key, config=provider_config)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


def display_sanitized_report(sanitized_text: str, entities: List[Dict], alias_snapshot: Dict) -> None:
    """
    Display sanitized text and redaction report.
    
    Args:
        sanitized_text: Text with placeholders
        entities: Detected entities
        alias_snapshot: Current alias mappings
    """
    print("\n" + "="*50)
    print("SANITIZED TEXT:")
    print("="*50)
    print(sanitized_text)
    
    print("\n" + "="*50)
    print("REDACTION REPORT:")
    print("="*50)
    
    # Count by type
    type_counts = {}
    type_examples = {}
    
    for entity in entities:
        entity_type = entity['type']
        type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        
        if entity_type not in type_examples:
            type_examples[entity_type] = []
        if len(type_examples[entity_type]) < 3:
            type_examples[entity_type].append(entity['text'])
    
    # Display counts and examples
    print(f"Total entities replaced: {len(entities)}")
    print()
    
    for entity_type in sorted(type_counts.keys()):
        print(f"{entity_type}: {type_counts[entity_type]} occurrences")
        examples = type_examples[entity_type]
        forward_maps = alias_snapshot.get('forward_maps', {})
        type_map = forward_maps.get(entity_type, {})
        
        for example in examples[:3]:
            placeholder = type_map.get(example, "???")
            print(f"  - {example} -> {placeholder}")
        if len(type_examples[entity_type]) > 3:
            print(f"  ... and {len(type_examples[entity_type]) - 3} more")
        print()


def display_help():
    """Display help information for interactive mode."""
    print("\nAvailable commands:")
    print("  help    - Show this help message")
    print("  stats   - Show session statistics")
    print("  clear   - Clear session mappings")
    print("  quit    - Exit the program")
    print("  exit    - Exit the program")
    print("\nOr enter any text to process through the pipeline.")


def display_session_stats(session: Session):
    """Display session statistics."""
    print("\nSession Statistics:")
    print(f"  Prompts processed: {session.stats['prompts_processed']}")
    print(f"  Entities detected: {session.stats['entities_detected']}")
    
    if session.provider and hasattr(session.provider, 'get_usage_stats'):
        usage_stats = session.provider.get_usage_stats()
        print(f"  API requests: {usage_stats.get('total_requests', 0)}")
        print(f"  Tokens used: {usage_stats.get('total_tokens', 0)}")
        print(f"  Estimated cost: ${usage_stats.get('estimated_cost', 0):.4f}")


def export_session_mappings(session: Session, filepath: str):
    """Export session mappings to file with security warning."""
    print("\n" + "!"*60)
    print("SECURITY WARNING: Exporting entity mappings!")
    print("This file will contain sensitive data mappings.")
    print("Ensure proper file permissions and secure handling.")
    print("!"*60)
    
    try:
        mappings = session.alias_manager.export_mappings()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(mappings, f, indent=2, ensure_ascii=False)
        
        # Set restrictive permissions
        os.chmod(filepath, 0o600)
        print(f"\nMappings exported to: {filepath}")
        print("File permissions set to 0600 (owner read/write only)")
        
    except Exception as e:
        print(f"Error exporting mappings: {e}")


def run_interactive_mode(args: argparse.Namespace, config, session: Session) -> None:
    """
    Run interactive prompt mode.
    
    Args:
        args: Command-line arguments
        config: System configuration
        session: Current session
    """
    if not args.quiet:
        print("LLM Pseudonymizer - Interactive Mode")
        print("Type 'quit' or 'exit' to end session")
        print("Type 'help' for available commands")
        print("-" * 50)
    
    while True:
        try:
            # Get user input
            prompt = input("\n> ").strip()
            
            # Handle special commands
            if prompt.lower() in ['quit', 'exit']:
                break
            elif prompt.lower() == 'help':
                display_help()
                continue
            elif prompt.lower() == 'stats':
                display_session_stats(session)
                continue
            elif prompt.lower() == 'clear':
                session.alias_manager.reset_session()
                print("Session cleared.")
                continue
            elif not prompt:
                continue
            
            # Process the prompt
            result = process_prompt(prompt, args, config, session)
            
            # Display results
            if result.final_response:
                print(f"\n{result.final_response}")
                
                if result.unknown_placeholders:
                    print(f"\nWarning: Unknown placeholders: {result.unknown_placeholders}")
                
                if result.validation_warnings:
                    print(f"\nValidation warnings: {result.validation_warnings}")
            elif args.no_send:
                print(f"\nSanitized text: {result.sanitized_text}")
            
        except KeyboardInterrupt:
            print("\nUse 'quit' or 'exit' to end session.")
        except Exception as e:
            print(f"\nError: {e}")
            if args.verbose:
                traceback.print_exc()
    
    # Export mappings if requested
    if args.export_mappings:
        export_session_mappings(session, args.export_mappings)
    
    if not args.quiet:
        print("\nSession ended.")


def run_batch_mode(args: argparse.Namespace, config, session: Session) -> None:
    """
    Process prompts from a file in batch mode.
    
    Args:
        args: Command-line arguments
        config: System configuration
        session: Current session
    """
    if not os.path.exists(args.batch):
        raise FileNotFoundError(f"Batch input file not found: {args.batch}")
    
    # Prepare output
    output_file = None
    if args.output:
        output_file = open(args.output, 'w', encoding='utf-8')
    
    try:
        with open(args.batch, 'r', encoding='utf-8') as f:
            prompts = f.readlines()
        
        if not args.quiet:
            print(f"Processing {len(prompts)} prompts...")
        
        for i, prompt in enumerate(prompts, 1):
            prompt = prompt.strip()
            if not prompt:
                continue
            
            if not args.quiet:
                print(f"\nProcessing prompt {i}/{len(prompts)}...")
            
            try:
                result = process_prompt(prompt, args, config, session)
                
                # Write results
                output_data = {
                    'prompt': prompt,
                    'sanitized': result.sanitized_text,
                    'response': result.final_response,
                    'unknown_placeholders': result.unknown_placeholders,
                    'detected_entities': result.detected_count
                }
                
                if output_file:
                    output_file.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                elif not args.quiet:
                    if result.final_response:
                        print(f"Response: {result.final_response}")
                    else:
                        print(f"Sanitized: {result.sanitized_text}")
                    
            except Exception as e:
                error_data = {
                    'prompt': prompt,
                    'error': str(e)
                }
                
                if output_file:
                    output_file.write(json.dumps(error_data, ensure_ascii=False) + '\n')
                else:
                    print(f"Error processing prompt: {e}")
        
        if not args.quiet:
            print(f"\nBatch processing complete.")
        
    finally:
        if output_file:
            output_file.close()


def run_rehydrate_mode(args: argparse.Namespace, session: Session) -> None:
    """
    Run rehydration test mode.
    
    Args:
        args: Command-line arguments
        session: Current session
    """
    try:
        from rehydrate import rehydrate_response
    except ImportError:
        raise ImportError("rehydrate.py module not found. Please implement rehydrate.py first.")
    
    print("Rehydration Test Mode")
    print("-" * 50)
    
    # For testing, we need some mappings
    # Check if session has mappings
    if not hasattr(session.alias_manager, 'has_mappings') or not session.alias_manager.has_mappings():
        print("Warning: No entity mappings in session.")
        print("Creating sample mappings for testing...")
        
        # Create sample mappings
        session.alias_manager.get_or_create_alias("John Doe", "PERSON")
        session.alias_manager.get_or_create_alias("jane@example.com", "EMAIL")
        session.alias_manager.get_or_create_alias("https://example.com", "URL")
        session.alias_manager.get_or_create_alias("Acme Corp", "ORG")
        
        print("\nSample mappings created:")
        print("  PERSON_1 -> John Doe")
        print("  EMAIL_1 -> jane@example.com")
        print("  URL_1 -> https://example.com")
        print("  ORG_1 -> Acme Corp")
    
    # Get alias snapshot
    if hasattr(session.alias_manager, 'create_snapshot'):
        alias_snapshot = session.alias_manager.create_snapshot()
    else:
        # Fallback for basic alias manager
        alias_snapshot = session.alias_manager
    
    # Rehydrate the provided text
    rehydrated, unknown = rehydrate_response(args.rehydrate, alias_snapshot)
    
    print(f"\nOriginal: {args.rehydrate}")
    print(f"Rehydrated: {rehydrated}")
    
    if unknown:
        print(f"\nUnknown placeholders: {unknown}")


def validate_environment(args: argparse.Namespace, config) -> None:
    """
    Validate environment and configuration.
    
    Args:
        args: Command-line arguments
        config: System configuration
        
    Raises:
        RuntimeError: If environment is invalid
    """
    # Check for API key if sending is enabled
    if not args.no_send and not args.rehydrate:
        provider = args.provider or config.get_default_provider()
        
        if provider == 'openai':
            api_key = args.api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise RuntimeError(
                    "OpenAI API key required for sending prompts. "
                    "Set OPENAI_API_KEY or use --api-key, or use --no-send mode."
                )
    
    # Warn about detection methods
    if args.disable_regex and args.disable_spacy:
        print("Warning: All detection methods disabled. No entities will be detected.")
    
    # Check for conflicting options
    if args.quiet and args.verbose:
        raise ValueError("Cannot use --quiet and --verbose together")


def main() -> None:
    """
    Main entry point for the LLM Pseudonymizer CLI.
    
    Handles:
    - Argument parsing
    - Configuration loading
    - Session initialization
    - Mode selection (interactive vs single-shot)
    - Graceful shutdown
    """
    try:
        args = parse_arguments()
        config = load_configuration(args.config)
        
        # Validate environment
        validate_environment(args, config)
        
        # Initialize session
        session = create_session(config)
        
        # Route to appropriate mode
        if args.rehydrate:
            run_rehydrate_mode(args, session)
        elif args.batch:
            run_batch_mode(args, config, session)
        else:
            run_interactive_mode(args, config, session)
            
    except KeyboardInterrupt:
        print("\nSession terminated by user.")
        sys.exit(0)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Please ensure all required modules are implemented.", file=sys.stderr)
        sys.exit(4)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()