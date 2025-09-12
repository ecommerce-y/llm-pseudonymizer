"""
Test suite for CLI Module

Tests all functionality of the CLI including argument parsing,
session management, pipeline orchestration, and various operational modes.
"""

import pytest
import os
import sys
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock, mock_open
from io import StringIO

# Import the CLI module
from cli import (
    Session, ProcessResult, parse_arguments, load_configuration,
    create_session, detect_all_entities, process_prompt, send_to_llm,
    initialize_provider, display_sanitized_report, display_help,
    display_session_stats, export_session_mappings, run_interactive_mode,
    run_batch_mode, run_rehydrate_mode, validate_environment, main
)


class TestSession:
    """Test Session class functionality."""
    
    def test_session_initialization(self):
        """Test session initialization."""
        session = Session()
        
        assert session.alias_manager is None
        assert session.config is None
        assert session.nlp_model is None
        assert session.provider is None
        assert session.stats['prompts_processed'] == 0
        assert session.stats['entities_detected'] == 0


class TestProcessResult:
    """Test ProcessResult class functionality."""
    
    def test_process_result_initialization(self):
        """Test process result initialization."""
        result = ProcessResult()
        
        assert result.sanitized_text == ""
        assert result.alias_snapshot == {}
        assert result.detected_count == 0
        assert result.llm_response == ""
        assert result.final_response == ""
        assert result.unknown_placeholders == []
        assert result.validation_warnings == []


class TestArgumentParsing:
    """Test command-line argument parsing."""
    
    def test_parse_arguments_required_config(self):
        """Test that --config is required."""
        with patch('sys.argv', ['cli.py']):
            with pytest.raises(SystemExit):
                parse_arguments()
    
    def test_parse_arguments_basic_config(self):
        """Test parsing with basic config."""
        with patch('sys.argv', ['cli.py', '--config', 'test.yaml']):
            args = parse_arguments()
            assert args.config == 'test.yaml'
            assert args.no_send is False
            assert args.verbose is False
    
    def test_parse_arguments_all_flags(self):
        """Test parsing all command-line flags."""
        with patch('sys.argv', [
            'cli.py',
            '--config', 'test.yaml',
            '--no-send',
            '--echo-sanitized',
            '--model', 'gpt-4',
            '--verbose',
            '--strict',
            '--disable-spacy'
        ]):
            args = parse_arguments()
            
            assert args.config == 'test.yaml'
            assert args.no_send is True
            assert args.echo_sanitized is True
            assert args.model == 'gpt-4'
            assert args.verbose is True
            assert args.strict is True
            assert args.disable_spacy is True
    
    def test_parse_arguments_rehydrate_mode(self):
        """Test parsing rehydrate mode."""
        with patch('sys.argv', ['cli.py', '--config', 'test.yaml', '--rehydrate', 'Hello PERSON_1']):
            args = parse_arguments()
            assert args.rehydrate == 'Hello PERSON_1'
    
    def test_parse_arguments_batch_mode(self):
        """Test parsing batch mode."""
        with patch('sys.argv', ['cli.py', '--config', 'test.yaml', '--batch', 'input.txt', '--output', 'output.json']):
            args = parse_arguments()
            assert args.batch == 'input.txt'
            assert args.output == 'output.json'


class TestConfigurationLoading:
    """Test configuration loading functionality."""
    
    def test_load_configuration_missing_file(self):
        """Test error on missing configuration file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_configuration("nonexistent.yaml")
    
    @patch('os.path.exists')
    @patch('cli.Config')
    def test_load_configuration_success(self, mock_config_class, mock_exists):
        """Test successful configuration loading."""
        mock_exists.return_value = True
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        
        result = load_configuration("test.yaml")
        
        assert result == mock_config
        mock_config_class.assert_called_once_with("test.yaml")
    
    @patch('os.path.exists')
    def test_load_configuration_import_error(self, mock_exists):
        """Test error when config module not found."""
        mock_exists.return_value = True
        
        with pytest.raises(ImportError, match="config.py module not found"):
            load_configuration("test.yaml")


class TestSessionCreation:
    """Test session creation functionality."""
    
    @patch('cli.AliasManager')
    def test_create_session_basic(self, mock_alias_manager_class):
        """Test basic session creation."""
        mock_alias_manager = Mock()
        mock_alias_manager_class.return_value = mock_alias_manager
        
        config = Mock()
        config.is_method_enabled.return_value = False
        
        session = create_session(config)
        
        assert session.alias_manager == mock_alias_manager
        assert session.config == config
        assert session.nlp_model is None
    
    @patch('cli.AliasManager')
    @patch('detector.spacy_ner.load_model')
    def test_create_session_with_spacy(self, mock_load_model, mock_alias_manager_class):
        """Test session creation with spaCy enabled."""
        mock_alias_manager = Mock()
        mock_alias_manager_class.return_value = mock_alias_manager
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        config = Mock()
        config.is_method_enabled.side_effect = lambda x: x == 'spacy'
        
        session = create_session(config)
        
        assert session.nlp_model == mock_model
        mock_load_model.assert_called_once()
    
    @patch('cli.AliasManager')
    def test_create_session_spacy_import_error(self, mock_alias_manager_class):
        """Test session creation with spaCy import error."""
        mock_alias_manager = Mock()
        mock_alias_manager_class.return_value = mock_alias_manager
        
        config = Mock()
        config.is_method_enabled.side_effect = lambda x: x == 'spacy'
        
        with patch('builtins.print') as mock_print:
            session = create_session(config)
        
        assert session.nlp_model is None
        mock_print.assert_called_with("Warning: detector.spacy_ner module not found.")


class TestEntityDetection:
    """Test entity detection pipeline."""
    
    @patch('detector.rules.detect_patterns')
    @patch('detector.spacy_ner.detect_entities')
    @patch('detector.merge.merge_detections')
    def test_detect_all_entities_both_methods(self, mock_merge, mock_spacy, mock_regex):
        """Test entity detection with both methods enabled."""
        # Setup mocks
        mock_regex.return_value = [{'text': 'test@example.com', 'type': 'EMAIL'}]
        mock_spacy.return_value = [{'text': 'John Doe', 'type': 'PERSON'}]
        mock_merge.return_value = [
            {'text': 'test@example.com', 'type': 'EMAIL'},
            {'text': 'John Doe', 'type': 'PERSON'}
        ]
        
        config = Mock()
        config.is_method_enabled.return_value = True
        
        session = Mock()
        session.nlp_model = Mock()
        
        args = Mock()
        args.disable_regex = False
        args.disable_spacy = False
        
        entities = detect_all_entities("John Doe email test@example.com", config, session, args)
        
        assert len(entities) == 2
        mock_regex.assert_called_once_with("John Doe email test@example.com")
        mock_spacy.assert_called_once_with("John Doe email test@example.com", session.nlp_model)
        mock_merge.assert_called_once()
    
    @patch('detector.rules.detect_patterns')
    @patch('detector.merge.merge_detections')
    def test_detect_all_entities_regex_only(self, mock_merge, mock_regex):
        """Test entity detection with only regex enabled."""
        mock_regex.return_value = [{'text': 'test@example.com', 'type': 'EMAIL'}]
        mock_merge.return_value = [{'text': 'test@example.com', 'type': 'EMAIL'}]
        
        config = Mock()
        config.is_method_enabled.side_effect = lambda x: x == 'regex'
        
        session = Mock()
        session.nlp_model = None
        
        args = Mock()
        args.disable_regex = False
        args.disable_spacy = False
        
        entities = detect_all_entities("Email test@example.com", config, session, args)
        
        assert len(entities) == 1
        mock_regex.assert_called_once()
    
    @patch('detector.rules.detect_patterns')
    @patch('detector.merge.merge_detections')
    def test_detect_all_entities_disabled_methods(self, mock_merge, mock_regex):
        """Test entity detection with methods disabled via args."""
        config = Mock()
        config.is_method_enabled.return_value = True
        
        session = Mock()
        session.nlp_model = Mock()
        
        args = Mock()
        args.disable_regex = True
        args.disable_spacy = True
        
        mock_merge.return_value = []
        
        entities = detect_all_entities("Test text", config, session, args)
        
        assert len(entities) == 0
        mock_regex.assert_not_called()
        mock_merge.assert_called_once_with([], [])


class TestProcessPrompt:
    """Test prompt processing pipeline."""
    
    @patch('cli.detect_all_entities')
    @patch('rewrite.rewrite_text')
    @patch('rewrite.leak_check')
    @patch('cli.send_to_llm')
    @patch('rehydrate.rehydrate_response')
    def test_process_prompt_full_pipeline(self, mock_rehydrate, mock_send, mock_leak, mock_rewrite, mock_detect):
        """Test complete prompt processing pipeline."""
        # Setup mocks
        mock_detect.return_value = [{'text': 'john@example.com', 'type': 'EMAIL'}]
        mock_rewrite.return_value = ("Hello EMAIL_1", {'forward_maps': {'EMAIL': {'john@example.com': 'EMAIL_1'}}}, {}, [])
        mock_leak.return_value = []
        mock_send.return_value = "Response with EMAIL_1"
        mock_rehydrate.return_value = ("Response with john@example.com", [])
        
        args = Mock()
        args.verbose = False
        args.echo_sanitized = False
        args.no_send = False
        args.strict = False
        
        config = Mock()
        session = Mock()
        session.stats = {'entities_detected': 0, 'prompts_processed': 0}
        
        result = process_prompt("Hello john@example.com", args, config, session)
        
        assert result.sanitized_text == "Hello EMAIL_1"
        assert result.final_response == "Response with john@example.com"
        assert result.detected_count == 1
        assert result.unknown_placeholders == []
        assert session.stats['entities_detected'] == 1
        assert session.stats['prompts_processed'] == 1
    
    @patch('cli.detect_all_entities')
    @patch('rewrite.rewrite_text')
    def test_process_prompt_no_send_mode(self, mock_rewrite, mock_detect):
        """Test prompt processing in no-send mode."""
        mock_detect.return_value = [{'text': 'john@example.com', 'type': 'EMAIL'}]
        mock_rewrite.return_value = ("Hello EMAIL_1", {}, {}, [])
        
        args = Mock()
        args.verbose = False
        args.echo_sanitized = False
        args.no_send = True
        args.strict = False
        
        config = Mock()
        session = Mock()
        session.stats = {'entities_detected': 0, 'prompts_processed': 0}
        
        result = process_prompt("Hello john@example.com", args, config, session)
        
        assert result.sanitized_text == "Hello EMAIL_1"
        assert result.final_response == ""
        assert result.llm_response == ""
    
    @patch('cli.detect_all_entities')
    @patch('rewrite.rewrite_text')
    @patch('rewrite.leak_check')
    def test_process_prompt_strict_leak_detection(self, mock_leak, mock_rewrite, mock_detect):
        """Test strict leak detection mode."""
        mock_detect.return_value = []
        mock_rewrite.return_value = ("Hello john@example.com", {}, {}, [])
        mock_leak.return_value = ["john@example.com"]
        
        args = Mock()
        args.verbose = False
        args.echo_sanitized = False
        args.no_send = False
        args.strict = True
        
        config = Mock()
        session = Mock()
        session.stats = {'entities_detected': 0, 'prompts_processed': 0}
        
        with pytest.raises(ValueError, match="Leak detected"):
            process_prompt("Hello john@example.com", args, config, session)


class TestProviderInitialization:
    """Test LLM provider initialization."""
    
    @patch('providers.openai_client.OpenAIProvider')
    def test_initialize_provider_openai(self, mock_provider_class):
        """Test OpenAI provider initialization."""
        args = Mock()
        args.provider = 'openai'
        args.api_key = 'test-key'
        
        config = Mock()
        config.get_provider_config.return_value = {'model': 'gpt-3.5-turbo'}
        
        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider
        
        provider = initialize_provider(args, config)
        
        assert provider == mock_provider
        mock_provider_class.assert_called_once_with(
            api_key='test-key',
            config={'model': 'gpt-3.5-turbo'}
        )
    
    def test_initialize_provider_unknown(self):
        """Test error on unknown provider."""
        args = Mock()
        args.provider = 'unknown'
        
        config = Mock()
        
        with pytest.raises(ValueError, match="Unknown provider"):
            initialize_provider(args, config)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'env-key'})
    @patch('providers.openai_client.OpenAIProvider')
    def test_initialize_provider_env_api_key(self, mock_provider_class):
        """Test provider initialization with API key from environment."""
        args = Mock()
        args.provider = 'openai'
        args.api_key = None
        
        config = Mock()
        config.get_provider_config.return_value = {}
        
        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider
        
        provider = initialize_provider(args, config)
        
        mock_provider_class.assert_called_once_with(
            api_key='env-key',
            config={}
        )
    
    @patch.dict(os.environ, {}, clear=True)
    def test_initialize_provider_no_api_key(self):
        """Test error when no API key is available."""
        args = Mock()
        args.provider = 'openai'
        args.api_key = None
        
        config = Mock()
        
        with pytest.raises(ValueError, match="OpenAI API key required"):
            initialize_provider(args, config)


class TestDisplayFunctions:
    """Test display and output functions."""
    
    @patch('builtins.print')
    def test_display_sanitized_report(self, mock_print):
        """Test sanitized text report display."""
        entities = [
            {'text': 'john@example.com', 'type': 'EMAIL'},
            {'text': 'Jane Doe', 'type': 'PERSON'},
            {'text': 'jane@example.com', 'type': 'EMAIL'}
        ]
        
        alias_snapshot = {
            'forward_maps': {
                'EMAIL': {
                    'john@example.com': 'EMAIL_1',
                    'jane@example.com': 'EMAIL_2'
                },
                'PERSON': {
                    'Jane Doe': 'PERSON_1'
                }
            }
        }
        
        display_sanitized_report("Hello EMAIL_1 and PERSON_1", entities, alias_snapshot)
        
        # Verify key information was printed
        print_calls = [call[0][0] for call in mock_print.call_args_list if call[0]]
        assert any("Total entities replaced: 3" in call for call in print_calls)
        assert any("EMAIL: 2 occurrences" in call for call in print_calls)
        assert any("PERSON: 1 occurrences" in call for call in print_calls)
    
    @patch('builtins.print')
    def test_display_help(self, mock_print):
        """Test help display."""
        display_help()
        
        print_calls = [call[0][0] for call in mock_print.call_args_list if call[0]]
        assert any("Available commands:" in call for call in print_calls)
        assert any("help" in call for call in print_calls)
        assert any("quit" in call for call in print_calls)
    
    @patch('builtins.print')
    def test_display_session_stats(self, mock_print):
        """Test session statistics display."""
        session = Mock()
        session.stats = {
            'prompts_processed': 5,
            'entities_detected': 12
        }
        session.provider = None
        
        display_session_stats(session)
        
        print_calls = [call[0][0] for call in mock_print.call_args_list if call[0]]
        assert any("Prompts processed: 5" in call for call in print_calls)
        assert any("Entities detected: 12" in call for call in print_calls)
    
    @patch('builtins.print')
    def test_display_session_stats_with_provider(self, mock_print):
        """Test session statistics display with provider stats."""
        session = Mock()
        session.stats = {
            'prompts_processed': 3,
            'entities_detected': 8
        }
        session.provider = Mock()
        session.provider.get_usage_stats.return_value = {
            'total_requests': 3,
            'total_tokens': 150,
            'estimated_cost': 0.0025
        }
        
        display_session_stats(session)
        
        print_calls = [call[0][0] for call in mock_print.call_args_list if call[0]]
        assert any("API requests: 3" in call for call in print_calls)
        assert any("Tokens used: 150" in call for call in print_calls)
        assert any("Estimated cost: $0.0025" in call for call in print_calls)


class TestInteractiveMode:
    """Test interactive mode functionality."""
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_interactive_mode_quit(self, mock_print, mock_input):
        """Test quit command in interactive mode."""
        mock_input.side_effect = ['quit']
        
        args = Mock()
        args.quiet = False
        args.export_mappings = None
        
        config = Mock()
        session = Mock()
        
        run_interactive_mode(args, config, session)
        
        # Should have printed welcome message
        print_calls = [call[0][0] for call in mock_print.call_args_list if call[0]]
        assert any("Interactive Mode" in call for call in print_calls)
    
    @patch('builtins.input')
    @patch('cli.display_session_stats')
    @patch('builtins.print')
    def test_interactive_mode_stats_command(self, mock_print, mock_stats, mock_input):
        """Test stats command in interactive mode."""
        mock_input.side_effect = ['stats', 'quit']
        
        args = Mock()
        args.quiet = False
        args.export_mappings = None
        
        config = Mock()
        session = Mock()
        
        run_interactive_mode(args, config, session)
        
        mock_stats.assert_called_once_with(session)
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_interactive_mode_clear_command(self, mock_print, mock_input):
        """Test clear command in interactive mode."""
        mock_input.side_effect = ['clear', 'quit']
        
        args = Mock()
        args.quiet = False
        args.export_mappings = None
        
        config = Mock()
        session = Mock()
        session.alias_manager = Mock()
        
        run_interactive_mode(args, config, session)
        
        session.alias_manager.reset_session.assert_called_once()
    
    @patch('builtins.input')
    @patch('cli.process_prompt')
    @patch('builtins.print')
    def test_interactive_mode_process_prompt(self, mock_print, mock_process, mock_input):
        """Test processing a prompt in interactive mode."""
        mock_input.side_effect = ['Hello world', 'quit']
        
        mock_result = Mock()
        mock_result.final_response = "Hello there!"
        mock_result.unknown_placeholders = []
        mock_result.validation_warnings = []
        mock_process.return_value = mock_result
        
        args = Mock()
        args.quiet = False
        args.export_mappings = None
        args.no_send = False
        
        config = Mock()
        session = Mock()
        
        run_interactive_mode(args, config, session)
        
        mock_process.assert_called_once_with('Hello world', args, config, session)
        
        # Check that response was printed
        print_calls = [call[0][0] for call in mock_print.call_args_list if call[0]]
        assert any("Hello there!" in call for call in print_calls)


class TestBatchMode:
    """Test batch processing functionality."""
    
    def test_batch_mode_missing_file(self):
        """Test error when batch input file is missing."""
        args = Mock()
        args.batch = 'nonexistent.txt'
        
        config = Mock()
        session = Mock()
        
        with pytest.raises(FileNotFoundError):
            run_batch_mode(args, config, session)
    
    @patch('builtins.open', new_callable=mock_open, read_data="First prompt\nSecond prompt\n")
    @patch('cli.process_prompt')
    @patch('builtins.print')
    def test_batch_mode_success(self, mock_print, mock_process, mock_file):
        """Test successful batch processing."""
        # Setup mock results
        mock_result1 = Mock()
        mock_result1.sanitized_text = "First sanitized"
        mock_result1.final_response = "First response"
        mock_result1.unknown_placeholders = []
        mock_result1.detected_count = 1
        
        mock_result2 = Mock()
        mock_result2.sanitized_text = "Second sanitized"
        mock_result2.final_response = "Second response"
        mock_result2.unknown_placeholders = []
        mock_result2.detected_count = 0
        
        mock_process.side_effect = [mock_result1, mock_result2]
        
        args = Mock()
        args.batch = 'input.txt'
        args.output = None
        args.quiet = False
        
        config = Mock()
        session = Mock()
        
        run_batch_mode(args, config, session)
        
        assert mock_process.call_count == 2
        mock_process.assert_any_call('First prompt', args, config, session)
        mock_process.assert_any_call('Second prompt', args, config, session)
    
    @patch('builtins.open')
    @patch('cli.process_prompt')
    def test_batch_mode_with_output_file(self, mock_process, mock_open_func):
        """Test batch processing with output file."""
        # Setup file mocks
        input_file_mock = mock_open(read_data="Test prompt\n")
        output_file_mock = mock_open()
        
        def open_side_effect(filename, mode='r', **kwargs):
            if 'input.txt' in filename:
                return input_file_mock.return_value
            elif 'output.json' in filename:
                return output_file_mock.return_value
            return mock_open().return_value
        
        mock_open_func.side_effect = open_side_effect
        
        # Setup process result
        mock_result = Mock()
        mock_result.sanitized_text = "Test sanitized"
        mock_result.final_response = "Test response"
        mock_result.unknown_placeholders = []
        mock_result.detected_count = 1
        mock_process.return_value = mock_result
        
        args = Mock()
        args.batch = 'input.txt'
        args.output = 'output.json'
        args.quiet = False
        
        config = Mock()
        session = Mock()
        
        run_batch_mode(args, config, session)
        
        # Verify output was written
        output_file_mock.return_value.write.assert_called()
        written_data = output_file_mock.return_value.write.call_args[0][0]
        assert 'Test prompt' in written_data
        assert 'Test response' in written_data


class TestRehydrateMode:
    """Test rehydration mode functionality."""
    
    @patch('rehydrate.rehydrate_response')
    @patch('builtins.print')
    def test_rehydrate_mode_with_mappings(self, mock_print, mock_rehydrate):
        """Test rehydrate mode with existing mappings."""
        args = Mock()
        args.rehydrate = "Hello PERSON_1 at EMAIL_1"
        
        session = Mock()
        alias_manager = Mock()
        alias_manager.has_mappings.return_value = True
        alias_manager.create_snapshot.return_value = {
            'forward_maps': {
                'PERSON': {'John Doe': 'PERSON_1'},
                'EMAIL': {'john@example.com': 'EMAIL_1'}
            }
        }
        session.alias_manager = alias_manager
        
        mock_rehydrate.return_value = ("Hello John Doe at john@example.com", [])
        
        run_rehydrate_mode(args, session)
        
        mock_rehydrate.assert_called_once_with("Hello PERSON_1 at EMAIL_1", alias_manager.create_snapshot())
        
        # Check output
        print_calls = [call[0][0] for call in mock_print.call_args_list if call[0]]
        assert any("Hello John Doe at john@example.com" in call for call in print_calls)
    
    @patch('rehydrate.rehydrate_response')
    @patch('builtins.print')
    def test_rehydrate_mode_no_mappings(self, mock_print, mock_rehydrate):
        """Test rehydrate mode without existing mappings."""
        args = Mock()
        args.rehydrate = "Hello PERSON_1"
        
        session = Mock()
        alias_manager = Mock()
        alias_manager.has_mappings.return_value = False
        alias_manager.get_or_create_alias = Mock()
        session.alias_manager = alias_manager
        
        mock_rehydrate.return_value = ("Hello John Doe", [])
        
        run_rehydrate_mode(args, session)
        
        # Should have created sample mappings
        assert alias_manager.get_or_create_alias.call_count == 4


class TestEnvironmentValidation:
    """Test environment validation functionality."""
    
    @patch.dict(os.environ, {}, clear=True)
    def test_validate_environment_no_api_key(self):
        """Test environment validation with missing API key."""
        args = Mock()
        args.no_send = False
        args.rehydrate = None
        args.api_key = None
        args.provider = None
        
        config = Mock()
        config.get_default_provider.return_value = 'openai'
        
        with pytest.raises(RuntimeError, match="OpenAI API key required"):
            validate_environment(args, config)
    
    def test_validate_environment_no_send_mode(self):
        """Test environment validation in no-send mode."""
        args = Mock()
        args.no_send = True
        args.api_key = None
        
        config = Mock()
        
        # Should not raise error
        validate_environment(args, config)
    
    def test_validate_environment_conflicting_options(self):
        """Test validation with conflicting options."""
        args = Mock()
        args.no_send = True
        args.quiet = True
        args.verbose = True
        
        config = Mock()
        
        with pytest.raises(ValueError, match="Cannot use --quiet and --verbose together"):
            validate_environment(args, config)
    
    @patch('builtins.print')
    def test_validate_environment_all_detection_disabled(self, mock_print):
        """Test warning when all detection methods are disabled."""
        args = Mock()
        args.no_send = True
        args.disable_regex = True
        args.disable_spacy = True
        args.quiet = False
        args.verbose = False
        
        config = Mock()
        
        validate_environment(args, config)
        
        mock_print.assert_called_with("Warning: All detection methods disabled. No entities will be detected.")


class TestMainFunction:
    """Test main function and error handling."""
    
    @patch('cli.parse_arguments')
    @patch('cli.load_configuration')
    @patch('cli.validate_environment')
    @patch('cli.create_session')
    @patch('cli.run_interactive_mode')
    def test_main_interactive_mode(self, mock_interactive, mock_session, mock_validate, mock_config, mock_args):
        """Test main function in interactive mode."""
        args = Mock()
        args.rehydrate = None
        args.batch = None
        mock_args.return_value = args
        
        config = Mock()
        mock_config.return_value = config
        
        session = Mock()
        mock_session.return_value = session
        
        main()
        
        mock_args.assert_called_once()
        mock_config.assert_called_once()
        mock_validate.assert_called_once_with(args, config)
        mock_session.assert_called_once_with(config)
        mock_interactive.assert_called_once_with(args, config, session)
    
    @patch('cli.parse_arguments')
    @patch('cli.load_configuration')
    @patch('cli.validate_environment')
    @patch('cli.create_session')
    @patch('cli.run_rehydrate_mode')
    def test_main_rehydrate_mode(self, mock_rehydrate, mock_session, mock_validate, mock_config, mock_args):
        """Test main function in rehydrate mode."""
        args = Mock()
        args.rehydrate = "Hello PERSON_1"
        args.batch = None
        mock_args.return_value = args
        
        config = Mock()
        mock_config.return_value = config
        
        session = Mock()
        mock_session.return_value = session
        
        main()
        
        mock_rehydrate.assert_called_once_with(args, session)
    
    @patch('cli.parse_arguments')
    def test_main_keyboard_interrupt(self, mock_args):
        """Test graceful handling of keyboard interrupt."""
        mock_args.side_effect = KeyboardInterrupt()
        
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 0
    
    @patch('cli.parse_arguments')
    @patch('cli.load_configuration')
    def test_main_file_not_found_error(self, mock_config, mock_args):
        """Test handling of file not found error."""
        mock_args.return_value = Mock()
        mock_config.side_effect = FileNotFoundError("Config not found")
        
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 2
    
    @patch('cli.parse_arguments')
    @patch('cli.load_configuration')
    def test_main_import_error(self, mock_config, mock_args):
        """Test handling of import error."""
        mock_args.return_value = Mock()
        mock_config.side_effect = ImportError("Module not found")
        
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 4
    
    @patch('cli.parse_arguments')
    @patch('cli.load_configuration')
    @patch('cli.validate_environment')
    def test_main_runtime_error(self, mock_validate, mock_config, mock_args):
        """Test handling of runtime error."""
        mock_args.return_value = Mock()
        mock_config.return_value = Mock()
        mock_validate.side_effect = RuntimeError("API key required")
        
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 3


class TestExportMappings:
    """Test mapping export functionality."""
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.chmod')
    @patch('builtins.print')
    def test_export_session_mappings(self, mock_print, mock_chmod, mock_file):
        """Test exporting session mappings."""
        session = Mock()
        session.alias_manager.export_mappings.return_value = {
            'PERSON': {'John Doe': 'PERSON_1'},
            'EMAIL': {'john@example.com': 'EMAIL_1'}
        }
        
        export_session_mappings(session, 'mappings.json')
        
        # Check file was opened for writing
        mock_file.assert_called_once_with('mappings.json', 'w', encoding='utf-8')
        
        # Check permissions were set
        mock_chmod.assert_called_once_with('mappings.json', 0o600)
        
        # Check security warning was printed
        print_calls = [call[0][0] for call in mock_print.call_args_list if call[0]]
        assert any("SECURITY WARNING" in call for call in print_calls)
        
        # Check JSON was written
        written_data = ''.join(call[0][0] for call in mock_file.return_value.write.call_args_list)
        assert 'John Doe' in written_data
        assert 'PERSON_1' in written_data


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_process_result_empty_initialization(self):
        """Test ProcessResult with empty values."""
        result = ProcessResult()
        
        # Should have sensible defaults
        assert isinstance(result.sanitized_text, str)
        assert isinstance(result.alias_snapshot, dict)
        assert isinstance(result.unknown_placeholders, list)
        assert isinstance(result.validation_warnings, list)
    
    @patch('builtins.print')
    def test_display_sanitized_report_empty_entities(self, mock_print):
        """Test display report with no entities."""
        display_sanitized_report("Hello world", [], {})
        
        print_calls = [call[0][0] for call in mock_print.call_args_list if call[0]]
        assert any("Total entities replaced: 0" in call for call in print_calls)
    
    @patch('builtins.input')
    @patch('builtins.print')
    def test_interactive_mode_empty_input(self, mock_print, mock_input):
        """Test interactive mode with empty input."""
        mock_input.side_effect = ['', '   ', 'quit']
        
        args = Mock()
        args.quiet = False
        args.export_mappings = None
        
        config = Mock()
        session = Mock()
        
        # Should handle empty input gracefully
        run_interactive_mode(args, config, session)
    
    @patch('builtins.input')
    @patch('cli.process_prompt')
    @patch('builtins.print')
    def test_interactive_mode_processing_error(self, mock_print, mock_process, mock_input):
        """Test interactive mode with processing error."""
        mock_input.side_effect = ['test prompt', 'quit']
        mock_process.side_effect = Exception("Processing failed")
        
        args = Mock()
        args.quiet = False
        args.export_mappings = None
        args.verbose = False
        
        config = Mock()
        session = Mock()
        
        # Should handle error gracefully and continue
        run_interactive_mode(args, config, session)
        
        # Check error was printed
        print_calls = [call[0][0] for call in mock_print.call_args_list if call[0]]
        assert any("Error: Processing failed" in call for call in print_calls)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])