"""
Test suite for OpenAI Client Module

Tests all functionality of the OpenAI provider including initialization,
API communication, error handling, retry logic, and usage statistics.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from providers.openai_client import OpenAIProvider, send_to_openai


class TestOpenAIProviderInitialization:
    """Test OpenAI provider initialization scenarios."""
    
    def test_initialization_with_api_key(self):
        """Test provider initialization with API key."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider.config['model'] == 'gpt-3.5-turbo'
        assert provider.config['temperature'] == 0.7
        assert provider.config['timeout'] == 30
        assert provider.total_requests == 0
        assert provider.total_tokens == 0
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'env-key'})
    def test_initialization_from_env(self):
        """Test API key from environment variable."""
        provider = OpenAIProvider()
        assert provider.api_key == 'env-key'
    
    @patch.dict(os.environ, {}, clear=True)
    def test_initialization_no_api_key(self):
        """Test initialization fails without API key."""
        with pytest.raises(ValueError, match="OpenAI API key required"):
            OpenAIProvider()
    
    def test_configuration_override(self):
        """Test configuration override."""
        config = {
            'model': 'gpt-4',
            'temperature': 0.5,
            'timeout': 60,
            'max_retries': 5
        }
        
        provider = OpenAIProvider(api_key="test", config=config)
        
        assert provider.config['model'] == 'gpt-4'
        assert provider.config['temperature'] == 0.5
        assert provider.config['timeout'] == 60
        assert provider.config['max_retries'] == 5
        assert provider.config['retry_delay'] == 1.0  # Default not overridden
    
    @patch('providers.openai_client.OpenAI')
    def test_client_initialization_success(self, mock_openai_class):
        """Test successful OpenAI client initialization."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        provider = OpenAIProvider(api_key="test")
        
        mock_openai_class.assert_called_once_with(
            api_key="test",
            timeout=30,
            max_retries=0
        )
        assert provider.client == mock_client
    
    def test_client_initialization_import_error(self):
        """Test client initialization with missing openai package."""
        with patch('builtins.__import__', side_effect=ImportError):
            with pytest.raises(ImportError, match="OpenAI package not installed"):
                OpenAIProvider(api_key="test")


class TestMessagePreparation:
    """Test message preparation with placeholder protection."""
    
    def test_prepare_messages_basic(self):
        """Test basic message preparation."""
        provider = OpenAIProvider(api_key="test")
        messages = provider._prepare_messages("Hello PERSON_1")
        
        assert len(messages) == 2
        assert messages[0]['role'] == 'system'
        assert 'opaque placeholders' in messages[0]['content']
        assert 'Do not modify, invent, or remove placeholders' in messages[0]['content']
        assert messages[1]['role'] == 'user'
        assert messages[1]['content'] == "Hello PERSON_1"
    
    def test_prepare_messages_custom_system(self):
        """Test message preparation with custom system message."""
        provider = OpenAIProvider(api_key="test")
        messages = provider._prepare_messages(
            "Hello PERSON_1",
            custom_system_message="Be concise"
        )
        
        assert len(messages) == 2
        assert messages[0]['role'] == 'system'
        assert 'opaque placeholders' in messages[0]['content']
        assert 'Be concise' in messages[0]['content']
        assert messages[1]['role'] == 'user'
        assert messages[1]['content'] == "Hello PERSON_1"
    
    def test_prepare_messages_empty_prompt(self):
        """Test message preparation with empty prompt."""
        provider = OpenAIProvider(api_key="test")
        messages = provider._prepare_messages("")
        
        assert len(messages) == 2
        assert messages[0]['role'] == 'system'
        assert messages[1]['role'] == 'user'
        assert messages[1]['content'] == ""


class TestAPIParameters:
    """Test API parameter preparation."""
    
    def test_prepare_api_params_defaults(self):
        """Test API parameter preparation with defaults."""
        provider = OpenAIProvider(api_key="test")
        params = provider._prepare_api_params(None, {})
        
        expected_params = {
            'model': 'gpt-3.5-turbo',
            'temperature': 0.7,
            'stream': False,
            'top_p': 1.0,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0
        }
        
        assert params == expected_params
    
    def test_prepare_api_params_model_override(self):
        """Test API parameter preparation with model override."""
        provider = OpenAIProvider(api_key="test")
        params = provider._prepare_api_params("gpt-4", {})
        
        assert params['model'] == 'gpt-4'
        assert params['temperature'] == 0.7  # Default preserved
    
    def test_prepare_api_params_kwargs_override(self):
        """Test API parameter preparation with kwargs override."""
        provider = OpenAIProvider(api_key="test")
        params = provider._prepare_api_params(None, {
            'temperature': 0.9,
            'max_tokens': 100,
            'custom_param': 'value'
        })
        
        assert params['temperature'] == 0.9  # Overridden
        assert params['max_tokens'] == 100   # Added
        assert params['custom_param'] == 'value'  # Custom param
        assert params['model'] == 'gpt-3.5-turbo'  # Default preserved
    
    def test_prepare_api_params_none_values_removed(self):
        """Test that None values are removed from API parameters."""
        config = {'model': 'gpt-3.5-turbo', 'max_tokens': None}
        provider = OpenAIProvider(api_key="test", config=config)
        params = provider._prepare_api_params(None, {})
        
        assert 'max_tokens' not in params


class TestSendPrompt:
    """Test prompt sending functionality."""
    
    @patch('providers.openai_client.OpenAI')
    def test_send_prompt_success(self, mock_openai_class):
        """Test successful prompt sending."""
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Hello John"))]
        mock_response.usage = Mock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15
        )
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        provider = OpenAIProvider(api_key="test")
        response = provider.send_prompt("Hello PERSON_1")
        
        assert response == "Hello John"
        assert provider.total_requests == 1
        assert provider.total_tokens == 15
        assert provider.total_prompt_tokens == 10
        assert provider.total_completion_tokens == 5
    
    @patch('providers.openai_client.OpenAI')
    def test_send_prompt_model_override(self, mock_openai_class):
        """Test prompt sending with model override."""
        mock_response = Mock(choices=[Mock(message=Mock(content="Response"))])
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        provider = OpenAIProvider(api_key="test")
        provider.send_prompt("Test", model="gpt-4")
        
        # Check model was overridden
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['model'] == 'gpt-4'
    
    @patch('providers.openai_client.OpenAI')
    def test_send_prompt_system_message_always_included(self, mock_openai_class):
        """Test system message is always included."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        provider = OpenAIProvider(api_key="test")
        
        # Capture the messages sent
        def capture_messages(**kwargs):
            messages = kwargs.get('messages', [])
            assert len(messages) >= 1
            assert any('opaque placeholders' in msg.get('content', '') 
                      for msg in messages if msg.get('role') == 'system')
            return Mock(choices=[Mock(message=Mock(content="OK"))])
        
        mock_client.chat.completions.create.side_effect = capture_messages
        
        provider.send_prompt("Test")
    
    @patch('providers.openai_client.OpenAI')
    def test_send_prompt_invalid_response_format(self, mock_openai_class):
        """Test handling of invalid response format."""
        mock_response = Mock()
        mock_response.choices = []  # Empty choices
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        provider = OpenAIProvider(api_key="test")
        
        with pytest.raises(ValueError, match="Invalid response format"):
            provider.send_prompt("Test")


class TestRetryLogic:
    """Test retry logic for API failures."""
    
    @patch('providers.openai_client.time.sleep')  # Speed up tests
    @patch('providers.openai_client.OpenAI')
    def test_retry_on_rate_limit(self, mock_openai_class, mock_sleep):
        """Test retry logic for rate limit errors."""
        # Mock openai module
        with patch('providers.openai_client.openai') as mock_openai_module:
            mock_rate_limit_error = Exception("Rate limit exceeded")
            mock_rate_limit_error.__class__.__name__ = 'RateLimitError'
            mock_openai_module.RateLimitError = type('RateLimitError', (Exception,), {})
            
            mock_client = Mock()
            
            # First two calls fail with rate limit, third succeeds
            mock_client.chat.completions.create.side_effect = [
                mock_openai_module.RateLimitError("Rate limit exceeded"),
                mock_openai_module.RateLimitError("Rate limit exceeded"),
                Mock(choices=[Mock(message=Mock(content="Success"))])
            ]
            mock_openai_class.return_value = mock_client
            
            provider = OpenAIProvider(api_key="test")
            provider.config['retry_delay'] = 0.01  # Fast retry for testing
            
            response = provider.send_prompt("Test")
            assert response == "Success"
            assert mock_client.chat.completions.create.call_count == 3
            assert mock_sleep.call_count == 2  # Two retries
    
    @patch('providers.openai_client.time.sleep')
    @patch('providers.openai_client.OpenAI')
    def test_retry_on_timeout(self, mock_openai_class, mock_sleep):
        """Test retry logic for timeout errors."""
        with patch('providers.openai_client.openai') as mock_openai_module:
            mock_openai_module.APITimeoutError = type('APITimeoutError', (Exception,), {})
            
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = [
                mock_openai_module.APITimeoutError("Timeout"),
                Mock(choices=[Mock(message=Mock(content="Success"))])
            ]
            mock_openai_class.return_value = mock_client
            
            provider = OpenAIProvider(api_key="test")
            provider.config['retry_delay'] = 0.01
            
            response = provider.send_prompt("Test")
            assert response == "Success"
            assert mock_client.chat.completions.create.call_count == 2
    
    @patch('providers.openai_client.OpenAI')
    def test_retry_exhausted_rate_limit(self, mock_openai_class):
        """Test all retries exhausted for rate limit."""
        with patch('providers.openai_client.openai') as mock_openai_module:
            mock_openai_module.RateLimitError = type('RateLimitError', (Exception,), {})
            
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = mock_openai_module.RateLimitError("Rate limit")
            mock_openai_class.return_value = mock_client
            
            provider = OpenAIProvider(api_key="test")
            provider.config['max_retries'] = 1
            provider.config['retry_delay'] = 0.01
            
            with pytest.raises(Exception):  # Should raise the RateLimitError
                provider.send_prompt("Test")
    
    @patch('providers.openai_client.OpenAI')
    def test_no_retry_on_other_errors(self, mock_openai_class):
        """Test no retry on non-retryable errors."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = ValueError("Some other error")
        mock_openai_class.return_value = mock_client
        
        provider = OpenAIProvider(api_key="test")
        
        with pytest.raises(ValueError, match="Some other error"):
            provider.send_prompt("Test")
        
        # Should only be called once (no retries)
        assert mock_client.chat.completions.create.call_count == 1


class TestErrorHandling:
    """Test specific error handling scenarios."""
    
    @patch('providers.openai_client.OpenAI')
    def test_handle_context_length_error(self, mock_openai_class):
        """Test context length error handling."""
        with patch('providers.openai_client.openai') as mock_openai_module:
            mock_openai_module.InvalidRequestError = type('InvalidRequestError', (Exception,), {})
            
            mock_client = Mock()
            error = mock_openai_module.InvalidRequestError(
                "This model's maximum context length is 4096 tokens"
            )
            mock_client.chat.completions.create.side_effect = error
            mock_openai_class.return_value = mock_client
            
            provider = OpenAIProvider(api_key="test")
            
            with pytest.raises(ValueError, match="Prompt too long"):
                provider.send_prompt("Test" * 1000)
    
    @patch('providers.openai_client.OpenAI')
    def test_handle_model_not_found_error(self, mock_openai_class):
        """Test model not found error handling."""
        with patch('providers.openai_client.openai') as mock_openai_module:
            mock_openai_module.InvalidRequestError = type('InvalidRequestError', (Exception,), {})
            
            mock_client = Mock()
            error = mock_openai_module.InvalidRequestError("Model 'invalid-model' not found")
            mock_client.chat.completions.create.side_effect = error
            mock_openai_class.return_value = mock_client
            
            provider = OpenAIProvider(api_key="test")
            
            with pytest.raises(ValueError, match="Model .* not found"):
                provider.send_prompt("Test", model="invalid-model")
    
    @patch('providers.openai_client.OpenAI')
    def test_handle_other_invalid_request_error(self, mock_openai_class):
        """Test other invalid request errors are not modified."""
        with patch('providers.openai_client.openai') as mock_openai_module:
            mock_openai_module.InvalidRequestError = type('InvalidRequestError', (Exception,), {})
            
            mock_client = Mock()
            error = mock_openai_module.InvalidRequestError("Some other invalid request")
            mock_client.chat.completions.create.side_effect = error
            mock_openai_class.return_value = mock_client
            
            provider = OpenAIProvider(api_key="test")
            
            with pytest.raises(Exception):  # Should raise the original error
                provider.send_prompt("Test")


class TestTokenEstimation:
    """Test token estimation functionality."""
    
    def test_estimate_tokens_with_tiktoken(self):
        """Test token estimation with tiktoken."""
        with patch('providers.openai_client.tiktoken') as mock_tiktoken:
            mock_encoding = Mock()
            mock_encoding.encode.return_value = ['token'] * 12  # 12 tokens
            mock_tiktoken.encoding_for_model.return_value = mock_encoding
            
            provider = OpenAIProvider(api_key="test")
            
            text = "Hello, this is a test message with PERSON_1 and EMAIL_1"
            tokens = provider.estimate_tokens(text)
            
            assert tokens == 12
            mock_tiktoken.encoding_for_model.assert_called_once_with('gpt-3.5-turbo')
            mock_encoding.encode.assert_called_once_with(text)
    
    def test_estimate_tokens_model_override(self):
        """Test token estimation with model override."""
        with patch('providers.openai_client.tiktoken') as mock_tiktoken:
            mock_encoding = Mock()
            mock_encoding.encode.return_value = ['token'] * 10
            mock_tiktoken.encoding_for_model.return_value = mock_encoding
            
            provider = OpenAIProvider(api_key="test")
            tokens = provider.estimate_tokens("test", model="gpt-4")
            
            assert tokens == 10
            mock_tiktoken.encoding_for_model.assert_called_once_with('gpt-4')
    
    def test_estimate_tokens_unknown_model_fallback(self):
        """Test token estimation fallback for unknown model."""
        with patch('providers.openai_client.tiktoken') as mock_tiktoken:
            mock_encoding = Mock()
            mock_encoding.encode.return_value = ['token'] * 8
            mock_tiktoken.encoding_for_model.side_effect = KeyError("Unknown model")
            mock_tiktoken.get_encoding.return_value = mock_encoding
            
            provider = OpenAIProvider(api_key="test")
            tokens = provider.estimate_tokens("test", model="unknown-model")
            
            assert tokens == 8
            mock_tiktoken.get_encoding.assert_called_once_with("cl100k_base")
    
    def test_estimate_tokens_fallback_without_tiktoken(self):
        """Test token estimation fallback without tiktoken."""
        with patch('builtins.__import__', side_effect=ImportError):
            provider = OpenAIProvider(api_key="test")
            
            text = "A" * 100  # 100 characters
            tokens = provider.estimate_tokens(text)
            
            # Fallback uses ~4 chars per token
            assert tokens == 25


class TestResponseValidation:
    """Test response validation functionality."""
    
    def test_validate_response_clean(self):
        """Test validation of clean response."""
        provider = OpenAIProvider(api_key="test")
        
        response = "Hello PERSON_1, please email EMAIL_1"
        warnings = provider.validate_response(response)
        
        assert len(warnings) == 0
    
    def test_validate_response_corrupted(self):
        """Test validation detects corrupted placeholders."""
        provider = OpenAIProvider(api_key="test")
        
        response = "Hello person_1, please email Email_1"
        warnings = provider.validate_response(response)
        
        assert len(warnings) == 2
        assert any("person_1" in w for w in warnings)
        assert any("Email_1" in w for w in warnings)
    
    def test_validate_response_mixed_case_valid(self):
        """Test validation allows properly formatted placeholders."""
        provider = OpenAIProvider(api_key="test")
        
        response = "Contact PERSON_1 at EMAIL_2 or visit URL_1 for ORG_3"
        warnings = provider.validate_response(response)
        
        assert len(warnings) == 0
    
    def test_validate_response_no_placeholders(self):
        """Test validation with no placeholders."""
        provider = OpenAIProvider(api_key="test")
        
        response = "This is a normal response with no placeholders."
        warnings = provider.validate_response(response)
        
        assert len(warnings) == 0


class TestUsageStatistics:
    """Test usage statistics tracking."""
    
    @patch('providers.openai_client.OpenAI')
    def test_usage_statistics_single_request(self, mock_openai_class):
        """Test usage statistics for single request."""
        mock_response = Mock(
            choices=[Mock(message=Mock(content="Response"))],
            usage=Mock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        )
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        provider = OpenAIProvider(api_key="test")
        provider.send_prompt("Test")
        
        stats = provider.get_usage_stats()
        assert stats['total_requests'] == 1
        assert stats['total_tokens'] == 15
        assert stats['total_prompt_tokens'] == 10
        assert stats['total_completion_tokens'] == 5
        assert 'estimated_cost' in stats
    
    @patch('providers.openai_client.OpenAI')
    def test_usage_statistics_multiple_requests(self, mock_openai_class):
        """Test usage statistics for multiple requests."""
        responses = [
            Mock(
                choices=[Mock(message=Mock(content="Response1"))],
                usage=Mock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
            ),
            Mock(
                choices=[Mock(message=Mock(content="Response2"))],
                usage=Mock(prompt_tokens=20, completion_tokens=10, total_tokens=30)
            )
        ]
        
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = responses
        mock_openai_class.return_value = mock_client
        
        provider = OpenAIProvider(api_key="test")
        provider.send_prompt("Test1")
        provider.send_prompt("Test2")
        
        stats = provider.get_usage_stats()
        assert stats['total_requests'] == 2
        assert stats['total_tokens'] == 45
        assert stats['total_prompt_tokens'] == 30
        assert stats['total_completion_tokens'] == 15
    
    @patch('providers.openai_client.OpenAI')
    def test_usage_statistics_no_usage_info(self, mock_openai_class):
        """Test usage statistics when response has no usage info."""
        mock_response = Mock(choices=[Mock(message=Mock(content="Response"))])
        # No usage attribute
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        provider = OpenAIProvider(api_key="test")
        provider.send_prompt("Test")
        
        stats = provider.get_usage_stats()
        assert stats['total_requests'] == 1
        assert stats['total_tokens'] == 0  # No usage info
        assert stats['total_prompt_tokens'] == 0
        assert stats['total_completion_tokens'] == 0


class TestCostEstimation:
    """Test cost estimation functionality."""
    
    def test_cost_estimation_gpt35_turbo(self):
        """Test cost estimation for GPT-3.5-turbo."""
        provider = OpenAIProvider(api_key="test")
        provider.total_prompt_tokens = 1000
        provider.total_completion_tokens = 500
        provider.config['model'] = 'gpt-3.5-turbo'
        
        cost = provider._estimate_cost()
        
        # GPT-3.5-turbo pricing
        # $0.0005 per 1K prompt tokens
        # $0.0015 per 1K completion tokens
        expected_cost = (1000 / 1000 * 0.0005) + (500 / 1000 * 0.0015)
        
        assert abs(cost - expected_cost) < 0.0001
    
    def test_cost_estimation_gpt4(self):
        """Test cost estimation for GPT-4."""
        provider = OpenAIProvider(api_key="test")
        provider.total_prompt_tokens = 1000
        provider.total_completion_tokens = 500
        provider.config['model'] = 'gpt-4'
        
        cost = provider._estimate_cost()
        
        # GPT-4 pricing
        # $0.03 per 1K prompt tokens
        # $0.06 per 1K completion tokens
        expected_cost = (1000 / 1000 * 0.03) + (500 / 1000 * 0.06)
        
        assert abs(cost - expected_cost) < 0.0001
    
    def test_cost_estimation_unknown_model(self):
        """Test cost estimation for unknown model defaults to GPT-3.5-turbo."""
        provider = OpenAIProvider(api_key="test")
        provider.total_prompt_tokens = 1000
        provider.total_completion_tokens = 500
        provider.config['model'] = 'unknown-model'
        
        cost = provider._estimate_cost()
        
        # Should default to GPT-3.5-turbo pricing
        expected_cost = (1000 / 1000 * 0.0005) + (500 / 1000 * 0.0015)
        
        assert abs(cost - expected_cost) < 0.0001


class TestConvenienceFunction:
    """Test the convenience function."""
    
    @patch('providers.openai_client.OpenAIProvider')
    def test_send_to_openai_convenience(self, mock_provider_class):
        """Test the send_to_openai convenience function."""
        mock_provider = Mock()
        mock_provider.send_prompt.return_value = "Response"
        mock_provider_class.return_value = mock_provider
        
        result = send_to_openai("Test prompt", api_key="test-key", model="gpt-4")
        
        assert result == "Response"
        mock_provider_class.assert_called_once_with(api_key="test-key")
        mock_provider.send_prompt.assert_called_once_with("Test prompt", model="gpt-4")
    
    @patch('providers.openai_client.OpenAIProvider')
    def test_send_to_openai_with_kwargs(self, mock_provider_class):
        """Test convenience function with additional kwargs."""
        mock_provider = Mock()
        mock_provider.send_prompt.return_value = "Response"
        mock_provider_class.return_value = mock_provider
        
        result = send_to_openai(
            "Test prompt", 
            api_key="test-key", 
            model="gpt-4",
            temperature=0.9,
            max_tokens=100
        )
        
        assert result == "Response"
        mock_provider.send_prompt.assert_called_once_with(
            "Test prompt", 
            model="gpt-4",
            temperature=0.9,
            max_tokens=100
        )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_api_key_string(self):
        """Test empty string API key is treated as missing."""
        with pytest.raises(ValueError, match="OpenAI API key required"):
            OpenAIProvider(api_key="")
    
    def test_whitespace_only_api_key(self):
        """Test whitespace-only API key is accepted (OpenAI will reject it)."""
        # This should not raise during initialization
        provider = OpenAIProvider(api_key="   ")
        assert provider.api_key == "   "
    
    @patch('providers.openai_client.OpenAI')
    def test_very_long_prompt(self, mock_openai_class):
        """Test handling of very long prompts."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="Response"))]
        )
        
        provider = OpenAIProvider(api_key="test")
        
        # Very long prompt
        long_prompt = "A" * 10000
        result = provider.send_prompt(long_prompt)
        
        assert result == "Response"
        
        # Check that the prompt was included in the call
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        assert any(long_prompt in msg['content'] for msg in messages)
    
    def test_zero_retries_configuration(self):
        """Test configuration with zero retries."""
        config = {'max_retries': 0}
        provider = OpenAIProvider(api_key="test", config=config)
        
        assert provider.config['max_retries'] == 0
    
    def test_negative_retry_delay(self):
        """Test configuration with negative retry delay."""
        config = {'retry_delay': -1.0}
        provider = OpenAIProvider(api_key="test", config=config)
        
        # Should accept negative delay (though not practical)
        assert provider.config['retry_delay'] == -1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])