"""
OpenAI Client Module for LLM Pseudonymizer

Provides interface between pseudonymized text and OpenAI API.
Handles API communication, error management, retry logic, and ensures
placeholder integrity is maintained throughout LLM interaction.
"""

import os
import time
import re
from typing import Optional, Dict, Any, List
from unittest.mock import Mock

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if available
except ImportError:
    # dotenv not installed, continue with system environment variables
    pass


class OpenAIProvider:
    """
    Manages OpenAI API communication with proper configuration and error handling.
    
    Automatically protects placeholders and implements robust retry logic.
    """
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OpenAI provider with API key and configuration.
        
        Args:
            api_key: OpenAI API key (falls back to OPENAI_API_KEY env var)
            config: Provider configuration from config.yaml
            
        Raises:
            ValueError: If no API key is provided or found
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Default configuration
        self.config = {
            'model': 'gpt-3.5-turbo',
            'temperature': 0.7,
            'max_tokens': None,
            'timeout': 30,
            'max_retries': 3,
            'retry_delay': 1.0,
            'retry_backoff': 2.0,
            'stream': False,
            'top_p': 1.0,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0
        }
        
        # Override with provided config
        if config:
            self.config.update(config)
        
        # Usage tracking
        self.total_requests = 0
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        
        # Initialize OpenAI client
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """
        Initialize OpenAI client with configuration.
        
        Returns:
            Configured OpenAI client
        """
        try:
            from openai import OpenAI
            
            # Create client with timeout
            client = OpenAI(
                api_key=self.api_key,
                timeout=self.config['timeout'],
                max_retries=0  # We handle retries ourselves
            )
            
            return client
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )
    
    def send_prompt(self, sanitized_text: str, model: Optional[str] = None, 
                    system_message: Optional[str] = None, **kwargs) -> str:
        """
        Send sanitized prompt to OpenAI API.
        
        Args:
            sanitized_text: Prompt text with placeholders
            model: Model to use (overrides config)
            system_message: Additional system instructions
            **kwargs: Additional API parameters
            
        Returns:
            Model response text
            
        Raises:
            OpenAIError: For API errors
            TimeoutError: For request timeouts
            ValueError: For invalid parameters
        """
        # Prepare messages
        messages = self._prepare_messages(sanitized_text, system_message)
        
        # Merge parameters
        api_params = self._prepare_api_params(model, kwargs)
        
        # Send with retry logic
        response = self._send_with_retry(messages, api_params)
        
        # Extract and return text
        response_text = self._extract_response_text(response)
        
        # Update usage statistics
        self._update_usage_stats(response)
        
        return response_text
    
    def _prepare_messages(self, prompt: str, custom_system_message: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Prepare messages with placeholder protection.
        
        Args:
            prompt: User prompt with placeholders
            custom_system_message: Additional system instructions
            
        Returns:
            List of message dictionaries
        """
        # Placeholder protection message
        protection_message = (
            "This prompt contains opaque placeholders (PERSON_1, EMAIL_2, etc.). "
            "Do not modify, invent, or remove placeholders."
        )
        
        # Combine system messages
        system_content = protection_message
        if custom_system_message:
            system_content = f"{protection_message}\n\n{custom_system_message}"
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]
        
        return messages
    
    def _prepare_api_params(self, model: Optional[str], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare API parameters with config defaults and overrides.
        
        Args:
            model: Model override
            kwargs: Additional parameters
            
        Returns:
            Merged API parameters
        """
        # Start with config defaults
        api_params = {
            'model': model or self.config['model'],
            'temperature': self.config['temperature'],
            'max_tokens': self.config['max_tokens'],
            'stream': self.config['stream'],
            'top_p': self.config['top_p'],
            'frequency_penalty': self.config['frequency_penalty'],
            'presence_penalty': self.config['presence_penalty']
        }
        
        # Remove None values
        api_params = {k: v for k, v in api_params.items() if v is not None}
        
        # Override with kwargs
        api_params.update(kwargs)
        
        return api_params
    
    def _send_with_retry(self, messages: List[Dict[str, str]], api_params: Dict[str, Any]):
        """
        Send request with retry logic.
        
        Args:
            messages: Formatted messages
            api_params: API parameters
            
        Returns:
            API response object
            
        Raises:
            OpenAIError: After all retries exhausted
        """
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        last_error = None
        delay = self.config['retry_delay']
        
        for attempt in range(self.config['max_retries']):
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    **api_params
                )
                return response
                
            except openai.RateLimitError as e:
                last_error = e
                if attempt < self.config['max_retries'] - 1:
                    time.sleep(delay)
                    delay *= self.config['retry_backoff']
                    continue
                
            except openai.APITimeoutError as e:
                last_error = e
                if attempt < self.config['max_retries'] - 1:
                    time.sleep(delay)
                    delay *= self.config['retry_backoff']
                    continue
                    
            except openai.InvalidRequestError as e:
                # Handle specific errors
                if "maximum context length" in str(e):
                    estimated_tokens = self.estimate_tokens(messages[1]['content'])
                    raise ValueError(
                        f"Prompt too long for model. Consider using a model with larger context window "
                        f"or reducing prompt length. Estimated tokens: {estimated_tokens}"
                    )
                elif "model" in str(e).lower() and "not found" in str(e).lower():
                    raise ValueError(f"Model '{api_params.get('model')}' not found. Available models: gpt-3.5-turbo, gpt-4, etc.")
                else:
                    # Don't retry on other invalid request errors
                    raise
                    
            except Exception as e:
                # Don't retry on other errors
                raise
        
        # All retries exhausted
        if last_error:
            raise last_error
        else:
            raise RuntimeError("All retries exhausted with no specific error")
    
    def _extract_response_text(self, response) -> str:
        """
        Extract text content from API response.
        
        Args:
            response: OpenAI API response object
            
        Returns:
            Response text content
        """
        try:
            return response.choices[0].message.content
        except (AttributeError, IndexError, KeyError):
            raise ValueError("Invalid response format from OpenAI API")
    
    def _update_usage_stats(self, response):
        """
        Update usage statistics from API response.
        
        Args:
            response: OpenAI API response object
        """
        self.total_requests += 1
        
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            self.total_tokens += getattr(usage, 'total_tokens', 0)
            self.total_prompt_tokens += getattr(usage, 'prompt_tokens', 0)
            self.total_completion_tokens += getattr(usage, 'completion_tokens', 0)
    
    def estimate_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Text to estimate
            model: Model to use for encoding
            
        Returns:
            Estimated token count
        """
        try:
            import tiktoken
            
            model_name = model or self.config['model']
            
            # Get encoding for model
            try:
                encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                # Fallback to cl100k_base for newer models
                encoding = tiktoken.get_encoding("cl100k_base")
            
            # Count tokens
            tokens = encoding.encode(text)
            return len(tokens)
            
        except ImportError:
            # Fallback estimation without tiktoken
            # Rough estimate: ~4 characters per token
            return len(text) // 4
    
    def validate_response(self, response_text: str) -> List[str]:
        """
        Validate placeholder integrity in response.
        
        Args:
            response_text: Response from model
            
        Returns:
            List of validation warnings
        """
        warnings = []
        
        # Pattern for valid placeholders
        valid_pattern = re.compile(r'\b(?:PERSON|ORG|EMAIL|URL)_[1-9]\d*\b')
        
        # Pattern for potentially corrupted placeholders (case-insensitive variations)
        corrupted_pattern = re.compile(r'\b(?:person|org|email|url)_[1-9]\d*\b', re.IGNORECASE)
        
        # Find valid placeholders
        valid_placeholders = set(valid_pattern.findall(response_text))
        
        # Find potentially corrupted placeholders
        corrupted_matches = corrupted_pattern.findall(response_text)
        
        # Check for corrupted placeholders (not in valid set and not properly formatted)
        for match in corrupted_matches:
            if match not in valid_placeholders and match.upper() != match:
                warnings.append(f"Potentially corrupted placeholder: {match}")
        
        return warnings
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for current session.
        
        Returns:
            Dictionary with usage stats
        """
        return {
            'total_requests': self.total_requests,
            'total_tokens': self.total_tokens,
            'total_prompt_tokens': self.total_prompt_tokens,
            'total_completion_tokens': self.total_completion_tokens,
            'estimated_cost': self._estimate_cost()
        }
    
    def _estimate_cost(self) -> float:
        """
        Estimate cost based on token usage and model pricing.
        
        Returns:
            Estimated cost in USD
        """
        # Pricing as of 2024 (approximate)
        pricing = {
            'gpt-3.5-turbo': {'prompt': 0.0005, 'completion': 0.0015},  # per 1K tokens
            'gpt-4': {'prompt': 0.03, 'completion': 0.06},
            'gpt-4-turbo': {'prompt': 0.01, 'completion': 0.03}
        }
        
        model = self.config['model']
        
        # Default to gpt-3.5-turbo pricing if model not found
        if model not in pricing:
            model = 'gpt-3.5-turbo'
        
        rates = pricing[model]
        
        prompt_cost = (self.total_prompt_tokens / 1000) * rates['prompt']
        completion_cost = (self.total_completion_tokens / 1000) * rates['completion']
        
        return prompt_cost + completion_cost


# Convenience function for simple usage
def send_to_openai(sanitized_text: str, api_key: Optional[str] = None, 
                   model: str = 'gpt-3.5-turbo', **kwargs) -> str:
    """
    Convenience function to send sanitized text to OpenAI.
    
    Args:
        sanitized_text: Text with placeholders
        api_key: OpenAI API key
        model: Model to use
        **kwargs: Additional parameters
        
    Returns:
        Model response text
    """
    provider = OpenAIProvider(api_key=api_key)
    return provider.send_prompt(sanitized_text, model=model, **kwargs)