"""
Configuration management for the LLM Pseudonymizer system.

This module provides centralized configuration loading and access for all
system components. It loads settings from YAML files and provides type-safe
methods to query detection settings, entity types, and provider configurations.
"""

import yaml
import os
from typing import Dict, Any, List
from pathlib import Path


class Config:
    """
    Main configuration class that loads and manages system settings.
    
    This class loads configuration from a YAML file and provides methods
    to query enabled entities, detection methods, and provider settings.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
            ValueError: If required configuration keys are missing
        """
        self.config_path = config_path
        self.config = self.load_config(config_path)
        self._validate_config()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load and validate configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Parsed configuration dictionary
            
        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If file contains invalid YAML
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            if config is None:
                raise ValueError("Configuration file is empty")
                
            return config
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in configuration file {config_path}: {e}")
    
    def _validate_config(self):
        """
        Validate that required configuration sections and keys exist.
        
        Raises:
            ValueError: If required configuration is missing or invalid
        """
        # Check required top-level sections
        required_sections = ['detection', 'provider']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Required configuration section '{section}' is missing")
        
        # Validate detection section
        detection = self.config['detection']
        if 'entities' not in detection:
            raise ValueError("Required 'detection.entities' section is missing")
        if 'methods' not in detection:
            raise ValueError("Required 'detection.methods' section is missing")
        
        # Validate at least one entity is configured
        entities = detection['entities']
        if not entities or not any(entities.values()):
            raise ValueError("At least one entity type must be enabled in detection.entities")
        
        # Validate at least one method is configured
        methods = detection['methods']
        if not methods or not any(methods.values()):
            raise ValueError("At least one detection method must be enabled in detection.methods")
        
        # Validate provider section
        provider = self.config['provider']
        if 'default' not in provider:
            raise ValueError("Required 'provider.default' setting is missing")
        
        default_provider = provider['default']
        if default_provider not in provider:
            raise ValueError(f"Default provider '{default_provider}' configuration is missing")
        
        # Validate provider-specific settings
        self._validate_provider_config(default_provider, provider[default_provider])
    
    def _validate_provider_config(self, provider_name: str, provider_config: Dict[str, Any]):
        """
        Validate provider-specific configuration.
        
        Args:
            provider_name: Name of the provider
            provider_config: Provider configuration dictionary
        """
        if provider_name == 'openai':
            # Validate OpenAI-specific settings
            if 'model' not in provider_config:
                raise ValueError("OpenAI provider requires 'model' setting")
            
            # Validate temperature if present
            if 'temperature' in provider_config:
                temp = provider_config['temperature']
                if not isinstance(temp, (int, float)) or temp < 0.0 or temp > 2.0:
                    raise ValueError("Temperature must be between 0.0 and 2.0")
            
            # Validate timeout if present
            if 'timeout' in provider_config:
                timeout = provider_config['timeout']
                if not isinstance(timeout, int) or timeout <= 0:
                    raise ValueError("Timeout must be a positive integer")
            
            # Validate max_retries if present
            if 'max_retries' in provider_config:
                retries = provider_config['max_retries']
                if not isinstance(retries, int) or retries < 0:
                    raise ValueError("Max retries must be a non-negative integer")
            
            # Validate retry_delay if present
            if 'retry_delay' in provider_config:
                delay = provider_config['retry_delay']
                if not isinstance(delay, (int, float)) or delay <= 0:
                    raise ValueError("Retry delay must be a positive number")
            
            # Validate retry_backoff if present
            if 'retry_backoff' in provider_config:
                backoff = provider_config['retry_backoff']
                if not isinstance(backoff, (int, float)) or backoff < 1.0:
                    raise ValueError("Retry backoff must be >= 1.0")
    
    def get_enabled_entities(self) -> List[str]:
        """
        Get list of entity types enabled for detection.
        
        Returns:
            List of enabled entity type names (e.g., ['PERSON', 'EMAIL'])
        """
        entities = self.config['detection']['entities']
        return [entity_type for entity_type, enabled in entities.items() if enabled]
    
    def is_entity_enabled(self, entity_type: str) -> bool:
        """
        Check if a specific entity type is enabled.
        
        Args:
            entity_type: Entity type to check (PERSON, ORG, EMAIL, URL)
            
        Returns:
            True if entity detection is enabled
        """
        entities = self.config['detection']['entities']
        return entities.get(entity_type, False)
    
    def is_method_enabled(self, method: str) -> bool:
        """
        Check if a detection method is enabled.
        
        Args:
            method: Detection method name ('regex' or 'spacy')
            
        Returns:
            True if method is enabled
        """
        methods = self.config['detection']['methods']
        return methods.get(method, False)
    
    def get_enabled_methods(self) -> List[str]:
        """
        Get list of all enabled detection methods.
        
        Returns:
            List of enabled method names
        """
        methods = self.config['detection']['methods']
        return [method for method, enabled in methods.items() if enabled]
    
    def get_provider_config(self, provider: str | None = None) -> Dict[str, Any]:
        """
        Get configuration for specified provider.
        
        Args:
            provider: Provider name (defaults to configured default)
            
        Returns:
            Provider-specific configuration dictionary
            
        Raises:
            KeyError: If provider not found in configuration
        """
        if provider is None:
            provider = self.get_default_provider()
        
        provider_configs = self.config['provider']
        if provider not in provider_configs:
            raise KeyError(f"Provider '{provider}' not found in configuration")
        
        return provider_configs[provider]
    
    def get_default_provider(self) -> str:
        """
        Get the default provider name.
        
        Returns:
            Default provider name from configuration
        """
        return self.config['provider']['default']
    
    def get_model(self, provider: str | None = None) -> str:
        """
        Get model name for provider.
        
        Args:
            provider: Provider name (defaults to default provider)
            
        Returns:
            Model name (e.g., 'gpt-3.5-turbo')
        """
        provider_config = self.get_provider_config(provider)
        return provider_config['model']
    
    def get_timeout(self, provider: str | None = None) -> int:
        """
        Get timeout value for provider.
        
        Args:
            provider: Provider name (defaults to default provider)
            
        Returns:
            Timeout in seconds
        """
        provider_config = self.get_provider_config(provider)
        return provider_config.get('timeout', 30)  # Default 30 seconds
    
    def get_temperature(self, provider: str | None = None) -> float:
        """
        Get temperature setting for provider.
        
        Args:
            provider: Provider name (defaults to default provider)
            
        Returns:
            Temperature value (0.0-2.0)
        """
        provider_config = self.get_provider_config(provider)
        return provider_config.get('temperature', 0.7)  # Default 0.7
    
    def get_max_tokens(self, provider: str | None = None) -> int | None:
        """
        Get maximum tokens setting for provider.
        
        Args:
            provider: Provider name (defaults to default provider)
            
        Returns:
            Maximum tokens (None means use model default)
        """
        provider_config = self.get_provider_config(provider)
        return provider_config.get('max_tokens', None)
    
    def get_max_retries(self, provider: str | None = None) -> int:
        """
        Get maximum retry count for provider.
        
        Args:
            provider: Provider name (defaults to default provider)
            
        Returns:
            Maximum number of retries
        """
        provider_config = self.get_provider_config(provider)
        return provider_config.get('max_retries', 3)  # Default 3 retries
    
    def get_retry_delay(self, provider: str | None = None) -> float:
        """
        Get initial retry delay for provider.
        
        Args:
            provider: Provider name (defaults to default provider)
            
        Returns:
            Initial retry delay in seconds
        """
        provider_config = self.get_provider_config(provider)
        return provider_config.get('retry_delay', 1.0)  # Default 1.0 second
    
    def get_retry_backoff(self, provider: str | None = None) -> float:
        """
        Get retry backoff multiplier for provider.
        
        Args:
            provider: Provider name (defaults to default provider)
            
        Returns:
            Backoff multiplier for retries
        """
        provider_config = self.get_provider_config(provider)
        return provider_config.get('retry_backoff', 2.0)  # Default 2.0x backoff