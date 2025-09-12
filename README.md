# LLM Pseudonymizer

A local system that pseudonymizes sensitive information before sending text to external LLMs, then rehydrates the placeholders in the response. All sensitive data remains local with per-session mapping tables.

## Overview

The LLM Pseudonymizer protects your privacy when using external LLM services by:
1. **Detecting** sensitive entities (emails, URLs, names, organizations) in your text
2. **Replacing** them with opaque placeholders (e.g., `EMAIL_1`, `PERSON_2`)
3. **Sending** the sanitized text to the LLM
4. **Restoring** the original entities in the response

All sensitive data stays on your machine - only placeholders are sent to external services.

## Features

- **Multi-layer Detection**: Regex patterns + spaCy NER for comprehensive entity detection
- **Configurable**: YAML-based configuration to control what gets detected
- **Privacy-First**: No persistence of sensitive data, memory-only session management
- **High Performance**: < 300ms total processing for 2-3KB text
- **Extensible**: Support for multiple LLM providers (OpenAI, future: Anthropic, etc.)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/llm-pseudonymizer.git
cd llm-pseudonymizer

# Install dependencies
pip install spacy pyyaml openai

# Download spaCy English model
python3 -m spacy download en_core_web_sm
```

### Configuration

Create a `config.yaml` file:

```yaml
detection:
  entities:
    PERSON: true      # Detect person names
    ORG: true         # Detect organizations  
    EMAIL: true       # Detect email addresses
    URL: true         # Detect URLs and domains
  methods:
    regex: true       # Use regex detection
    spacy: true       # Use spaCy NER

provider:
  default: "openai"
  openai:
    model: "gpt-3.5-turbo"
    timeout: 30
    temperature: 0.7
    max_retries: 3

session:
  echo_sanitized: false
  strict_mode: false
```

### Usage

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Run the pseudonymizer
python3 cli.py --config config.yaml

# Test without sending to LLM
python3 cli.py --config config.yaml --no-send --echo-sanitized
```

### Example Session

```
Enter prompt: Hi, I'm John Doe from john.doe@company.com. Visit https://company.com for more info.

Sanitized: Hi, I'm PERSON_1 from EMAIL_1. Visit URL_1 for more info.

Response: Hello PERSON_1! I'd be happy to help. I'll check out URL_1 and can reach you at EMAIL_1 if needed.

Final: Hello John Doe! I'd be happy to help. I'll check out https://company.com and can reach you at john.doe@company.com if needed.
```

## Architecture

### Detection Pipeline
1. **Regex Detection** (`detector/rules.py`): High-precision patterns for emails and URLs
2. **spaCy NER** (`detector/spacy_ner.py`): ML-based detection for person and organization names
3. **Merge & Resolve** (`detector/merge.py`): Combines results and resolves conflicts

### Processing Flow
1. **Configuration** (`config.py`): Loads settings from YAML
2. **Alias Management** (`aliases.py`): Maps entities to placeholders
3. **Text Rewriting** (`rewrite.py`): Replaces entities with placeholders
4. **Provider Integration** (`providers/`): Sends to external LLM
5. **Rehydration** (`rehydrate.py`): Restores original entities

## Entity Types

| Type | Source | Examples |
|------|--------|----------|
| EMAIL | Regex | `user@example.com`, `first.last+tag@domain.co.uk` |
| URL | Regex | `https://example.com/path`, `example.com`, `ftp://files.example.com` |
| PERSON | spaCy NER | `John Smith`, `Dr. Jane Doe` |
| ORG | spaCy NER | `Microsoft Corporation`, `The Walt Disney Company` |

## Precedence Rules

When entities overlap, the system resolves conflicts using these rules:
1. **All regex entities > all NER entities**
2. **Longer spans > shorter spans** (within same source)
3. **Earlier position wins** (for same length)

Example: `john@example.com` detected as both EMAIL (regex) and "john" as PERSON (spaCy) → EMAIL wins.

## Configuration Options

### Detection Control
```yaml
detection:
  entities:
    PERSON: false    # Disable person detection
    EMAIL: true      # Keep email detection
  methods:
    spacy: false     # Disable spaCy (regex-only mode)
```

### Provider Settings
```yaml
provider:
  openai:
    model: "gpt-4"           # Use GPT-4
    temperature: 0.3         # Lower randomness
    timeout: 60              # Longer timeout
    max_retries: 5           # More retries
```

## Command Line Options

```bash
python3 cli.py --config config.yaml [OPTIONS]

Options:
  --config PATH          Path to configuration file (required)
  --no-send             Process without sending to LLM
  --echo-sanitized      Display sanitized text and redaction report
  --provider PROVIDER   Override default provider
  --model MODEL         Override default model
  --timeout SECONDS     Override request timeout
  --strict              Fail on pre-send leak detection
```

## Development

### Project Structure
```
llm-pseudonymizer/
├── cli.py                    # Command-line interface
├── config.py                 # Configuration management
├── aliases.py                # Entity-to-placeholder mapping
├── rewrite.py                # Text sanitization
├── rehydrate.py              # Placeholder restoration
├── detector/
│   ├── rules.py              # Regex-based detection
│   ├── spacy_ner.py          # spaCy NER detection
│   └── merge.py              # Conflict resolution
├── providers/
│   └── openai_client.py      # OpenAI API integration
├── dev_docs/                 # Detailed specifications
└── tests/                    # Unit tests
```

### Running Tests

```bash
# Test individual modules
python3 -m pytest test_aliases.py -v
python3 -m pytest test_spacy_ner.py -v
python3 -m pytest test_merge.py -v

# Test all modules
python3 -m pytest -v
```

### Development Order

Follow this order when implementing or modifying:
1. `config.py` - Configuration management
2. `aliases.py` - Core data structures
3. `detector/rules.py` - Regex detection
4. `detector/spacy_ner.py` - NER detection
5. `detector/merge.py` - Conflict resolution
6. `rewrite.py` - Text sanitization
7. `rehydrate.py` - Response restoration
8. `providers/openai_client.py` - API integration
9. `cli.py` - User interface

## Security Considerations

- **Local Processing**: All sensitive data remains on your machine
- **No Persistence**: Entity mappings exist only in memory during session
- **No Logging**: No logging of entity mappings or prompts/responses by default
- **API Key Security**: Use environment variables for API keys
- **Not Cryptographic**: Reduces exposure risk, not adversarial re-identification protection

## Performance

### Benchmarks (8GB RAM laptop)
- Regex detection: < 5ms for 2KB text
- spaCy NER: 100-150ms for 2KB text
- Total sanitization: < 300ms for 2-3KB text
- Memory usage: < 500MB resident

### Optimization Features
- Compiled regex patterns cached at import
- Lazy loading of spaCy model
- Efficient string operations
- Model caching across requests

## Troubleshooting

### Common Issues

**spaCy model not found:**
```bash
python3 -m spacy download en_core_web_sm
```

**Missing API key:**
```bash
export OPENAI_API_KEY="your-key-here"
```

**Configuration errors:**
- Ensure config.yaml has required sections: `detection`, `provider`
- Check YAML syntax with a validator
- Verify boolean values are lowercase: `true`/`false`

### Debug Mode

```bash
# See what entities are detected
python3 cli.py --config config.yaml --echo-sanitized

# Test without sending to LLM
python3 cli.py --config config.yaml --no-send
```

## Contributing

1. Read the specifications in `dev_docs/`
2. Follow the development order
3. Add tests for new functionality
4. Update relevant specification files
5. Ensure all tests pass

## License

[Add your license here]

## Documentation

Detailed technical specifications are available in the `dev_docs/` directory:

- [`system_specifications.md`](dev_docs/system_specifications.md) - Overall system architecture
- [`implementation_plan.md`](dev_docs/implementation_plan.md) - Development roadmap
- [`config_specifications.md`](dev_docs/config_specifications.md) - Configuration system
- [`aliases_specifications.md`](dev_docs/aliases_specifications.md) - Entity mapping
- [`rules_specifications.md`](dev_docs/rules_specifications.md) - Regex detection
- [`spacy_ner_specifications.md`](dev_docs/spacy_ner_specifications.md) - NER detection
- [`merge_specifications.md`](dev_docs/merge_specifications.md) - Conflict resolution
