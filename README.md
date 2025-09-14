# LLM Pseudonymizer

LLM Pseudonymizer protects sensitive information when using external LLMs by replacing private data with placeholders before sending text out, then restoring it locally after. No sensitive data leaves your machine.

## How It Works
1. Detects sensitive entities (emails, URLs, names, orgs, phone numbers)
2. Replaces them with placeholders (`EMAIL_1`, `PERSON_2`, `PHONE_3`, etc.)
3. Sends sanitized text to the LLM  
4. Restores original values in the response  

## Features
- Regex + spaCy NER for detection  
- Configurable YAML-based settings  
- In-memory only (no persistence)  
- Works with OpenAI, extensible to others  
- Fast (<300ms for typical input)  

## Quick Start

### Install
```bash
git clone https://github.com/your-username/llm-pseudonymizer.git
cd llm-pseudonymizer
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

### Configure
Create `.env` with your API key:
```bash
OPENAI_API_KEY=sk-your-api-key
```
Remember to commit your `.env` file to version control so I can have your API keys.

Check `config.yaml` to adjust detection (defaults provided).

### Run
```bash
# Interactive
python3 cli.py --config config.yaml  

# Show sanitized content
python3 cli.py --config config.yaml --echo-sanitized  

# Test mode (show each step)
python3 cli.py --config config.yaml --test  
```

Example:
```
Input:   Hi, I'm John Doe (john.doe@company.com). Call me at 555-123-4567.
Sanitized → Hi, I'm PERSON_1 (EMAIL_1). Call me at PHONE_1.
Response  → Hello PERSON_1! I'll call you at PHONE_1...
Rehydrated → Hello John Doe! I'll call you at 555-123-4567...
```

## Project Structure
```
cli.py            # Entry point
config.py         # Config loader
aliases.py        # Placeholder mappings
rewrite.py        # Text sanitization
rehydrate.py      # Placeholder restoration
detector/         # Entity detection (regex + spaCy)
providers/        # LLM provider integrations
```

## Security
- All pseudonymization happens locally  
- No data stored beyond session memory  
- API keys via `.env` or environment vars  

## Troubleshooting
- Missing spaCy model → `python3 -m spacy download en_core_web_sm`  
- Missing API key → add to `.env` or `--api-key` arg  
- View sanitized text → `--echo-sanitized`  