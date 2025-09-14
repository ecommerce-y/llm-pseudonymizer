from detector.rules import detect_phones

# Test different phone formats
test_cases = [
    "Call me at 555-123-4567",
    "My number is (555) 123-4567",
    "Contact: 555.123.4567",
    "Phone: +1-555-123-4567",
    "Reach me at 1 555 123 4567",
]

for text in test_cases:
    entities = detect_phones(text)
    print(f"\nText: {text}")
    print(f"Entities found: {len(entities)}")
    for e in entities:
        print(f"  - '{e['text']}' at [{e['start']}:{e['end']}]")