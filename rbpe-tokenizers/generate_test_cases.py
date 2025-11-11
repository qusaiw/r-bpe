#!/usr/bin/env python3
"""
Generate reference test cases from Python R-BPE tokenizer for Rust comparison.

This script creates a JSON file with test cases containing:
- Input text
- Expected token IDs from encoding
- Expected decoded text

The Rust tests can load this file and verify exact matching behavior.
"""

import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

# Add parent directory to path to import rbpe module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rbpe.mapping_tokenizer import MappingTokenizer

def load_json_map(path):
    """Load a JSON mapping file."""
    with open(path, 'r') as f:
        data = json.load(f)
        # Convert string keys to integers
        return {int(k): v for k, v in data.items()}

def create_test_cases():
    """Generate comprehensive test cases using the Python tokenizer."""
    
    print("Loading R-BPE tokenizer...")
    
    # Load tokenizers
    new_tokenizer = AutoTokenizer.from_pretrained("../rbpe_tokenizer/new_tokenizer")
    old_tokenizer = AutoTokenizer.from_pretrained("../rbpe_tokenizer/old_tokenizer")
    
    # Load mappings manually
    new_to_old_map_path = "../rbpe_tokenizer/metadata/new_to_old_map.json"
    old_to_new_map_path = "../rbpe_tokenizer/metadata/old_to_new_map.json"
    replacement_char_path = "../rbpe_tokenizer/metadata/replacement_character_map.json"
    
    # Define target language ranges (Arabic)
    # Unicode ranges for Arabic script: U+0600-U+06FF, U+0750-U+077F, U+08A0-U+08FF, U+FB50-U+FDFF, U+FE70-U+FEFF
    target_language_scripts_ranges = [
        (0x0600, 0x06FF),  # Arabic
        (0x0750, 0x077F),  # Arabic Supplement
        (0x08A0, 0x08FF),  # Arabic Extended-A
        (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
        (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
    ]
    
    # Initialize tokenizer with required parameters
    tokenizer = MappingTokenizer(
        new_tokenizer=new_tokenizer,
        old_tokenizer=old_tokenizer,
        token_id_language_map={},  # Empty for testing
        reusable_languages=[],  # Empty for testing
        target_language_scripts_ranges=target_language_scripts_ranges,
        new_to_old_map_path=new_to_old_map_path,
        old_to_new_map_path=old_to_new_map_path,
        replacement_character_map_path=replacement_char_path,
        new_tokenizer_additional_special_tokens=[],
        apply_normalization=False,
        debug_mode=False,
    )
    
    print("✓ Tokenizer loaded\n")
    
    # Define test cases
    test_cases = [
        # Basic cases
        ("pure_english_short", "Hello World"),
        ("pure_english_sentence", "The quick brown fox jumps over the lazy dog"),
        ("pure_arabic_short", "مرحبا"),
        ("pure_arabic_sentence", "مرحبا! كيف حالك اليوم؟"),
        
        # Mixed language
        ("mixed_simple", "Hello مرحبا World"),
        ("mixed_complex", "Hello مرحبا World عالم!"),
        ("code_switching", "This is a test هذا اختبار متعدد اللغات multilingual"),
        
        # Numbers and punctuation
        ("numbers", "The year is 2024"),
        ("arabic_numbers", "السنة ٢٠٢٤"),
        ("mixed_numbers", "The year is 2024 and السنة هي ٢٠٢٤"),
        ("punctuation", "Hello! مرحبا؟ World... عالم!"),
        
        # Special cases
        ("single_word", "hello"),
        ("single_arabic_word", "مرحبا"),
        ("whitespace_only", "   "),
        ("newlines", "Line 1\nLine 2\nLine 3"),
        ("tabs", "Column1\tColumn2\tColumn3"),
        
        # Email and URLs (common in real data)
        ("email", "Contact: test@example.com"),
        ("url", "Visit https://example.com for more"),
        
        # Long text
        ("long_english", "This is a longer piece of text that contains multiple sentences. " * 5),
        ("long_arabic", "هذا نص طويل يحتوي على جمل متعددة. " * 5),
        ("long_mixed", "This is mixed text هذا نص مختلط " * 10),
        
        # Edge cases
        ("single_char", "a"),
        ("single_arabic_char", "ا"),
        ("repeated_chars", "aaaaaaa"),
        ("repeated_arabic", "ااااااا"),
        
        # Common phrases
        ("greeting", "Hello, how are you?"),
        ("arabic_greeting", "مرحبا، كيف حالك؟"),
        ("question", "What is your name?"),
        ("arabic_question", "ما اسمك؟"),
    ]
    
    results = []
    
    print("Generating test cases:")
    print("=" * 80)
    
    for test_name, text in test_cases:
        print(f"\n{test_name}:")
        print(f"  Input: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Encode
        ids = tokenizer.encode(text, add_special_tokens=False)
        print(f"  Tokens: {len(ids)}")
        
        # Decode with basic method
        basic_decoded = tokenizer.basic_decode(ids, skip_special_tokens=False)
        
        # Decode with advanced method
        advanced_decoded = tokenizer.decode(ids, skip_special_tokens=False)
        
        # Check for replacement characters
        has_replacement_basic = "�" in basic_decoded
        has_replacement_advanced = "�" in advanced_decoded
        
        if has_replacement_basic or has_replacement_advanced:
            print(f"  ⚠️  Replacement chars: basic={has_replacement_basic}, advanced={has_replacement_advanced}")
        
        # Store result
        results.append({
            "name": test_name,
            "input": text,
            "token_ids": ids,
            "num_tokens": len(ids),
            "basic_decoded": basic_decoded,
            "advanced_decoded": advanced_decoded,
            "has_replacement_basic": has_replacement_basic,
            "has_replacement_advanced": has_replacement_advanced,
            "matches_input": advanced_decoded.strip() == text.strip(),
        })
        
        print(f"  ✓ Generated")
    
    print("\n" + "=" * 80)
    print(f"\n✓ Generated {len(results)} test cases")
    
    # Save to JSON
    output_file = Path(__file__).parent / "tests" / "python_reference_outputs.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved to {output_file}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"  Total test cases: {len(results)}")
    print(f"  Total tokens: {sum(r['num_tokens'] for r in results)}")
    print(f"  Cases with replacement chars (basic): {sum(r['has_replacement_basic'] for r in results)}")
    print(f"  Cases with replacement chars (advanced): {sum(r['has_replacement_advanced'] for r in results)}")
    print(f"  Perfect matches: {sum(r['matches_input'] for r in results)}")
    
    return results

if __name__ == "__main__":
    try:
        create_test_cases()
        print("\n✓ Done!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
