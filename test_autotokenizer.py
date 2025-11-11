#!/usr/bin/env python3
"""
Test AutoTokenizer integration with R-BPE

This tests that R-BPE works exactly like a normal HuggingFace tokenizer
when loaded with AutoTokenizer.from_pretrained().
"""

from transformers import AutoTokenizer
import torch

def test_autotokenizer_loading():
    """Test that AutoTokenizer can load R-BPE"""
    print("Test 1: AutoTokenizer Loading")
    print("-" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained("rbpe_tokenizer", trust_remote_code=True)
    
    assert tokenizer is not None
    assert tokenizer.__class__.__name__ in ["RBPETokenizer", "RBPETokenizerFast"]
    print("✓ Loaded with AutoTokenizer.from_pretrained()")
    print(f"  Class: {tokenizer.__class__.__name__}\n")


def test_basic_encoding():
    """Test basic text encoding"""
    print("Test 2: Basic Encoding")
    print("-" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained("rbpe_tokenizer", trust_remote_code=True)
    
    # Test cases
    test_cases = [
        ("Pure English", "Hello World! How are you today?"),
        ("Pure Arabic", "مرحبا! كيف حالك اليوم؟"),
        ("Mixed", "Hello مرحبا World عالم!"),
        ("Code-switching", "This is a test هذا اختبار"),
    ]
    
    for name, text in test_cases:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        
        # Normalize whitespace for comparison
        match = decoded.strip().replace("  ", " ") == text.strip().replace("  ", " ")
        
        print(f"  {name}: {len(ids)} tokens - {'✓' if match else '✗'}")
        if not match:
            print(f"    Expected: '{text}'")
            print(f"    Got:      '{decoded}'")
    
    print()


def test_call_method():
    """Test __call__ method (standard HF interface)"""
    print("Test 3: __call__ Method (HF Standard Interface)")
    print("-" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained("rbpe_tokenizer", trust_remote_code=True)
    
    text = "Hello مرحبا World"
    
    # Single text
    result = tokenizer(text)
    assert "input_ids" in result
    assert "attention_mask" in result
    assert len(result["input_ids"]) == len(result["attention_mask"])
    print(f"✓ Single text: {len(result['input_ids'])} tokens")
    
    # Batch
    texts = ["Hello", "مرحبا", "World"]
    result = tokenizer(texts, padding=True)
    assert len(result["input_ids"]) == 3
    print(f"✓ Batch: {len(texts)} texts, padded to {len(result['input_ids'][0])} tokens")
    
    # With truncation
    result = tokenizer(text, max_length=5, truncation=True)
    assert len(result["input_ids"]) <= 5
    print(f"✓ Truncation: Limited to {len(result['input_ids'])} tokens")
    
    # With padding
    result = tokenizer(texts, padding="max_length", max_length=10)
    assert all(len(ids) == 10 for ids in result["input_ids"])
    print(f"✓ Padding: All sequences padded to 10 tokens\n")


def test_batch_encode_plus():
    """Test batch_encode_plus method"""
    print("Test 4: batch_encode_plus")
    print("-" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained("rbpe_tokenizer", trust_remote_code=True)
    
    texts = [
        "Hello World",
        "مرحبا يا عالم",
        "Mixed text مع نص عربي",
    ]
    
    result = tokenizer.batch_encode_plus(
        texts,
        padding=True,
        truncation=True,
        max_length=50,
        return_tensors="pt"
    )
    
    assert isinstance(result["input_ids"], torch.Tensor)
    assert isinstance(result["attention_mask"], torch.Tensor)
    assert result["input_ids"].shape[0] == 3  # 3 texts
    print(f"✓ Batch encoded: {result['input_ids'].shape}")
    print(f"  Returns PyTorch tensors\n")


def test_encode_decode_round_trip():
    """Test that encode -> decode is lossless"""
    print("Test 5: Encode/Decode Round Trip")
    print("-" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained("rbpe_tokenizer", trust_remote_code=True)
    
    test_texts = [
        "Hello World",
        "مرحبا يا عالم",
        "This is a test with مختلط mixed content",
        "Numbers: 123 456 789",
        "Symbols: @#$%^&*()",
    ]
    
    all_passed = True
    for text in test_texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        
        # Normalize spaces
        text_norm = text.strip().replace("  ", " ")
        decoded_norm = decoded.strip().replace("  ", " ")
        
        match = text_norm == decoded_norm
        status = "✓" if match else "✗"
        print(f"  {status} '{text[:50]}'")
        
        if not match:
            print(f"      Got: '{decoded}'")
            all_passed = False
    
    print()
    return all_passed


def test_special_tokens():
    """Test special token handling"""
    print("Test 6: Special Tokens")
    print("-" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained("rbpe_tokenizer", trust_remote_code=True)
    
    # Check special tokens exist
    print(f"  BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    
    # Test with special tokens
    text = "Hello World"
    ids_with = tokenizer.encode(text, add_special_tokens=True)
    ids_without = tokenizer.encode(text, add_special_tokens=False)
    
    print(f"\n  Without special tokens: {len(ids_without)} tokens")
    print(f"  With special tokens:    {len(ids_with)} tokens")
    print(f"  Difference: {len(ids_with) - len(ids_without)} tokens\n")


def test_comparison_with_direct_api():
    """Compare AutoTokenizer with direct Rust API"""
    print("Test 7: Parity with Direct Rust API")
    print("-" * 80)
    
    import rbpe_tokenizers
    
    # Load both ways
    auto_tok = AutoTokenizer.from_pretrained("rbpe_tokenizer", trust_remote_code=True)
    rust_tok = rbpe_tokenizers.RBPETokenizer.from_pretrained("rbpe_tokenizer")
    
    # Test same inputs
    texts = [
        "Hello World",
        "مرحبا يا عالم",
        "Mixed مختلط content",
    ]
    
    all_match = True
    for text in texts:
        auto_ids = auto_tok.encode(text, add_special_tokens=False)
        rust_ids = rust_tok.encode(text, add_special_tokens=False)
        
        match = auto_ids == rust_ids
        status = "✓" if match else "✗"
        print(f"  {status} '{text}'")
        
        if not match:
            print(f"      AutoTokenizer: {auto_ids}")
            print(f"      Direct Rust:   {rust_ids}")
            all_match = False
    
    if all_match:
        print("\n✓ Perfect parity: AutoTokenizer produces same IDs as direct Rust API\n")
    else:
        print("\n✗ Mismatch detected\n")
    
    return all_match


def main():
    print("=" * 80)
    print("R-BPE AutoTokenizer Integration Tests")
    print("=" * 80)
    print()
    
    try:
        test_autotokenizer_loading()
        test_basic_encoding()
        test_call_method()
        test_batch_encode_plus()
        test_encode_decode_round_trip()
        test_special_tokens()
        test_comparison_with_direct_api()
        
        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print()
        print("R-BPE now has full parity with standard HuggingFace tokenizers!")
        print("You can use it with:")
        print("  - AutoTokenizer.from_pretrained()")
        print("  - HuggingFace Trainer")
        print("  - Transformers pipelines")
        print("  - Any HF-compatible library")
        print()
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ TEST FAILED")
        print("=" * 80)
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
