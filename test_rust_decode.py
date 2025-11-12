#!/usr/bin/env python3
"""
Test Rust tokenizer encode-decode cycle directly
"""

try:
    import rbpe_tokenizers
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("ERROR: Rust tokenizer not available")
    print("Build with: cd rbpe-tokenizers && maturin develop --release")
    exit(1)

def test_basic_encode_decode():
    """Test basic encode-decode cycle"""
    print("Test 1: Basic Encode-Decode Cycle (Rust Direct)")
    print("-" * 80)
    
    tokenizer = rbpe_tokenizers.RBPETokenizer.from_pretrained("rbpe_tokenizer")
    
    test_cases = [
        "Hello World! How are you today?",
        "مرحبا! كيف حالك اليوم؟",
        "This is a test هذا اختبار",
        "Mixed content with numbers 123 and symbols !@#",
    ]
    
    all_passed = True
    for i, prompt in enumerate(test_cases, 1):
        print(f"\n  Test case {i}: {prompt[:50]}...")
        
        # Encode
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        print(f"    Encoded: {len(ids)} tokens")
        print(f"    Token IDs: {ids[:10]}..." if len(ids) > 10 else f"    Token IDs: {ids}")
        
        # Decode
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"    Decoded: {decoded[:50]}...")
        
        # Check match
        match = decoded.strip() == prompt.strip()
        print(f"    Match: {'✓' if match else '✗'}")
        
        if not match:
            print(f"    EXPECTED: '{prompt}'")
            print(f"    GOT:      '{decoded}'")
            all_passed = False
    
    print("\n" + "=" * 80)
    return all_passed


def test_advanced_decoder():
    """Test advanced decoder"""
    print("\nTest 2: Advanced Decoder")
    print("-" * 80)
    
    tokenizer = rbpe_tokenizers.RBPETokenizer.from_pretrained("rbpe_tokenizer")
    
    test_cases = [
        "Hello World! How are you today?",
        "مرحبا! كيف حالك اليوم؟",
        "This is a test هذا اختبار",
    ]
    
    all_passed = True
    for i, prompt in enumerate(test_cases, 1):
        print(f"\n  Test case {i}: {prompt[:50]}...")
        
        # Encode
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        
        # Decode with basic decoder
        decoded_basic = tokenizer.decode(ids, skip_special_tokens=True)
        
        # Decode with advanced decoder
        decoded_advanced = tokenizer.decode_advanced(ids, skip_special_tokens=True)
        
        print(f"    Basic decode: {decoded_basic[:50]}...")
        print(f"    Advanced decode: {decoded_advanced[:50]}...")
        
        # Check matches
        match_basic = decoded_basic.strip() == prompt.strip()
        match_advanced = decoded_advanced.strip() == prompt.strip()
        
        print(f"    Basic match: {'✓' if match_basic else '✗'}")
        print(f"    Advanced match: {'✓' if match_advanced else '✗'}")
        
        if not match_basic:
            print(f"    BASIC - Expected: '{prompt}'")
            print(f"    BASIC - Got:      '{decoded_basic}'")
        
        if not match_advanced:
            print(f"    ADVANCED - Expected: '{prompt}'")
            print(f"    ADVANCED - Got:      '{decoded_advanced}'")
            all_passed = False
    
    print("\n" + "=" * 80)
    return all_passed


def test_with_special_tokens():
    """Test encoding with special tokens"""
    print("\nTest 3: With Special Tokens")
    print("-" * 80)
    
    tokenizer = rbpe_tokenizers.RBPETokenizer.from_pretrained("rbpe_tokenizer")
    
    prompt = "Hello World مرحبا"
    
    print(f"\n  Prompt: {prompt}")
    
    # Without special tokens
    ids_without = tokenizer.encode(prompt, add_special_tokens=False)
    decoded_without = tokenizer.decode(ids_without, skip_special_tokens=True)
    
    # With special tokens
    ids_with = tokenizer.encode(prompt, add_special_tokens=True)
    decoded_with_skip = tokenizer.decode(ids_with, skip_special_tokens=True)
    decoded_with_keep = tokenizer.decode(ids_with, skip_special_tokens=False)
    
    print(f"\n  Without special tokens:")
    print(f"    IDs: {ids_without[:10]}... (len={len(ids_without)})")
    print(f"    Decoded: '{decoded_without}'")
    
    print(f"\n  With special tokens:")
    print(f"    IDs: {ids_with[:10]}... (len={len(ids_with)})")
    print(f"    Decoded (skip special): '{decoded_with_skip}'")
    print(f"    Decoded (keep special): '{decoded_with_keep}'")
    
    match_without = decoded_without.strip() == prompt.strip()
    match_with = decoded_with_skip.strip() == prompt.strip()
    
    print(f"\n  Match (without special): {'✓' if match_without else '✗'}")
    print(f"  Match (with special, skip decode): {'✓' if match_with else '✗'}")
    
    print("\n" + "=" * 80)
    return match_without and match_with


def test_batch_encoding():
    """Test batch encoding"""
    print("\nTest 4: Batch Encoding/Decoding")
    print("-" * 80)
    
    tokenizer = rbpe_tokenizers.RBPETokenizer.from_pretrained("rbpe_tokenizer")
    
    prompts = [
        "Hello World",
        "مرحبا يا عالم",
        "Mixed test مختلط",
    ]
    
    print(f"\n  Encoding {len(prompts)} prompts...")
    
    # Batch encode
    batch_ids = tokenizer.encode_batch(prompts, add_special_tokens=False)
    
    print(f"  Encoded {len(batch_ids)} sequences")
    
    # Decode each
    all_passed = True
    for i, (prompt, ids) in enumerate(zip(prompts, batch_ids)):
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        match = decoded.strip() == prompt.strip()
        
        print(f"\n  Sequence {i+1}:")
        print(f"    Original: {prompt}")
        print(f"    Tokens: {len(ids)}")
        print(f"    Decoded: {decoded}")
        print(f"    Match: {'✓' if match else '✗'}")
        
        if not match:
            all_passed = False
    
    print("\n" + "=" * 80)
    return all_passed


def test_edge_cases():
    """Test edge cases"""
    print("\nTest 5: Edge Cases")
    print("-" * 80)
    
    tokenizer = rbpe_tokenizers.RBPETokenizer.from_pretrained("rbpe_tokenizer")
    
    edge_cases = [
        ("Single char", "a"),
        ("Single Arabic char", "أ"),
        ("Only spaces", "   "),
        ("Special chars", "!@#$%^&*()"),
        ("Numbers", "0123456789"),
        ("Arabic numbers", "٠١٢٣٤٥٦٧٨٩"),
    ]
    
    all_passed = True
    for name, text in edge_cases:
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            
            # For space strings, decoded might normalize differently
            if text.strip() == "":
                match = decoded.strip() == ""
            else:
                match = decoded.strip() == text.strip()
            
            status = "✓" if match else "✗"
            print(f"  {status} {name}: '{text[:30]}'")
            
            if not match and text.strip() != "":
                print(f"      Expected: '{text}'")
                print(f"      Got:      '{decoded}'")
                all_passed = False
                
        except Exception as e:
            print(f"  ✗ {name}: ERROR - {e}")
            all_passed = False
    
    print("\n" + "=" * 80)
    return all_passed


def main():
    print("=" * 80)
    print("R-BPE Rust Tokenizer Encode-Decode Tests")
    print("=" * 80)
    print()
    
    if not RUST_AVAILABLE:
        return 1
    
    tests = [
        ("Basic Encode-Decode", test_basic_encode_decode),
        ("Advanced Decoder", test_advanced_decoder),
        ("Special Tokens", test_with_special_tokens),
        ("Batch Encoding", test_batch_encoding),
        ("Edge Cases", test_edge_cases),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✅ {name} PASSED\n")
            else:
                failed += 1
                print(f"❌ {name} FAILED\n")
        except Exception as e:
            failed += 1
            print(f"❌ {name} FAILED with exception:")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 80)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 80)
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED!")
        print("\nThe Rust tokenizer encode-decode cycle works correctly!")
        print()
        return 0
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
        print("\nThere are issues with the encode-decode cycle.")
        print("Please review the failures above.\n")
        return 1


if __name__ == "__main__":
    exit(main())
