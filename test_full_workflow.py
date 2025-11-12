#!/usr/bin/env python3
"""
Comprehensive R-BPE Test Suite

Tests the full workflow:
1. Check installations
2. Test Rust tokenizer directly
3. Test HuggingFace AutoTokenizer loading
4. Test encoding/decoding
5. Test batch operations
6. Verify performance
"""

import sys
import time
from pathlib import Path

def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def test_installations():
    """Test that all components are installed."""
    print_section("1. Testing Installations")
    
    try:
        from rbpe import RBPETokenizer
        print("✓ Python training components installed")
    except ImportError as e:
        print(f"✗ Failed to import rbpe: {e}")
        return False
    
    try:
        import rbpe_tokenizers
        print("✓ Rust tokenizer installed")
        print(f"  Version: {rbpe_tokenizers.__version__ if hasattr(rbpe_tokenizers, '__version__') else 'unknown'}")
    except ImportError as e:
        print(f"✗ Failed to import rbpe_tokenizers: {e}")
        print("  Run: cd rbpe-tokenizers && maturin develop --release")
        return False
    
    try:
        from transformers import AutoTokenizer
        print("✓ HuggingFace transformers installed")
    except ImportError as e:
        print(f"✗ Failed to import transformers: {e}")
        return False
    
    return True

def test_rust_tokenizer_direct():
    """Test loading and using Rust tokenizer directly."""
    print_section("2. Testing Rust Tokenizer (Direct API)")
    
    try:
        import rbpe_tokenizers
        
        tokenizer_path = "rbpe_tokenizer"
        if not Path(tokenizer_path).exists():
            print(f"✗ Tokenizer not found at: {tokenizer_path}")
            print("  Please ensure a trained tokenizer exists")
            return False
        
        print(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer = rbpe_tokenizers.RBPETokenizer.from_pretrained(tokenizer_path)
        print(f"✓ Loaded: {tokenizer}")
        
        # Test encoding
        test_texts = [
            ("Pure English", "Hello World!"),
            ("Pure Arabic", "مرحبا يا عالم"),
            ("Mixed", "Hello مرحبا World"),
        ]
        
        for name, text in test_texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            decoded_adv = tokenizer.decode_advanced(ids, skip_special_tokens=True)
            
            match = decoded.strip() == text.strip()
            match_adv = decoded_adv.strip() == text.strip()
            
            print(f"\n  {name}: '{text}'")
            print(f"    Tokens: {len(ids)}")
            print(f"    Basic decode:    {'✓' if match else '✗'} '{decoded}'")
            print(f"    Advanced decode: {'✓' if match_adv else '✗'} '{decoded_adv}'")
        
        # Test batch
        print("\n  Batch operations:")
        batch_texts = ["Hello", "مرحبا", "World", "عالم"]
        batch_ids = tokenizer.encode_batch(batch_texts, add_special_tokens=False)
        batch_decoded = tokenizer.decode_batch(batch_ids, skip_special_tokens=True)
        
        print(f"    Encoded {len(batch_texts)} texts")
        for orig, dec in zip(batch_texts, batch_decoded):
            match = orig.strip() == dec.strip()
            print(f"      '{orig}' -> '{dec}' {'✓' if match else '✗'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_autotokenizer():
    """Test loading with HuggingFace AutoTokenizer."""
    print_section("3. Testing HuggingFace AutoTokenizer")
    
    try:
        from transformers import AutoTokenizer
        
        tokenizer_path = "rbpe_tokenizer"
        print(f"Loading with AutoTokenizer from: {tokenizer_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )
        
        print(f"✓ Loaded successfully")
        print(f"  Class: {tokenizer.__class__.__name__}")
        print(f"  Module: {tokenizer.__class__.__module__}")
        
        # Test basic operations
        text = "Hello مرحبا World"
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        
        print(f"\n  Single text encoding:")
        print(f"    Input:   '{text}'")
        print(f"    Tokens:  {len(ids)}")
        print(f"    Decoded: '{decoded}'")
        
        # Test __call__ method
        print(f"\n  Testing __call__ method:")
        result = tokenizer(text, add_special_tokens=False)
        print(f"    ✓ input_ids: {len(result['input_ids'])} tokens")
        print(f"    ✓ attention_mask: {len(result['attention_mask'])} values")
        
        # Test batch
        print(f"\n  Testing batch encoding:")
        batch_texts = ["Hello", "مرحبا", "World"]
        result = tokenizer(
            batch_texts,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt"
        )
        print(f"    ✓ Batch shape: {result['input_ids'].shape}")
        print(f"    ✓ Attention mask shape: {result['attention_mask'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance():
    """Test performance of Rust tokenizer."""
    print_section("4. Performance Test")
    
    try:
        import rbpe_tokenizers
        
        tokenizer = rbpe_tokenizers.RBPETokenizer.from_pretrained("rbpe_tokenizer")
        
        # Warm up
        for _ in range(10):
            tokenizer.encode("Hello مرحبا World", add_special_tokens=False)
        
        # Single text performance
        text = "Hello مرحبا World عالم! This is a test هذا اختبار"
        iterations = 1000
        
        print(f"Testing {iterations} iterations of encode+decode")
        print(f"Text length: {len(text)} chars")
        
        start = time.time()
        for _ in range(iterations):
            ids = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
        elapsed = time.time() - start
        
        ops_per_sec = iterations / elapsed
        us_per_op = (elapsed / iterations) * 1_000_000
        
        print(f"\n  Results:")
        print(f"    Total time: {elapsed:.3f}s")
        print(f"    Operations/sec: {ops_per_sec:,.0f}")
        print(f"    Time per operation: {us_per_op:.1f} µs")
        
        # Batch performance
        batch_texts = ["Test text " + str(i) for i in range(100)]
        
        start = time.time()
        batch_ids = tokenizer.encode_batch(batch_texts, add_special_tokens=False)
        batch_decoded = tokenizer.decode_batch(batch_ids, skip_special_tokens=True)
        elapsed = time.time() - start
        
        throughput = len(batch_texts) / elapsed
        
        print(f"\n  Batch performance (100 texts):")
        print(f"    Total time: {elapsed*1000:.1f}ms")
        print(f"    Throughput: {throughput:,.0f} texts/sec")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        return False

def test_tokenizer_structure():
    """Test that saved tokenizer has correct structure."""
    print_section("5. Testing Tokenizer Structure")
    
    tokenizer_path = Path("rbpe_tokenizer")
    
    required_files = [
        "new_tokenizer/tokenizer.json",
        "old_tokenizer/tokenizer.json",
        "metadata/new_to_old_map.json",
        "metadata/old_to_new_map.json",
        "tokenization_rbpe.py",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    
    print("Checking required files:")
    all_exist = True
    for file in required_files:
        file_path = tokenizer_path / file
        exists = file_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")
        if not exists:
            all_exist = False
    
    # Check tokenization_rbpe.py content
    tokenization_file = tokenizer_path / "tokenization_rbpe.py"
    if tokenization_file.exists():
        content = tokenization_file.read_text()
        has_rust_import = "import rbpe_tokenizers" in content
        has_rust_class = "RBPETokenizer" in content
        
        print(f"\n  Checking tokenization_rbpe.py:")
        print(f"    {'✓' if has_rust_import else '✗'} Has Rust import")
        print(f"    {'✓' if has_rust_class else '✗'} Has RBPETokenizer class")
        
        if not (has_rust_import and has_rust_class):
            print(f"    ⚠ Warning: tokenization_rbpe.py may be from old format")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests."""
    print("=" * 80)
    print("  R-BPE Comprehensive Test Suite")
    print("=" * 80)
    
    tests = [
        ("Installations", test_installations),
        ("Rust Tokenizer Direct", test_rust_tokenizer_direct),
        ("HuggingFace AutoTokenizer", test_autotokenizer),
        ("Performance", test_performance),
        ("Tokenizer Structure", test_tokenizer_structure),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results[name] = False
    
    # Summary
    print_section("Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! R-BPE is working correctly.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
