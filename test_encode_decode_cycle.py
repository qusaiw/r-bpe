#!/usr/bin/env python3
"""
Test encode-decode cycle for R-BPE tokenizer
Specifically tests: inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
Then verifies decoding works correctly.
"""

from transformers import AutoTokenizer
import torch

def test_basic_encode_decode():
    """Test basic encode-decode cycle"""
    print("Test 1: Basic Encode-Decode Cycle")
    print("-" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained("rbpe_tokenizer", trust_remote_code=True)
    
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
        ids = tokenizer.encode(prompt)
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


def test_tensor_encode_decode():
    """Test encode-decode with PyTorch tensors (model input pattern)"""
    print("\nTest 2: Tensor Encode-Decode (Model Input Pattern)")
    print("-" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained("rbpe_tokenizer", trust_remote_code=True)
    
    test_cases = [
        "Hello World! How are you today?",
        "مرحبا! كيف حالك اليوم؟",
        "This is a test هذا اختبار",
    ]
    
    all_passed = True
    for i, prompt in enumerate(test_cases, 1):
        print(f"\n  Test case {i}: {prompt[:50]}...")
        
        # Encode as tensors (like model input)
        inputs = tokenizer(prompt, return_tensors="pt")
        print(f"    Input shape: {inputs['input_ids'].shape}")
        print(f"    Token IDs (tensor): {inputs['input_ids'][0][:10]}...")
        
        # Decode from tensor
        decoded = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
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


def test_batch_tensor_decode():
    """Test batch decoding with tensors"""
    print("\nTest 3: Batch Tensor Decode")
    print("-" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained("rbpe_tokenizer", trust_remote_code=True)
    
    prompts = [
        "Hello World",
        "مرحبا يا عالم",
        "Mixed test مختلط",
    ]
    
    print(f"\n  Encoding {len(prompts)} prompts...")
    
    # Encode as batch with padding
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    print(f"    Batch shape: {inputs['input_ids'].shape}")
    
    # Decode each sequence
    all_passed = True
    for i, (prompt, input_ids) in enumerate(zip(prompts, inputs['input_ids'])):
        decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
        match = decoded.strip() == prompt.strip()
        
        print(f"\n  Sequence {i+1}:")
        print(f"    Original: {prompt}")
        print(f"    Decoded:  {decoded}")
        print(f"    Match: {'✓' if match else '✗'}")
        
        if not match:
            all_passed = False
    
    print("\n" + "=" * 80)
    return all_passed


def test_model_device_pattern():
    """Test the exact pattern: tokenizer(prompt, return_tensors="pt").to(device)"""
    print("\nTest 4: Model Device Pattern")
    print("-" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained("rbpe_tokenizer", trust_remote_code=True)
    
    # Simulate model device
    device = torch.device("cpu")  # Use CPU for testing
    
    prompts = [
        "Hello World! How are you?",
        "مرحبا! كيف حالك؟",
        "Test with mixed مختلط content",
    ]
    
    all_passed = True
    for i, prompt in enumerate(prompts, 1):
        print(f"\n  Test case {i}: {prompt[:50]}...")
        
        # Exact pattern from user's question
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        print(f"    Device: {inputs['input_ids'].device}")
        print(f"    Shape: {inputs['input_ids'].shape}")
        
        # Decode
        decoded = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
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


def test_decode_with_special_tokens():
    """Test decoding with and without special tokens"""
    print("\nTest 5: Decode with Special Tokens")
    print("-" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained("rbpe_tokenizer", trust_remote_code=True)
    
    prompt = "Hello World مرحبا"
    
    print(f"\n  Prompt: {prompt}")
    
    # Encode with special tokens
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    print(f"    Tokens (with special): {inputs['input_ids'].shape}")
    
    # Decode with special tokens
    decoded_with = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
    print(f"    Decoded (keep special): {decoded_with}")
    
    # Decode without special tokens
    decoded_without = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    print(f"    Decoded (skip special): {decoded_without}")
    
    # Check match
    match = decoded_without.strip() == prompt.strip()
    print(f"    Match: {'✓' if match else '✗'}")
    
    print("\n" + "=" * 80)
    return match


def test_edge_cases():
    """Test edge cases that might break decoding"""
    print("\nTest 6: Edge Cases")
    print("-" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained("rbpe_tokenizer", trust_remote_code=True)
    
    edge_cases = [
        ("Empty string", ""),
        ("Single char", "a"),
        ("Single Arabic char", "أ"),
        ("Only spaces", "   "),
        ("Long text", "Hello " * 100),
        ("Special chars", "!@#$%^&*()"),
        ("Numbers", "0123456789"),
        ("Arabic numbers", "٠١٢٣٤٥٦٧٨٩"),
    ]
    
    all_passed = True
    for name, text in edge_cases:
        try:
            inputs = tokenizer(text, return_tensors="pt")
            decoded = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
            
            # For empty/space strings, decoded might normalize differently
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


def test_batch_decode():
    """Test batch_decode method"""
    print("\nTest 7: Batch Decode Method")
    print("-" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained("rbpe_tokenizer", trust_remote_code=True)
    
    prompts = [
        "Hello World",
        "مرحبا يا عالم",
        "Test مختلط mixed",
    ]
    
    # Encode batch
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    
    # Batch decode
    decoded_batch = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
    
    print(f"  Original prompts: {len(prompts)}")
    print(f"  Decoded prompts: {len(decoded_batch)}")
    
    all_passed = True
    for i, (original, decoded) in enumerate(zip(prompts, decoded_batch)):
        match = decoded.strip() == original.strip()
        status = "✓" if match else "✗"
        print(f"\n  {status} Prompt {i+1}:")
        print(f"      Original: {original}")
        print(f"      Decoded:  {decoded}")
        
        if not match:
            all_passed = False
    
    print("\n" + "=" * 80)
    return all_passed


def main():
    print("=" * 80)
    print("R-BPE Encode-Decode Cycle Tests")
    print("=" * 80)
    
    tests = [
        ("Basic Encode-Decode", test_basic_encode_decode),
        ("Tensor Encode-Decode", test_tensor_encode_decode),
        ("Batch Tensor Decode", test_batch_tensor_decode),
        ("Model Device Pattern", test_model_device_pattern),
        ("Special Tokens", test_decode_with_special_tokens),
        ("Edge Cases", test_edge_cases),
        ("Batch Decode", test_batch_decode),
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
        print("\nThe encode-decode cycle works correctly for:")
        print("  ✓ Basic encode/decode")
        print("  ✓ Tensor inputs (model pattern)")
        print("  ✓ Batch processing")
        print("  ✓ Device transfers (.to(device))")
        print("  ✓ Special tokens")
        print("  ✓ Edge cases")
        print()
        return 0
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
        print("\nThere may be issues with the encode-decode cycle.")
        print("Please review the failures above.\n")
        return 1


if __name__ == "__main__":
    exit(main())
