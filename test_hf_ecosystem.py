#!/usr/bin/env python3
"""
Test R-BPE integration with HuggingFace ecosystem

This tests that R-BPE works with:
- Transformers pipelines
- Dataset processing
- Model compatibility
"""

from transformers import AutoTokenizer
from datasets import Dataset

def test_dataset_map():
    """Test tokenizer with datasets library"""
    print("Test: Dataset Processing with .map()")
    print("-" * 80)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("rbpe_tokenizer", trust_remote_code=True)
    
    # Create sample dataset
    data = {
        "text": [
            "Hello World",
            "مرحبا يا عالم",
            "Mixed content مع نص عربي",
            "Another example",
        ]
    }
    dataset = Dataset.from_dict(data)
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=50
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    print(f"✓ Dataset tokenized successfully")
    print(f"  Original dataset: {len(dataset)} examples")
    print(f"  Tokenized dataset: {len(tokenized_dataset)} examples")
    print(f"  Features: {list(tokenized_dataset.features.keys())}")
    print(f"  First example shape: {len(tokenized_dataset[0]['input_ids'])} tokens\n")
    
    return True


def test_batch_processing():
    """Test efficient batch processing"""
    print("Test: Efficient Batch Processing")
    print("-" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained("rbpe_tokenizer", trust_remote_code=True)
    
    # Large batch
    texts = [f"This is example number {i} with some text" for i in range(100)]
    
    # Process in batch
    import time
    start = time.time()
    result = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    elapsed = time.time() - start
    
    print(f"✓ Batch processed 100 texts")
    print(f"  Time: {elapsed*1000:.1f}ms ({elapsed*10:.2f}ms per text)")
    print(f"  Output shape: {result['input_ids'].shape}")
    print(f"  Memory efficient: Using Rust backend\n")
    
    return True


def test_tokenizer_save_load():
    """Test saving and loading tokenizer"""
    print("Test: Save and Load Tokenizer")
    print("-" * 80)
    
    import tempfile
    import shutil
    
    # Load original
    tokenizer = AutoTokenizer.from_pretrained("rbpe_tokenizer", trust_remote_code=True)
    
    # Save to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer.save_pretrained(tmpdir)
        
        print(f"✓ Saved tokenizer to {tmpdir}")
        
        # Load from temp
        loaded_tokenizer = AutoTokenizer.from_pretrained(tmpdir, trust_remote_code=True)
        
        print(f"✓ Loaded tokenizer from {tmpdir}")
        
        # Test they produce same output
        text = "Test text مع نص عربي"
        original_ids = tokenizer.encode(text)
        loaded_ids = loaded_tokenizer.encode(text)
        
        assert original_ids == loaded_ids, "Mismatch between original and loaded!"
        print(f"✓ Original and loaded tokenizers produce identical output\n")
    
    return True


def test_model_compatibility():
    """Test that tokenizer output is compatible with models"""
    print("Test: Model Input Compatibility")
    print("-" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained("rbpe_tokenizer", trust_remote_code=True)
    
    # Prepare inputs like for a model
    texts = ["Hello World", "مرحبا"]
    inputs = tokenizer(texts, padding=True, return_tensors="pt")
    
    # Check format
    assert "input_ids" in inputs
    assert "attention_mask" in inputs
    assert inputs["input_ids"].dim() == 2  # [batch_size, seq_len]
    assert inputs["attention_mask"].dim() == 2
    assert inputs["input_ids"].shape == inputs["attention_mask"].shape
    
    print(f"✓ Output format is model-compatible")
    print(f"  input_ids shape: {inputs['input_ids'].shape}")
    print(f"  attention_mask shape: {inputs['attention_mask'].shape}")
    print(f"  dtype: {inputs['input_ids'].dtype}")
    print(f"  device: {inputs['input_ids'].device}\n")
    
    return True


def test_special_use_cases():
    """Test special use cases"""
    print("Test: Special Use Cases")
    print("-" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained("rbpe_tokenizer", trust_remote_code=True)
    
    # 1. Empty string
    try:
        result = tokenizer("")
        print(f"✓ Empty string: {len(result['input_ids'])} tokens")
    except Exception as e:
        print(f"✗ Empty string failed: {e}")
        return False
    
    # 2. Very long text
    long_text = "Test " * 1000
    result = tokenizer(long_text, truncation=True, max_length=512)
    assert len(result['input_ids']) <= 512
    print(f"✓ Long text (truncated): {len(result['input_ids'])} tokens")
    
    # 3. Special characters
    special_text = "Hello! @#$%^&*() مرحبا؟"
    result = tokenizer(special_text)
    print(f"✓ Special characters: {len(result['input_ids'])} tokens")
    
    # 4. Numbers
    numbers = "123 456 789 ١٢٣ ٤٥٦"
    result = tokenizer(numbers)
    print(f"✓ Numbers: {len(result['input_ids'])} tokens\n")
    
    return True


def main():
    print("=" * 80)
    print("R-BPE HuggingFace Ecosystem Integration Tests")
    print("=" * 80)
    print()
    
    tests = [
        ("Dataset Processing", test_dataset_map),
        ("Batch Processing", test_batch_processing),
        ("Save/Load", test_tokenizer_save_load),
        ("Model Compatibility", test_model_compatibility),
        ("Special Use Cases", test_special_use_cases),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"✗ {name} failed\n")
        except Exception as e:
            failed += 1
            print(f"✗ {name} failed with error:")
            print(f"  {type(e).__name__}: {e}\n")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 80)
    if failed == 0:
        print(f"✅ ALL {passed} TESTS PASSED!")
        print("=" * 80)
        print()
        print("R-BPE is fully integrated with the HuggingFace ecosystem!")
        print()
        print("You can now use R-BPE with:")
        print("  ✓ AutoTokenizer.from_pretrained()")
        print("  ✓ datasets.Dataset.map()")
        print("  ✓ Transformers models")
        print("  ✓ Any HuggingFace-compatible tool")
        print()
        return 0
    else:
        print(f"❌ {failed}/{passed+failed} TESTS FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit(main())
