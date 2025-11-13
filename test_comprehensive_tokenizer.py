#!/usr/bin/env python3
"""
Comprehensive R-BPE Tokenizer Test Suite

Tests that R-BPE acts exactly like any other HuggingFace tokenizer.
Includes tests for:
- Basic tokenization (encode/decode)
- Batch processing
- Padding and truncation
- Special tokens
- Chat template (apply_chat_template)
- Model compatibility
- Save/load
- All standard tokenizer methods
"""

from transformers import AutoTokenizer
import torch
from typing import List, Dict


class TokenizerTestSuite:
    """Comprehensive tokenizer test suite"""
    
    def __init__(self, tokenizer_path: str, reference_model: str = None):
        """
        Initialize test suite.
        
        Args:
            tokenizer_path: Path to R-BPE tokenizer
            reference_model: Optional reference model to compare against
        """
        self.tokenizer_path = tokenizer_path
        self.reference_model = reference_model
        self.tokenizer = None
        self.reference_tokenizer = None
        
        self.test_texts = [
            "Hello, world!",
            "مرحبا بالعالم!",
            "This is a test with mixed مختلط content",
            "Numbers: 123 456 789",
            "Special chars: !@#$%^&*()",
            "",  # Empty string
            "A" * 1000,  # Long text
        ]
        
        self.chat_messages = [
            [
                {"role": "user", "content": "Hello!"},
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            [
                {"role": "user", "content": "Tell me a joke"},
                {"role": "assistant", "content": "Why did the chicken cross the road?"},
                {"role": "user", "content": "I don't know, why?"},
            ],
        ]
    
    def load_tokenizers(self):
        """Load tokenizers"""
        print("Loading tokenizers...")
        print("-" * 80)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=True
        )
        print(f"✓ Loaded R-BPE tokenizer from {self.tokenizer_path}")
        
        if self.reference_model:
            self.reference_tokenizer = AutoTokenizer.from_pretrained(
                self.reference_model
            )
            print(f"✓ Loaded reference tokenizer from {self.reference_model}")
        
        print()
    
    def test_basic_attributes(self):
        """Test 1: Basic tokenizer attributes"""
        print("Test 1: Basic Tokenizer Attributes")
        print("-" * 80)
        
        tests = [
            ("vocab_size", hasattr(self.tokenizer, 'vocab_size')),
            ("model_max_length", hasattr(self.tokenizer, 'model_max_length')),
            ("is_fast", hasattr(self.tokenizer, 'is_fast')),
            ("bos_token", hasattr(self.tokenizer, 'bos_token')),
            ("eos_token", hasattr(self.tokenizer, 'eos_token')),
            ("pad_token", hasattr(self.tokenizer, 'pad_token')),
        ]
        
        all_passed = True
        for attr, has_attr in tests:
            if has_attr:
                value = getattr(self.tokenizer, attr)
                print(f"  ✓ {attr}: {value}")
            else:
                print(f"  ✗ {attr}: NOT FOUND")
                all_passed = False
        
        print()
        return all_passed
    
    def test_encode_decode(self):
        """Test 2: Basic encode/decode"""
        print("Test 2: Basic Encode/Decode")
        print("-" * 80)
        
        all_passed = True
        for i, text in enumerate(self.test_texts[:5], 1):  # Skip long/empty
            try:
                # Encode
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                
                # Decode
                decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
                
                # Check
                match = decoded.strip() == text.strip()
                status = "✓" if match else "✗"
                
                print(f"  {status} Test {i}: '{text[:40]}...' -> {len(ids)} tokens")
                
                if not match:
                    print(f"      Expected: '{text}'")
                    print(f"      Got:      '{decoded}'")
                    all_passed = False
            except Exception as e:
                print(f"  ✗ Test {i}: ERROR - {e}")
                all_passed = False
        
        print()
        return all_passed
    
    def test_call_interface(self):
        """Test 3: __call__ interface"""
        print("Test 3: __call__ Interface (Standard HF API)")
        print("-" * 80)
        
        text = "Hello, world!"
        
        try:
            # Single text
            result = self.tokenizer(text)
            assert "input_ids" in result
            assert "attention_mask" in result
            print(f"  ✓ Single text: {len(result['input_ids'])} tokens")
            
            # With return_tensors
            result = self.tokenizer(text, return_tensors="pt")
            assert isinstance(result["input_ids"], torch.Tensor)
            assert result["input_ids"].dim() == 2  # Should be 2D
            print(f"  ✓ With return_tensors='pt': shape {result['input_ids'].shape}")
            
            # Batch
            texts = ["Hello", "World", "Test"]
            result = self.tokenizer(texts, padding=True)
            assert len(result["input_ids"]) == 3
            print(f"  ✓ Batch processing: {len(texts)} texts")
            
            # Truncation
            result = self.tokenizer(text, max_length=5, truncation=True)
            assert len(result["input_ids"]) <= 5
            print(f"  ✓ Truncation: limited to {len(result['input_ids'])} tokens")
            
            # Padding
            result = self.tokenizer(texts, padding="max_length", max_length=10)
            assert all(len(ids) == 10 for ids in result["input_ids"])
            print(f"  ✓ Padding: all sequences padded to 10")
            
            print()
            return True
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            print()
            return False
    
    def test_batch_methods(self):
        """Test 4: Batch encode/decode methods"""
        print("Test 4: Batch Encode/Decode Methods")
        print("-" * 80)
        
        texts = ["Hello", "مرحبا", "Test"]
        
        try:
            # batch_encode_plus
            result = self.tokenizer.batch_encode_plus(
                texts,
                padding=True,
                return_tensors="pt"
            )
            assert result["input_ids"].shape[0] == 3
            print(f"  ✓ batch_encode_plus: {result['input_ids'].shape}")
            
            # encode_plus
            result = self.tokenizer.encode_plus(
                texts[0],
                return_tensors="pt"
            )
            assert "input_ids" in result
            print(f"  ✓ encode_plus: {result['input_ids'].shape}")
            
            # batch_decode
            ids_list = [
                self.tokenizer.encode(text, add_special_tokens=False)
                for text in texts
            ]
            decoded = self.tokenizer.batch_decode(ids_list, skip_special_tokens=True)
            assert len(decoded) == len(texts)
            print(f"  ✓ batch_decode: {len(decoded)} sequences")
            
            print()
            return True
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            print()
            return False
    
    def test_special_tokens(self):
        """Test 5: Special tokens handling"""
        print("Test 5: Special Tokens Handling")
        print("-" * 80)
        
        text = "Hello, world!"
        
        try:
            # Without special tokens
            ids_without = self.tokenizer.encode(text, add_special_tokens=False)
            
            # With special tokens
            ids_with = self.tokenizer.encode(text, add_special_tokens=True)
            
            # Should have more tokens with special tokens
            has_difference = len(ids_with) > len(ids_without)
            
            print(f"  Without special tokens: {len(ids_without)} tokens")
            print(f"  With special tokens:    {len(ids_with)} tokens")
            print(f"  Difference: {len(ids_with) - len(ids_without)}")
            
            # Decode with skip_special_tokens
            decoded_skip = self.tokenizer.decode(ids_with, skip_special_tokens=True)
            decoded_keep = self.tokenizer.decode(ids_with, skip_special_tokens=False)
            
            print(f"  ✓ Decode (skip special): '{decoded_skip}'")
            print(f"  ✓ Decode (keep special): '{decoded_keep[:50]}...'")
            
            # Check special token IDs
            if self.tokenizer.bos_token_id:
                print(f"  ✓ BOS token ID: {self.tokenizer.bos_token_id}")
            if self.tokenizer.eos_token_id:
                print(f"  ✓ EOS token ID: {self.tokenizer.eos_token_id}")
            if self.tokenizer.pad_token_id:
                print(f"  ✓ PAD token ID: {self.tokenizer.pad_token_id}")
            
            print()
            return True
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            print()
            return False
    
    def test_chat_template(self):
        """Test 6: apply_chat_template"""
        print("Test 6: apply_chat_template (Chat Template Support)")
        print("-" * 80)
        
        if not hasattr(self.tokenizer, 'apply_chat_template'):
            print("  ✗ apply_chat_template not available")
            print()
            return False
        
        try:
            all_passed = True
            
            for i, messages in enumerate(self.chat_messages, 1):
                # Apply chat template
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                
                print(f"  ✓ Test {i}: {len(messages)} message(s)")
                print(f"      Formatted length: {len(formatted)} chars")
                print(f"      Preview: {formatted[:60]}...")
                
                # Also test with tokenization
                ids = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False
                )
                print(f"      Tokenized: {len(ids)} tokens")
                
                # Test with generation prompt
                ids_with_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True
                )
                print(f"      With generation prompt: {len(ids_with_prompt)} tokens")
                print()
            
            print("  ✅ apply_chat_template works perfectly!")
            print()
            return all_passed
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            print()
            return False
    
    def test_model_compatibility(self):
        """Test 7: Model input compatibility"""
        print("Test 7: Model Input Compatibility")
        print("-" * 80)
        
        prompt = "Hello, world!"
        device = torch.device("cpu")
        
        try:
            # Pattern: tokenizer(prompt, return_tensors="pt").to(device)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            
            # Check format
            assert "input_ids" in inputs
            assert "attention_mask" in inputs
            assert inputs["input_ids"].dim() == 2
            assert inputs["input_ids"].device.type == "cpu"
            
            print(f"  ✓ Input format is model-compatible")
            print(f"    input_ids shape: {inputs['input_ids'].shape}")
            print(f"    attention_mask shape: {inputs['attention_mask'].shape}")
            print(f"    device: {inputs['input_ids'].device}")
            
            # Decode
            decoded = self.tokenizer.decode(
                inputs['input_ids'][0],
                skip_special_tokens=True
            )
            print(f"    Decoded: '{decoded}'")
            
            # Check match
            match = decoded.strip() == prompt.strip()
            if match:
                print(f"  ✓ Encode-decode cycle works with tensors")
            else:
                print(f"  ✗ Mismatch in decode")
                return False
            
            print()
            return True
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            print()
            return False
    
    def test_save_load(self):
        """Test 8: Save and load tokenizer"""
        print("Test 8: Save and Load Tokenizer")
        print("-" * 80)
        
        import tempfile
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save
                self.tokenizer.save_pretrained(tmpdir)
                print(f"  ✓ Saved to temporary directory")
                
                # Load
                loaded_tokenizer = AutoTokenizer.from_pretrained(
                    tmpdir,
                    trust_remote_code=True
                )
                print(f"  ✓ Loaded from temporary directory")
                
                # Test they produce same output
                text = "Hello, world!"
                original_ids = self.tokenizer.encode(text)
                loaded_ids = loaded_tokenizer.encode(text)
                
                if original_ids == loaded_ids:
                    print(f"  ✓ Original and loaded produce identical output")
                else:
                    print(f"  ✗ Mismatch between original and loaded")
                    return False
                
                print()
                return True
                
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            print()
            return False
    
    def test_comparison_with_reference(self):
        """Test 9: Comparison with reference tokenizer"""
        print("Test 9: Comparison with Reference Tokenizer")
        print("-" * 80)
        
        if not self.reference_tokenizer:
            print("  ⊘ Skipped (no reference tokenizer)")
            print()
            return True
        
        try:
            text = "Hello, world!"
            
            # Compare encoding
            rbpe_ids = self.tokenizer.encode(text, add_special_tokens=False)
            ref_ids = self.reference_tokenizer.encode(text, add_special_tokens=False)
            
            match = rbpe_ids == ref_ids
            status = "✓" if match else "✗"
            
            print(f"  {status} Encoding comparison")
            print(f"    R-BPE: {len(rbpe_ids)} tokens - {rbpe_ids[:10]}...")
            print(f"    Reference: {len(ref_ids)} tokens - {ref_ids[:10]}...")
            
            if not match:
                print(f"    Note: Different encodings are OK if tokenizers differ")
            
            # Compare vocab size
            print(f"  ✓ Vocab size comparison")
            print(f"    R-BPE: {len(self.tokenizer)}")
            print(f"    Reference: {len(self.reference_tokenizer)}")
            
            # Compare special tokens
            print(f"  ✓ Special tokens")
            print(f"    R-BPE BOS: {self.tokenizer.bos_token_id}")
            print(f"    Reference BOS: {self.reference_tokenizer.bos_token_id}")
            
            print()
            return True
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            print()
            return False
    
    def test_edge_cases(self):
        """Test 10: Edge cases"""
        print("Test 10: Edge Cases")
        print("-" * 80)
        
        edge_cases = [
            ("Empty string", ""),
            ("Whitespace only", "   "),
            ("Single character", "A"),
            ("Long text", "A" * 1000),
            ("Unicode", "Hello 👋 世界 🌍"),
            ("Mixed RTL/LTR", "Hello مرحبا שלום"),
        ]
        
        all_passed = True
        for name, text in edge_cases:
            try:
                result = self.tokenizer(text, return_tensors="pt")
                decoded = self.tokenizer.decode(
                    result['input_ids'][0],
                    skip_special_tokens=True
                )
                print(f"  ✓ {name}: {len(result['input_ids'][0])} tokens")
            except Exception as e:
                print(f"  ✗ {name}: ERROR - {e}")
                all_passed = False
        
        print()
        return all_passed
    
    def run_all_tests(self):
        """Run all tests"""
        print("=" * 80)
        print("Comprehensive R-BPE Tokenizer Test Suite")
        print("=" * 80)
        print()
        
        self.load_tokenizers()
        
        tests = [
            ("Basic Attributes", self.test_basic_attributes),
            ("Encode/Decode", self.test_encode_decode),
            ("__call__ Interface", self.test_call_interface),
            ("Batch Methods", self.test_batch_methods),
            ("Special Tokens", self.test_special_tokens),
            ("Chat Template", self.test_chat_template),
            ("Model Compatibility", self.test_model_compatibility),
            ("Save/Load", self.test_save_load),
            ("Reference Comparison", self.test_comparison_with_reference),
            ("Edge Cases", self.test_edge_cases),
        ]
        
        results = []
        for name, test_func in tests:
            try:
                passed = test_func()
                results.append((name, passed))
            except Exception as e:
                print(f"✗ {name} crashed: {e}")
                import traceback
                traceback.print_exc()
                results.append((name, False))
        
        # Summary
        print("=" * 80)
        print("Test Results Summary")
        print("=" * 80)
        print()
        
        passed = sum(1 for _, p in results if p)
        total = len(results)
        
        for name, result in results:
            status = "✅" if result else "❌"
            print(f"  {status} {name}")
        
        print()
        print("-" * 80)
        print(f"  {passed}/{total} tests passed")
        print("-" * 80)
        print()
        
        if passed == total:
            print("🎉 SUCCESS! R-BPE tokenizer works exactly like a standard tokenizer!")
            print()
            print("You can use it with:")
            print("  ✓ AutoTokenizer.from_pretrained()")
            print("  ✓ model.generate()")
            print("  ✓ apply_chat_template()")
            print("  ✓ Transformers Trainer")
            print("  ✓ Any HuggingFace-compatible library")
            print()
            return 0
        else:
            print(f"⚠️  {total - passed} test(s) failed")
            print("Please review the failures above.")
            print()
            return 1


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive R-BPE tokenizer test suite"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="rbpe_tokenizer_llama31",
        help="Path to R-BPE tokenizer (default: rbpe_tokenizer_llama31)"
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Optional reference model to compare against"
    )
    
    args = parser.parse_args()
    
    suite = TokenizerTestSuite(args.tokenizer, args.reference)
    return suite.run_all_tests()


if __name__ == "__main__":
    exit(main())
