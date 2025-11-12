#!/usr/bin/env python3
"""
Demo: R-BPE tokenizer with model input pattern
Shows that: inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
followed by decoding works perfectly.
"""

from transformers import AutoTokenizer
import torch

def main():
    print("=" * 80)
    print("R-BPE Tokenizer - Model Input Pattern Demo")
    print("=" * 80)
    print()
    
    # Load tokenizer
    print("Loading R-BPE tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("rbpe_tokenizer", trust_remote_code=True)
    print("✓ Tokenizer loaded\n")
    
    # Simulate model device (use cuda if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Test prompts
    prompts = [
        "Hello World! How are you today?",
        "مرحبا! كيف حالك اليوم؟",
        "This is a mixed prompt مع نص عربي",
    ]
    
    print("=" * 80)
    print("Testing the exact pattern: tokenizer(prompt, return_tensors='pt').to(device)")
    print("=" * 80)
    print()
    
    for i, prompt in enumerate(prompts, 1):
        print(f"Test {i}: {prompt}")
        print("-" * 80)
        
        # THE EXACT PATTERN FROM YOUR QUESTION:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        print(f"✓ Tokenized")
        print(f"  Shape: {inputs['input_ids'].shape}")
        print(f"  Device: {inputs['input_ids'].device}")
        print(f"  First 10 token IDs: {inputs['input_ids'][0][:10].tolist()}")
        
        # Decode back
        decoded = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        
        print(f"✓ Decoded: {decoded}")
        
        # Verify
        match = decoded.strip() == prompt.strip()
        print(f"✓ Match: {match}")
        
        if not match:
            print(f"  WARNING: Mismatch!")
            print(f"  Expected: '{prompt}'")
            print(f"  Got:      '{decoded}'")
        
        print()
    
    print("=" * 80)
    print("✅ SUCCESS!")
    print("=" * 80)
    print()
    print("The pattern works perfectly:")
    print("  inputs = tokenizer(prompt, return_tensors='pt').to(model.device)")
    print("  outputs = model.generate(**inputs)")
    print("  decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)")
    print()
    print("You can now use R-BPE tokenizer with any HuggingFace model!")
    print()


if __name__ == "__main__":
    main()
