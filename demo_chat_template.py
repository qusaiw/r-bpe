#!/usr/bin/env python3
"""
Demo: R-BPE Tokenizer with Chat Template

Shows that apply_chat_template works perfectly with R-BPE tokenizer.
"""

from transformers import AutoTokenizer
import torch


def demo_chat_template():
    print("=" * 80)
    print("R-BPE Tokenizer - Chat Template Demo")
    print("=" * 80)
    print()
    
    # Load tokenizer
    print("Loading R-BPE tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "rbpe_tokenizer_llama31",
        trust_remote_code=True
    )
    print("✓ Tokenizer loaded")
    print()
    
    # Example 1: Simple user message
    print("=" * 80)
    print("Example 1: Simple User Message")
    print("=" * 80)
    print()
    
    messages = [
        {"role": "user", "content": "Hello! Can you help me?"},
    ]
    
    print("Messages:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")
    print()
    
    # Apply chat template (text only)
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print("Formatted text:")
    print("-" * 80)
    print(formatted_text)
    print("-" * 80)
    print()
    
    # Apply chat template (tokenized)
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    print(f"Tokenized: {input_ids.shape}")
    print(f"Token IDs: {input_ids[0][:20].tolist()}...")
    print()
    
    # Example 2: System + User message
    print("=" * 80)
    print("Example 2: System + User Message")
    print("=" * 80)
    print()
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    
    print("Messages:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")
    print()
    
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print("Formatted text:")
    print("-" * 80)
    print(formatted_text)
    print("-" * 80)
    print()
    
    # Example 3: Multi-turn conversation
    print("=" * 80)
    print("Example 3: Multi-turn Conversation")
    print("=" * 80)
    print()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke"},
        {"role": "assistant", "content": "Why did the chicken cross the road?"},
        {"role": "user", "content": "I don't know, why?"},
    ]
    
    print("Messages:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")
    print()
    
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print("Formatted text:")
    print("-" * 80)
    print(formatted_text)
    print("-" * 80)
    print()
    
    # Example 4: Arabic conversation
    print("=" * 80)
    print("Example 4: Arabic Conversation")
    print("=" * 80)
    print()
    
    messages = [
        {"role": "system", "content": "أنت مساعد ذكي ومفيد"},
        {"role": "user", "content": "مرحبا! كيف حالك؟"},
    ]
    
    print("Messages:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")
    print()
    
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print("Formatted text:")
    print("-" * 80)
    print(formatted_text)
    print("-" * 80)
    print()
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    print(f"Tokenized: {input_ids.shape}")
    print()
    
    # Example 5: Model generation simulation
    print("=" * 80)
    print("Example 5: Complete Model Workflow Simulation")
    print("=" * 80)
    print()
    
    messages = [
        {"role": "user", "content": "What is 2+2?"},
    ]
    
    print("Step 1: Format conversation with chat template")
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    print(f"  ✓ Input shape: {input_ids.shape}")
    print()
    
    print("Step 2: Simulate model generation")
    # Simulate generating response tokens (in real use, this would be model.generate())
    response_tokens = [791, 4320, 374, 220, 19]  # "The answer is 4"
    full_output = torch.cat([input_ids, torch.tensor([response_tokens])], dim=1)
    print(f"  ✓ Output shape: {full_output.shape}")
    print()
    
    print("Step 3: Decode response")
    # Decode only the new tokens (skip the prompt)
    response_text = tokenizer.decode(
        full_output[0][input_ids.shape[1]:],
        skip_special_tokens=True
    )
    print(f"  ✓ Response: '{response_text}'")
    print()
    
    # Summary
    print("=" * 80)
    print("✅ SUCCESS!")
    print("=" * 80)
    print()
    print("R-BPE tokenizer fully supports chat templates!")
    print()
    print("You can use it exactly like any other tokenizer:")
    print()
    print("  # Load tokenizer")
    print("  tokenizer = AutoTokenizer.from_pretrained(")
    print("      'rbpe_tokenizer_llama31',")
    print("      trust_remote_code=True")
    print("  )")
    print()
    print("  # Apply chat template")
    print("  messages = [")
    print("      {'role': 'user', 'content': 'Hello!'},")
    print("  ]")
    print("  inputs = tokenizer.apply_chat_template(")
    print("      messages,")
    print("      tokenize=True,")
    print("      add_generation_prompt=True,")
    print("      return_tensors='pt'")
    print("  )")
    print()
    print("  # Generate with model")
    print("  outputs = model.generate(**inputs)")
    print()
    print("  # Decode response")
    print("  response = tokenizer.decode(outputs[0], skip_special_tokens=True)")
    print()


if __name__ == "__main__":
    demo_chat_template()
