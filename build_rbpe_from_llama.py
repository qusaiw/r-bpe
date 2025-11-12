#!/usr/bin/env python3
"""
Build R-BPE tokenizer from meta-llama/Llama-3.1-8B-Instruct

This script:
1. Downloads the base tokenizer from meta-llama/Llama-3.1-8B-Instruct
2. Creates a new R-BPE tokenizer structure
3. Preserves all special tokens and chat template
4. Saves to rbpe_tokenizer_llama31/ directory
"""

import os
import json
import shutil
from pathlib import Path
from transformers import AutoTokenizer

def build_rbpe_tokenizer(base_model: str, output_dir: str):
    """
    Build R-BPE tokenizer from a base model.
    
    Args:
        base_model: HuggingFace model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        output_dir: Directory to save the R-BPE tokenizer
    """
    print("=" * 80)
    print("Building R-BPE Tokenizer from Base Model")
    print("=" * 80)
    print()
    
    # Step 1: Load base tokenizer
    print(f"Step 1: Loading base tokenizer from {base_model}")
    print("-" * 80)
    try:
        base_tokenizer = AutoTokenizer.from_pretrained(base_model)
        print(f"✓ Loaded tokenizer")
        print(f"  Vocab size: {len(base_tokenizer)}")
        print(f"  Model max length: {base_tokenizer.model_max_length}")
        print()
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        print()
        print("Note: For gated models like Llama, you need to:")
        print("  1. Accept the license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
        print("  2. Login with: huggingface-cli login")
        return False
    
    # Step 2: Create output directory structure
    print(f"Step 2: Creating R-BPE directory structure at {output_dir}")
    print("-" * 80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / "new_tokenizer").mkdir(exist_ok=True)
    (output_path / "old_tokenizer").mkdir(exist_ok=True)
    (output_path / "metadata").mkdir(exist_ok=True)
    
    print(f"✓ Created directory structure")
    print()
    
    # Step 3: Save base tokenizer as "old" tokenizer
    print("Step 3: Saving base tokenizer as 'old' tokenizer")
    print("-" * 80)
    
    old_tokenizer_dir = output_path / "old_tokenizer"
    base_tokenizer.save_pretrained(old_tokenizer_dir)
    print(f"✓ Saved to {old_tokenizer_dir}")
    print()
    
    # Step 4: For now, use same tokenizer as "new" tokenizer (can be replaced later)
    print("Step 4: Copying to 'new' tokenizer (can be replaced with trained version)")
    print("-" * 80)
    
    new_tokenizer_dir = output_path / "new_tokenizer"
    base_tokenizer.save_pretrained(new_tokenizer_dir)
    print(f"✓ Saved to {new_tokenizer_dir}")
    print()
    
    # Step 5: Create identity mappings (since we're using same tokenizer for both)
    print("Step 5: Creating vocabulary mappings")
    print("-" * 80)
    
    vocab_size = len(base_tokenizer)
    
    # Identity mapping: each token maps to itself
    new_to_old_map = {str(i): i for i in range(vocab_size)}
    old_to_new_map = {str(i): i for i in range(vocab_size)}
    
    # Save mappings
    metadata_dir = output_path / "metadata"
    
    with open(metadata_dir / "new_to_old_map.json", "w") as f:
        json.dump(new_to_old_map, f, indent=2)
    
    with open(metadata_dir / "old_to_new_map.json", "w") as f:
        json.dump(old_to_new_map, f, indent=2)
    
    # Create empty replacement character map
    with open(metadata_dir / "replacement_character_map.json", "w") as f:
        json.dump({}, f, indent=2)
    
    print(f"✓ Created mappings for {vocab_size} tokens")
    print()
    
    # Step 6: Create tokenizer config
    print("Step 6: Creating R-BPE tokenizer config")
    print("-" * 80)
    
    # Get special tokens
    special_tokens = {}
    if base_tokenizer.bos_token:
        special_tokens["bos_token"] = base_tokenizer.bos_token
    if base_tokenizer.eos_token:
        special_tokens["eos_token"] = base_tokenizer.eos_token
    if base_tokenizer.unk_token:
        special_tokens["unk_token"] = base_tokenizer.unk_token
    if base_tokenizer.pad_token:
        special_tokens["pad_token"] = base_tokenizer.pad_token
    
    # Get chat template if available
    chat_template = None
    if hasattr(base_tokenizer, 'chat_template') and base_tokenizer.chat_template:
        chat_template = base_tokenizer.chat_template
        print(f"✓ Found chat template")
    
    # Create tokenizer config
    config = {
        "tokenizer_class": "RBPETokenizer",
        "auto_map": {
            "AutoTokenizer": ["tokenization.RBPETokenizer", None]
        },
        "model_max_length": base_tokenizer.model_max_length,
        **special_tokens,
    }
    
    # Add chat template if available
    if chat_template:
        config["chat_template"] = chat_template
    
    # Save config
    with open(output_path / "tokenizer_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Created tokenizer_config.json")
    print()
    
    # Step 7: Copy tokenization.py
    print("Step 7: Copying tokenization implementation")
    print("-" * 80)
    
    source_file = Path("rbpe_tokenizer/tokenization.py")
    if source_file.exists():
        shutil.copy2(source_file, output_path / "tokenization.py")
        print(f"✓ Copied tokenization.py")
    else:
        print(f"✗ Warning: tokenization.py not found at {source_file}")
    print()
    
    # Step 8: Create special_tokens_map.json
    print("Step 8: Creating special tokens map")
    print("-" * 80)
    
    special_tokens_map = {}
    for key in ["bos_token", "eos_token", "unk_token", "pad_token"]:
        token = getattr(base_tokenizer, key, None)
        if token:
            special_tokens_map[key] = {
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False
            }
    
    with open(output_path / "special_tokens_map.json", "w") as f:
        json.dump(special_tokens_map, f, indent=2)
    
    print(f"✓ Created special_tokens_map.json")
    print()
    
    # Step 9: Summary
    print("=" * 80)
    print("✅ R-BPE Tokenizer Built Successfully!")
    print("=" * 80)
    print()
    print(f"Location: {output_dir}")
    print()
    print("Directory structure:")
    print(f"  {output_dir}/")
    print(f"    ├── tokenization.py")
    print(f"    ├── tokenizer_config.json")
    print(f"    ├── special_tokens_map.json")
    print(f"    ├── new_tokenizer/")
    print(f"    │   ├── tokenizer.json")
    print(f"    │   └── ...")
    print(f"    ├── old_tokenizer/")
    print(f"    │   ├── tokenizer.json")
    print(f"    │   └── ...")
    print(f"    └── metadata/")
    print(f"        ├── new_to_old_map.json")
    print(f"        ├── old_to_new_map.json")
    print(f"        └── replacement_character_map.json")
    print()
    print("You can now load this tokenizer with:")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{output_dir}', trust_remote_code=True)")
    print()
    
    if chat_template:
        print("✓ Chat template preserved - apply_chat_template() will work!")
        print()
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build R-BPE tokenizer from a base model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model ID (default: meta-llama/Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rbpe_tokenizer_llama31",
        help="Output directory (default: rbpe_tokenizer_llama31)"
    )
    
    args = parser.parse_args()
    
    success = build_rbpe_tokenizer(args.model, args.output)
    
    if success:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
