from typing import List, Optional, Union
from transformers import PreTrainedTokenizerBase, AutoTokenizer

import os
import sys
import shutil
import pkg_resources
import importlib
from pathlib import Path
from datetime import datetime

from .token_classifier import TokenClassifier
from .data_cleaner import DataCleaner
from .bpe_tokenizer_trainer import BPETokenizerTrainer
from .mapping_tokenizer import MappingTokenizer
import os
import json
import yaml
import argparse

from huggingface_hub import login

from .logger_config import setup_logger

logger = setup_logger('BPE')


class RBPETokenizer:
    """Factory class to create and prepare a custom R-BPE tokenizer with minimal configuration requirements."""
    def __init__(self, model_id, training_data_dir, clean_data=True, cleaned_data_dir=None,
                         hf_token=None, min_reusable_count=20000, target_language_scripts=['arabic'], preserved_languages_scripts=['latin', 'greek'],
                         special_tokens={}, additional_special_tokens=[], apply_rbpe_arabic_norm=True):
        """Initialize an R-BPE tokenizer from parameters.
        
        Args:
            model_id (str): The HuggingFace model id of the original tokenizer.
            training_data_dir (str): The directory where the training data for the new tokenizer is stored.
            clean_data (bool): Whether to clean the training data or not.
            cleaned_data_dir (str): The directory where the cleaned training data for the new tokenizer should be saved. Optional, will only process in memory if not provided.
            hf_token (str): The HuggingFace access token.
            min_reusable_count (int): The minimum number of tokens needed for reuse (threshold ***_h_*** in the paper).
            target_language_scripts (list): The list of the unicode script names or aliases of the target language.
            preserved_languages_scripts (list): the unicode script names or aliases of the languages that must be preserved.
            special_tokens (dict): The dictionary of custom special tokens values for the main special tokens: pad_token, unk_token, bos_token, mask_token, sep_token, cls_token.
            additional_special_tokens (list): The list of additional special tokens the new tokenizer will have.
            apply_rbpe_arabic_norm (bool): Whether to apply the R-BPE Arabic normalization during encoding or not.
        """
        
        self.token_classifier = None
        self.tokenizer = None
        self.old_tokenizer = None
        self.new_tokenizer = None
        self.mapping_tokenizer = None
        self.reusable_languages_dict = None
        self.target_language_scripts_ranges = None

        # Validate required parameters
        if not model_id:
            raise ValueError("model_id is required")
        self.model_id = model_id
        if not training_data_dir:
            raise ValueError("training_data_dir is required to train the new tokenizer")
        if clean_data and not cleaned_data_dir:
            logger.warning("cleaned_data_dir was not provided. Cleaned data will not be saved to disk.")
        self.training_data_dir = training_data_dir
        self.clean_data = clean_data
        self.cleaned_data_dir = cleaned_data_dir if clean_data else None

        if not hf_token:
            raise ValueError("hf_token is required to log in to Hugging Face Hub")
        self.hf_token = hf_token
        self.min_reusable_count = min_reusable_count
        self.target_language_scripts = target_language_scripts
        self.preserved_languages_scripts = preserved_languages_scripts
        self.special_tokens = special_tokens
        self.additional_special_tokens = additional_special_tokens
        self.apply_rbpe_arabic_norm = apply_rbpe_arabic_norm
        
        # Login to HF
        try:
            login(token=hf_token)
            logger.debug("Successfully logged in to Hugging Face Hub")
        except Exception as e:
            error_msg = f"Failed to log in to Hugging Face Hub: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    @classmethod
    def from_config(cls, config_path: str):
        """Initialize an R-BPE tokenizer from a YAML config file.
        
        Args:
            config_path (str): Path to YAML config file with simplified parameters
            
        Returns:
            RBPETokenizer: Initialized tokenizer instance
        """
        # Load config from YAML
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract parameters from config
        model_id = config.get('model_id')
        training_data_dir = config.get('training_data_dir')
        clean_data = config.get('clean_data', True)
        cleaned_data_dir = config.get('cleaned_data_dir')
        # TODO: make hf_token an environment variable!
        hf_token = config.get('hf_token')
        min_reusable_count = config.get('min_reusable_count', 20000)
        target_language_scripts = config.get('target_language_scripts', [])
        preserved_languages_scripts = config.get('preserved_languages_scripts', [])
        apply_rbpe_arabic_norm = config.get('apply_rbpe_arabic_norm', True)
        
        # Extract special tokens
        special_tokens = config.get('special_tokens', {})
        additional_special_tokens = config.get('additional_special_tokens', [])
        
        # Create instance with parameters
        instance = cls.__new__(cls)
        instance.__init__(
            model_id=model_id,
            training_data_dir=training_data_dir,
            clean_data=clean_data,
            cleaned_data_dir=cleaned_data_dir,
            hf_token=hf_token,
            min_reusable_count=min_reusable_count,
            target_language_scripts=target_language_scripts,
            preserved_languages_scripts=preserved_languages_scripts,
            apply_rbpe_arabic_norm=apply_rbpe_arabic_norm,
            special_tokens=special_tokens,
            additional_special_tokens=additional_special_tokens,
        )
        return instance
    
    def prepare(self) -> PreTrainedTokenizerBase:
        """
        Orchestrates the complete tokenizer preparation process:
        1. Classifies tokens using TokenClassifier
        2. Cleans data using DataCleaner (if needed)
        3. Trains new tokenizer using BPETokenizerTrainer
        4. Creates mappings using MappingTokenizer
        5. Returns final RBPETokenizer instance
        
        Returns:
            PreTrainedTokenizerBase: The prepared tokenizer
        """
        logger.info("Starting tokenizer preparation process...")

        # Token Classification
        logger.info("Initializing TokenClassifier...")
        self.token_classifier = TokenClassifier(
            min_reusable_ids=self.min_reusable_count,
            target_language_scripts=self.target_language_scripts,
            preserved_languages_scripts=self.preserved_languages_scripts,
            old_tokenizer_model_id=self.model_id,
            hf_api_key=self.hf_token,
        )

        # Get reusable languages and ranges
        self.reusable_languages_dict, total_reusable_count = self.token_classifier.get_reusable_languages_and_count()
        self.target_language_scripts_ranges = self.token_classifier.get_target_language_scripts_ranges()
        
        cleaned_dataset = None
        if self.clean_data:
            # Clean Data
            logger.info("Starting data cleaning process...")
            
            cleaner = DataCleaner(
                data_dir=self.training_data_dir,
                reusable_languages_with_ranges=self.reusable_languages_dict,
                cleaned_data_dir=self.cleaned_data_dir
            )
            cleaned_dataset = cleaner.process()
            logger.info("Data cleaning completed")
        
        # Train the new tokenizer
        logger.info("Training new BPE tokenizer...")
        
        special_tokens_dict = {
            'additional_special_tokens': self.additional_special_tokens
        }
        
        if cleaned_dataset:
            trainer = BPETokenizerTrainer(
                dataset=cleaned_dataset,
                vocab_size=total_reusable_count,
                model_id=self.model_id,
                special_tokens_dict=special_tokens_dict,
            )
        else:
            trainer = BPETokenizerTrainer(
                dataset_dir=self.training_data_dir,
                vocab_size=total_reusable_count,
                model_id=self.model_id,
                special_tokens_dict=special_tokens_dict,
            )
        self.new_tokenizer = trainer.run()
        
        logger.info("Tokenizer training completed successfully.")
        
        # TODO: update var names to match paper
        # Create mapping layer
        logger.info("Creating mapping tokenizer...")
        
        self.old_tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.mapping_tokenizer = MappingTokenizer(
            new_tokenizer=self.new_tokenizer,
            old_tokenizer=self.old_tokenizer,
            token_id_language_map=self.token_classifier.classified_ids_with_ranges,
            reusable_languages=list(self.reusable_languages_dict.keys()),
            target_language_scripts_ranges=self.target_language_scripts_ranges,
            new_tokenizer_additional_special_tokens=self.additional_special_tokens,
            apply_normalization=self.apply_rbpe_arabic_norm,
        )
        
        logger.info("Mapping creation completed successfully.")
        
        # Create a simple wrapper that knows how to save the tokenizer in the right format
        logger.info("Creating R-BPE tokenizer wrapper for saving...")
        
        class RBPETokenizerSaver:
            """Simple wrapper that saves tokenizer files in format compatible with Rust backend."""
            def __init__(self, factory):
                self.factory = factory
                self.old_tokenizer = factory.old_tokenizer
                self.new_tokenizer = factory.new_tokenizer
                self.mapping_tokenizer = factory.mapping_tokenizer
                self.token_classifier = factory.token_classifier
                self.reusable_languages_dict = factory.reusable_languages_dict
                self.target_language_scripts_ranges = factory.target_language_scripts_ranges
                self.special_tokens = factory.special_tokens
                self.model_id = factory.model_id
            
            def save_pretrained(self, save_directory: str):
                """Save tokenizer in format compatible with Rust backend and AutoTokenizer."""
                os.makedirs(save_directory, exist_ok=True)
                
                # Save new and old tokenizers
                new_tok_dir = os.path.join(save_directory, "new_tokenizer")
                old_tok_dir = os.path.join(save_directory, "old_tokenizer")
                os.makedirs(new_tok_dir, exist_ok=True)
                os.makedirs(old_tok_dir, exist_ok=True)
                
                self.new_tokenizer.save_pretrained(new_tok_dir)
                self.old_tokenizer.save_pretrained(old_tok_dir)
                
                # Save metadata files
                meta_dir = os.path.join(save_directory, "metadata")
                os.makedirs(meta_dir, exist_ok=True)
                
                with open(os.path.join(meta_dir, "new_to_old_map.json"), "w") as f:
                    json.dump({str(k): v for k, v in self.mapping_tokenizer.new_to_old_map.items()}, f, indent=2)
                
                with open(os.path.join(meta_dir, "old_to_new_map.json"), "w") as f:
                    json.dump({str(k): v for k, v in self.mapping_tokenizer.old_to_new_map.items()}, f, indent=2)
                
                with open(os.path.join(meta_dir, "replacement_character_map.json"), "w") as f:
                    json.dump({str(k): v for k, v in self.mapping_tokenizer.replacement_character_map.items()}, f, indent=2)
                
                # Optional metadata for reference
                with open(os.path.join(meta_dir, "token_id_language_map.json"), "w") as f:
                    json.dump(self.token_classifier.classified_ids_with_ranges, f, indent=2)
                
                with open(os.path.join(meta_dir, "token_text_language_map.json"), "w") as f:
                    json.dump(self.token_classifier.classified_tokens_with_ranges, f, indent=2)
                
                with open(os.path.join(meta_dir, "vocabulary_languages.txt"), "w") as f:
                    sorted_langs = sorted(self.token_classifier.all_languages_data, key=lambda x: x[1], reverse=False)
                    for language, id_count in sorted_langs:
                        f.write(f"{language}\t{id_count}\n")
                
                # Copy tokenization.py to enable AutoTokenizer.from_pretrained()
                tokenization_file = Path(__file__).parent / "tokenization.py"
                if tokenization_file.exists():
                    import shutil
                    shutil.copy2(tokenization_file, os.path.join(save_directory, "tokenization.py"))
                
                # Save config compatible with AutoTokenizer
                config = {
                    "auto_map": {
                        "AutoTokenizer": ["tokenization.RBPETokenizer", None]
                    },
                    "model_type": "rbpe",
                    "tokenizer_class": "RBPETokenizer",
                    "target_language": "arabic",
                }
                
                # Add special tokens to config
                if self.old_tokenizer.pad_token:
                    config["pad_token"] = self.old_tokenizer.pad_token
                if self.old_tokenizer.eos_token:
                    config["eos_token"] = self.old_tokenizer.eos_token
                if self.old_tokenizer.bos_token:
                    config["bos_token"] = self.old_tokenizer.bos_token
                if self.old_tokenizer.unk_token:
                    config["unk_token"] = self.old_tokenizer.unk_token
                
                with open(os.path.join(save_directory, "tokenizer_config.json"), "w") as f:
                    json.dump(config, f, indent=2)
                
                # Save special tokens map
                special_tokens_map = {}
                if self.old_tokenizer.pad_token:
                    special_tokens_map["pad_token"] = self.old_tokenizer.pad_token
                if self.old_tokenizer.eos_token:
                    special_tokens_map["eos_token"] = self.old_tokenizer.eos_token
                if self.old_tokenizer.bos_token:
                    special_tokens_map["bos_token"] = self.old_tokenizer.bos_token
                if self.old_tokenizer.unk_token:
                    special_tokens_map["unk_token"] = self.old_tokenizer.unk_token
                
                with open(os.path.join(save_directory, "special_tokens_map.json"), "w") as f:
                    json.dump(special_tokens_map, f, indent=2)
                
                logger.info(f"R-BPE tokenizer saved to {save_directory}")
                logger.info("Tokenizer can be loaded with: AutoTokenizer.from_pretrained(path, trust_remote_code=True)")
        
        self.tokenizer = RBPETokenizerSaver(self)
        
        logger.info("Tokenizer preparation completed successfully!")
        
        return self.tokenizer
    
    @classmethod
    def from_pretrained(cls, pretrained_path: str, **kwargs):
        """
        Load a pretrained R-BPE tokenizer.
        
        Note: This method is deprecated. Use AutoTokenizer.from_pretrained() instead:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(pretrained_path, trust_remote_code=True)
        
        Args:
            pretrained_path: Path to the saved tokenizer directory
            **kwargs: Additional arguments (unused)
        
        Returns:
            Reference to use AutoTokenizer instead
        """
        logger.warning(
            "RBPETokenizer.from_pretrained() is deprecated. "
            "Please use AutoTokenizer.from_pretrained() instead:\n"
            "  from transformers import AutoTokenizer\n"
            f"  tokenizer = AutoTokenizer.from_pretrained('{pretrained_path}', trust_remote_code=True)"
        )
        
        # Try to import and return the Rust-based tokenizer
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(pretrained_path, trust_remote_code=True)
        except Exception as e:
            raise ImportError(
                f"Failed to load tokenizer. Make sure:\n"
                f"1. Rust tokenizer is built: cd rbpe-tokenizers && maturin develop --release\n"
                f"2. Tokenizer directory exists: {pretrained_path}\n"
                f"Error: {e}"
            )
