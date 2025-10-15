"""
Data Utilities

This module provides utilities for data processing, tokenization, and dataset management
for the hallucination mitigation project.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import logging
import pandas as pd
from pathlib import Path


class Tokenizer:
    """
    Wrapper for tokenizer with additional utilities for hallucination detection.
    """
    
    def __init__(self, model_name: str = "gpt2", max_length: int = 512):
        """
        Initialize tokenizer.
        
        Args:
            model_name: Name of the pretrained model
            max_length: Maximum sequence length
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.logger = logging.getLogger(__name__)
    
    def tokenize(self, 
                texts: Union[str, List[str]], 
                return_tensors: str = "pt",
                padding: bool = True,
                truncation: bool = True) -> Dict[str, torch.Tensor]:
        """
        Tokenize input texts.
        
        Args:
            texts: Input text or list of texts
            return_tensors: Format of returned tensors
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            
        Returns:
            Tokenized inputs
        """
        return self.tokenizer(
            texts,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length
        )
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded texts
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


class HallucinationDataset(Dataset):
    """
    Dataset class for hallucination detection and mitigation.
    """
    
    def __init__(self, 
                 data: List[Dict[str, Any]],
                 tokenizer: Tokenizer,
                 label_key: str = "hallucination_label",
                 text_key: str = "text"):
        """
        Initialize dataset.
        
        Args:
            data: List of data samples
            tokenizer: Tokenizer instance
            label_key: Key for hallucination labels in data
            text_key: Key for text in data
        """
        self.data = data
        self.tokenizer = tokenizer
        self.label_key = label_key
        self.text_key = text_key
        
        self.logger = logging.getLogger(__name__)
    
    def __len__(self):
        """Return dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dataset item
        """
        item = self.data[idx]
        
        # Tokenize text
        text = item[self.text_key]
        tokenized = self.tokenizer.tokenize(text)
        
        # Prepare item
        sample = {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'text': text
        }
        
        # Add labels if available
        if self.label_key in item:
            sample['labels'] = torch.tensor(item[self.label_key], dtype=torch.long)
        
        # Add any additional fields
        for key, value in item.items():
            if key not in [self.text_key, self.label_key]:
                sample[key] = value
        
        return sample


class DataProcessor:
    """
    Data processor for hallucination detection and mitigation datasets.
    """
    
    def __init__(self, tokenizer: Tokenizer):
        """
        Initialize data processor.
        
        Args:
            tokenizer: Tokenizer instance
        """
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
    
    def load_json_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load data from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of data samples
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.logger.info(f"Loaded {len(data)} samples from {file_path}")
        return data
    
    def load_csv_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of data samples
        """
        df = pd.read_csv(file_path)
        data = df.to_dict('records')
        
        self.logger.info(f"Loaded {len(data)} samples from {file_path}")
        return data
    
    def create_data_loader(self, 
                          data: List[Dict[str, Any]],
                          batch_size: int = 8,
                          shuffle: bool = True,
                          num_workers: int = 0) -> DataLoader:
        """
        Create data loader from data.
        
        Args:
            data: List of data samples
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            
        Returns:
            DataLoader instance
        """
        dataset = HallucinationDataset(data, self.tokenizer)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn
        )
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for batching samples.
        
        Args:
            batch: List of samples
            
        Returns:
            Batched data
        """
        # Get all keys from the first sample
        keys = batch[0].keys()
        
        # Create batched dictionary
        batched = {}
        for key in keys:
            if key == 'text':
                # Keep texts as list
                batched[key] = [sample[key] for sample in batch]
            elif isinstance(batch[0][key], torch.Tensor):
                # Stack tensors
                batched[key] = torch.stack([sample[key] for sample in batch])
            else:
                # Keep as list
                batched[key] = [sample[key] for sample in batch]
        
        return batched
    
    def filter_by_length(self, 
                        data: List[Dict[str, Any]], 
                        text_key: str = "text",
                        min_length: int = 10,
                        max_length: int = 1000) -> List[Dict[str, Any]]:
        """
        Filter data by text length.
        
        Args:
            data: List of data samples
            text_key: Key for text in data
            min_length: Minimum text length
            max_length: Maximum text length
            
        Returns:
            Filtered data
        """
        filtered_data = []
        
        for sample in data:
            text = sample[text_key]
            text_length = len(text.split())
            
            if min_length <= text_length <= max_length:
                filtered_data.append(sample)
        
        self.logger.info(f"Filtered {len(data)} samples to {len(filtered_data)} by length")
        return filtered_data
    
    def balance_dataset(self, 
                       data: List[Dict[str, Any]], 
                       label_key: str = "hallucination_label") -> List[Dict[str, Any]]:
        """
        Balance dataset by label distribution.
        
        Args:
            data: List of data samples
            label_key: Key for labels in data
            
        Returns:
            Balanced dataset
        """
        # Count labels
        label_counts = {}
        for sample in data:
            label = sample[label_key]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Find minimum count
        min_count = min(label_counts.values())
        
        # Sample data to balance
        balanced_data = []
        label_indices = {label: [] for label in label_counts.keys()}
        
        # Group samples by label
        for idx, sample in enumerate(data):
            label = sample[label_key]
            label_indices[label].append(idx)
        
        # Sample equal number from each label
        for label, indices in label_indices.items():
            sampled_indices = indices[:min_count]
            for idx in sampled_indices:
                balanced_data.append(data[idx])
        
        self.logger.info(f"Balanced dataset from {len(data)} to {len(balanced_data)} samples")
        return balanced_data
    
    def split_data(self, 
                  data: List[Dict[str, Any]], 
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1,
                  test_ratio: float = 0.1) -> Tuple[List[Dict[str, Any]], ...]:
        """
        Split data into train/validation/test sets.
        
        Args:
            data: List of data samples
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        total_size = len(data)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        # Shuffle data
        import random
        random.shuffle(data)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        self.logger.info(f"Split data: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        return train_data, val_data, test_data
