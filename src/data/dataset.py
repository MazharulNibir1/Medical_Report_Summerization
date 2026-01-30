"""
PyTorch Dataset class for medical report summarization
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Optional
from transformers import PreTrainedTokenizer


class MedicalSummarizationDataset(Dataset):
    """
    Dataset class for medical report summarization
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_input_length: int = 512,
        max_target_length: int = 128,
        prefix: str = "summarize: "
    ):
        """
        Initialize dataset
        
        Args:
            data_path: Path to CSV file with 'report' and 'summary' columns
            tokenizer: Hugging Face tokenizer
            max_input_length: Maximum length for input sequences
            max_target_length: Maximum length for target sequences
            prefix: Prefix to add to input text (for T5)
        """
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.prefix = prefix
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single item from dataset
        
        Args:
            idx: Index of item
            
        Returns:
            Dictionary with tokenized inputs and targets
        """
        row = self.data.iloc[idx]
        
        # Add prefix to input (important for T5)
        source_text = self.prefix + row['report']
        target_text = row['summary']
        
        # Tokenize input
        source = self.tokenizer(
            source_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare labels (replace padding token id with -100 so it's ignored in loss)
        labels = target['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': source['input_ids'].squeeze(),
            'attention_mask': source['attention_mask'].squeeze(),
            'labels': labels
        }


def create_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_input_length: int = 512,
    max_target_length: int = 128,
    num_workers: int = 0
):
    """
    Create train, validation, and test dataloaders
    
    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        test_path: Path to test CSV
        tokenizer: Hugging Face tokenizer
        batch_size: Batch size
        max_input_length: Maximum input sequence length
        max_target_length: Maximum target sequence length
        num_workers: Number of workers for DataLoader
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = MedicalSummarizationDataset(
        train_path, tokenizer, max_input_length, max_target_length
    )
    
    val_dataset = MedicalSummarizationDataset(
        val_path, tokenizer, max_input_length, max_target_length
    )
    
    test_dataset = MedicalSummarizationDataset(
        test_path, tokenizer, max_input_length, max_target_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    
    dataset = MedicalSummarizationDataset(
        data_path="./data/processed/train.csv",
        tokenizer=tokenizer
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"\nSample item:")
    sample = dataset[0]
    for key, value in sample.items():
        print(f"{key}: {value.shape}")