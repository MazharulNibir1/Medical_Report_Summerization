"""
Script to train summarization model
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import argparse
import torch
from transformers import AutoTokenizer

from src.data.dataset import create_dataloaders
from src.training.train import train_model


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train summarization model")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/t5_config.yaml",
        help="Path to config file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer
    model_name = config['model']['name']
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_path=config['data']['processed_dir'] + "/train.csv",
        val_path=config['data']['processed_dir'] + "/val.csv",
        test_path=config['data']['processed_dir'] + "/test.csv",
        tokenizer=tokenizer,
        batch_size=int(config['training']['batch_size']),
        max_input_length=int(config['data']['max_input_length']),
        max_target_length=int(config['data']['max_target_length'])
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Train model
    history = train_model(
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=config['training']['output_dir'],
        num_epochs=int(config['training']['num_epochs']),
        learning_rate=float(config['training']['learning_rate']),
        warmup_steps=int(config['training']['warmup_steps']),
        gradient_accumulation_steps=int(config['training']['gradient_accumulation_steps']),
        device=device,
        save_steps=int(config['training']['save_steps']),
        eval_steps=int(config['training']['eval_steps']),
        logging_steps=int(config['training']['logging_steps']),
        save_total_limit=int(config['training']['save_total_limit'])
    )
    
    print("\n✓ Training completed successfully!")


if __name__ == "__main__":
    main()