"""
Training loop for summarization models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import json


def train_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: str = "./models/fine_tuned",
    num_epochs: int = 5,
    learning_rate: float = 5e-5,
    warmup_steps: int = 500,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_steps: int = 500,
    eval_steps: int = 500,
    logging_steps: int = 100,
    save_total_limit: int = 3
) -> Dict:
    """
    Train a seq2seq model for summarization
    
    Args:
        model_name: Pretrained model name
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        output_dir: Directory to save model checkpoints
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        gradient_accumulation_steps: Gradient accumulation steps
        max_grad_norm: Maximum gradient norm for clipping
        device: Device to use
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        logging_steps: Log every N steps
        save_total_limit: Maximum number of checkpoints to keep
        
    Returns:
        Dictionary with training history
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model.to(device)
    
    # Calculate total steps
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    # Training loop
    print(f"\nStarting training...")
    print(f"  Total epochs: {num_epochs}")
    print(f"  Train batches per epoch: {len(train_loader)}")
    print(f"  Total optimization steps: {total_steps}")
    print(f"  Device: {device}")
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            train_loss += loss.item() * gradient_accumulation_steps
            train_steps += 1
            
            # Update weights
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': train_loss / train_steps,
                    'lr': scheduler.get_last_lr()[0]
                })
                
                # Logging
                if global_step % logging_steps == 0:
                    avg_train_loss = train_loss / train_steps
                    current_lr = scheduler.get_last_lr()[0]
                    history['train_loss'].append(avg_train_loss)
                    history['learning_rates'].append(current_lr)
                
                # Evaluation
                if global_step % eval_steps == 0:
                    val_loss = evaluate(model, val_loader, device)
                    history['val_loss'].append(val_loss)
                    
                    print(f"\nStep {global_step}: Val Loss = {val_loss:.4f}")
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_dir = output_dir / "best_model"
                        best_model_dir.mkdir(exist_ok=True)
                        
                        model.save_pretrained(best_model_dir)
                        tokenizer.save_pretrained(best_model_dir)
                        
                        print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
                    
                    model.train()
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    checkpoint_dir.mkdir(exist_ok=True)
                    
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    
                    # Remove old checkpoints
                    cleanup_checkpoints(output_dir, save_total_limit)
        
        # End of epoch validation
        avg_train_loss = train_loss / train_steps
        val_loss = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir / 'best_model'}")
    print(f"{'='*60}")
    
    return history


def evaluate(model, dataloader, device):
    """
    Evaluate model on validation set
    
    Args:
        model: Model to evaluate
        dataloader: Validation DataLoader
        device: Device to use
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0
    total_steps = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            total_steps += 1
    
    return total_loss / total_steps


def cleanup_checkpoints(output_dir: Path, save_total_limit: int):
    """
    Remove old checkpoints to keep only the most recent ones
    
    Args:
        output_dir: Directory containing checkpoints
        save_total_limit: Maximum number of checkpoints to keep
    """
    checkpoints = sorted(
        [d for d in output_dir.glob("checkpoint-*")],
        key=lambda x: int(x.name.split("-")[1])
    )
    
    if len(checkpoints) > save_total_limit:
        for checkpoint in checkpoints[:-save_total_limit]:
            print(f"Removing old checkpoint: {checkpoint.name}")
            import shutil
            shutil.rmtree(checkpoint)


if __name__ == "__main__":
    print("This module should be imported, not run directly.")
    print("Use scripts/train_model.py to train a model.")