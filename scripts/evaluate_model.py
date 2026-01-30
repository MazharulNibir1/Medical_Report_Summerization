"""
Script to evaluate trained model
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import argparse
import torch
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.data.dataset import create_dataloaders
from src.training.evaluate import evaluate_model, print_evaluation_results


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Evaluate summarization model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/t5_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/predictions/test_predictions.csv",
        help="Path to save predictions"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model from: {args.model_path}")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Create test dataloader
    print("Creating test dataloader...")
    _, _, test_loader = create_dataloaders(
        train_path=config['data']['processed_dir'] + "/train.csv",
        val_path=config['data']['processed_dir'] + "/val.csv",
        test_path=config['data']['processed_dir'] + "/test.csv",
        tokenizer=tokenizer,
        batch_size=config['evaluation']['batch_size'],
        max_input_length=config['data']['max_input_length'],
        max_target_length=config['data']['max_target_length']
    )
    
    # Evaluate
    print("\nEvaluating model...")
    metrics, predictions, references = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        dataloader=test_loader,
        device=device,
        max_length=config['generation']['max_length'],
        num_beams=config['generation']['num_beams']
    )
    
    # Print results
    print_evaluation_results(metrics)
    
    # Save predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame({
        'reference': references,
        'prediction': predictions
    })
    
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to: {output_path}")
    
    # Save metrics
    metrics_path = output_path.parent.parent / "metrics" / "test_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()