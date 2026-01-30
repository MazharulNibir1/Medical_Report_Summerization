"""Training module for medical report summarization"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.train import train_model
from src.training.evaluate import evaluate_model, calculate_rouge, calculate_bertscore

__all__ = [
    "train_model",
    "evaluate_model",
    "calculate_rouge",
    "calculate_bertscore",
]