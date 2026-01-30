"""
Evaluation metrics for summarization models
"""

import torch
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from typing import List, Dict, Tuple
from tqdm import tqdm


def calculate_rouge(
    predictions: List[str],
    references: List[str],
    rouge_types: List[str] = ['rouge1', 'rouge2', 'rougeL']
) -> Dict[str, float]:
    """
    Calculate ROUGE scores
    
    Args:
        predictions: List of predicted summaries
        references: List of reference summaries
        rouge_types: Types of ROUGE to calculate
        
    Returns:
        Dictionary of ROUGE scores
    """
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    
    scores = {rouge_type: {'precision': [], 'recall': [], 'fmeasure': []} 
              for rouge_type in rouge_types}
    
    for pred, ref in zip(predictions, references):
        rouge_scores = scorer.score(ref, pred)
        
        for rouge_type in rouge_types:
            scores[rouge_type]['precision'].append(rouge_scores[rouge_type].precision)
            scores[rouge_type]['recall'].append(rouge_scores[rouge_type].recall)
            scores[rouge_type]['fmeasure'].append(rouge_scores[rouge_type].fmeasure)
    
    # Calculate averages
    avg_scores = {}
    for rouge_type in rouge_types:
        avg_scores[f'{rouge_type}_precision'] = np.mean(scores[rouge_type]['precision'])
        avg_scores[f'{rouge_type}_recall'] = np.mean(scores[rouge_type]['recall'])
        avg_scores[f'{rouge_type}_fmeasure'] = np.mean(scores[rouge_type]['fmeasure'])
    
    return avg_scores


def calculate_bertscore(
    predictions: List[str],
    references: List[str],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_size: int = 32
) -> Dict[str, float]:
    """
    Calculate BERTScore
    
    Args:
        predictions: List of predicted summaries
        references: List of reference summaries
        device: Device to use
        batch_size: Batch size for processing
        
    Returns:
        Dictionary of BERTScore metrics
    """
    P, R, F1 = bert_score(
        predictions,
        references,
        lang='en',
        device=device,
        batch_size=batch_size,
        verbose=False
    )
    
    return {
        'bertscore_precision': P.mean().item(),
        'bertscore_recall': R.mean().item(),
        'bertscore_f1': F1.mean().item()
    }


def evaluate_model(
    model,
    tokenizer,
    dataloader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    max_length: int = 128,
    num_beams: int = 4,
    calculate_bert: bool = True
) -> Tuple[Dict[str, float], List[str], List[str]]:
    """
    Evaluate model on a dataset
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer
        dataloader: DataLoader for evaluation data
        device: Device to use
        max_length: Maximum generation length
        num_beams: Number of beams for beam search
        calculate_bert: Whether to calculate BERTScore (can be slow)
        
    Returns:
        Tuple of (metrics_dict, predictions, references)
    """
    model.eval()
    model.to(device)
    
    predictions = []
    references = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Generate predictions
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            
            # Decode predictions
            pred_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(pred_texts)
            
            # Decode references
            # Replace -100 with pad_token_id for decoding
            labels[labels == -100] = tokenizer.pad_token_id
            ref_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
            references.extend(ref_texts)
    
    # Calculate metrics
    print("\nCalculating ROUGE scores...")
    rouge_scores = calculate_rouge(predictions, references)
    
    metrics = rouge_scores.copy()
    
    if calculate_bert:
        print("Calculating BERTScore...")
        bert_scores = calculate_bertscore(predictions, references, device=device)
        metrics.update(bert_scores)
    
    return metrics, predictions, references


def print_evaluation_results(metrics: Dict[str, float]):
    """
    Pretty print evaluation results
    
    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # ROUGE scores
    print("\nROUGE Scores:")
    print("-" * 60)
    for key, value in sorted(metrics.items()):
        if 'rouge' in key:
            print(f"{key:30s}: {value:.4f}")
    
    # BERTScore
    if any('bertscore' in key for key in metrics.keys()):
        print("\nBERTScore:")
        print("-" * 60)
        for key, value in sorted(metrics.items()):
            if 'bertscore' in key:
                print(f"{key:30s}: {value:.4f}")
    
    print("="*60)


if __name__ == "__main__":
    # Test ROUGE calculation
    predictions = [
        "The patient has a fever and cough.",
        "Surgery was successful without complications."
    ]
    references = [
        "Patient presents with fever, cough, and fatigue.",
        "The surgical procedure was completed successfully."
    ]
    
    rouge_scores = calculate_rouge(predictions, references)
    print("ROUGE Scores:")
    for key, value in rouge_scores.items():
        print(f"{key}: {value:.4f}")