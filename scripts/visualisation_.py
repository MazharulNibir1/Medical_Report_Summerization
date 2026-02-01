"""
Generate visualizations for model results
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_training_history(history_path: str, output_dir: str = "./results/visualizations"):
    """Plot training and validation loss"""
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load training history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Plot training loss
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Training Loss
    if 'train_loss' in history and len(history['train_loss']) > 0:
        axes[0].plot(history['train_loss'], label='Training Loss', linewidth=2)
        axes[0].set_xlabel('Steps')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Validation Loss
    if 'val_loss' in history and len(history['val_loss']) > 0:
        axes[1].plot(history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
        axes[1].set_xlabel('Evaluation Steps')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Validation Loss Over Time', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_history.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/training_history.png")
    plt.close()


def plot_rouge_scores(metrics_path: str, output_dir: str = "./results/visualizations"):
    """Plot ROUGE scores as bar chart"""
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Extract ROUGE F-measure scores
    rouge_scores = {
        'ROUGE-1': metrics.get('rouge1_fmeasure', 0),
        'ROUGE-2': metrics.get('rouge2_fmeasure', 0),
        'ROUGE-L': metrics.get('rougeL_fmeasure', 0)
    }
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(rouge_scores.keys(), rouge_scores.values(), 
                   color=['#2ecc71', '#3498db', '#9b59b6'], 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('ROUGE Scores on Test Set', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rouge_scores.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/rouge_scores.png")
    plt.close()


def plot_all_metrics(metrics_path: str, output_dir: str = "./results/visualizations"):
    """Plot all metrics including precision, recall, F1"""
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Prepare data
    rouge_types = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    precision = [
        metrics.get('rouge1_precision', 0),
        metrics.get('rouge2_precision', 0),
        metrics.get('rougeL_precision', 0)
    ]
    recall = [
        metrics.get('rouge1_recall', 0),
        metrics.get('rouge2_recall', 0),
        metrics.get('rougeL_recall', 0)
    ]
    f1 = [
        metrics.get('rouge1_fmeasure', 0),
        metrics.get('rouge2_fmeasure', 0),
        metrics.get('rougeL_fmeasure', 0)
    ]
    
    # Create grouped bar plot
    x = np.arange(len(rouge_types))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', 
                   color='#3498db', edgecolor='black', linewidth=1, alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', 
                   color='#2ecc71', edgecolor='black', linewidth=1, alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', 
                   color='#e74c3c', edgecolor='black', linewidth=1, alpha=0.8)
    
    ax.set_xlabel('ROUGE Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Detailed ROUGE Metrics (Precision, Recall, F1)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(rouge_types)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/detailed_metrics.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/detailed_metrics.png")
    plt.close()


def plot_bertscore(metrics_path: str, output_dir: str = "./results/visualizations"):
    """Plot BERTScore metrics"""
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Extract BERTScore
    bertscore_metrics = {
        'Precision': metrics.get('bertscore_precision', 0),
        'Recall': metrics.get('bertscore_recall', 0),
        'F1': metrics.get('bertscore_f1', 0)
    }
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(bertscore_metrics.keys(), bertscore_metrics.values(),
                   color=['#e67e22', '#16a085', '#c0392b'],
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('BERTScore Metrics on Test Set', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/bertscore.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/bertscore.png")
    plt.close()


def plot_sample_comparisons(predictions_path: str, num_samples: int = 3, 
                            output_dir: str = "./results/visualizations"):
    """Create a visualization of sample predictions vs references"""
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load predictions
    df = pd.read_csv(predictions_path)
    
    # Sample random examples
    samples = df.sample(n=min(num_samples, len(df)))
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 1, figsize=(14, 4*num_samples))
    
    if num_samples == 1:
        axes = [axes]
    
    for idx, (i, row) in enumerate(samples.iterrows()):
        ax = axes[idx]
        
        reference = row['reference'][:200] + "..." if len(row['reference']) > 200 else row['reference']
        prediction = row['prediction'][:200] + "..." if len(row['prediction']) > 200 else row['prediction']
        
        # Create text visualization
        ax.text(0.05, 0.7, f"Reference:\n{reference}", 
                fontsize=10, verticalalignment='top', wrap=True,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        ax.text(0.05, 0.3, f"Prediction:\n{prediction}", 
                fontsize=10, verticalalignment='top', wrap=True,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f'Sample {idx+1}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sample_comparisons.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/sample_comparisons.png")
    plt.close()


def generate_summary_table(metrics_path: str, output_dir: str = "./results/visualizations"):
    """Generate a summary table image"""
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Prepare data for table
    data = {
        'Metric': [
            'ROUGE-1 (F1)',
            'ROUGE-2 (F1)',
            'ROUGE-L (F1)',
            'BERTScore (F1)',
            'BERTScore (Precision)',
            'BERTScore (Recall)'
        ],
        'Score': [
            f"{metrics.get('rouge1_fmeasure', 0):.4f}",
            f"{metrics.get('rouge2_fmeasure', 0):.4f}",
            f"{metrics.get('rougeL_fmeasure', 0):.4f}",
            f"{metrics.get('bertscore_f1', 0):.4f}",
            f"{metrics.get('bertscore_precision', 0):.4f}",
            f"{metrics.get('bertscore_recall', 0):.4f}"
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center',
                     colWidths=[0.6, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style cells
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    plt.title('Model Performance Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(f"{output_dir}/metrics_table.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/metrics_table.png")
    plt.close()


def main():
    print("Generating visualizations...\n")
    
    # Paths
    history_path = "./models/fine_tuned/t5/training_history.json"
    metrics_path = "./results/metrics/test_metrics.json"
    predictions_path = "./results/predictions/test_predictions.csv"
    output_dir = "./results/visualizations"
    
    # Generate all visualizations
    try:
        if Path(history_path).exists():
            plot_training_history(history_path, output_dir)
        else:
            print(f"⚠ Warning: {history_path} not found, skipping training history plot")
    except Exception as e:
        print(f"⚠ Error plotting training history: {e}")
    
    try:
        if Path(metrics_path).exists():
            plot_rouge_scores(metrics_path, output_dir)
            plot_all_metrics(metrics_path, output_dir)
            plot_bertscore(metrics_path, output_dir)
            generate_summary_table(metrics_path, output_dir)
        else:
            print(f"⚠ Warning: {metrics_path} not found, skipping metrics plots")
    except Exception as e:
        print(f"⚠ Error plotting metrics: {e}")
    
    try:
        if Path(predictions_path).exists():
            plot_sample_comparisons(predictions_path, num_samples=3, output_dir=output_dir)
        else:
            print(f"⚠ Warning: {predictions_path} not found, skipping sample comparisons")
    except Exception as e:
        print(f"⚠ Error plotting sample comparisons: {e}")
    
    print("\n" + "="*60)
    print("✓ All visualizations generated successfully!")
    print(f"✓ Saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()