"""
Data preprocessing for medical report summarization
"""

import re
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """
    Clean medical text
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text
    """
    if pd.isna(text):
        return ""
    
    # Convert to string
    text = str(text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def validate_pair(report: str, summary: str, 
                  min_report_length: int = 50,
                  max_report_length: int = 5000,
                  min_summary_length: int = 10,
                  max_summary_length: int = 500) -> bool:
    """
    Validate report-summary pair
    
    Args:
        report: Medical report text
        summary: Summary text
        min_report_length: Minimum report length in characters
        max_report_length: Maximum report length in characters
        min_summary_length: Minimum summary length in characters
        max_summary_length: Maximum summary length in characters
        
    Returns:
        True if valid, False otherwise
    """
    # Check if both exist
    if not report or not summary:
        return False
    
    # Check lengths
    report_len = len(report)
    summary_len = len(summary)
    
    if report_len < min_report_length or report_len > max_report_length:
        return False
    
    if summary_len < min_summary_length or summary_len > max_summary_length:
        return False
    
    # Summary should be shorter than report
    if summary_len >= report_len:
        return False
    
    return True


def preprocess_data(
    input_path: str = "./data/raw/mtsamples.csv",
    output_dir: str = "./data/processed",
    text_col: str = "transcription",
    summary_col: str = "description",
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess and split medical text data
    
    Args:
        input_path: Path to raw data file
        output_dir: Directory to save processed data
        text_col: Column name containing medical reports
        summary_col: Column name containing summaries
        train_split: Training set proportion
        val_split: Validation set proportion
        test_split: Test set proportion
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"Initial dataset size: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check if columns exist
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Available: {df.columns.tolist()}")
    if summary_col not in df.columns:
        raise ValueError(f"Column '{summary_col}' not found. Available: {df.columns.tolist()}")
    
    # Select and rename columns
    df = df[[text_col, summary_col]].copy()
    df = df.rename(columns={text_col: 'report', summary_col: 'summary'})
    
    # Remove missing values
    df = df.dropna()
    print(f"After removing missing values: {len(df)}")
    
    # Clean text
    print("Cleaning text...")
    df['report'] = df['report'].apply(clean_text)
    df['summary'] = df['summary'].apply(clean_text)
    
    # Validate pairs
    print("Validating report-summary pairs...")
    df['valid'] = df.apply(
        lambda row: validate_pair(row['report'], row['summary']),
        axis=1
    )
    
    df = df[df['valid']].drop(columns=['valid'])
    print(f"Valid pairs: {len(df)}")
    
    # Calculate statistics
    df['report_length'] = df['report'].str.len()
    df['summary_length'] = df['summary'].str.len()
    df['compression_ratio'] = df['summary_length'] / df['report_length']
    
    print("\nDataset Statistics:")
    print(f"  Report length - Mean: {df['report_length'].mean():.0f}, "
          f"Median: {df['report_length'].median():.0f}")
    print(f"  Summary length - Mean: {df['summary_length'].mean():.0f}, "
          f"Median: {df['summary_length'].median():.0f}")
    print(f"  Compression ratio - Mean: {df['compression_ratio'].mean():.2f}")
    
    # Drop statistics columns
    df = df[['report', 'summary']]
    
    # Split data
    print("\nSplitting data...")
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_split,
        random_state=random_state
    )
    
    # Second split: train vs val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_split / (train_split + val_split),
        random_state=random_state
    )
    
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
    # Save processed data
    train_path = Path(output_dir) / "train.csv"
    val_path = Path(output_dir) / "val.csv"
    test_path = Path(output_dir) / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n✓ Processed data saved to {output_dir}")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess medical text data")
    parser.add_argument(
        "--input",
        type=str,
        default="./data/raw/mtsamples.csv",
        help="Path to raw data file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/processed",
        help="Output directory"
    )
    parser.add_argument(
        "--text_col",
        type=str,
        default="transcription",
        help="Column name for medical reports"
    )
    parser.add_argument(
        "--summary_col",
        type=str,
        default="description",
        help="Column name for summaries"
    )
    
    args = parser.parse_args()
    
    preprocess_data(
        input_path=args.input,
        output_dir=args.output,
        text_col=args.text_col,
        summary_col=args.summary_col
    )