"""
Transformer-based summarization models (T5, BART)
"""

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig
)
from typing import List, Optional


class TransformerSummarizer:
    """
    Wrapper for transformer-based summarization models
    """
    
    def __init__(
        self,
        model_name: str = "t5-small",
        device: Optional[str] = None
    ):
        """
        Initialize transformer model
        
        Args:
            model_name: Hugging Face model name
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_name = model_name
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def summarize(
        self,
        text: str,
        max_length: int = 128,
        min_length: int = 30,
        num_beams: int = 4,
        length_penalty: float = 2.0,
        no_repeat_ngram_size: int = 3,
        prefix: str = "summarize: "
    ) -> str:
        """
        Generate summary using transformer model
        
        Args:
            text: Input text
            max_length: Maximum summary length
            min_length: Minimum summary length
            num_beams: Number of beams for beam search
            length_penalty: Length penalty
            no_repeat_ngram_size: No repeat n-gram size
            prefix: Prefix for input (for T5 models)
            
        Returns:
            Generated summary
        """
        # Add prefix for T5 models
        if "t5" in self.model_name.lower():
            input_text = prefix + text
        else:
            input_text = text
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True
            )
        
        # Decode
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return summary
    
    def summarize_batch(
        self,
        texts: List[str],
        batch_size: int = 8,
        **kwargs
    ) -> List[str]:
        """
        Generate summaries for multiple texts
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            **kwargs: Additional arguments for summarize()
            
        Returns:
            List of summaries
        """
        summaries = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_summaries = [self.summarize(text, **kwargs) for text in batch]
            summaries.extend(batch_summaries)
        
        return summaries


if __name__ == "__main__":
    # Test the transformer model
    summarizer = TransformerSummarizer("t5-small")
    
    test_text = """
    The patient is a 23-year-old white female presenting with abdominal pain.
    The pain started 2 days ago and is located in the lower right quadrant.
    She denies fever, nausea, or vomiting.
    Physical examination reveals tenderness in the right lower quadrant.
    Laboratory tests show elevated white blood cell count.
    CT scan suggests possible appendicitis.
    Patient is scheduled for appendectomy.
    """
    
    summary = summarizer.summarize(test_text)
    print("Original text:")
    print(test_text)
    print("\nGenerated summary:")
    print(summary)