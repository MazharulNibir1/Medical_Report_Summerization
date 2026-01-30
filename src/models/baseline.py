"""
Extractive summarization baseline using TextRank
"""

import nltk
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class ExtractiveSummarizer:
    """
    Extractive summarization using TextRank algorithm
    """
    
    def __init__(self, num_sentences: int = 3):
        """
        Initialize extractive summarizer
        
        Args:
            num_sentences: Number of sentences to extract
        """
        self.num_sentences = num_sentences
        self.vectorizer = TfidfVectorizer()
        
    def _sentence_tokenize(self, text: str) -> List[str]:
        """Split text into sentences"""
        return nltk.sent_tokenize(text)
    
    def _build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """
        Build sentence similarity matrix using TF-IDF and cosine similarity
        
        Args:
            sentences: List of sentences
            
        Returns:
            Similarity matrix
        """
        # Create TF-IDF vectors
        try:
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            return similarity_matrix
        except:
            # If only one sentence or error, return zero matrix
            return np.zeros((len(sentences), len(sentences)))
    
    def _textrank(self, similarity_matrix: np.ndarray, 
                  damping: float = 0.85, 
                  max_iter: int = 100) -> np.ndarray:
        """
        Apply TextRank algorithm to rank sentences
        
        Args:
            similarity_matrix: Sentence similarity matrix
            damping: Damping factor
            max_iter: Maximum iterations
            
        Returns:
            Sentence scores
        """
        n = len(similarity_matrix)
        
        # Initialize scores
        scores = np.ones(n) / n
        
        # Normalize similarity matrix
        norm_matrix = similarity_matrix / (similarity_matrix.sum(axis=1, keepdims=True) + 1e-10)
        
        # Iterate
        for _ in range(max_iter):
            prev_scores = scores.copy()
            scores = (1 - damping) / n + damping * norm_matrix.T.dot(scores)
            
            # Check convergence
            if np.allclose(scores, prev_scores, atol=1e-6):
                break
        
        return scores
    
    def summarize(self, text: str, num_sentences: int = None) -> str:
        """
        Generate extractive summary
        
        Args:
            text: Input text to summarize
            num_sentences: Number of sentences in summary (default: self.num_sentences)
            
        Returns:
            Extractive summary
        """
        if num_sentences is None:
            num_sentences = self.num_sentences
        
        # Tokenize into sentences
        sentences = self._sentence_tokenize(text)
        
        # Handle edge cases
        if len(sentences) <= num_sentences:
            return text
        
        # Build similarity matrix
        similarity_matrix = self._build_similarity_matrix(sentences)
        
        # Rank sentences
        scores = self._textrank(similarity_matrix)
        
        # Select top sentences
        ranked_indices = np.argsort(scores)[::-1][:num_sentences]
        
        # Sort by original order to maintain coherence
        ranked_indices = sorted(ranked_indices)
        
        # Create summary
        summary = ' '.join([sentences[i] for i in ranked_indices])
        
        return summary


if __name__ == "__main__":
    # Test the baseline
    summarizer = ExtractiveSummarizer(num_sentences=2)
    
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
    print("\nExtracted summary:")
    print(summary)