"""
OpenAI embedding model implementation.
Simplified OpenAI embeddings with retries and batch processing.
"""
import time
import random
import logging
import numpy as np
import os
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

from .embedding_config import EmbeddingConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# Suppress HTTP request logs from OpenAI client libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Load .env file from project root
env_path = Path(__file__).parent.parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)


class OpenAIEmbeddingModel:
    """
    OpenAI embedding model wrapper with retry logic and batch processing.
    
    Features:
    - OpenAI text-embedding-3-small (or configurable model)
    - Exponential backoff retry logic
    - Batch and parallel batch processing
    """
    
    def __init__(self, config: EmbeddingConfig = None):
        """Initialize the OpenAI embedding model."""
        self.config = config or DEFAULT_CONFIG
        self.provider = self.config.provider
        self.model_name = self.config.get_model_name()
        
        if self.provider == "openai":
            self.embeddings = OpenAIEmbeddings(
                model=self.model_name,
                max_retries=0,  # We handle retries ourselves
                request_timeout=self.config.request_timeout,
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")
    
    def _make_single_api_call(self, text: str, is_batch: bool = False, texts: List[str] = None):
        """Make a single API call without retries."""
        if is_batch and texts:
            return self.embeddings.embed_documents(texts)
        else:
            return self.embeddings.embed_query(text.strip())
    
    def _call_embedding_api(self, text: str, is_batch: bool = False, texts: List[str] = None):
        """Call the embedding API with retry logic."""
        max_attempts = self.config.max_retries
        base_delay = self.config.retry_min_seconds
        max_delay = self.config.retry_max_seconds
        
        for attempt in range(max_attempts):
            try:
                return self._make_single_api_call(text, is_batch, texts)
            except Exception as e:
                if attempt == max_attempts - 1:
                    logger.error(f"All {max_attempts} embedding API attempts failed: {type(e).__name__}: {str(e)}")
                    raise
                
                delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                if self.config.log_similarity_matches:
                    logger.warning(f"Embedding API attempt {attempt + 1}/{max_attempts} failed, retrying in {delay:.2f}s")
                time.sleep(delay)
        
        raise RuntimeError("Unexpected end of retry loop")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        if not text or not text.strip():
            return np.zeros(self.config.get_embedding_dimension())
        
        try:
            vector = self._call_embedding_api(text)
            return np.array(vector)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.config.get_embedding_dimension())
    
    def batch_get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts efficiently."""
        if not texts:
            return []
        
        # Filter out empty texts
        non_empty_texts = [t for t in texts if t and t.strip()]
        if not non_empty_texts:
            return [np.zeros(self.config.get_embedding_dimension()) for _ in texts]
        
        try:
            vectors = self._call_embedding_api("", is_batch=True, texts=non_empty_texts)
            
            # Handle case where some texts were empty
            result = []
            empty_idx = 0
            for i, text in enumerate(texts):
                if text and text.strip():
                    result.append(np.array(vectors[empty_idx]))
                    empty_idx += 1
                else:
                    result.append(np.zeros(self.config.get_embedding_dimension()))
            
            return result
        except Exception as e:
            logger.error(f"Batch embedding error for {len(texts)} texts: {type(e).__name__}: {str(e)}")
            # Return zero vectors on error to prevent complete failure
            return [np.zeros(self.config.get_embedding_dimension()) for _ in texts]
    
    def batch_get_embeddings_parallel(
        self, 
        texts: List[str], 
        batch_size: int = 100,
        max_workers: int = None
    ) -> List[np.ndarray]:
        """
        Get embeddings for multiple texts using parallel batch processing.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch (default: config batch_size)
            max_workers: Max concurrent threads (default: config max_concurrency)
        
        Returns:
            List of embedding vectors in the same order as input texts
        """
        if not texts:
            return []
        
        batch_size = batch_size or self.config.batch_size
        max_workers = max_workers or self.config.max_concurrency
        
        # Split into batches
        batches = []
        batch_indices = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batches.append(batch)
            batch_indices.append(i)
        
        # Process batches in parallel
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            future_to_batch_idx = {
                executor.submit(self.batch_get_embeddings, batch): idx 
                for idx, batch in enumerate(batches)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch_idx):
                batch_idx = future_to_batch_idx[future]
                try:
                    batch_results = future.result()
                    results[batch_idx] = batch_results
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx + 1}: {type(e).__name__}: {str(e)}")
                    # Return zero vectors for failed batch
                    results[batch_idx] = [
                        np.zeros(self.config.get_embedding_dimension()) 
                        for _ in batches[batch_idx]
                    ]
        
        # Combine results in order
        all_embeddings = []
        for idx in range(len(batches)):
            all_embeddings.extend(results[idx])
        
        return all_embeddings


def create_embedding_model(config: EmbeddingConfig = None) -> OpenAIEmbeddingModel:
    """Create an OpenAI embedding model instance."""
    return OpenAIEmbeddingModel(config=config)

