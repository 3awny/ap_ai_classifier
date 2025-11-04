"""
Unified semantic retrieval index supporting both SentenceTransformer and OpenAI embeddings.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
import logging
import numpy as np
from sklearn.neighbors import NearestNeighbors
from ap_ai_classifier.config import EMBEDDING_MODEL_NAME, TOP_K
from ap_ai_classifier.embeddings import create_embedding_model

logger = logging.getLogger(__name__)


@dataclass
class RetrievedExample:
    """Unified retrieval result format."""
    idx: int
    text: str
    nominal: str
    department: str
    tax_code: str
    distance: float
    similarity: Optional[float] = None
    
    def __post_init__(self):
        """Calculate similarity from distance if not provided."""
        if self.similarity is None:
            self.similarity = 1.0 - self.distance


class SemanticIndex:
    """
    Unified semantic retrieval index supporting multiple embedding backends.
    
    Supports:
    - SentenceTransformer (for semantic splitting, local model)
    - OpenAI embeddings (for LLM retrieval, API-based)
    """
    
    def __init__(
        self,
        backend: str = 'sentence_transformer',  # 'sentence_transformer' or 'openai'
        model_name: Optional[str] = None,  # For sentence_transformer backend
        use_parallel: bool = True  # For openai backend
    ):
        """
        Initialize semantic index.
        
        Args:
            backend: 'sentence_transformer' (local) or 'openai' (API-based)
            model_name: Model name for sentence_transformer (defaults to EMBEDDING_MODEL_NAME)
            use_parallel: Use parallel processing for OpenAI embeddings (default: True)
        """
        self.backend = backend
        self.use_parallel = use_parallel
        self.nn = None
        self.texts = None
        self.labels = None
        self.embeddings = None
        
        if backend == 'sentence_transformer':
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name or EMBEDDING_MODEL_NAME)
            self.embedding_model = None
        elif backend == 'openai':
            self.model = None
            self.embedding_model = create_embedding_model()
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'sentence_transformer' or 'openai'")
    
    def fit(self, texts: List[str], labels: List[Tuple[str, str, str]], use_parallel: Optional[bool] = None):
        """
        Build index from training data.
        
        Args:
            texts: List of text strings to index
            labels: List of label tuples (nominal, department, tax_code)
            use_parallel: Override parallel processing setting (for OpenAI backend only)
        """
        self.texts = texts
        self.labels = labels
        
        if self.backend == 'sentence_transformer':
            self.embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        else:  # openai
            batch_size = 100
            num_batches = (len(texts) + batch_size - 1) // batch_size
            use_parallel = use_parallel if use_parallel is not None else self.use_parallel
            
            if use_parallel:
                logger.info(f"🔮 Building retrieval index ({num_batches} batches, {self.embedding_model.config.max_concurrency} threads)...")
                all_embeddings = self.embedding_model.batch_get_embeddings_parallel(
                    texts, batch_size=batch_size
                )
            else:
                logger.info(f"🔮 Building retrieval index ({num_batches} batches)...")
                all_embeddings = []
                from tqdm import tqdm
                for i in tqdm(range(0, len(texts), batch_size), desc="Indexing"):
                    batch = texts[i:i+batch_size]
                    batch_embeddings = self.embedding_model.batch_get_embeddings(batch)
                    all_embeddings.extend(batch_embeddings)
            
            self.embeddings = np.array(all_embeddings)
        
        # Build k-NN index
        self.nn = NearestNeighbors(n_neighbors=min(TOP_K, len(texts)), metric="cosine")
        self.nn.fit(self.embeddings)
    
    def encode(self, text: str) -> np.ndarray:
        """Encode a single text into embedding vector."""
        if self.backend == 'sentence_transformer':
            return self.model.encode([text], convert_to_numpy=True)[0]
        else:  # openai
            return self.embedding_model.get_embedding(text)
    
    def retrieve(self, query_text: str, top_k: int = TOP_K) -> List[RetrievedExample]:
        """
        Retrieve similar examples using k-NN.
        
        Args:
            query_text: Query text to search for
            top_k: Number of results to return
            
        Returns:
            List of RetrievedExample objects
        """
        q_emb = self.encode(query_text).reshape(1, -1)
        distances, indices = self.nn.kneighbors(q_emb, n_neighbors=top_k, return_distance=True)
        
        results = []
        for d, i in zip(distances[0], indices[0]):
            nominal, dept, tc = self.labels[i]
            similarity = 1.0 - float(d)  # Convert distance to similarity
            results.append(
                RetrievedExample(
                    idx=int(i),
                    text=self.texts[i],
                    nominal=nominal,
                    department=dept,
                    tax_code=tc,
                    distance=float(d),
                    similarity=similarity
                )
            )
        return results
    
    def search(self, query: str, k: int = 5):
        """
        Search for similar examples. Returns list of dicts (compatible with OpenAIRetrievalIndex interface).
        
        Args:
            query: Query text
            k: Number of results
            
        Returns:
            List of dicts with 'text', 'labels', 'similarity' keys
        """
        results = self.retrieve(query, k)
        return [
            {
                'text': r.text,
                'labels': (r.nominal, r.department, r.tax_code),
                'similarity': r.similarity
            }
            for r in results
        ]


# Alias for backward compatibility
OpenAIRetrievalIndex = SemanticIndex
