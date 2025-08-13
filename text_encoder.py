"""
Text Encoder Module for Drift Detection
Handles encoding of text data using pre-trained transformer models
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')


class TextEncoder:
    """Text encoder using pre-trained transformer models"""
    
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 512):
        """
        Initialize text encoder with specified model
        
        Args:
            model_name: HuggingFace model name
            max_length: Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded model: {model_name} on device: {self.device}")
        except Exception as e:
            raise ValueError(f"Failed to load model {model_name}: {str(e)}")
        
        # Get embedding dimension
        with torch.no_grad():
            dummy_input = self.tokenizer("test", return_tensors='pt', 
                                       padding=True, truncation=True)
            dummy_input = {k: v.to(self.device) for k, v in dummy_input.items()}
            dummy_output = self.model(**dummy_input)
            self.embedding_dim = dummy_output.last_hidden_state.shape[-1]
    
    def encode_texts(self, texts: List[str], batch_size: int = 32, 
                    pooling_strategy: str = 'cls') -> np.ndarray:
        """
        Encode a list of texts into embeddings
        
        Args:
            texts: List of text strings to encode
            batch_size: Batch size for processing
            pooling_strategy: Strategy for pooling tokens ('cls', 'mean', 'max')
            
        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)
        
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self._encode_batch(batch_texts, pooling_strategy)
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def _encode_batch(self, texts: List[str], pooling_strategy: str) -> np.ndarray:
        """Encode a batch of texts"""
        # Tokenize
        inputs = self.tokenizer(
            texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=self.max_length
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Apply pooling strategy
            if pooling_strategy == 'cls':
                # Use CLS token (first token)
                embeddings = hidden_states[:, 0, :]
            elif pooling_strategy == 'mean':
                # Mean pooling with attention mask
                embeddings = self._mean_pooling(hidden_states, attention_mask)
            elif pooling_strategy == 'max':
                # Max pooling
                embeddings = torch.max(hidden_states, dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
        
        return embeddings.cpu().numpy()
    
    def _mean_pooling(self, hidden_states: torch.Tensor, 
                     attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply mean pooling with attention mask"""
        # Expand attention mask to match hidden states
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        
        # Sum embeddings and mask
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def encode_batch_average(self, texts: List[str], 
                           pooling_strategy: str = 'cls') -> np.ndarray:
        """
        Encode texts and return average embedding across all texts
        
        Args:
            texts: List of text strings
            pooling_strategy: Pooling strategy for individual texts
            
        Returns:
            Average embedding vector
        """
        embeddings = self.encode_texts(texts, pooling_strategy=pooling_strategy)
        if len(embeddings) == 0:
            return np.zeros(self.embedding_dim)
        return np.mean(embeddings, axis=0)
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings"""
        return self.embedding_dim
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'max_length': self.max_length,
            'device': str(self.device)
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the encoder
    encoder = TextEncoder("bert-base-uncased")
    
    test_texts = [
        "I want to buy a new phone",
        "Looking for the best laptop deals", 
        "Machine learning is fascinating",
        "Data science requires statistical knowledge"
    ]
    
    print("Testing TextEncoder...")
    print(f"Model info: {encoder.get_model_info()}")
    
    # Test encoding
    embeddings = encoder.encode_texts(test_texts)
    print(f"Encoded {len(test_texts)} texts into embeddings of shape: {embeddings.shape}")
    
    # Test batch average
    avg_embedding = encoder.encode_batch_average(test_texts)
    print(f"Average embedding shape: {avg_embedding.shape}")
    
    # Test different pooling strategies
    for strategy in ['cls', 'mean', 'max']:
        emb = encoder.encode_texts(test_texts[:2], pooling_strategy=strategy)
        print(f"Pooling strategy '{strategy}': {emb.shape}")
