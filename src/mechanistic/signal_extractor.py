"""
Mechanistic Signal Extractor

This module extracts various mechanistic signals from language model computations
that can be used to detect and mitigate hallucinations.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging


class MechanisticSignalExtractor:
    """
    Extracts mechanistic signals from language model forward passes.
    
    This class analyzes various internal representations and computations
    to identify patterns that may indicate hallucination or uncertainty.
    """
    
    def __init__(self):
        """Initialize the signal extractor."""
        self.logger = logging.getLogger(__name__)
        
    def compute_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of attention weights as a measure of attention uncertainty.
        
        Args:
            attention_weights: Attention weights tensor [batch, heads, seq_len, seq_len]
            
        Returns:
            Attention entropy scores [batch]
        """
        # Compute entropy for each head
        head_entropies = []
        for head_idx in range(attention_weights.size(1)):
            head_weights = attention_weights[:, head_idx]  # [batch, seq_len, seq_len]
            
            # Compute entropy across the sequence dimension
            head_entropy = -torch.sum(head_weights * torch.log(head_weights + 1e-8), dim=-1)
            head_entropies.append(head_entropy.mean(dim=-1))  # Average over sequence
        
        # Average across heads
        attention_entropy = torch.stack(head_entropies, dim=-1).mean(dim=-1)
        
        return attention_entropy
    
    def compute_activation_magnitude(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute magnitude of activations as a measure of internal model confidence.
        
        Args:
            hidden_states: List of hidden states from each layer
            
        Returns:
            Activation magnitude scores [batch]
        """
        # Use the last layer hidden states
        last_hidden = hidden_states[-1]  # [batch, seq_len, hidden_dim]
        
        # Compute L2 norm of hidden states
        activation_norm = torch.norm(last_hidden, dim=-1)  # [batch, seq_len]
        
        # Average over sequence length
        activation_magnitude = activation_norm.mean(dim=-1)  # [batch]
        
        return activation_magnitude
    
    def compute_confidence_variance(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute variance in confidence scores across tokens.
        
        Args:
            logits: Model output logits [batch, seq_len, vocab_size]
            
        Returns:
            Confidence variance scores [batch]
        """
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]
        
        # Get max probability (confidence) for each token
        max_probs, _ = torch.max(probs, dim=-1)  # [batch, seq_len]
        
        # Compute variance in confidence across sequence
        confidence_variance = torch.var(max_probs, dim=-1)  # [batch]
        
        return confidence_variance
    
    def compute_gradient_magnitude(self, 
                                 logits: torch.Tensor,
                                 targets: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient magnitude as a measure of optimization difficulty.
        
        Args:
            logits: Model output logits
            targets: Target tokens
            
        Returns:
            Gradient magnitude scores
        """
        # Compute loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
        loss = loss.view(logits.size(0), -1)  # Reshape to [batch, seq_len]
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=loss.sum(),
            inputs=logits,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Compute gradient magnitude
        gradient_magnitude = torch.norm(gradients, dim=-1).mean(dim=-1)
        
        return gradient_magnitude
    
    def compute_layer_consistency(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute consistency across layers as a measure of internal coherence.
        
        Args:
            hidden_states: List of hidden states from each layer
            
        Returns:
            Layer consistency scores [batch]
        """
        if len(hidden_states) < 2:
            return torch.zeros(hidden_states[0].size(0), device=hidden_states[0].device)
        
        # Compute cosine similarity between consecutive layers
        similarities = []
        for i in range(len(hidden_states) - 1):
            # Average pool to get sentence-level representations
            layer1 = hidden_states[i].mean(dim=1)  # [batch, hidden_dim]
            layer2 = hidden_states[i + 1].mean(dim=1)  # [batch, hidden_dim]
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(layer1, layer2, dim=-1)
            similarities.append(similarity)
        
        # Average similarity across layers
        layer_consistency = torch.stack(similarities, dim=-1).mean(dim=-1)
        
        return layer_consistency
    
    def compute_token_surprisal(self, 
                              logits: torch.Tensor,
                              input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute surprisal (negative log probability) of input tokens.
        
        Args:
            logits: Model output logits [batch, seq_len, vocab_size]
            input_ids: Input token IDs [batch, seq_len]
            
        Returns:
            Token surprisal scores [batch]
        """
        # Shift logits and targets for next token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather probabilities for actual tokens
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Compute surprisal (negative log probability)
        surprisal = -token_log_probs.mean(dim=-1)  # Average over sequence
        
        return surprisal
    
    def extract_signals(self, 
                       hidden_states: List[torch.Tensor],
                       attentions: List[torch.Tensor],
                       logits: torch.Tensor,
                       input_ids: Optional[torch.Tensor] = None,
                       targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Extract all mechanistic signals from model outputs.
        
        Args:
            hidden_states: List of hidden states from each layer
            attentions: List of attention weights from each layer
            logits: Model output logits
            input_ids: Optional input token IDs
            targets: Optional target tokens
            
        Returns:
            Dictionary of extracted signals
        """
        signals = {}
        
        # Extract attention-based signals
        if attentions and len(attentions) > 0:
            # Use attention from the last layer
            last_attention = attentions[-1]  # [batch, heads, seq_len, seq_len]
            signals['attention_entropy'] = self.compute_attention_entropy(last_attention)
        
        # Extract activation-based signals
        if hidden_states and len(hidden_states) > 0:
            signals['activation_magnitude'] = self.compute_activation_magnitude(hidden_states)
            signals['layer_consistency'] = self.compute_layer_consistency(hidden_states)
        
        # Extract confidence-based signals
        if logits is not None:
            signals['confidence_variance'] = self.compute_confidence_variance(logits)
            
            # Token surprisal if input_ids provided
            if input_ids is not None:
                signals['token_surprisal'] = self.compute_token_surprisal(logits, input_ids)
        
        # Extract gradient-based signals (if targets provided)
        if targets is not None and logits is not None:
            try:
                signals['gradient_magnitude'] = self.compute_gradient_magnitude(logits, targets)
            except RuntimeError:
                # Skip gradient computation if not in training mode
                pass
        
        return signals
    
    def get_signal_statistics(self, signals: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics for extracted signals.
        
        Args:
            signals: Dictionary of extracted signals
            
        Returns:
            Dictionary of signal statistics
        """
        stats = {}
        
        for signal_name, signal_values in signals.items():
            if isinstance(signal_values, torch.Tensor):
                stats[signal_name] = {
                    'mean': signal_values.mean().item(),
                    'std': signal_values.std().item(),
                    'min': signal_values.min().item(),
                    'max': signal_values.max().item(),
                    'median': signal_values.median().item()
                }
        
        return stats
