"""
Attention Analyzer

This module provides detailed analysis of attention patterns that may
indicate hallucination or uncertainty in language model outputs.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging


class AttentionAnalyzer:
    """
    Analyzes attention patterns for hallucination detection.
    
    This class provides various methods to analyze attention weights
    and identify patterns that may indicate hallucinations or uncertainty.
    """
    
    def __init__(self):
        """Initialize the attention analyzer."""
        self.logger = logging.getLogger(__name__)
    
    def compute_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of attention weights.
        
        Args:
            attention_weights: Attention weights [batch, heads, seq_len, seq_len]
            
        Returns:
            Attention entropy [batch, heads]
        """
        # Add small epsilon to avoid log(0)
        attention_weights = attention_weights + 1e-8
        
        # Compute entropy
        entropy = -torch.sum(attention_weights * torch.log(attention_weights), dim=-1)
        
        return entropy
    
    def compute_attention_diversity(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute attention diversity across heads.
        
        Args:
            attention_weights: Attention weights [batch, heads, seq_len, seq_len]
            
        Returns:
            Attention diversity scores [batch]
        """
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # Compute pairwise KL divergence between heads
        diversity_scores = []
        
        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                head_i = attention_weights[:, i]  # [batch, seq_len, seq_len]
                head_j = attention_weights[:, j]  # [batch, seq_len, seq_len]
                
                # Compute KL divergence
                kl_div = F.kl_div(
                    F.log_softmax(head_i.view(-1, seq_len), dim=-1),
                    F.softmax(head_j.view(-1, seq_len), dim=-1),
                    reduction='none'
                )
                
                # Average KL divergence
                avg_kl = kl_div.view(batch_size, -1).mean(dim=-1)
                diversity_scores.append(avg_kl)
        
        # Average across all head pairs
        if diversity_scores:
            diversity = torch.stack(diversity_scores, dim=-1).mean(dim=-1)
        else:
            diversity = torch.zeros(batch_size, device=attention_weights.device)
        
        return diversity
    
    def compute_attention_focus(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute attention focus (how concentrated attention is).
        
        Args:
            attention_weights: Attention weights [batch, heads, seq_len, seq_len]
            
        Returns:
            Attention focus scores [batch, heads]
        """
        # Compute max attention weight for each position
        max_attention, _ = torch.max(attention_weights, dim=-1)  # [batch, heads, seq_len]
        
        # Average across sequence length
        focus_scores = max_attention.mean(dim=-1)  # [batch, heads]
        
        return focus_scores
    
    def compute_attention_consistency(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute consistency of attention patterns across layers.
        
        Args:
            attention_weights: List of attention weights from multiple layers
            
        Returns:
            Attention consistency scores [batch]
        """
        if len(attention_weights) < 2:
            return torch.zeros(attention_weights[0].size(0), device=attention_weights[0].device)
        
        # Use the last two layers for consistency analysis
        layer1 = attention_weights[-2]  # [batch, heads, seq_len, seq_len]
        layer2 = attention_weights[-1]  # [batch, heads, seq_len, seq_len]
        
        # Compute cosine similarity between attention patterns
        # Flatten attention matrices
        flat1 = layer1.view(layer1.size(0), layer1.size(1), -1)  # [batch, heads, seq_len*seq_len]
        flat2 = layer2.view(layer2.size(0), layer2.size(1), -1)  # [batch, heads, seq_len*seq_len]
        
        # Compute cosine similarity for each head
        similarities = F.cosine_similarity(flat1, flat2, dim=-1)  # [batch, heads]
        
        # Average across heads
        consistency = similarities.mean(dim=-1)  # [batch]
        
        return consistency
    
    def analyze_attention_patterns(self, attention_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Comprehensive analysis of attention patterns.
        
        Args:
            attention_weights: Attention weights [batch, heads, seq_len, seq_len]
            
        Returns:
            Dictionary of attention analysis results
        """
        results = {}
        
        # Basic attention statistics
        results['attention_entropy'] = self.compute_attention_entropy(attention_weights)
        results['attention_diversity'] = self.compute_attention_diversity(attention_weights)
        results['attention_focus'] = self.compute_attention_focus(attention_weights)
        
        # Attention statistics across heads
        results['mean_attention_entropy'] = results['attention_entropy'].mean(dim=-1)
        results['std_attention_entropy'] = results['attention_entropy'].std(dim=-1)
        results['mean_attention_focus'] = results['attention_focus'].mean(dim=-1)
        results['std_attention_focus'] = results['attention_focus'].std(dim=-1)
        
        return results
    
    def identify_attention_anomalies(self, 
                                   attention_analysis: Dict[str, torch.Tensor],
                                   threshold_std: float = 2.0) -> torch.Tensor:
        """
        Identify attention anomalies that may indicate hallucinations.
        
        Args:
            attention_analysis: Results from attention pattern analysis
            threshold_std: Standard deviation threshold for anomaly detection
            
        Returns:
            Binary tensor indicating attention anomalies [batch]
        """
        batch_size = list(attention_analysis.values())[0].size(0)
        anomaly_scores = torch.zeros(batch_size, device=list(attention_analysis.values())[0].device)
        
        # Check for anomalies in different attention metrics
        metrics_to_check = [
            'mean_attention_entropy',
            'attention_diversity',
            'mean_attention_focus'
        ]
        
        for metric in metrics_to_check:
            if metric in attention_analysis:
                values = attention_analysis[metric]
                
                # Compute z-score
                mean_val = values.mean()
                std_val = values.std()
                
                if std_val > 0:
                    z_scores = torch.abs((values - mean_val) / std_val)
                    anomaly_scores += (z_scores > threshold_std).float()
        
        # Normalize by number of metrics
        anomaly_scores = anomaly_scores / len(metrics_to_check)
        
        return anomaly_scores
