"""
Activation Tracker

This module tracks and analyzes neural network activations to identify
patterns that may indicate hallucinations or model uncertainty.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging


class ActivationTracker:
    """
    Tracks and analyzes neural network activations for hallucination detection.
    
    This class monitors internal model activations and identifies patterns
    that may indicate hallucinations or uncertainty in model outputs.
    """
    
    def __init__(self):
        """Initialize the activation tracker."""
        self.logger = logging.getLogger(__name__)
        self.activation_history = []
    
    def track_activations(self, 
                         hidden_states: List[torch.Tensor],
                         layer_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Track activations across all layers.
        
        Args:
            hidden_states: List of hidden states from each layer
            layer_names: Optional names for layers
            
        Returns:
            Dictionary of activation statistics
        """
        activation_stats = {}
        
        for layer_idx, hidden_state in enumerate(hidden_states):
            layer_name = layer_names[layer_idx] if layer_names else f"layer_{layer_idx}"
            
            # Basic statistics
            activation_stats[f"{layer_name}_mean"] = hidden_state.mean(dim=-1).mean(dim=-1)  # Average over sequence and hidden dim
            activation_stats[f"{layer_name}_std"] = hidden_state.std(dim=-1).mean(dim=-1)
            activation_stats[f"{layer_name}_max"] = hidden_state.max(dim=-1).values.mean(dim=-1)
            activation_stats[f"{layer_name}_min"] = hidden_state.min(dim=-1).values.mean(dim=-1)
            
            # Activation magnitude
            activation_stats[f"{layer_name}_magnitude"] = torch.norm(hidden_state, dim=-1).mean(dim=-1)
            
            # Activation sparsity (fraction of near-zero activations)
            near_zero = torch.abs(hidden_state) < 0.01
            activation_stats[f"{layer_name}_sparsity"] = near_zero.float().mean(dim=-1).mean(dim=-1)
        
        return activation_stats
    
    def compute_activation_consistency(self, 
                                     hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute consistency of activations across layers.
        
        Args:
            hidden_states: List of hidden states from each layer
            
        Returns:
            Activation consistency scores [batch]
        """
        if len(hidden_states) < 2:
            return torch.zeros(hidden_states[0].size(0), device=hidden_states[0].device)
        
        consistency_scores = []
        
        # Compute consistency between consecutive layers
        for i in range(len(hidden_states) - 1):
            layer1 = hidden_states[i]
            layer2 = hidden_states[i + 1]
            
            # Average pool to get sentence-level representations
            repr1 = layer1.mean(dim=1)  # [batch, hidden_dim]
            repr2 = layer2.mean(dim=1)  # [batch, hidden_dim]
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(repr1, repr2, dim=-1)
            consistency_scores.append(similarity)
        
        # Average consistency across layer pairs
        if consistency_scores:
            consistency = torch.stack(consistency_scores, dim=-1).mean(dim=-1)
        else:
            consistency = torch.zeros(hidden_states[0].size(0), device=hidden_states[0].device)
        
        return consistency
    
    def detect_activation_anomalies(self, 
                                  activation_stats: Dict[str, torch.Tensor],
                                  threshold_std: float = 2.0) -> torch.Tensor:
        """
        Detect activation anomalies that may indicate hallucinations.
        
        Args:
            activation_stats: Activation statistics from tracking
            threshold_std: Standard deviation threshold for anomaly detection
            
        Returns:
            Binary tensor indicating activation anomalies [batch]
        """
        batch_size = list(activation_stats.values())[0].size(0)
        anomaly_scores = torch.zeros(batch_size, device=list(activation_stats.values())[0].device)
        
        # Check for anomalies in different activation metrics
        metrics_to_check = [
            'magnitude', 'sparsity', 'std'
        ]
        
        for metric in metrics_to_check:
            # Find all stats containing this metric
            metric_stats = {k: v for k, v in activation_stats.items() if metric in k}
            
            if metric_stats:
                # Combine all layers for this metric
                combined_values = torch.stack(list(metric_stats.values()), dim=-1).mean(dim=-1)
                
                # Compute z-score
                mean_val = combined_values.mean()
                std_val = combined_values.std()
                
                if std_val > 0:
                    z_scores = torch.abs((combined_values - mean_val) / std_val)
                    anomaly_scores += (z_scores > threshold_std).float()
        
        # Normalize by number of metrics
        if metrics_to_check:
            anomaly_scores = anomaly_scores / len(metrics_to_check)
        
        return anomaly_scores
    
    def compute_activation_entropy(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of activations as a measure of uncertainty.
        
        Args:
            hidden_states: Hidden states [batch, seq_len, hidden_dim]
            
        Returns:
            Activation entropy scores [batch]
        """
        # Discretize activations into bins
        num_bins = 10
        min_val = hidden_states.min()
        max_val = hidden_states.max()
        
        # Create bins
        bin_edges = torch.linspace(min_val, max_val, num_bins + 1, device=hidden_states.device)
        
        # Digitize activations
        digitized = torch.bucketize(hidden_states, bin_edges)
        
        # Compute entropy for each sample
        batch_size = hidden_states.size(0)
        entropies = []
        
        for i in range(batch_size):
            # Get histogram for this sample
            sample_digits = digitized[i].flatten()  # Flatten to 1D
            hist = torch.bincount(sample_digits, minlength=num_bins + 1).float()
            
            # Normalize to probabilities
            probs = hist / hist.sum()
            probs = probs[probs > 0]  # Remove zero probabilities
            
            # Compute entropy
            entropy = -torch.sum(probs * torch.log(probs))
            entropies.append(entropy)
        
        return torch.stack(entropies)
    
    def analyze_activation_patterns(self, 
                                  hidden_states: List[torch.Tensor],
                                  layer_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Comprehensive analysis of activation patterns.
        
        Args:
            hidden_states: List of hidden states from each layer
            layer_names: Optional names for layers
            
        Returns:
            Dictionary of activation analysis results
        """
        results = {}
        
        # Track basic activation statistics
        activation_stats = self.track_activations(hidden_states, layer_names)
        results.update(activation_stats)
        
        # Compute consistency across layers
        results['activation_consistency'] = self.compute_activation_consistency(hidden_states)
        
        # Compute entropy for each layer
        for layer_idx, hidden_state in enumerate(hidden_states):
            layer_name = layer_names[layer_idx] if layer_names else f"layer_{layer_idx}"
            results[f"{layer_name}_entropy"] = self.compute_activation_entropy(hidden_state)
        
        # Detect anomalies
        results['activation_anomalies'] = self.detect_activation_anomalies(activation_stats)
        
        return results
