"""
Group Selector Implementation

This module implements the group selection strategy for GRPO,
which groups samples based on mechanistic signals and hallucination scores.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np
from sklearn.cluster import KMeans
import logging


class GroupSelector:
    """
    Group Selector for GRPO Algorithm.
    
    This class implements various strategies for grouping samples based on
    mechanistic signals, hallucination scores, and other criteria.
    """
    
    def __init__(self, 
                 num_groups: int = 4,
                 group_size: int = 2,
                 strategy: str = "hallucination_based"):
        """
        Initialize the group selector.
        
        Args:
            num_groups: Number of groups to form
            group_size: Minimum size of each group
            strategy: Grouping strategy ('hallucination_based', 'signal_clustering', 'random')
        """
        self.num_groups = num_groups
        self.group_size = group_size
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
        
    def hallucination_based_grouping(self, 
                                   batch: Dict[str, torch.Tensor],
                                   hallucination_scores: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Group samples based on hallucination scores.
        
        Args:
            batch: Input batch
            hallucination_scores: Hallucination probability scores
            
        Returns:
            List of grouped batches
        """
        batch_size = hallucination_scores.size(0)
        
        # Sort samples by hallucination score
        sorted_indices = torch.argsort(hallucination_scores)
        
        # Create groups with balanced hallucination scores
        groups = []
        samples_per_group = batch_size // self.num_groups
        remainder = batch_size % self.num_groups
        
        start_idx = 0
        for group_id in range(self.num_groups):
            # Distribute remainder samples across first few groups
            group_size = samples_per_group + (1 if group_id < remainder else 0)
            end_idx = start_idx + group_size
            
            group_indices = sorted_indices[start_idx:end_idx]
            
            # Create group batch
            group_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    group_batch[key] = value[group_indices]
                else:
                    group_batch[key] = value
            
            # Add group metadata
            group_batch['group_id'] = torch.full((group_size,), group_id, dtype=torch.long)
            group_batch['group_mask'] = torch.full((group_size,), group_id, dtype=torch.long)
            group_batch['hallucination_scores'] = hallucination_scores[group_indices]
            
            groups.append(group_batch)
            start_idx = end_idx
        
        return groups
    
    def signal_clustering_grouping(self, 
                                 batch: Dict[str, torch.Tensor],
                                 hallucination_scores: torch.Tensor,
                                 signals: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Group samples using clustering on mechanistic signals.
        
        Args:
            batch: Input batch
            hallucination_scores: Hallucination probability scores
            signals: Mechanistic signals
            
        Returns:
            List of grouped batches
        """
        batch_size = hallucination_scores.size(0)
        
        # Combine signals into feature vector
        signal_features = []
        
        if 'attention_entropy' in signals:
            signal_features.append(signals['attention_entropy'].cpu().numpy())
        
        if 'activation_magnitude' in signals:
            signal_features.append(signals['activation_magnitude'].cpu().numpy())
        
        if 'confidence_variance' in signals:
            signal_features.append(signals['confidence_variance'].cpu().numpy())
        
        # Add hallucination scores as a feature
        signal_features.append(hallucination_scores.cpu().numpy())
        
        if not signal_features:
            # Fallback to hallucination-based grouping
            return self.hallucination_based_grouping(batch, hallucination_scores)
        
        # Stack features
        feature_matrix = np.column_stack(signal_features)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=self.num_groups, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix)
        
        # Create groups based on clusters
        groups = []
        for group_id in range(self.num_groups):
            group_mask = cluster_labels == group_id
            group_indices = torch.tensor(np.where(group_mask)[0], dtype=torch.long)
            
            if len(group_indices) == 0:
                continue  # Skip empty groups
            
            # Create group batch
            group_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    group_batch[key] = value[group_indices]
                else:
                    group_batch[key] = value
            
            # Add group metadata
            group_size = len(group_indices)
            group_batch['group_id'] = torch.full((group_size,), group_id, dtype=torch.long)
            group_batch['group_mask'] = torch.full((group_size,), group_id, dtype=torch.long)
            group_batch['hallucination_scores'] = hallucination_scores[group_indices]
            
            groups.append(group_batch)
        
        return groups
    
    def random_grouping(self, 
                      batch: Dict[str, torch.Tensor],
                      hallucination_scores: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Group samples randomly (for baseline comparison).
        
        Args:
            batch: Input batch
            hallucination_scores: Hallucination probability scores
            
        Returns:
            List of grouped batches
        """
        batch_size = hallucination_scores.size(0)
        
        # Random permutation of indices
        random_indices = torch.randperm(batch_size)
        
        # Create groups
        groups = []
        samples_per_group = batch_size // self.num_groups
        remainder = batch_size % self.num_groups
        
        start_idx = 0
        for group_id in range(self.num_groups):
            group_size = samples_per_group + (1 if group_id < remainder else 0)
            end_idx = start_idx + group_size
            
            group_indices = random_indices[start_idx:end_idx]
            
            # Create group batch
            group_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    group_batch[key] = value[group_indices]
                else:
                    group_batch[key] = value
            
            # Add group metadata
            group_batch['group_id'] = torch.full((group_size,), group_id, dtype=torch.long)
            group_batch['group_mask'] = torch.full((group_size,), group_id, dtype=torch.long)
            group_batch['hallucination_scores'] = hallucination_scores[group_indices]
            
            groups.append(group_batch)
            start_idx = end_idx
        
        return groups
    
    def form_groups(self, 
                   batch: Dict[str, torch.Tensor],
                   hallucination_scores: torch.Tensor,
                   signals: Optional[Dict[str, torch.Tensor]] = None) -> List[Dict[str, torch.Tensor]]:
        """
        Form groups based on the selected strategy.
        
        Args:
            batch: Input batch
            hallucination_scores: Hallucination probability scores
            signals: Optional mechanistic signals
            
        Returns:
            List of grouped batches
        """
        if self.strategy == "hallucination_based":
            return self.hallucination_based_grouping(batch, hallucination_scores)
        elif self.strategy == "signal_clustering" and signals is not None:
            return self.signal_clustering_grouping(batch, hallucination_scores, signals)
        elif self.strategy == "random":
            return self.random_grouping(batch, hallucination_scores)
        else:
            # Default to hallucination-based grouping
            self.logger.warning(f"Unknown strategy {self.strategy}, falling back to hallucination_based")
            return self.hallucination_based_grouping(batch, hallucination_scores)
    
    def get_group_statistics(self, groups: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """
        Compute statistics for the formed groups.
        
        Args:
            groups: List of grouped batches
            
        Returns:
            Group statistics
        """
        stats = {
            'num_groups': len(groups),
            'avg_group_size': 0.0,
            'group_size_std': 0.0,
            'avg_hallucination_score': 0.0,
            'hallucination_score_std': 0.0
        }
        
        if not groups:
            return stats
        
        group_sizes = [len(group['hallucination_scores']) for group in groups]
        hallucination_scores = [group['hallucination_scores'].mean().item() for group in groups]
        
        stats['avg_group_size'] = np.mean(group_sizes)
        stats['group_size_std'] = np.std(group_sizes)
        stats['avg_hallucination_score'] = np.mean(hallucination_scores)
        stats['hallucination_score_std'] = np.std(hallucination_scores)
        
        return stats
