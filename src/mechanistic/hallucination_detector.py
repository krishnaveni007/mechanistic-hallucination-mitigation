"""
Hallucination Detector

This module implements various strategies for detecting hallucinations
using mechanistic signals extracted from language model computations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging


class HallucinationDetector:
    """
    Detects potential hallucinations using mechanistic signals.
    
    This class implements multiple strategies for hallucination detection
    based on internal model representations and computational patterns.
    """
    
    def __init__(self, 
                 method: str = "threshold_based",
                 threshold: float = 0.5,
                 use_ml_detector: bool = False):
        """
        Initialize the hallucination detector.
        
        Args:
            method: Detection method ('threshold_based', 'ml_based', 'ensemble')
            threshold: Threshold for threshold-based detection
            use_ml_detector: Whether to use machine learning-based detection
        """
        self.method = method
        self.threshold = threshold
        self.use_ml_detector = use_ml_detector
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML-based detector if requested
        if self.use_ml_detector or method == "ml_based":
            self.ml_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            self.scaler = StandardScaler()
            self.is_fitted = False
        
        # Initialize ensemble weights
        self.ensemble_weights = {
            'attention_entropy': 0.25,
            'activation_magnitude': 0.25,
            'confidence_variance': 0.25,
            'layer_consistency': 0.25
        }
    
    def threshold_based_detection(self, 
                                signals: Dict[str, torch.Tensor],
                                logits: torch.Tensor) -> torch.Tensor:
        """
        Detect hallucinations using threshold-based approach.
        
        Args:
            signals: Mechanistic signals
            logits: Model output logits
            
        Returns:
            Hallucination scores [batch]
        """
        batch_size = logits.size(0)
        hallucination_scores = torch.zeros(batch_size, device=logits.device)
        
        # Combine multiple signals with weights
        for signal_name, signal_values in signals.items():
            if signal_name in self.ensemble_weights:
                weight = self.ensemble_weights[signal_name]
                
                # Normalize signal to [0, 1] range
                if signal_name == 'attention_entropy':
                    # Higher entropy = more uncertain attention
                    normalized_signal = torch.sigmoid(signal_values - 2.0)
                elif signal_name == 'activation_magnitude':
                    # Higher magnitude = potentially unusual activations
                    normalized_signal = torch.sigmoid(signal_values - 1.0)
                elif signal_name == 'confidence_variance':
                    # Higher variance = more uncertain predictions
                    normalized_signal = torch.sigmoid(signal_values * 10.0)
                elif signal_name == 'layer_consistency':
                    # Lower consistency = more inconsistent representations
                    normalized_signal = 1.0 - torch.sigmoid(signal_values)
                else:
                    # Default normalization
                    normalized_signal = torch.sigmoid(signal_values)
                
                hallucination_scores += weight * normalized_signal
        
        # Apply threshold
        hallucination_detected = (hallucination_scores > self.threshold).float()
        
        return hallucination_scores
    
    def ml_based_detection(self, 
                         signals: Dict[str, torch.Tensor],
                         logits: torch.Tensor) -> torch.Tensor:
        """
        Detect hallucinations using machine learning approach.
        
        Args:
            signals: Mechanistic signals
            logits: Model output logits
            
        Returns:
            Hallucination scores [batch]
        """
        batch_size = logits.size(0)
        
        # Prepare features
        features = []
        for signal_name in ['attention_entropy', 'activation_magnitude', 
                           'confidence_variance', 'layer_consistency']:
            if signal_name in signals:
                features.append(signals[signal_name].cpu().numpy())
            else:
                # Use zeros if signal not available
                features.append(np.zeros(batch_size))
        
        if not features:
            # Fallback to threshold-based detection
            return self.threshold_based_detection(signals, logits)
        
        # Stack features
        feature_matrix = np.column_stack(features)
        
        if not self.is_fitted:
            # Fit the detector on current batch (for simplicity)
            # In practice, this should be fitted on a larger dataset
            self.scaler.fit(feature_matrix)
            scaled_features = self.scaler.transform(feature_matrix)
            self.ml_detector.fit(scaled_features)
            self.is_fitted = True
        
        # Transform features
        scaled_features = self.scaler.transform(feature_matrix)
        
        # Get anomaly scores
        anomaly_scores = self.ml_detector.decision_function(scaled_features)
        
        # Convert to hallucination scores (higher = more likely hallucination)
        hallucination_scores = torch.tensor(
            -anomaly_scores,  # Negative because IsolationForest returns lower scores for anomalies
            device=logits.device,
            dtype=torch.float32
        )
        
        # Normalize to [0, 1] range
        hallucination_scores = torch.sigmoid(hallucination_scores)
        
        return hallucination_scores
    
    def ensemble_detection(self, 
                         signals: Dict[str, torch.Tensor],
                         logits: torch.Tensor) -> torch.Tensor:
        """
        Detect hallucinations using ensemble of methods.
        
        Args:
            signals: Mechanistic signals
            logits: Model output logits
            
        Returns:
            Hallucination scores [batch]
        """
        # Get predictions from different methods
        threshold_scores = self.threshold_based_detection(signals, logits)
        
        if self.use_ml_detector:
            ml_scores = self.ml_based_detection(signals, logits)
            # Combine with equal weights
            ensemble_scores = 0.5 * threshold_scores + 0.5 * ml_scores
        else:
            ensemble_scores = threshold_scores
        
        return ensemble_scores
    
    def detect(self, 
              signals: Dict[str, torch.Tensor],
              logits: torch.Tensor) -> torch.Tensor:
        """
        Detect hallucinations using the specified method.
        
        Args:
            signals: Mechanistic signals
            logits: Model output logits
            
        Returns:
            Hallucination probability scores [batch]
        """
        if self.method == "threshold_based":
            return self.threshold_based_detection(signals, logits)
        elif self.method == "ml_based":
            return self.ml_based_detection(signals, logits)
        elif self.method == "ensemble":
            return self.ensemble_detection(signals, logits)
        else:
            self.logger.warning(f"Unknown method {self.method}, falling back to threshold_based")
            return self.threshold_based_detection(signals, logits)
    
    def update_ensemble_weights(self, weights: Dict[str, float]):
        """
        Update ensemble weights for signal combination.
        
        Args:
            weights: New weights for signals
        """
        self.ensemble_weights.update(weights)
        self.logger.info(f"Updated ensemble weights: {self.ensemble_weights}")
    
    def get_detection_statistics(self, 
                               hallucination_scores: torch.Tensor,
                               threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Compute statistics for hallucination detection results.
        
        Args:
            hallucination_scores: Hallucination probability scores
            threshold: Optional threshold for binary classification
            
        Returns:
            Detection statistics
        """
        stats = {
            'mean_score': hallucination_scores.mean().item(),
            'std_score': hallucination_scores.std().item(),
            'min_score': hallucination_scores.min().item(),
            'max_score': hallucination_scores.max().item(),
            'median_score': hallucination_scores.median().item()
        }
        
        if threshold is not None:
            binary_predictions = (hallucination_scores > threshold).float()
            stats['detection_rate'] = binary_predictions.mean().item()
            stats['num_detected'] = binary_predictions.sum().item()
        
        return stats
    
    def calibrate_threshold(self, 
                          signals_list: List[Dict[str, torch.Tensor]],
                          logits_list: List[torch.Tensor],
                          target_fpr: float = 0.1) -> float:
        """
        Calibrate detection threshold based on validation data.
        
        Args:
            signals_list: List of signal dictionaries from validation data
            logits_list: List of logits from validation data
            target_fpr: Target false positive rate
            
        Returns:
            Calibrated threshold
        """
        all_scores = []
        
        for signals, logits in zip(signals_list, logits_list):
            scores = self.detect(signals, logits)
            all_scores.append(scores)
        
        # Combine all scores
        combined_scores = torch.cat(all_scores, dim=0)
        
        # Find threshold for target FPR
        sorted_scores, _ = torch.sort(combined_scores)
        threshold_idx = int((1 - target_fpr) * len(sorted_scores))
        calibrated_threshold = sorted_scores[threshold_idx].item()
        
        self.threshold = calibrated_threshold
        self.logger.info(f"Calibrated threshold to {calibrated_threshold} for FPR={target_fpr}")
        
        return calibrated_threshold
