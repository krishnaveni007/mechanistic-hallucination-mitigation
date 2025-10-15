"""
Hallucination Metrics

This module implements various metrics for evaluating hallucination detection
and mitigation effectiveness.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score,
    confusion_matrix
)
import logging


class HallucinationMetrics:
    """
    Comprehensive metrics for evaluating hallucination detection and mitigation.
    
    This class provides various metrics to assess the effectiveness of
    hallucination detection and mitigation techniques.
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.logger = logging.getLogger(__name__)
    
    def compute_detection_metrics(self, 
                                predictions: torch.Tensor,
                                targets: torch.Tensor,
                                threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute binary classification metrics for hallucination detection.
        
        Args:
            predictions: Predicted hallucination scores [batch]
            targets: Ground truth hallucination labels [batch]
            threshold: Threshold for binary classification
            
        Returns:
            Dictionary of detection metrics
        """
        # Convert to numpy for sklearn metrics
        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        # Binary predictions
        binary_preds = (pred_np > threshold).astype(int)
        
        # Compute metrics
        metrics = {
            'precision': precision_score(target_np, binary_preds, zero_division=0),
            'recall': recall_score(target_np, binary_preds, zero_division=0),
            'f1': f1_score(target_np, binary_preds, zero_division=0),
            'accuracy': (binary_preds == target_np).mean()
        }
        
        # ROC-AUC and PR-AUC
        if len(np.unique(target_np)) > 1:
            metrics['roc_auc'] = roc_auc_score(target_np, pred_np)
            metrics['pr_auc'] = average_precision_score(target_np, pred_np)
        else:
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(target_np, binary_preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            metrics['specificity'] = 0.0
            metrics['sensitivity'] = 0.0
        
        return metrics
    
    def compute_regression_metrics(self, 
                                 predictions: torch.Tensor,
                                 targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute regression metrics for hallucination score prediction.
        
        Args:
            predictions: Predicted hallucination scores [batch]
            targets: Ground truth hallucination scores [batch]
            
        Returns:
            Dictionary of regression metrics
        """
        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        # Mean Squared Error
        mse = np.mean((pred_np - target_np) ** 2)
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(pred_np - target_np))
        
        # R-squared
        ss_res = np.sum((target_np - pred_np) ** 2)
        ss_tot = np.sum((target_np - np.mean(target_np)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Pearson correlation
        correlation = np.corrcoef(pred_np, target_np)[0, 1] if len(pred_np) > 1 else 0.0
        correlation = correlation if not np.isnan(correlation) else 0.0
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'correlation': correlation
        }
        
        return metrics
    
    def compute_mitigation_metrics(self, 
                                 original_scores: torch.Tensor,
                                 mitigated_scores: torch.Tensor,
                                 ground_truth: torch.Tensor) -> Dict[str, float]:
        """
        Compute metrics for hallucination mitigation effectiveness.
        
        Args:
            original_scores: Original hallucination scores [batch]
            mitigated_scores: Mitigated hallucination scores [batch]
            ground_truth: Ground truth hallucination labels [batch]
            
        Returns:
            Dictionary of mitigation metrics
        """
        # Reduction in hallucination scores
        reduction = original_scores - mitigated_scores
        avg_reduction = reduction.mean().item()
        
        # Percentage of samples with reduced hallucination scores
        improvement_rate = (reduction > 0).float().mean().item()
        
        # Improvement for hallucinating samples
        hallucinating_mask = ground_truth.bool()
        if hallucinating_mask.sum() > 0:
            hallucination_reduction = reduction[hallucinating_mask].mean().item()
            hallucination_improvement_rate = (reduction[hallucinating_mask] > 0).float().mean().item()
        else:
            hallucination_reduction = 0.0
            hallucination_improvement_rate = 0.0
        
        # False positive rate (non-hallucinating samples that were "fixed")
        non_hallucinating_mask = ~hallucinating_mask
        if non_hallucinating_mask.sum() > 0:
            fp_rate = (reduction[non_hallucinating_mask] < 0).float().mean().item()
        else:
            fp_rate = 0.0
        
        metrics = {
            'avg_reduction': avg_reduction,
            'improvement_rate': improvement_rate,
            'hallucination_reduction': hallucination_reduction,
            'hallucination_improvement_rate': hallucination_improvement_rate,
            'false_positive_rate': fp_rate
        }
        
        return metrics
    
    def compute_confidence_metrics(self, 
                                 predictions: torch.Tensor,
                                 confidence_scores: torch.Tensor,
                                 targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute metrics for prediction confidence calibration.
        
        Args:
            predictions: Predicted hallucination scores [batch]
            confidence_scores: Confidence scores [batch]
            targets: Ground truth labels [batch]
            
        Returns:
            Dictionary of confidence metrics
        """
        pred_np = predictions.cpu().numpy()
        conf_np = confidence_scores.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        # Expected Calibration Error (ECE)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (conf_np > bin_lower) & (conf_np <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = target_np[in_bin].mean()
                avg_confidence_in_bin = conf_np[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Maximum Calibration Error (MCE)
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (conf_np > bin_lower) & (conf_np <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = target_np[in_bin].mean()
                avg_confidence_in_bin = conf_np[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        # Reliability diagram data
        reliability_data = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (conf_np > bin_lower) & (conf_np <= bin_upper)
            if in_bin.sum() > 0:
                accuracy_in_bin = target_np[in_bin].mean()
                avg_confidence_in_bin = conf_np[in_bin].mean()
                reliability_data.append({
                    'confidence': avg_confidence_in_bin,
                    'accuracy': accuracy_in_bin,
                    'count': in_bin.sum()
                })
        
        metrics = {
            'ece': ece,
            'mce': mce,
            'reliability_data': reliability_data
        }
        
        return metrics
    
    def compute_comprehensive_metrics(self, 
                                    predictions: torch.Tensor,
                                    targets: torch.Tensor,
                                    original_scores: Optional[torch.Tensor] = None,
                                    confidence_scores: Optional[torch.Tensor] = None,
                                    threshold: float = 0.5) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            predictions: Predicted hallucination scores [batch]
            targets: Ground truth labels [batch]
            original_scores: Optional original scores for mitigation evaluation
            confidence_scores: Optional confidence scores for calibration evaluation
            threshold: Threshold for binary classification
            
        Returns:
            Dictionary of comprehensive metrics
        """
        metrics = {}
        
        # Detection metrics
        metrics['detection'] = self.compute_detection_metrics(predictions, targets, threshold)
        
        # Regression metrics
        metrics['regression'] = self.compute_regression_metrics(predictions, targets.float())
        
        # Mitigation metrics (if original scores provided)
        if original_scores is not None:
            metrics['mitigation'] = self.compute_mitigation_metrics(
                original_scores, predictions, targets
            )
        
        # Confidence metrics (if confidence scores provided)
        if confidence_scores is not None:
            metrics['confidence'] = self.compute_confidence_metrics(
                predictions, confidence_scores, targets
            )
        
        return metrics
