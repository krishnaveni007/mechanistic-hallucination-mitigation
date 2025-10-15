"""
Logging utilities for experiment tracking and monitoring.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
import wandb
import torch


def setup_logger(name: str, 
                 level: str = "INFO",
                 log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class ExperimentTracker:
    """
    Track experiments using wandb and local logging.
    """
    
    def __init__(self, 
                 project_name: str,
                 experiment_name: Optional[str] = None,
                 use_wandb: bool = False,
                 log_dir: str = "logs"):
        """
        Initialize experiment tracker.
        
        Args:
            project_name: Name of the project
            experiment_name: Name of the experiment
            use_wandb: Whether to use wandb logging
            log_dir: Directory for local logs
        """
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project=project_name,
                name=experiment_name,
                config={}
            )
        
        # Setup local logging
        self.logger = setup_logger(
            name="experiment",
            log_file=str(self.log_dir / f"{experiment_name or 'experiment'}.log")
        )
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to wandb and local logger.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        # Log to wandb
        if self.use_wandb:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
        
        # Log to local logger
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Metrics - {metrics_str}")
    
    def log_config(self, config: Dict[str, Any]):
        """
        Log configuration.
        
        Args:
            config: Configuration dictionary
        """
        if self.use_wandb:
            wandb.config.update(config)
        
        self.logger.info(f"Configuration: {config}")
    
    def log_model(self, model: torch.nn.Module, model_name: str = "model"):
        """
        Log model architecture.
        
        Args:
            model: Model to log
            model_name: Name for the model
        """
        if self.use_wandb:
            wandb.watch(model, log="all", log_freq=100)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"{model_name} - Total params: {total_params:,}, "
                        f"Trainable params: {trainable_params:,}")
    
    def log_artifact(self, file_path: str, artifact_type: str = "model"):
        """
        Log artifact to wandb.
        
        Args:
            file_path: Path to the artifact
            artifact_type: Type of artifact
        """
        if self.use_wandb:
            artifact = wandb.Artifact(
                name=f"{self.experiment_name}_{artifact_type}",
                type=artifact_type
            )
            artifact.add_file(file_path)
            wandb.log_artifact(artifact)
        
        self.logger.info(f"Logged artifact: {file_path}")
    
    def finish(self):
        """Finish experiment tracking."""
        if self.use_wandb:
            wandb.finish()
        
        self.logger.info("Experiment tracking finished")
