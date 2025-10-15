"""
Model utilities for loading, saving, and managing models.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
from pathlib import Path
import json
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelManager:
    """
    Manager for model operations including loading, saving, and checkpointing.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize model manager.
        
        Args:
            model_dir: Directory for storing models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def save_model(self, 
                  model: nn.Module,
                  tokenizer,
                  model_name: str,
                  metadata: Optional[Dict[str, Any]] = None):
        """
        Save model and tokenizer.
        
        Args:
            model: Model to save
            tokenizer: Tokenizer to save
            model_name: Name for the saved model
            metadata: Optional metadata to save
        """
        model_path = self.model_dir / f"{model_name}.pt"
        tokenizer_path = self.model_dir / f"{model_name}_tokenizer"
        metadata_path = self.model_dir / f"{model_name}_metadata.json"
        
        # Save model state dict
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'model_config': getattr(model, 'config', None)
        }, model_path)
        
        # Save tokenizer
        tokenizer.save_pretrained(tokenizer_path)
        
        # Save metadata
        if metadata:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved model to {model_path}")
        self.logger.info(f"Saved tokenizer to {tokenizer_path}")
    
    def load_model(self, 
                  model_name: str,
                  model_class: Optional[nn.Module] = None) -> tuple:
        """
        Load model and tokenizer.
        
        Args:
            model_name: Name of the saved model
            model_class: Optional model class to instantiate
            
        Returns:
            Tuple of (model, tokenizer, metadata)
        """
        model_path = self.model_dir / f"{model_name}.pt"
        tokenizer_path = self.model_dir / f"{model_name}_tokenizer"
        metadata_path = self.model_dir / f"{model_name}_metadata.json"
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if model_class is None:
            # Try to load from transformers
            model_config = checkpoint.get('model_config')
            if model_config and hasattr(model_config, 'model_type'):
                model_class = AutoModelForCausalLM
            else:
                raise ValueError("Must provide model_class if not using transformers model")
        
        model = model_class.from_pretrained(
            model_name if isinstance(model_name, str) and '/' in model_name else "gpt2",
            state_dict=checkpoint['model_state_dict']
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Load metadata
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        self.logger.info(f"Loaded model from {model_path}")
        return model, tokenizer, metadata


class CheckpointManager:
    """
    Manager for training checkpoints.
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for storing checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(self,
                       epoch: int,
                       model: nn.Module,
                       optimizer,
                       loss: float,
                       metrics: Dict[str, float],
                       is_best: bool = False,
                       checkpoint_name: Optional[str] = None):
        """
        Save training checkpoint.
        
        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer to save
            loss: Current loss
            metrics: Current metrics
            is_best: Whether this is the best checkpoint
            checkpoint_name: Optional custom checkpoint name
        """
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_epoch_{epoch}"
        
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pt"
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint_data, best_path)
            self.logger.info(f"Saved best checkpoint to {best_path}")
        
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self,
                       checkpoint_name: str = "best_checkpoint") -> Dict[str, Any]:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint to load
            
        Returns:
            Checkpoint data
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pt"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint_data
    
    def list_checkpoints(self) -> list:
        """
        List available checkpoints.
        
        Returns:
            List of checkpoint names
        """
        checkpoints = []
        for checkpoint_file in self.checkpoint_dir.glob("*.pt"):
            checkpoints.append(checkpoint_file.stem)
        
        return sorted(checkpoints)
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """
        Clean up old checkpoints, keeping only the last N.
        
        Args:
            keep_last_n: Number of checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        
        # Remove non-best checkpoints
        checkpoint_files = [cp for cp in checkpoints if cp.startswith("checkpoint_epoch_")]
        checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split("_")[-1]))
        
        # Remove old checkpoints
        for checkpoint_file in checkpoint_files[:-keep_last_n]:
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_file}.pt"
            checkpoint_path.unlink()
            self.logger.info(f"Removed old checkpoint: {checkpoint_path}")
