#!/usr/bin/env python3
"""
Training script for GRPO hallucination mitigation.

This script demonstrates how to train a model using the GRPO pipeline
with mechanistic signal integration for hallucination mitigation.
"""

import argparse
import logging
import yaml
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.grpo import GRPOPipeline, GRPOConfig
from src.utils.data_utils import DataProcessor, Tokenizer
from src.evaluation.metrics import HallucinationMetrics


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model_and_tokenizer(model_name: str, device: str):
    """Create model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train GRPO model for hallucination mitigation")
    parser.add_argument("--config", type=str, default="config/default_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="models",
                       help="Path to output directory")
    parser.add_argument("--log_level", type=str, default="INFO",
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = config['model']['device']
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Create model and tokenizer
    model_name = config['model']['name']
    logger.info(f"Loading model: {model_name}")
    model, tokenizer = create_model_and_tokenizer(model_name, device)
    
    # Create GRPO configuration
    grpo_config = GRPOConfig(
        model_name=model_name,
        learning_rate=config['grpo']['learning_rate'],
        batch_size=config['grpo']['batch_size'],
        num_groups=config['grpo']['num_groups'],
        group_size=config['grpo']['group_size'],
        beta=config['grpo']['beta'],
        gamma=config['grpo']['gamma'],
        max_length=config['model']['max_length'],
        device=device
    )
    
    # Create GRPO pipeline
    pipeline = GRPOPipeline(grpo_config)
    pipeline.initialize_model(model)
    
    # Create data processor
    tokenizer_wrapper = Tokenizer(model_name, config['model']['max_length'])
    data_processor = DataProcessor(tokenizer_wrapper)
    
    # Load data
    data_dir = Path(args.data_dir)
    train_data = data_processor.load_json_data(str(data_dir / "train.json"))
    val_data = data_processor.load_json_data(str(data_dir / "val.json"))
    
    # Create data loaders
    train_loader = data_processor.create_data_loader(
        train_data, 
        batch_size=config['grpo']['batch_size'],
        shuffle=True
    )
    val_loader = data_processor.create_data_loader(
        val_data,
        batch_size=config['grpo']['batch_size'],
        shuffle=False
    )
    
    # Create metrics calculator
    metrics_calculator = HallucinationMetrics()
    
    # Training loop
    num_epochs = config['grpo']['num_epochs']
    logger.info(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        model.train()
        total_loss = 0.0
        total_hallucination_score = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Training step
            metrics = pipeline.train_step(batch)
            
            total_loss += metrics['loss']
            total_hallucination_score += metrics['avg_hallucination_score']
            num_batches += 1
            
            # Log progress
            if batch_idx % config['training']['logging_steps'] == 0:
                logger.info(f"Batch {batch_idx}: Loss = {metrics['loss']:.4f}, "
                           f"Hallucination Score = {metrics['avg_hallucination_score']:.4f}")
        
        # Epoch summary
        avg_loss = total_loss / num_batches
        avg_hallucination_score = total_hallucination_score / num_batches
        
        logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_loss:.4f}, "
                   f"Avg Hallucination Score: {avg_hallucination_score:.4f}")
        
        # Validation phase
        if epoch % config['training']['eval_steps'] == 0:
            logger.info("Running validation...")
            val_metrics = pipeline.evaluate(val_loader)
            
            logger.info(f"Validation - Loss: {val_metrics['eval_loss']:.4f}, "
                       f"Hallucination Score: {val_metrics['eval_hallucination_score']:.4f}")
        
        # Save checkpoint
        if epoch % config['training']['save_steps'] == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': pipeline.optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_model_path = output_dir / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer': tokenizer
    }, final_model_path)
    logger.info(f"Saved final model to {final_model_path}")


if __name__ == "__main__":
    main()
