"""
Basic usage example for GRPO hallucination mitigation.

This example demonstrates how to use the GRPO pipeline for training
a language model with mechanistic signal integration.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.grpo import GRPOPipeline, GRPOConfig
from src.mechanistic import MechanisticSignalExtractor, HallucinationDetector
from src.evaluation.metrics import HallucinationMetrics


def main():
    """Basic usage example."""
    print("GRPO Hallucination Mitigation - Basic Usage Example")
    
    # Configuration
    model_name = "gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create GRPO configuration
    grpo_config = GRPOConfig(
        model_name=model_name,
        learning_rate=1e-5,
        batch_size=4,
        num_groups=2,
        group_size=2,
        beta=0.1,
        gamma=0.5,
        max_length=128,
        device=device
    )
    
    # Create GRPO pipeline
    print("Creating GRPO pipeline...")
    pipeline = GRPOPipeline(grpo_config)
    pipeline.initialize_model(model)
    
    # Create sample data
    print("Creating sample data...")
    sample_texts = [
        "The capital of France is Paris.",
        "The moon is made of cheese.",
        "Water boils at 100 degrees Celsius.",
        "Unicorns live in the forest."
    ]
    
    # Tokenize sample texts
    inputs = tokenizer(
        sample_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Add dummy labels (0 = not hallucination, 1 = hallucination)
    labels = torch.tensor([0, 1, 0, 1], device=device)  # Sample labels
    inputs['labels'] = labels
    
    print("Sample data prepared:")
    for i, text in enumerate(sample_texts):
        print(f"  {i}: {text} (Label: {labels[i].item()})")
    
    # Extract mechanistic signals
    print("\nExtracting mechanistic signals...")
    signals = pipeline.extract_mechanistic_signals(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask']
    )
    
    print("Extracted signals:")
    for signal_name, signal_values in signals.items():
        print(f"  {signal_name}: {signal_values.mean().item():.4f} ± {signal_values.std().item():.4f}")
    
    # Detect hallucinations
    print("\nDetecting hallucinations...")
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            return_dict=True
        )
        hallucination_scores = pipeline.detect_hallucinations(signals, outputs.logits)
    
    print("Hallucination scores:")
    for i, score in enumerate(hallucination_scores):
        print(f"  Sample {i}: {score.item():.4f}")
    
    # Form groups
    print("\nForming groups...")
    groups = pipeline.form_groups(inputs, hallucination_scores)
    
    print(f"Formed {len(groups)} groups:")
    for i, group in enumerate(groups):
        group_size = len(group['input_ids'])
        avg_score = group['hallucination_scores'].mean().item()
        print(f"  Group {i}: {group_size} samples, avg hallucination score: {avg_score:.4f}")
    
    # Training step
    print("\nPerforming training step...")
    model.train()
    metrics = pipeline.train_step(inputs)
    
    print("Training metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Evaluation
    print("\nRunning evaluation...")
    model.eval()
    eval_metrics = pipeline.evaluate([inputs])  # Single batch evaluation
    
    print("Evaluation metrics:")
    for metric_name, metric_value in eval_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Compute comprehensive metrics
    print("\nComputing comprehensive metrics...")
    metrics_calculator = HallucinationMetrics()
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            return_dict=True
        )
        signals = pipeline.extract_mechanistic_signals(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        predictions = pipeline.detect_hallucinations(signals, outputs.logits)
    
    comprehensive_metrics = metrics_calculator.compute_comprehensive_metrics(
        predictions=predictions,
        targets=labels,
        threshold=0.5
    )
    
    print("Comprehensive metrics:")
    for category, metrics in comprehensive_metrics.items():
        print(f"  {category}:")
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                print(f"    {metric_name}: {metric_value:.4f}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
