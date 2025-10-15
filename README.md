# Mechanistic Hallucination Mitigation

Mitigating LLM hallucinations by leveraging mechanistic signals in GRPO (Group Relative Policy Optimization) pipelines.

## Overview

This project implements a novel approach to mitigating hallucinations in large language models by:

1. **Extracting mechanistic signals** from internal model computations (attention patterns, activations, confidence scores)
2. **Detecting potential hallucinations** using these signals
3. **Applying GRPO optimization** with group-based training that incorporates mechanistic feedback
4. **Evaluating mitigation effectiveness** using comprehensive metrics

## Key Features

- **Mechanistic Signal Extraction**: Analyzes attention entropy, activation magnitudes, confidence variance, and layer consistency
- **Hallucination Detection**: Multiple detection strategies including threshold-based, ML-based, and ensemble methods
- **GRPO Pipeline**: Group Relative Policy Optimization with mechanistic signal integration
- **Comprehensive Evaluation**: Extensive metrics for detection and mitigation effectiveness
- **Modular Design**: Clean, extensible architecture for research and experimentation

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/mechanistic-hallucination-mitigation.git
cd mechanistic-hallucination-mitigation

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

### Basic Usage

```python
from src.grpo import GRPOPipeline, GRPOConfig
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Create GRPO configuration
config = GRPOConfig(
    model_name="gpt2",
    learning_rate=1e-5,
    batch_size=8,
    num_groups=4,
    gamma=0.5  # Mechanistic signal weight
)

# Initialize pipeline
pipeline = GRPOPipeline(config)
pipeline.initialize_model(model)

# Train with mechanistic signals
metrics = pipeline.train_step(batch)
```

### Training Script

```bash
# Train with default configuration
python scripts/train.py --config config/default_config.yaml

# Train with custom data directory
python scripts/train.py --data_dir /path/to/data --output_dir /path/to/models
```

### Example Usage

See `examples/basic_usage.py` for a complete example demonstrating:
- Mechanistic signal extraction
- Hallucination detection
- Group formation
- Training and evaluation

## Architecture

### Core Components

1. **GRPO Pipeline** (`src/grpo/`)
   - `pipeline.py`: Main GRPO training pipeline
   - `optimizer.py`: GRPO optimizer with mechanistic integration
   - `group_selector.py`: Group formation strategies

2. **Mechanistic Analysis** (`src/mechanistic/`)
   - `signal_extractor.py`: Extract mechanistic signals
   - `hallucination_detector.py`: Detect hallucinations
   - `attention_analyzer.py`: Analyze attention patterns
   - `activation_tracker.py`: Track activation patterns

3. **Evaluation** (`src/evaluation/`)
   - `metrics.py`: Comprehensive evaluation metrics
   - `benchmark.py`: Benchmarking framework
   - `evaluator.py`: Model evaluation

4. **Utilities** (`src/utils/`)
   - `data_utils.py`: Data processing and tokenization
   - `model_utils.py`: Model management utilities
   - `logging.py`: Logging and experiment tracking

### Mechanistic Signals

The framework extracts several types of mechanistic signals:

- **Attention Entropy**: Measures uncertainty in attention patterns
- **Activation Magnitude**: Tracks unusual activation patterns
- **Confidence Variance**: Monitors prediction confidence consistency
- **Layer Consistency**: Measures coherence across model layers
- **Token Surprisal**: Computes unexpectedness of generated tokens

### GRPO Algorithm

The Group Relative Policy Optimization algorithm:

1. **Signal Extraction**: Extract mechanistic signals from model forward pass
2. **Hallucination Detection**: Identify potential hallucinations using signals
3. **Group Formation**: Group samples based on mechanistic characteristics
4. **Group-Relative Loss**: Optimize with group-specific objectives
5. **Mechanistic Penalty**: Apply penalties based on detected patterns

## Configuration

Configuration is managed through YAML files. See `config/default_config.yaml` for all available options:

```yaml
# Model configuration
model:
  name: "gpt2"
  max_length: 512

# GRPO configuration
grpo:
  learning_rate: 1e-5
  batch_size: 8
  num_groups: 4
  beta: 0.1  # KL regularization
  gamma: 0.5  # Mechanistic signal weight

# Detection configuration
detection:
  method: "threshold_based"
  threshold: 0.5
```

## Data Format

The framework expects data in JSON format:

```json
[
  {
    "text": "The capital of France is Paris.",
    "hallucination_label": 0
  },
  {
    "text": "The moon is made of green cheese.",
    "hallucination_label": 1
  }
]
```

Required fields:
- `text`: Input text
- `hallucination_label`: Binary label (0=not hallucination, 1=hallucination)

## Evaluation Metrics

The framework provides comprehensive evaluation metrics:

### Detection Metrics
- Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Sensitivity, Specificity

### Regression Metrics
- MSE, RMSE, MAE
- R-squared, Correlation

### Mitigation Metrics
- Average reduction in hallucination scores
- Improvement rate
- False positive rate

### Confidence Metrics
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)

## Research Applications

This framework enables research into:

- **Mechanistic Interpretability**: Understanding how LLMs generate hallucinations
- **Training Interventions**: Improving model training with mechanistic feedback
- **Evaluation Methods**: Developing better hallucination detection metrics
- **Model Architectures**: Designing models with built-in hallucination resistance

## Contributing

We welcome contributions! Please see our contributing guidelines for details on:

- Code style and standards
- Testing requirements
- Documentation standards
- Pull request process

## Citation

If you use this work in your research, please cite:

```bibtex
@article{mechanistic-hallucination-mitigation,
  title={Mitigating LLM Hallucinations by Leveraging Mechanistic Signals in GRPO Pipelines},
  author={Research Team},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Transformers library by Hugging Face
- PyTorch framework
- The broader AI research community

## Contact

For questions or support, please open an issue or contact the research team.