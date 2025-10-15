# Data Directory

This directory contains sample datasets for the mechanistic hallucination mitigation project.

## Sample Data

- `sample_train.json`: Training data with 20 examples
- `sample_val.json`: Validation data with 10 examples  
- `sample_test.json`: Test data with 10 examples

## Data Format

Each JSON file contains a list of objects with the following structure:

```json
{
  "text": "The statement to evaluate",
  "hallucination_label": 0  // 0 = not hallucination, 1 = hallucination
}
```

## Labels

- `0`: Factual/true statements
- `1`: Hallucinated/false statements

## Usage

To use your own data, replace the sample files with your datasets in the same JSON format, or update the configuration to point to your data files.

## Data Preparation

For training on your own data:

1. Prepare your data in the JSON format shown above
2. Ensure balanced representation of hallucination and non-hallucination examples
3. Update the file paths in your configuration
4. Consider data augmentation techniques for small datasets
