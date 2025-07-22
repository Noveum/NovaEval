# Usage Guide: Optimized SDK Comparison

## Prerequisites

```bash
pip install deepeval novaeval datasets openai
export OPENAI_API_KEY="your_openai_api_key"
```

## Running NovaEval (Optimized)

```bash
python scripts/novaeval_evaluator_fixed.py
```

**Expected Output**:
- Accuracy: ~90%+
- Time: ~25 seconds for 50 samples
- Results saved to: `novaeval_evaluator_results.json`

## Running DeepEval (Optimized)

```bash
python scripts/deepeval_final_comparison.py
```

**Expected Output**:
- Accuracy: 92.0%
- Time: ~118 seconds for 50 samples
- Results saved to: `deepeval_final_results.json`

## Key Configuration Differences

### NovaEval Configuration
```python
# Uses native MMLU integration
dataset = MMLUDataset(subset="elementary_mathematics", num_samples=50, split="test")
model = OpenAIModel(model_name="gpt-4.1-mini", temperature=0.0, max_tokens=1000)
scorer = AccuracyScorer()
evaluator = Evaluator(dataset=dataset, models=[model], scorers=[scorer])
results = evaluator.run()  # Proper SDK pattern
```

### DeepEval Configuration (Optimized)
```python
# Uses optimized GEval configuration
correctness_metric = GEval(
    name="MMLU_Correctness",
    criteria="Focus on final answer choice (A, B, C, D), not explanation format",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    model=model,
    threshold=0.5,      # Lower threshold
    strict_mode=False   # Non-strict mode
)
results = eval_dataset.evaluate(metrics=[correctness_metric])  # Proper SDK pattern
```

## Customization Options

### Changing Sample Size
Both scripts can be modified to use different sample sizes:
```python
# Change this line in both scripts
samples = dataset.select(range(100))  # For 100 samples instead of 50
```

### Using Different MMLU Subjects
```python
# NovaEval
dataset = MMLUDataset(subset="high_school_mathematics", num_samples=50, split="test")

# DeepEval
dataset = load_dataset("cais/mmlu", "high_school_mathematics", split="test")
```

### Different Models
```python
# Both frameworks support different models
model = OpenAIModel(model_name="gpt-4", ...)  # NovaEval
model = GPTModel(model="gpt-4", ...)          # DeepEval
```

## Troubleshooting

### Common Issues

1. **API Key Error**:
   ```bash
   export OPENAI_API_KEY="sk-your_actual_key_here"
   ```

2. **Import Errors**:
   ```bash
   pip install --upgrade deepeval novaeval
   ```

3. **Slow Performance**:
   - Reduce sample size for testing
   - Check internet connection
   - Verify API rate limits

### Performance Expectations

| Samples | NovaEval Time | DeepEval Time |
|---------|---------------|---------------|
| 10      | ~5s          | ~24s          |
| 50      | ~25s         | ~118s         |
| 100     | ~50s         | ~236s         |

## Results Analysis

Both scripts generate detailed JSON results that can be analyzed:

```python
import json

# Load results
with open('results/novaeval_evaluator_results.json', 'r') as f:
    nova_results = json.load(f)

with open('results/deepeval_final_results.json', 'r') as f:
    deep_results = json.load(f)

# Compare performance
print(f"NovaEval time: {nova_results['evaluation_time']:.2f}s")
print(f"DeepEval time: {deep_results['evaluation_time']:.2f}s")
```
