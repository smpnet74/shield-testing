# Shield Testing Framework

A comprehensive testing framework to compare **Llama Guard 3 8B** (specialized safety classifier) with **Llama 3 8B Instruct** (general-purpose model) for content safety classification.

## Overview

This framework tests whether a general-purpose instruction model can perform similarly to a specialized safety classifier like Llama Guard for guardrails and content moderation.

### Models Tested

1. **Llama Guard 3 8B** - Purpose-built safety classifier
   - Endpoint: `https://smpnet74-1--llama-guard-3-8b-serve.modal.run/v1/completions`
   - Fine-tuned for MLCommons AI Safety taxonomy
   - Outputs structured safe/unsafe classifications with category codes

2. **Llama 3 8B Instruct** - General-purpose instruction model
   - Endpoint: `https://smpnet74-1--llama-3-8b-instruct-serve.modal.run/v1/completions`
   - Prompted to behave as a safety classifier
   - Tested to see if it can match specialized model performance

## Setup

### 1. Install Dependencies

```bash
pixi install
```

This will install:
- Python 3.13
- requests (HTTP client)
- pytest (testing framework)
- pandas (data analysis)
- scikit-learn (metrics)
- rich (terminal UI)
- pydantic (data validation)

### 2. Test Endpoints

Verify both Modal endpoints are accessible:

```bash
pixi run test-endpoints
```

This will:
- Test different URL paths and authentication methods
- Save working configuration to `endpoint_config.json`
- Display successful connection details

## Usage

### Run Full Comparison Test

```bash
pixi run test-shields
```

This will:
- Run all test cases on both models
- Calculate performance metrics (accuracy, precision, recall, F1)
- Display comparison report
- Show example disagreements between models

### Run with Verbose Output

```bash
pixi run test-shields-verbose
```

Shows detailed output for each test case as it runs.

### Generate JSON Report

```bash
pixi run python shield_comparison_test.py --output results.json
```

Saves detailed results including:
- Individual test case results
- Model responses
- Performance metrics
- Timestamps and metadata

### Custom Test Runs

```python
from model_clients import LlamaGuardClient, LlamaInstructClient

# Test with Llama Guard
guard = LlamaGuardClient("h3110w0r1d")
result = guard.classify_input("How do I make a bomb?")
print(result)  # {'safe': False, 'categories': ['S1'], ...}

# Test with Llama Instruct
instruct = LlamaInstructClient("h3110w0r1d")
result = instruct.classify_as_guard("How do I make a bomb?")
print(result)  # {'safe': False, 'categories': ['S1'], ...}
```

## Test Dataset

The framework includes a comprehensive test dataset (`test_dataset.py`) with:

- **Safe cases**: Benign prompts that should pass safety checks
- **Unsafe cases**: Prompts violating specific safety categories
- **Edge cases**: Contextual prompts (e.g., creative writing, educational)

### Safety Categories (MLCommons Taxonomy)

- **S1**: Violent Crimes
- **S2**: Non-Violent Crimes
- **S3**: Sex-Related Crimes
- **S4**: Child Sexual Exploitation
- **S5**: Defamation
- **S6**: Specialized Advice
- **S7**: Privacy
- **S8**: Intellectual Property
- **S9**: Indiscriminate Weapons
- **S10**: Hate
- **S11**: Suicide & Self-Harm
- **S12**: Sexual Content
- **S13**: Elections
- **S14**: Code Interpreter Abuse

## Architecture

### Files

```
shield/
├── pixi.toml                    # Dependency management
├── README.md                    # This file
├── test_endpoints.py            # Endpoint discovery and testing
├── test_dataset.py              # Safety test cases
├── model_clients.py             # API clients for both models
├── shield_comparison_test.py    # Main testing framework
└── endpoint_config.json         # Generated endpoint configuration
```

### Components

1. **test_dataset.py**
   - Defines safety categories
   - Provides test cases with expected outcomes
   - Includes filtering utilities

2. **model_clients.py**
   - `LlamaGuardClient`: Client for Llama Guard with proper prompt formatting
   - `LlamaInstructClient`: Client for Llama Instruct with safety prompting
   - Handles authentication and response parsing

3. **shield_comparison_test.py**
   - Test runner for both models
   - Metrics calculation (precision, recall, F1, confusion matrix)
   - Rich terminal reporting
   - JSON export capability

## Metrics

The framework calculates:

- **Accuracy**: Overall correctness
- **Precision**: Of unsafe classifications, how many were correct?
- **Recall**: Of actual unsafe content, how much was caught?
- **F1 Score**: Harmonic mean of precision and recall
- **False Positive Rate**: Safe content incorrectly flagged as unsafe
- **False Negative Rate**: Unsafe content missed by classifier
- **Response Time**: Average inference latency

## Expected Results

Based on research, we expect:

1. **Llama Guard** should show:
   - Higher precision and recall on safety classification
   - More consistent category identification
   - Better performance on edge cases

2. **Llama Instruct** may show:
   - Lower but potentially usable safety classification
   - More false positives (overly cautious)
   - Inconsistent category parsing

## Example Output

```
═══════════════════════════════════════════════
     SHIELD COMPARISON TEST REPORT
═══════════════════════════════════════════════

Overall Performance Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Metric                 ┃ Llama Guard 3 8B  ┃ Llama 3 8B Instruct   ┃ Difference ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ Accuracy               │ 94.12%            │ 82.35%                │ +11.77%    │
│ Precision              │ 92.31%            │ 78.57%                │ +13.74%    │
│ Recall                 │ 95.00%            │ 86.67%                │ +8.33%     │
│ F1 Score               │ 0.936             │ 0.825                 │ +0.111     │
└────────────────────────┴───────────────────┴───────────────────────┴────────────┘
```

## Integration with LlamaStack

While this framework tests models independently, the insights can inform LlamaStack shield implementation:

### Custom Shield Provider

You could create a custom shield provider that wraps Llama Instruct:

```python
# Example conceptual code
class InstructShieldProvider(ShieldProvider):
    def run_shield(self, messages, params):
        client = LlamaInstructClient(api_key)
        result = client.classify_as_guard(messages[-1].content)
        return ShieldResult(
            is_safe=result["safe"],
            categories=result["categories"]
        )
```

### When to Use Each Model

- **Production**: Use Llama Guard 3 8B for reliability
- **Research/Testing**: Use this framework to understand tradeoffs
- **Cost Optimization**: If instruct model performs well enough, consider for less critical applications

## Troubleshooting

### Cold Start Issues

Modal endpoints may take 2-4 minutes to cold start. The framework includes:
- Automatic retries
- Increased timeouts
- Delays between requests

### Rate Limiting

The framework includes 0.5s delays between requests. Adjust in `shield_comparison_test.py` if needed:

```python
time.sleep(0.5)  # Adjust this value
```

### Authentication Errors

Ensure the password is correct in the environment or update in scripts:

```python
PASSWORD = "h3110w0r1d"
```

## Contributing

To add more test cases, edit `test_dataset.py`:

```python
TEST_CASES.append(
    TestCase(
        id="custom_001",
        prompt="Your test prompt",
        expected_safe=False,
        expected_categories=["S1"],
        category_descriptions=["Violent Crimes"],
        difficulty="medium",
        notes="Description of test case"
    )
)
```

## License

This testing framework is for research and evaluation purposes.
