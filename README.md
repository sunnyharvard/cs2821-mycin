# MYCIN Expert System – Diagnosis Pipeline

A MYCIN-style expert system for diagnosing bacterial infections and recommending antibiotic treatments, integrated with an evaluation pipeline for medical diagnosis datasets.

## Architecture

This system implements MYCIN using a **hybrid approach**:
- **LLM**: Only answers individual questions about patient data (not rule evaluation)
- **Backend**: Programmatically evaluates rules and performs logical inference

The system includes:
- 37 MYCIN rules for organism identification, infection site determination, and treatment recommendation
- Programmatic rule evaluation engine with certainty factor combination
- Integration with existing evaluation pipeline
- LLM integration for answering questions when data is missing

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the pipeline
```bash
chmod +x script.sh
./script.sh
```

Or run directly:
```bash
python test_set_pipeline.py \
    --patients data/test_patients.csv \
    --evidences data/evidences.json \
    --conditions data/conditions.json \
    --label-col PATHOLOGY \
    --out-dir outputs
```

## Key Files

- **`mycin_rules.py`**: Complete MYCIN rule base (37 rules, 34 questions)
- **`mycin_inference_engine.py`**: Core inference engine (rule evaluation, certainty combination)
- **`mycin_patient_mapper.py`**: Maps patient data to MYCIN parameters
- **`mycin_pipeline_integration.py`**: Integration with evaluation pipeline
- **`test_set_pipeline.py`**: Main evaluation pipeline (updated to use MYCIN)

## Documentation

- **`MYCIN_RULES_README.md`**: Complete documentation of MYCIN rules
- **`MYCIN_INTEGRATION_README.md`**: Detailed integration guide

## How It Works

1. **Patient Data Processing**: Evidence codes are mapped to MYCIN parameters
2. **Question Answering**: LLM answers MYCIN questions if data is missing
3. **Rule Evaluation**: Rules are evaluated programmatically (not via LLM)
4. **Inference**: Forward chaining applies all applicable rules
5. **Diagnosis**: Output formatted for evaluation pipeline

## Customizing LLM Integration

To use your own LLM API, modify `mycin_pipeline_integration.py`:

```python
def your_llm_call(prompt: str) -> str:
    import openai
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

Then update `test_set_pipeline.py` to use your function.

## Testing

Run the integration test:
```bash
python test_mycin_integration.py
```