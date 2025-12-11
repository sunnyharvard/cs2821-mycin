# MYCIN Medical Diagnosis System

A MYCIN-style expert system for general medical diagnosis, combining rule-based inference with LLM-powered differential diagnosis.

## Architecture

This system implements MYCIN using a **hybrid approach**:
- **MYCIN Rules**: 59 rules covering all 49 diseases in the evaluation dataset
- **LLM Integration**: GPT-4o provides comprehensive differential diagnosis
- **Combined Prediction**: Rule-based matches boost LLM predictions by 20%

The system:
- Predicts multiple diseases per patient (average ~5-10, matching ground truth)
- Uses backward chaining inference with certainty factors
- Validates all disease names against the allowed list
- Outputs probability distributions over all 49 possible diseases

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API key
```bash
export OPENAI_API_KEY='your-api-key-here'
```

### 3. Run evaluation
```bash
python evaluation/run_full_evaluation.py
```

Or use the main pipeline:
```bash
python tests/test_set_pipeline.py \
    --patients data/test_patients.csv \
    --evidences data/evidences.json \
    --conditions data/conditions.json \
    --label-col PATHOLOGY \
    --out-dir outputs
```

## Directory Structure

```
.
├── src/                     # Core MYCIN system modules
│   ├── mycin_medical_rules.py          # MYCIN-style rules for all diseases
│   ├── mycin_medical_pipeline.py       # Main pipeline (combines LLM + rules)
│   ├── mycin_medical_mapper.py         # Maps patient evidence to MYCIN parameters
│   ├── mycin_inference_engine.py       # Core inference engine (backward chaining)
│   └── oneshot_llm.py                  # LLM-only baseline for comparison
├── evaluation/              # Evaluation scripts and metrics
│   ├── evaluation.py                   # Evaluation metrics (KL divergence, cross-entropy, etc.)
│   ├── run_full_evaluation.py          # Full evaluation script
│   ├── evaluate_explanations.py        # Clinical quality evaluation (LLM judge)
│   └── evaluate_patient_satisfaction.py # Patient satisfaction evaluation (LLM judge)
├── tests/                   # Test and debug scripts
│   ├── test_set_pipeline.py            # Main evaluation pipeline
│   └── test_sample_prompt.py           # Sample prompt testing
├── scripts/                 # Utility scripts
│   └── script.sh                       # Pipeline runner script
├── logs/                    # Execution logs
├── results/                 # Prediction and evaluation results
├── outputs/                 # Intermediate patient data
├── data_extraction/         # Ground truth and data extraction
└── engine/                  # Original MYCIN LISP code
```

## Key Files

### Core System (src/)
- **`mycin_medical_rules.py`**: MYCIN-style rules for all diseases
- **`mycin_medical_pipeline.py`**: Main pipeline (combines LLM + rules)
- **`mycin_medical_mapper.py`**: Maps patient evidence to MYCIN parameters
- **`mycin_inference_engine.py`**: Core inference engine (backward chaining)
- **`oneshot_llm.py`**: LLM-only baseline for comparison

### Evaluation (evaluation/)
- **`evaluation.py`**: Evaluation metrics (KL divergence, cross-entropy, etc.)
- **`run_full_evaluation.py`**: Full evaluation script
- **`evaluate_explanations.py`**: Evaluate explanation quality from clinical perspective
- **`evaluate_patient_satisfaction.py`**: Evaluate explanations from patient's perspective

### Tests (tests/)
- **`test_set_pipeline.py`**: Main evaluation pipeline

## Output Format

### Predictions
Predictions are saved in `results/mycin_medical_differentials.jsonl`:
```json
{"row_index": 0, "differential_probs": {"GERD": 0.67, "Boerhaave": 0.17, ...}}
```

### Explanations
Explanations are saved in CSV format:
- `results/llm_differentials_explanations.csv`: One-shot LLM explanations
- `results/mycin_medical_explanations.csv`: MYCIN explanations with reasoning

### Evaluation Results
- `results/explanation_evaluations.csv`: Clinical quality scores (0-100)
- `results/patient_satisfaction_evaluations.csv`: Patient satisfaction scores (0-100)

### Running Evaluations
```bash
# Evaluate explanation quality (clinical perspective)
python evaluation/evaluate_explanations.py

# Evaluate patient satisfaction
python evaluation/evaluate_patient_satisfaction.py
```

## Performance

- **Average diseases per prediction**: ~5.26 (ground truth: ~9.91)
- **Average KL Divergence**: ~13.09 (lower is better)
- **Average L1 Distance**: ~1.23 (lower is better)

## How It Works

1. **Patient Data Mapping**: Evidence questions mapped to MYCIN parameters
2. **LLM Differential Diagnosis**: GPT-4o generates comprehensive differential (up to 10 diseases)
3. **MYCIN Rule Evaluation**: Rules fire based on patient parameters
4. **Combination**: Diseases matching rules get 20% probability boost
5. **Output**: Normalized probability distribution over all diseases
