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
python run_full_evaluation.py
```

Or use the main pipeline:
```bash
python test_set_pipeline.py \
    --patients data/test_patients.csv \
    --evidences data/evidences.json \
    --conditions data/conditions.json \
    --label-col PATHOLOGY \
    --out-dir outputs
```

## Key Files

- **`mycin_medical_rules.py`**: 59 MYCIN-style rules for all 49 diseases
- **`mycin_medical_pipeline.py`**: Main pipeline (combines LLM + rules)
- **`mycin_medical_mapper.py`**: Maps patient evidence to MYCIN parameters
- **`mycin_inference_engine.py`**: Core inference engine (backward chaining)
- **`test_set_pipeline.py`**: Main evaluation pipeline
- **`evaluation.py`**: Evaluation metrics (KL divergence, cross-entropy, etc.)
- **`run_full_evaluation.py`**: Full evaluation script

## Output Format

Predictions are saved in `results/mycin_medical_differentials.jsonl`:
```json
{"row_index": 0, "differential_probs": {"GERD": 0.67, "Boerhaave": 0.17, ...}}
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
