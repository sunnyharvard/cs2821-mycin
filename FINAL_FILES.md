# Final MYCIN Medical Diagnosis System Files

## Core System Files
- `mycin_medical_pipeline.py` - Main pipeline (combines LLM + MYCIN rules)
- `mycin_medical_rules.py` - 59 MYCIN-style rules for all 49 diseases
- `mycin_medical_mapper.py` - Maps patient evidence to MYCIN parameters
- `mycin_inference_engine.py` - Core inference engine (backward chaining)

## Evaluation & Testing
- `test_set_pipeline.py` - Main evaluation pipeline
- `run_full_evaluation.py` - Full evaluation script
- `evaluation.py` - Evaluation metrics

## Baseline Comparison
- `oneshot_llm.py` - LLM-only baseline for comparison

## Data Files
- `data_extraction/` - Disease list, ground truth, questions
- `outputs/` - Patient payloads
- `results/` - Predictions and evaluation results

## Documentation
- `README.md` - Updated documentation
- `requirements.txt` - Dependencies
- `script.sh` - Optional runner script

## Removed Files
- Old organism-based MYCIN files (mycin_rules.py, mycin_pipeline_integration.py, etc.)
- Test files (test_mycin_*.py, test_medical_mycin.py)
- Outdated documentation (PROPOSAL_ALIGNMENT.md, QUESTION_MAPPING_README.md, SYSTEM_FLOW.md)
