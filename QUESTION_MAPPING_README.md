# Question Mapping Integration

## Overview

The MYCIN system has been updated to work with the actual questions from `question_en_output.txt`. The system now maps actual question text to MYCIN parameters.

## Changes Made

### 1. New Module: `mycin_question_mapping.py`

This module provides:
- **Question-to-Parameter Mapping**: Maps actual question text to MYCIN parameter names
- **Parameter-to-Questions Mapping**: Reverse mapping for reference
- **Question Loading**: Loads all 223 questions from `question_en_output.txt`

**Key Functions**:
- `get_parameter_from_question(question_text)`: Maps question text to MYCIN parameter
- `get_questions_for_parameter(parameter)`: Gets all questions for a parameter
- `load_questions_from_file()`: Loads questions from file

### 2. Updated: `mycin_patient_mapper.py`

Now uses the question mapping to:
- Map actual question text in patient evidence to MYCIN parameters
- Works with questions like "Do you have a fever (either felt or measured with a thermometer)?" → `fever`
- Falls back to keyword matching for unmapped questions

### 3. Updated: `mycin_rules.py`

Updated the `QUESTIONS` dictionary with notes about:
- Which parameters are mapped from actual questions
- Which parameters require LLM inference (like `gram_stain`, `morphology`)
- Which parameters come from demographics

## Question Mappings

### Directly Mapped Questions

These questions map directly to MYCIN parameters:

| Question Text | MYCIN Parameter |
|--------------|----------------|
| "Do you have a fever (either felt or measured with a thermometer)?" | `fever` |
| "Do you have a cough?" | `cough` |
| "Are you experiencing shortness of breath or difficulty breathing in a significant way?" | `dyspnea` |
| "Do you have chest pain even at rest?" | `chest_pain` |
| "Are you immunosuppressed?" | `immunocompromised` |
| "Have you had surgery within the last month?" | `recent_surgery` |
| "Do you have a sore throat?" | `sore_throat` |
| "Have you traveled out of the country in the last 4 weeks?" | `travel_history` |
| ... and more |

### LLM-Inferred Parameters

These MYCIN parameters are not directly available in questions and require LLM inference:

- `gram_stain`: Gram stain results (positive/negative)
- `morphology`: Organism morphology (coccus/rod/spirillum)
- `growth_pattern`: How organism grows (chains/pairs/clusters)
- `culture_site`: Site of culture (blood/urine/sputum/CSF)
- `penicillin_sensitive`: Drug sensitivity
- `allergy_penicillin`: Patient allergies
- ... and other lab-specific parameters

## How It Works

### Example Patient Data

```python
{
    "demographics": {"AGE": 45, "SEX": "M"},
    "evidence": {
        "Do you have a fever (either felt or measured with a thermometer)?": True,
        "Do you have a cough?": True,
        "Are you experiencing shortness of breath or difficulty breathing in a significant way?": True
    }
}
```

### Mapping Process

1. **Question Mapping**: Each question text is looked up in `QUESTION_TO_PARAMETER`
   - "Do you have a fever..." → `fever: True`
   - "Do you have a cough?" → `cough: True`
   - "Are you experiencing shortness of breath..." → `dyspnea: True`

2. **MYCIN Parameter Extraction**: Mapped parameters are extracted
   ```python
   {
       "fever": True,
       "cough": True,
       "dyspnea": True,
       "age": 45,
       "sex": "M"
   }
   ```

3. **Rule Evaluation**: Rules use these parameters
   - Rules that need `gram_stain` or `morphology` will ask LLM if available
   - Rules that have `fever`, `cough`, `dyspnea` can fire directly

## Adding New Mappings

To add new question mappings, edit `mycin_question_mapping.py`:

```python
QUESTION_TO_PARAMETER = {
    # ... existing mappings ...
    "Your new question text?": "mycin_parameter",
}
```

## Limitations

1. **Lab Results**: Many MYCIN parameters (gram_stain, morphology, etc.) are not in the question list. These require LLM inference.

2. **Incomplete Coverage**: Not all 223 questions are mapped. Only questions relevant to MYCIN diagnosis are mapped.

3. **Question Variations**: Some questions may have slight variations. The mapping uses exact match first, then case-insensitive match.

## Testing

Test the mapping:

```python
from mycin_question_mapping import get_parameter_from_question

question = "Do you have a fever (either felt or measured with a thermometer)?"
param = get_parameter_from_question(question)
print(param)  # Output: "fever"
```

Test the full pipeline:

```python
from mycin_pipeline_integration import run_mycin_pipeline

patient = {
    "demographics": {"AGE": 45, "SEX": "M"},
    "evidence": {
        "Do you have a fever (either felt or measured with a thermometer)?": True,
        "Do you have a cough?": True
    }
}

result = run_mycin_pipeline([patient])
```

## Next Steps

1. **Expand Mappings**: Add more question mappings as needed
2. **LLM Integration**: Implement LLM inference for unmapped parameters
3. **Rule Updates**: Update rules to work better with available questions
4. **Validation**: Test with actual patient data from the dataset

