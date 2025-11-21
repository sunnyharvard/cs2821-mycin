# MYCIN System Flow: What Happens When a User Profile is Received

## Overview

When a patient profile is received, the system processes it through several stages to produce a diagnosis using MYCIN's rule-based inference engine.

## Input Format

A patient profile comes in as:
```python
{
    "demographics": {"AGE": 45, "SEX": "M"},
    "evidence": {
        "Do you have a cough?": True,
        "Do you have a fever?": True,
        "gram_stain": "positive",  # May be direct MYCIN parameter
        ...
    }
}
```

## Processing Pipeline

### Step 1: Patient Data Mapping (`mycin_patient_mapper.py`)

**Function**: `map_to_mycin_format()`

**What happens**:
1. **Check for direct MYCIN parameters**: If evidence already contains MYCIN parameter names (like `gram_stain`, `morphology`), they're extracted directly.

2. **Keyword extraction**: For general medical questions, the system uses keyword matching:
   - Searches evidence text for keywords like "fever", "cough", "hospital", etc.
   - Maps them to MYCIN parameters (`fever`, `cough`, `hospital_acquired`, etc.)

3. **LLM extraction (optional)**: If LLM is enabled and parameters are missing:
   - Creates a prompt asking LLM to extract specific MYCIN parameters from patient data
   - LLM returns JSON with extracted values
   - Missing parameters are filled in

**Output**: Dictionary of MYCIN parameters
```python
{
    "age": 45,
    "sex": "M",
    "cough": True,
    "fever": True,
    "gram_stain": "positive",
    "morphology": "coccus",
    ...
}
```

### Step 2: Inference Engine Initialization (`mycin_inference_engine.py`)

**Function**: `MYCINInferenceEngine.__init__()`

**What happens**:
- Creates a new inference engine instance
- Initializes empty fact database (`known_facts`)
- Sets up LLM question-answering function (if provided)

### Step 3: Fact Initialization

**Function**: `engine.set_fact()`

**What happens**:
- All extracted MYCIN parameters are set as initial facts
- Each fact has:
  - Parameter name (e.g., `gram_stain`)
  - Value (e.g., `"positive"`)
  - Certainty (default: 1.0 for direct observations)

**Example facts created**:
```python
Fact(parameter="gram_stain", value="positive", certainty=1.0)
Fact(parameter="cough", value=True, certainty=1.0)
Fact(parameter="fever", value=True, certainty=1.0)
```

### Step 4: Forward Chaining Inference

**Function**: `engine.forward_chain()`

**What happens** (iteratively):

#### 4a. Rule Evaluation Loop
For each of the 37 MYCIN rules:

1. **Check if rule already evaluated**: Skip if already processed

2. **Evaluate each condition** (`evaluate_condition()`):
   - For each condition in the rule's IF clause:
     - Check if we have the fact in `known_facts`
     - If not, check `patient_data` directly
     - If still not found, **ask LLM** to answer the question
     - Compare actual value with expected value using operator (`is`, `greater_than`, etc.)
     - Return (satisfied: True/False, certainty: float)

3. **Check if all conditions met**:
   - If ANY condition fails → rule doesn't fire
   - If ALL conditions satisfied → proceed to apply rule

4. **Apply rule** (`apply_rule()`):
   - Calculate conclusion certainty: `rule_certainty_factor × min(condition_certainties)`
   - For each conclusion in the rule's THEN clause:
     - Set new fact with calculated certainty
     - If fact already exists with same value → combine certainties
     - If fact exists with different value → keep higher certainty

5. **Mark rule as evaluated**

#### 4b. Iteration
- Process all rules
- If new facts were created, iterate again (up to 10 times)
- This allows rules to build on each other

**Example rule application**:
```
Rule R037:
  IF gram_stain is positive AND morphology is coccus AND growth_pattern is chains
  THEN identity = streptococcus (CF=0.7)

Evaluation:
  - gram_stain = "positive" ✓ (certainty: 1.0)
  - morphology = "coccus" ✓ (certainty: 1.0)
  - growth_pattern = "chains" → Not in facts, ask LLM
    → LLM: "How does the organism grow?" → "chains" (certainty: 0.9)
  
  All conditions met! Min certainty = 0.9
  Conclusion CF = 0.7 × 0.9 = 0.63
  
  New fact created:
    Fact(parameter="identity", value="streptococcus", certainty=0.63, source_rules=["R037"])
```

### Step 5: Treatment Rules

After organism identification, treatment rules fire:

```
Rule R200:
  IF identity is streptococcus AND penicillin_sensitive is True AND allergy_penicillin is False
  THEN recommended_drug = penicillin (CF=0.8)

Evaluation:
  - identity = "streptococcus" ✓ (from previous rule)
  - penicillin_sensitive → Not in facts, ask LLM
    → LLM: "Is the organism sensitive to penicillin?" → "yes" (certainty: 0.8)
  - allergy_penicillin → Not in facts, ask LLM
    → LLM: "Is the patient allergic to penicillin?" → "no" (certainty: 1.0)
  
  All conditions met! Min certainty = 0.8
  Conclusion CF = 0.8 × 0.8 = 0.64
  
  New fact:
    Fact(parameter="recommended_drug", value="penicillin", certainty=0.64, source_rules=["R200"])
```

### Step 6: Diagnosis Aggregation

**Function**: `engine.get_diagnosis()`

**What happens**:
1. **Collect all organism identities**: Gather all facts with `parameter="identity"`
2. **Collect all infection sites**: Gather all facts with `parameter="infection_site"`
3. **Collect all treatments**: Gather all facts with `parameter="recommended_drug"`

4. **Select top predictions**:
   - Organism: Highest certainty
   - Infection site: Highest certainty
   - Treatment: Highest certainty

5. **Convert to probabilities**:
   - Normalize certainty factors to probabilities (sum to 1.0)
   - Used for evaluation metrics

**Output structure**:
```python
{
    "organism_identity": {
        "name": "streptococcus",
        "certainty": 0.63,
        "all_organisms": {"streptococcus": 0.63, "staphylococcus": 0.45},
        "probabilities": {"streptococcus": 0.58, "staphylococcus": 0.42}
    },
    "infection_site": {...},
    "treatment": {
        "recommended_drug": "penicillin",
        "details": {"certainty": 0.64, "dose": "standard"},
        "all_treatments": {...}
    }
}
```

### Step 7: Format for Evaluation

**Function**: `run_mycin_pipeline()`

**What happens**:
- Extracts top organism as `predicted_label`
- Converts organism certainties to probability distribution
- Formats output for evaluation pipeline

**Final output**:
```python
{
    "predicted_label": "streptococcus",
    "probs": {
        "streptococcus": 0.58,
        "staphylococcus": 0.42
    },
    "diagnosis_details": {
        "organism": {...},
        "infection_site": {...},
        "treatment": {...},
        "applied_rules": ["R037", "R060", "R200"]
    }
}
```

## Key Design Principles

### 1. LLM Only Answers Questions
- LLM is **never** given rules to evaluate
- LLM is only asked: "What is the gram stain?" or "Is the patient allergic to penicillin?"
- Rule evaluation is **100% programmatic**

### 2. Certainty Factor Combination
When multiple rules support the same conclusion:
```
CF_combined = CF1 + CF2 × (1 - |CF1|)
```
This allows evidence to accumulate.

### 3. Forward Chaining
- Rules fire in any order
- New facts can trigger more rules
- Iterates until no new rules fire

### 4. Missing Data Handling
- If a rule needs a parameter that's missing:
  1. Check known facts
  2. Check patient data
  3. Ask LLM (if available)
  4. If still unknown, rule doesn't fire

## Example Complete Flow

**Input**:
```python
{
    "demographics": {"AGE": 45, "SEX": "M"},
    "evidence": {
        "Do you have a cough?": True,
        "Do you have a fever?": True,
        "gram_stain": "positive",
        "morphology": "coccus"
    }
}
```

**Processing**:
1. Mapper extracts: `{age: 45, cough: True, fever: True, gram_stain: "positive", morphology: "coccus"}`
2. Facts initialized: All above parameters set as facts
3. Rule R037 evaluated:
   - Needs: `gram_stain`, `morphology`, `growth_pattern`
   - Has: `gram_stain`, `morphology` ✓
   - Missing: `growth_pattern` → Ask LLM → "chains"
   - All conditions met → Fire rule → `identity = streptococcus` (CF=0.63)
4. Rule R060 evaluated:
   - Needs: `gram_stain`, `morphology`, `culture_site`, `cough`, `fever`
   - Has all ✓ → Fire → `identity = streptococcus_pneumoniae` (CF=0.56)
5. Certainty combination: `streptococcus` CF=0.63 + `streptococcus_pneumoniae` CF=0.56
6. Treatment rules fire based on organism identity
7. Final diagnosis: Top organism with probabilities

**Output**:
```python
{
    "predicted_label": "streptococcus",
    "probs": {"streptococcus": 0.53, "streptococcus_pneumoniae": 0.47},
    "diagnosis_details": {
        "applied_rules": ["R037", "R060", "R215"]
    }
}
```

## Summary

The system is a **hybrid expert system**:
- **Backend (programmatic)**: Rule evaluation, inference, certainty combination
- **LLM (question answering)**: Only answers individual questions when data is missing
- **No LLM rule evaluation**: Rules are never passed to LLM for evaluation

This ensures the system is:
- **Interpretable**: You can see exactly which rules fired
- **Deterministic**: Same inputs → same rule evaluations
- **Efficient**: LLM only called when needed for missing data
- **Accurate**: Uses proven MYCIN rule structure and certainty factors


