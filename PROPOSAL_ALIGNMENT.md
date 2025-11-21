# Alignment with Project Proposal

## Proposal Requirements

The proposal describes a hybrid architecture where:

1. **LLM as Knowledge Generator**: LLM "suggests candidate facts, explanations, or inferences along with weighted confidences"
2. **Constraint Layer**: "Validates and prunes these outputs to ensure that all required conditions are met"
3. **Certainty Translation**: "Translating probabilistic or confidence-based language model responses into structured certainty weights"
4. **Soft Reasoning Component**: LLM functions as a "soft reasoning component" that suggests facts/inferences
5. **Formal Logical Syntax**: "Modeling the LLM's output in formal logical syntax"

## Current Implementation

### ✅ What Aligns

1. **Hybrid Architecture**: ✅ We have LLM + symbolic rule layer
2. **Certainty Factors**: ✅ We use MYCIN-style certainty factors
3. **Symbolic Rule Evaluation**: ✅ Rules are evaluated programmatically
4. **Interpretability**: ✅ Rules are explicit and traceable
5. **Modularity**: ✅ LLM and rule engine are separate components

### ❌ What's Missing (Key Gaps)

1. **LLM as Knowledge Generator**: ❌ Currently, LLM is **passive** - it only answers questions when asked. The proposal wants LLM to **proactively generate candidate facts/inferences**.

2. **Constraint Validation/Pruning**: ❌ We don't have a separate "constraint layer" that validates and prunes LLM outputs. Rules just evaluate conditions.

3. **LLM-Generated Inferences**: ❌ LLM doesn't generate inferences or explanations - it just answers specific questions.

4. **Structured Certainty Translation**: ❌ We don't extract confidence from LLM responses systematically. We just use 1.0 or ask LLM.

5. **Formal Logical Syntax**: ❌ LLM outputs aren't modeled in formal logical syntax - they're just values.

## Architectural Mismatch

### Current Flow (Reactive):
```
Rule needs parameter → Ask LLM question → Get answer → Evaluate rule → Fire if conditions met
```

### Proposed Flow (Generative):
```
LLM generates candidate facts/inferences with confidences → Constraint layer validates/prunes → Facts added → Rules evaluate
```

## What Needs to Change

### 1. Add LLM Knowledge Generation Module

The LLM should proactively generate candidate facts, not just answer questions:

```python
def generate_candidate_facts(patient_data: Dict[str, Any]) -> List[Fact]:
    """
    LLM generates candidate facts/inferences from patient data.
    Returns list of facts with confidences.
    """
    prompt = f"""
    Based on this patient data, generate candidate medical facts:
    {patient_data}
    
    For each fact, provide:
    - Parameter name
    - Value
    - Confidence (0.0-1.0)
    - Reasoning
    
    Return as structured facts in logical syntax.
    """
    # LLM returns candidate facts
    return candidate_facts
```

### 2. Add Constraint Validation Layer

A layer that validates LLM-generated facts against domain constraints:

```python
class ConstraintLayer:
    def validate_and_prune(self, candidate_facts: List[Fact], 
                          domain_rules: List[Rule]) -> List[Fact]:
        """
        Validates candidate facts against domain rules.
        Prunes invalid or conflicting facts.
        """
        validated_facts = []
        for fact in candidate_facts:
            # Check against domain constraints
            if self.satisfies_constraints(fact, domain_rules):
                validated_facts.append(fact)
        return validated_facts
```

### 3. Extract Confidence from LLM

Systematically extract confidence/uncertainty from LLM responses:

```python
def extract_llm_confidence(llm_response: str) -> float:
    """
    Extract confidence from LLM response.
    Handles phrases like "likely", "probably", "certain", etc.
    """
    # Parse confidence indicators
    # "very likely" → 0.9
    # "probably" → 0.7
    # "possibly" → 0.5
    # etc.
```

### 4. Model LLM Output in Logical Syntax

Convert LLM outputs to formal logical statements:

```python
def to_logical_syntax(fact: Fact) -> str:
    """
    Convert fact to formal logical syntax.
    Example: Fact("gram_stain", "positive", 0.8) 
    → "gram_stain(patient) = positive [CF=0.8]"
    """
```

## Recommended Architecture Changes

### New Flow:
```
1. Patient Data Input
   ↓
2. LLM Knowledge Generator
   - Generates candidate facts/inferences
   - Provides confidences
   - Outputs in structured format
   ↓
3. Constraint Validation Layer
   - Validates facts against domain rules
   - Prunes invalid/conflicting facts
   - Ensures logical consistency
   ↓
4. Fact Database
   - Stores validated facts with certainty factors
   ↓
5. Rule Evaluation Engine
   - Evaluates MYCIN rules
   - Combines certainties
   - Performs forward chaining
   ↓
6. Diagnosis Output
```

## Implementation Priority

### High Priority (Core to Proposal):
1. ✅ **LLM Knowledge Generator**: Make LLM generate candidate facts proactively
2. ✅ **Constraint Validation Layer**: Add validation/pruning of LLM outputs
3. ✅ **Confidence Extraction**: Extract confidence from LLM responses systematically

### Medium Priority:
4. **Logical Syntax Modeling**: Convert facts to formal logical syntax
5. **Explanation Generation**: LLM generates explanations for inferences

### Low Priority (Nice to Have):
6. **Uncertainty Propagation**: Better modeling of LLM uncertainty
7. **Conflict Resolution**: More sophisticated conflict resolution

## Summary

**Current State**: The system has the right components (LLM + symbolic rules) but the **architecture is inverted**:
- Rules drive the process (reactive)
- LLM is passive (question-answering service)

**Proposed State**: The proposal wants:
- LLM to drive knowledge generation (proactive)
- Constraint layer to validate/prune (filtering)
- Rules to evaluate validated facts (reasoning)

**Alignment Score**: ~60%
- ✅ Has hybrid architecture
- ✅ Uses certainty factors
- ✅ Symbolic rule evaluation
- ❌ LLM not generative enough
- ❌ Missing constraint validation layer
- ❌ LLM confidence not systematically extracted


