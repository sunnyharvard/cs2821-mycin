"""
MYCIN Patient Data Mapper
=========================

Maps general patient evidence data to MYCIN-specific parameters.
Uses LLM to extract MYCIN-relevant information from patient data.
Now works with actual question text from question_en_output.txt
"""

import json
from typing import Dict, Any, Optional, Callable, List

from mycin_question_mapping import get_parameter_from_question, ALL_AVAILABLE_QUESTIONS
from mycin_rules import QUESTIONS


def create_llm_extraction_prompt(patient_data: Dict[str, Any], target_parameters: List[str]) -> str:
    """
    Create a prompt for LLM to extract MYCIN parameters from patient data.
    """
    questions = {param: QUESTIONS.get(param, param) for param in target_parameters}
    
    prompt = f"""You are a medical assistant extracting specific clinical parameters from patient data.

Patient Data:
{json.dumps(patient_data, indent=2)}

Extract the following parameters if available in the patient data. For each parameter, provide:
- The value if available
- "UNKNOWN" if not available
- For yes/no questions: true or false
- For categorical questions: the category value
- For numerical questions: the number

Parameters to extract:
{json.dumps(questions, indent=2)}

Respond in JSON format:
{{
    "gram_stain": "positive/negative/UNKNOWN",
    "morphology": "coccus/rod/spirillum/UNKNOWN",
    "culture_site": "blood/urine/sputum/cerebrospinal_fluid/wound/abdominal/UNKNOWN",
    "fever": true/false/UNKNOWN,
    "cough": true/false/UNKNOWN,
    ...
}}

JSON:"""
    return prompt


def extract_mycin_parameters(patient_data: Dict[str, Any], 
                           llm_extraction_fn: Optional[Callable] = None,
                           target_params: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Extract MYCIN parameters from general patient data.
    
    Args:
        patient_data: Patient data dictionary (from pipeline)
        llm_extraction_fn: Function that takes prompt and returns JSON dict
        target_params: List of MYCIN parameters to extract (default: all)
    
    Returns:
        Dictionary mapping MYCIN parameters to values
    """
    if target_params is None:
        # Default: extract commonly needed parameters
        target_params = [
            "gram_stain", "morphology", "culture_site", "fever", "cough",
            "dyspnea", "headache", "nuchal_rigidity", "urinary_symptoms",
            "abdominal_pain", "white_blood_count", "hospital_acquired",
            "immunocompromised", "burn", "recent_surgery"
        ]
    
    mycin_data = {}
    
    # First, try simple keyword matching
    evidence_text = json.dumps(patient_data.get("evidence", {})).lower()
    demographics = patient_data.get("demographics", {})
    
    # Extract age
    if "age" in demographics:
        mycin_data["age"] = demographics["age"]
    
    # Extract sex
    if "sex" in demographics:
        mycin_data["sex"] = demographics["sex"]
    
    # Simple keyword matching for common parameters
    if any(kw in evidence_text for kw in ["fever", "temperature", "temp"]):
        mycin_data["fever"] = True
    
    if any(kw in evidence_text for kw in ["cough", "coughing"]):
        mycin_data["cough"] = True
    
    if any(kw in evidence_text for kw in ["shortness of breath", "dyspnea", "difficulty breathing"]):
        mycin_data["dyspnea"] = True
    
    if any(kw in evidence_text for kw in ["headache", "head pain"]):
        mycin_data["headache"] = True
    
    if any(kw in evidence_text for kw in ["nuchal rigidity", "stiff neck"]):
        mycin_data["nuchal_rigidity"] = True
    
    if any(kw in evidence_text for kw in ["urinary", "dysuria", "frequency"]):
        mycin_data["urinary_symptoms"] = True
    
    if any(kw in evidence_text for kw in ["abdominal pain", "stomach pain"]):
        mycin_data["abdominal_pain"] = True
    
    if any(kw in evidence_text for kw in ["hospital", "nosocomial"]):
        mycin_data["hospital_acquired"] = True
    
    if any(kw in evidence_text for kw in ["immunocompromised", "immunosuppressed"]):
        mycin_data["immunocompromised"] = True
    
    if any(kw in evidence_text for kw in ["burn", "burns"]):
        mycin_data["burn"] = "serious"  # Default assumption
    
    if any(kw in evidence_text for kw in ["surgery", "surgical", "post-operative"]):
        mycin_data["recent_surgery"] = True
    
    # Use LLM for more complex extraction if available
    if llm_extraction_fn:
        missing_params = [p for p in target_params if p not in mycin_data]
        if missing_params:
            prompt = create_llm_extraction_prompt(patient_data, missing_params)
            llm_result = llm_extraction_fn(prompt)
            
            if isinstance(llm_result, dict):
                # Merge LLM results
                for param, value in llm_result.items():
                    if value != "UNKNOWN" and value is not None:
                        mycin_data[param] = value
            elif isinstance(llm_result, str):
                # Try to parse JSON
                try:
                    parsed = json.loads(llm_result)
                    for param, value in parsed.items():
                        if value != "UNKNOWN" and value is not None:
                            mycin_data[param] = value
                except json.JSONDecodeError:
                    pass
    
    return mycin_data


def map_to_mycin_format(patient_payload: Dict[str, Any],
                       llm_extraction_fn: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Map a patient payload from the pipeline to MYCIN format.
    Now works with actual question text from question_en_output.txt.
    
    Args:
        patient_payload: Patient payload from test_set_pipeline.py
        llm_extraction_fn: Optional LLM function for extraction
    
    Returns:
        Dictionary with MYCIN parameters
    """
    mycin_params = {}
    
    # Map evidence questions to MYCIN parameters
    evidence = patient_payload.get("evidence", {})
    
    # First pass: map actual question text to MYCIN parameters
    for question_text, answer in evidence.items():
        param = get_parameter_from_question(question_text)
        if param:
            # Found a mapping - use the answer
            mycin_params[param] = answer
        else:
            # Check if it's already a MYCIN parameter name
            from mycin_rules import QUESTIONS
            if question_text in QUESTIONS:
                mycin_params[question_text] = answer
    
    # Extract additional MYCIN parameters from general evidence using keyword matching
    extracted = extract_mycin_parameters(patient_payload, llm_extraction_fn)
    
    # Merge (extracted values don't override direct mappings)
    for key, value in extracted.items():
        if key not in mycin_params and value is not None:
            mycin_params[key] = value
    
    # Add demographics
    demographics = patient_payload.get("demographics", {})
    if "AGE" in demographics:
        mycin_params["age"] = demographics["AGE"]
    if "SEX" in demographics:
        mycin_params["sex"] = demographics["SEX"]
    
    return mycin_params

