"""
MYCIN Pipeline Integration
===========================

Integrates MYCIN inference engine with the existing test_set_pipeline.py
"""

from typing import Dict, List, Any, Optional, Callable
import json

from mycin_inference_engine import (
    MYCINInferenceEngine,
    simple_llm_qa_function,
    create_llm_question_prompt
)
from mycin_patient_mapper import map_to_mycin_format, extract_mycin_parameters
from mycin_rules import QUESTIONS


def run_mycin_pipeline(
    patient_payloads: List[Dict[str, Any]],
    llm_call_fn: Optional[Callable] = None,
    use_llm_for_extraction: bool = True,
    use_llm_for_questions: bool = True
) -> List[Dict[str, Any]]:
    """
    Run MYCIN inference on patient payloads.
    
    Args:
        patient_payloads: List of patient payloads from test_set_pipeline.py
        llm_call_fn: Function that takes a prompt string and returns response
        use_llm_for_extraction: Whether to use LLM for parameter extraction
        use_llm_for_questions: Whether to use LLM for answering questions
    
    Returns:
        List of prediction dictionaries in format:
        {
            "predicted_label": "organism_name",
            "probs": {"organism_name": 0.72, ...}
        }
    """
    predictions = []
    
    # Create LLM question answering function
    if use_llm_for_questions and llm_call_fn:
        def llm_qa_fn(question_key: str, patient_data: Dict[str, Any]) -> tuple:
            return simple_llm_qa_function(question_key, patient_data, llm_call_fn)
    else:
        llm_qa_fn = None
    
    # Create LLM extraction function
    if use_llm_for_extraction and llm_call_fn:
        def llm_extraction_fn(prompt: str) -> Dict[str, Any]:
            response = llm_call_fn(prompt)
            if isinstance(response, dict):
                return response
            elif isinstance(response, str):
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    return {}
            return {}
    else:
        llm_extraction_fn = None
    
    for patient_payload in patient_payloads:
        # Map patient data to MYCIN format
        mycin_data = map_to_mycin_format(patient_payload, llm_extraction_fn)
        
        # Create inference engine
        engine = MYCINInferenceEngine(llm_question_answering_fn=llm_qa_fn)
        
        # Initialize with known facts
        for param, value in mycin_data.items():
            if value is not None:
                engine.set_fact(param, value, certainty=1.0)
        
        # Run forward chaining inference
        applied_rules = engine.forward_chain(mycin_data)
        
        # Get diagnosis
        diagnosis = engine.get_diagnosis()
        
        # Format output for evaluation
        organism_name = diagnosis["organism_identity"]["name"]
        organism_probs = diagnosis["organism_identity"]["probabilities"]
        
        # Map organism names to condition names if needed
        # (You may need to adjust this mapping based on your condition names)
        predicted_label = organism_name
        
        # Create probabilities dict
        probs = {}
        if organism_probs:
            # Use organism probabilities
            probs = organism_probs
        elif organism_name:
            # Single prediction
            certainty = diagnosis["organism_identity"]["certainty"]
            probs[organism_name] = max(0.0, min(1.0, certainty))
        
        predictions.append({
            "predicted_label": predicted_label or "Unknown",
            "probs": probs,
            "diagnosis_details": {
                "organism": diagnosis["organism_identity"],
                "infection_site": diagnosis["infection_site"],
                "treatment": diagnosis["treatment"],
                "applied_rules": applied_rules
            }
        })
    
    return predictions


# Example LLM call function (placeholder - replace with your actual LLM API)
def example_llm_call(prompt: str) -> str:
    """
    Placeholder LLM call function.
    Replace this with your actual LLM API call (OpenAI, Anthropic, etc.)
    """
    # Example: using OpenAI
    # import openai
    # response = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # return response.choices[0].message.content
    
    # For now, return a placeholder
    return "UNKNOWN"


if __name__ == "__main__":
    # Example usage
    example_patient = {
        "demographics": {"AGE": 45, "SEX": "M"},
        "evidence": {
            "Do you have a cough?": True,
            "Do you have a fever?": True,
            "Are you experiencing shortness of breath?": True
        }
    }
    
    predictions = run_mycin_pipeline(
        [example_patient],
        llm_call_fn=example_llm_call,
        use_llm_for_extraction=True,
        use_llm_for_questions=True
    )
    
    print(json.dumps(predictions[0], indent=2))


