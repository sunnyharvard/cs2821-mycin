import json
from mycin_inference_engine import MYCINInferenceEngine

def mock_llm_qa(question, patient_data):
    """
    Mock LLM that answers based on patient data.
    """
    print(f"  [System asked]: {question}")
    
    # Simple mapping for the test case
    if question == "gram_stain":
        return "positive", 1.0
    if question == "morphology":
        return "coccus", 1.0
    if question == "growth_pattern":
        return "chains", 1.0
    if question == "site": # culture_site
        return "blood", 1.0
    if question == "culture_site":
        return "blood", 1.0
        
    # Default unknown
    return None, 0.0

def test_streptococcus_case():
    print("\n=== Testing Streptococcus Case (Backward Chaining) ===")
    
    # Patient data (minimal, forcing system to ask)
    patient_data = {
        "name": "Test Patient",
        # We deliberately omit lab results to see if it asks
    }
    
    engine = MYCINInferenceEngine(llm_question_answering_fn=mock_llm_qa)
    
    print("Starting diagnosis for 'identity'...")
    engine.backward_chain("identity", patient_data)
    
    diagnosis = engine.get_diagnosis()
    organism = diagnosis["organism_identity"]["name"]
    certainty = diagnosis["organism_identity"]["certainty"]
    
    print(f"\nDiagnosis Result:")
    print(f"  Organism: {organism}")
    print(f"  Certainty: {certainty}")
    
    # Verification
    if organism == "streptococcus":
        print("SUCCESS: Correctly diagnosed Streptococcus.")
    else:
        print(f"FAILURE: Expected Streptococcus, got {organism}")

if __name__ == "__main__":
    test_streptococcus_case()
