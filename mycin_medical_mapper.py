"""
MYCIN Medical Patient Mapper
=============================

Maps patient evidence to MYCIN medical diagnosis parameters.
"""

from typing import Dict, List, Any, Optional, Callable
import re
import json


def map_evidence_to_parameters(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map patient evidence to MYCIN medical diagnosis parameters.
    Uses keyword matching and pattern recognition.
    """
    params = {}
    evidence_lower = {k.lower(): v for k, v in evidence.items()}
    
    # Helper to check if a question contains keywords
    def has_keyword(keywords: List[str]) -> bool:
        for q, v in evidence.items():
            q_lower = q.lower()
            if any(kw in q_lower for kw in keywords) and v:
                return True
        return False
    
    # Helper to get value for a question containing keywords
    def get_value(keywords: List[str], default=None):
        for q, v in evidence.items():
            q_lower = q.lower()
            if any(kw in q_lower for kw in keywords):
                return v
        return default
    
    # Demographics (from demographics dict, not evidence)
    
    # Respiratory symptoms
    params["cough"] = has_keyword(["cough"])
    params["productive_cough"] = has_keyword(["cough that produces", "colored", "abundant sputum"])
    params["dyspnea"] = has_keyword(["shortness of breath", "difficulty breathing"])
    params["wheezing"] = has_keyword(["wheezing", "wheeze"])
    params["fever"] = has_keyword(["fever"])
    params["sore_throat"] = has_keyword(["sore throat"])
    params["nasal_congestion"] = has_keyword(["nasal congestion", "runny nose"])
    params["hoarse_voice"] = has_keyword(["tone of your voice", "deeper", "softer", "hoarse"])
    
    # Pain symptoms
    params["chest_pain"] = has_keyword(["chest pain"])
    params["chest_pain_at_rest"] = has_keyword(["chest pain even at rest"])
    params["chest_pain_exertion"] = has_keyword(["physical exertion", "alleviated with rest"])
    params["chest_pain_breathing"] = has_keyword(["pain that is increased when you breathe", "breathe in deeply"])
    params["abdominal_pain"] = has_keyword(["pain somewhere"]) and not has_keyword(["chest"])
    
    # Get pain location and character
    pain_location = get_value(["do you feel pain somewhere?"])
    pain_character = get_value(["characterize your pain"])
    pain_radiates = get_value(["does the pain radiate"])
    
    if pain_location:
        params["pain_location"] = str(pain_location).lower()
    if pain_character:
        params["pain_character"] = str(pain_character).lower()
    if pain_radiates:
        params["pain_radiates"] = str(pain_radiates).lower()
    
    # GI symptoms
    params["heartburn"] = has_keyword(["burning sensation", "stomach", "throat", "bitter taste"])
    params["burning_sensation"] = has_keyword(["burning sensation"])
    params["worse_lying_down"] = has_keyword(["worse when lying down", "alleviated while sitting up"])
    params["worse_after_eating"] = has_keyword(["worse after eating"])
    params["black_stools"] = has_keyword(["black", "stools", "coal"])
    params["hiatal_hernia"] = has_keyword(["hiatal hernia"])
    params["alcohol_abuse"] = has_keyword(["alcohol", "addiction"])
    params["overweight"] = has_keyword(["overweight"])
    
    # Cardiac symptoms
    params["palpitations"] = has_keyword(["palpitations"])
    params["syncope"] = has_keyword(["syncope", "fainting"])
    
    # Neurological symptoms
    params["headache"] = has_keyword(["headache"])
    params["confusion"] = has_keyword(["confused", "disorientated"])
    params["muscle_spasms"] = has_keyword(["muscle spasms", "spasms"])
    params["tongue_protrusion"] = has_keyword(["trouble keeping your tongue", "tongue in your mouth"])
    params["eyelid_droop"] = has_keyword(["hard time opening", "raising", "eyelids", "eyelid"])
    params["recent_antipsychotics"] = has_keyword(["antipsychotic medication", "last 7 days"])
    
    # Other symptoms
    params["contact_exposure"] = has_keyword(["contact with", "similar symptoms"])
    params["travel_history"] = get_value(["traveled out of the country"])
    params["smoking"] = has_keyword(["smoke cigarettes", "smoking"])
    params["copd"] = has_keyword(["chronic obstructive pulmonary disease", "copd"])
    params["asthma"] = has_keyword(["asthma", "bronchodilator"])
    params["daycare_exposure"] = has_keyword(["daycare"])
    params["household_size"] = get_value(["live with", "people"])
    params["pneumothorax_history"] = has_keyword(["spontaneous pneumothorax", "ever had"])
    params["family_pneumothorax"] = has_keyword(["family members", "pneumothorax"])
    params["intense_coughing_fits"] = has_keyword(["intense coughing fits"])
    params["premature_birth"] = has_keyword(["born prematurely", "complication at birth"])
    
    # Additional parameters for new rules
    params["ear_pain"] = has_keyword(["ear pain", "earache"])
    params["recent_cold"] = has_keyword(["cold in the last", "recent cold"])
    params["dyspnea_at_rest"] = has_keyword(["out of breath", "minimal physical effort"]) or (params.get("dyspnea") and has_keyword(["at rest"]))
    params["heart_failure"] = has_keyword(["heart failure"])
    params["facial_pain"] = has_keyword(["facial pain", "sinus pain"])
    params["greenish_discharge"] = has_keyword(["greenish", "yellowish", "nasal discharge"])
    params["itchy_nose"] = has_keyword(["itchy", "nose", "throat"])
    params["allergy_history"] = has_keyword(["allergy", "allergies", "hay fever", "eczema"])
    params["allergy_exposure"] = has_keyword(["contact with", "ate something", "allergy to"])
    params["swelling"] = has_keyword(["swelling", "swollen"])
    params["pale_skin"] = has_keyword(["pale", "paler than usual"])
    params["fatigue"] = has_keyword(["fatigue", "tired", "exhausted"])
    params["anemia_history"] = has_keyword(["anemia", "diagnosed with anemia"])
    params["irregular_heartbeat"] = has_keyword(["irregularly", "missing a beat", "irregular pattern", "disorganized pattern"])
    params["vomiting_blood"] = has_keyword(["thrown up blood", "vomiting blood", "coffee beans"])
    params["chronic_cough"] = params.get("cough") and has_keyword(["chronic", "persistent"])
    params["recurrent_infections"] = has_keyword(["recurrent", "repeated infections"])
    params["cardiac_symptoms"] = params.get("chest_pain") or params.get("palpitations") or params.get("dyspnea")
    params["chronic_sinusitis"] = has_keyword(["chronic sinusitis", "chronic rhinosinusitis"])
    params["nasal_polyps"] = has_keyword(["polyps", "nasal polyps"])
    params["severe_headache"] = has_keyword(["headache"]) and has_keyword(["severe", "intense", "violent"])
    params["family_cluster_headache"] = has_keyword(["family", "cluster headaches"])
    params["stridor"] = has_keyword(["high pitched sound", "stridor", "stridor"])
    params["ebola_contact"] = has_keyword(["ebola", "contact with anyone infected"])
    params["bleeding"] = has_keyword(["bleeding", "bruising", "unusual bleeding"])
    params["weakness_limbs"] = has_keyword(["weakness", "both arms", "both legs", "limbs"])
    params["numbness"] = has_keyword(["numbness", "loss of sensation", "tingling"])
    params["recent_infection"] = has_keyword(["recent infection", "viral infection", "recently had"])
    params["hiv_risk"] = has_keyword(["hiv", "unprotected sex", "hiv-positive partner", "intravenous drugs"])
    params["groin_pain"] = has_keyword(["groin", "inguinal"]) and params.get("abdominal_pain")
    params["groin_swelling"] = has_keyword(["groin", "inguinal"]) and params.get("swelling")
    params["pain_with_coughing"] = has_keyword(["increased with coughing", "coughing", "effort like lifting"])
    params["suffocating_feeling"] = has_keyword(["suffocating", "choking", "inability to breathe"])
    params["inability_to_breathe"] = has_keyword(["inability to breathe", "unable to breathe", "suffocating"])
    params["localized_swelling"] = params.get("swelling") and not has_keyword(["widespread", "diffuse"])
    params["pain_at_site"] = params.get("swelling") and params.get("abdominal_pain")
    params["muscle_weakness"] = has_keyword(["weakness", "muscle weakness"])
    params["weakness_worse_fatigue"] = params.get("muscle_weakness") and has_keyword(["increase with fatigue", "worse with fatigue"])
    params["recent_viral_infection"] = has_keyword(["recent viral infection", "viral infection"])
    params["rapid_heartbeat"] = has_keyword(["beating fast", "racing", "rapid"])
    params["weight_loss"] = has_keyword(["weight loss", "losing weight", "involuntary weight loss"])
    params["family_pancreatic_cancer"] = has_keyword(["family", "pancreatic cancer"])
    params["anxiety"] = has_keyword(["anxious", "anxiety", "panic"])
    params["feeling_dying"] = has_keyword(["dying", "about to die", "afraid"])
    params["chest_pain_improves_forward"] = params.get("chest_pain") and has_keyword(["improves when you lean forward", "lean forward"])
    params["pericarditis_history"] = has_keyword(["pericarditis", "ever had a pericarditis"])
    params["dvt_history"] = has_keyword(["deep vein thrombosis", "dvt"])
    params["joint_pain"] = has_keyword(["joint", "arthritis"])
    params["rash"] = has_keyword(["rash", "lesions", "redness"])
    params["fish_consumption"] = has_keyword(["fish", "tuna", "swiss cheese", "dark-fleshed fish"])
    params["nausea"] = has_keyword(["nauseous", "nausea", "vomiting"])
    params["flushing"] = has_keyword(["flushing", "red", "suddenly turn red"])
    params["chest_pain_movement"] = params.get("chest_pain") and has_keyword(["increased with movement", "pain that is increased with movement"])
    
    # Clean up: remove None values, convert True/False to proper booleans
    cleaned_params = {}
    for k, v in params.items():
        if v is not None:
            if isinstance(v, str):
                # Convert string booleans
                if v.lower() in ["y", "yes", "true", "1"]:
                    cleaned_params[k] = True
                elif v.lower() in ["n", "no", "false", "0"]:
                    cleaned_params[k] = False
                else:
                    cleaned_params[k] = v
            else:
                cleaned_params[k] = v
    
    return cleaned_params


def map_to_mycin_medical_format(
    patient_payload: Dict[str, Any],
    llm_extraction_fn: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Map patient payload to MYCIN medical diagnosis format.
    
    Args:
        patient_payload: Patient data with demographics and evidence
        llm_extraction_fn: Optional LLM function for extracting missing parameters
    
    Returns:
        Dictionary of MYCIN parameters
    """
    demographics = patient_payload.get("demographics", {})
    evidence = patient_payload.get("evidence", {})
    
    # Start with demographics
    mycin_params = {}
    if "AGE" in demographics:
        try:
            mycin_params["age"] = int(demographics["AGE"])
        except (ValueError, TypeError):
            mycin_params["age"] = demographics["AGE"]
    if "SEX" in demographics:
        mycin_params["sex"] = str(demographics["SEX"]).upper()
    
    # Map evidence to parameters
    evidence_params = map_evidence_to_parameters(evidence)
    mycin_params.update(evidence_params)
    
    # Use LLM to extract additional parameters if needed
    if llm_extraction_fn:
        try:
            # Use LLM to extract additional parameters from evidence
            evidence_str = json.dumps(evidence, indent=2)
            extraction_prompt = f"""Extract medical parameters from this patient evidence:

{evidence_str}

Return JSON with boolean parameters: {{"fever": true/false, "cough": true/false, ...}}
Only include parameters that are clearly present in the evidence.
"""
            response = llm_extraction_fn(extraction_prompt)
            if isinstance(response, str):
                response = re.sub(r'```json\s*', '', response)
                response = re.sub(r'```\s*', '', response)
                json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                if json_match:
                    extracted = json.loads(json_match.group(0))
                    # Merge extracted parameters (don't overwrite existing)
                    for k, v in extracted.items():
                        if k not in mycin_params:
                            mycin_params[k] = v
        except:
            pass
    
    return mycin_params

