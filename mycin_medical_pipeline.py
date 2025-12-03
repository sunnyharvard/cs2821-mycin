"""
MYCIN Medical Diagnosis Pipeline (Hybrid Approach)
=================================================

Hybrid approach combining rule-based inference with LLM intelligence:
- Rules provide structured, interpretable diagnosis
- LLM extracts parameters, answers questions, and provides comprehensive differential
- Intelligent combination: rules boost confidence, LLM fills gaps
"""

from typing import Dict, List, Any, Optional, Callable
import json
import os
import re

from mycin_inference_engine import MYCINInferenceEngine, simple_llm_qa_function
from mycin_medical_mapper import map_to_mycin_medical_format
from mycin_medical_rules import ALL_RULES, QUESTIONS, ASK_FIRST_PARAMETERS, Rule, RuleCondition, CertaintyFactor


def run_mycin_medical_pipeline(
    patient_payloads: List[Dict[str, Any]],
    llm_call_fn: Optional[Callable] = None,
    use_llm_for_extraction: bool = True,
    use_llm_for_questions: bool = True
) -> List[Dict[str, Any]]:
    """
    Run MYCIN medical diagnosis inference on patient payloads.
    
    Hybrid approach:
    - Rules evaluate programmatically to determine diagnoses
    - LLM extracts parameters, answers questions, and provides comprehensive differential
    - Intelligent combination: rules boost confidence, LLM fills gaps
    """
    predictions = []
    
    # Load all possible diseases
    all_diseases = []
    try:
        with open("data_extraction/diagnoses_from_json.txt", "r") as f:
            all_diseases = [line.strip() for line in f if line.strip()]
    except:
        pass
    
    # Create LLM question answering function
    if use_llm_for_questions and llm_call_fn:
        def llm_qa_fn(question_key: str, patient_data: Dict[str, Any]) -> tuple:
            return simple_llm_qa_function(question_key, patient_data, llm_call_fn)
    else:
        llm_qa_fn = None
    
    for patient_payload in patient_payloads:
        row_index = patient_payload.get("row_index", len(predictions))
        
        # Step 1: Use LLM to extract additional parameters from evidence
        enhanced_patient_data = patient_payload.copy()
        if use_llm_for_extraction and llm_call_fn:
            try:
                evidence = patient_payload.get("evidence", {})
                demographics = patient_payload.get("demographics", {})
                
                # Use LLM to extract additional parameters
                extraction_prompt = f"""You are a medical assistant extracting structured parameters from patient evidence.

Patient Demographics:
{json.dumps(demographics, indent=2)}

Patient Evidence:
{json.dumps(evidence, indent=2)}

Extract the following parameters if present in the evidence (respond with JSON only):
- fever: boolean
- cough: boolean
- productive_cough: boolean
- dyspnea: boolean
- wheezing: boolean
- sore_throat: boolean
- nasal_congestion: boolean
- chest_pain: boolean
- abdominal_pain: boolean
- heartburn: boolean
- headache: boolean
- fatigue: boolean
- contact_exposure: boolean
- smoking: boolean
- copd: boolean
- asthma: boolean

Return ONLY valid JSON: {{"fever": true/false, "cough": true/false, ...}}
"""
                extraction_response = llm_call_fn(extraction_prompt)
                try:
                    if isinstance(extraction_response, str):
                        extraction_response = re.sub(r'```json\s*', '', extraction_response)
                        extraction_response = re.sub(r'```\s*', '', extraction_response)
                        json_match = re.search(r'\{[^{}]*\}', extraction_response, re.DOTALL)
                        if json_match:
                            extracted_params = json.loads(json_match.group(0))
                            # Merge into patient data
                            if "mycin_params" not in enhanced_patient_data:
                                enhanced_patient_data["mycin_params"] = {}
                            enhanced_patient_data["mycin_params"].update(extracted_params)
                except:
                    pass
            except Exception as e:
                pass
        
        # Step 2: Get comprehensive LLM differential diagnosis
        llm_probs = {}
        if llm_call_fn and all_diseases:
            try:
                evidence = patient_payload.get("evidence", {})
                demographics = patient_payload.get("demographics", {})
                
                ev_lines = []
                for k, v in evidence.items():
                    ev_lines.append(f"- {k}: {v}")
                
                disease_list_str = "\n".join([f"- {d}" for d in all_diseases])
                
                prompt = f"""You are a senior clinician performing differential diagnosis.

You are given:
- A patient's demographics.
- A list of symptoms and clinical evidence.
- A list of possible diseases that you MUST choose from.

Your task:
1. Identify the most plausible diseases from the allowed list.
2. Select **no more than 10 diseases**.
3. Assign a probability to each selected disease.
4. Ensure:
   - Each probability is a **float between 0 and 1**.
   - The **sum of all probabilities is exactly 1.0**.
   - You **only** output diseases from the allowed disease list.
5. Keep the output concise—no explanations, no narrative, no medical reasoning.

Your response **MUST** be **only** valid JSON in the following structure:

{{
  "row_index": {row_index},
  "differential_probs": {{
    "Disease A": 0.40,
    "Disease B": 0.25,
    "Disease C": 0.35
  }}
}}

No comments, no markdown, no backticks, no explanation. Only the JSON object.

--------------------
PATIENT DATA
--------------------
Row index: {row_index}

Demographics:
{json.dumps(demographics, indent=2)}

Evidence:
{chr(10).join(ev_lines)}

--------------------
ALLOWED DISEASE LIST
(you may choose at most 10 from this list)
--------------------
{disease_list_str}"""
                
                response = llm_call_fn(prompt)
                
                # Parse LLM response
                if isinstance(response, str):
                    response = re.sub(r'```json\s*', '', response)
                    response = re.sub(r'```\s*', '', response)
                    response = response.strip()
                    
                    json_match = re.search(r'\{[^{}]*"differential_probs"[^{}]*\{[^{}]*\}[^{}]*\}', response, re.DOTALL)
                    if json_match:
                        response = json_match.group(0)
                    
                    try:
                        llm_result = json.loads(response)
                        raw_probs = llm_result.get("differential_probs", {})
                        
                        # Validate disease names match allowed list exactly
                        for disease, prob in raw_probs.items():
                            matched_disease = None
                            for allowed in all_diseases:
                                if disease.lower() == allowed.lower():
                                    matched_disease = allowed
                                    break
                            
                            if matched_disease:
                                llm_probs[matched_disease] = prob
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                pass
        
        # Step 3: Generate patient-specific rules using LLM
        augmented_rules = ALL_RULES.copy()
        if llm_call_fn:
            try:
                evidence = patient_payload.get("evidence", {})
                demographics = patient_payload.get("demographics", {})
                mycin_data = map_to_mycin_medical_format(enhanced_patient_data, None)
                
                # Identify key symptoms present
                present_symptoms = [k for k, v in mycin_data.items() if v is True and k in QUESTIONS]
                present_symptoms_str = ", ".join(present_symptoms[:10])  # Limit for prompt
                
                # Get list of all diseases
                disease_list_str = "\n".join([f"- {d}" for d in all_diseases]) if all_diseases else ""
                
                # Generate patient-specific rules
                rule_generation_prompt = f"""You are a medical expert creating MYCIN-style diagnostic rules for a specific patient.

Patient Demographics:
{json.dumps(demographics, indent=2)}

Key Symptoms Present:
{present_symptoms_str}

Available Diseases:
{disease_list_str}

Generate 2-5 patient-specific diagnostic rules in JSON format. Each rule should:
1. Use symptoms that are present in this patient
2. Conclude a diagnosis from the available disease list
3. Have appropriate certainty factors (0.2-0.8)

Format (JSON array):
[
  {{
    "rule_id": "DYNAMIC001",
    "category": "Dynamic",
    "conditions": [
      {{"parameter": "fever", "operator": "is", "value": true}},
      {{"parameter": "cough", "operator": "is", "value": true}}
    ],
    "conclusion": {{"diagnosis": "Influenza"}},
    "certainty_factor": 0.7,
    "description": "Patient-specific rule: fever + cough → Influenza"
  }}
]

Return ONLY valid JSON array, no markdown, no explanation."""
                
                rule_response = llm_call_fn(rule_generation_prompt)
                
                # Parse and add dynamic rules
                if isinstance(rule_response, str):
                    rule_response = re.sub(r'```json\s*', '', rule_response)
                    rule_response = re.sub(r'```\s*', '', rule_response)
                    rule_response = rule_response.strip()
                    
                    json_match = re.search(r'\[[^\]]*\{[^}]*\}[^\]]*\]', rule_response, re.DOTALL)
                    if json_match:
                        rule_response = json_match.group(0)
                    
                    try:
                        dynamic_rules_data = json.loads(rule_response)
                        if isinstance(dynamic_rules_data, list):
                            for rule_data in dynamic_rules_data:
                                try:
                                    # Validate and create Rule object
                                    conditions = [
                                        RuleCondition(
                                            parameter=c["parameter"],
                                            operator=c.get("operator", "is"),
                                            value=c["value"]
                                        )
                                        for c in rule_data.get("conditions", [])
                                    ]
                                    
                                    # Validate disease name
                                    diagnosis = rule_data.get("conclusion", {}).get("diagnosis", "")
                                    if diagnosis and all_diseases:
                                        # Check if diagnosis is in allowed list
                                        matched_disease = None
                                        for allowed in all_diseases:
                                            if diagnosis.lower() == allowed.lower():
                                                matched_disease = allowed
                                                break
                                        
                                        if matched_disease:
                                            dynamic_rule = Rule(
                                                rule_id=rule_data.get("rule_id", f"DYNAMIC{len(augmented_rules)}"),
                                                category=rule_data.get("category", "Dynamic"),
                                                conditions=conditions,
                                                conclusion={"diagnosis": matched_disease},
                                                certainty_factor=min(0.8, max(0.2, rule_data.get("certainty_factor", 0.5))),
                                                description=rule_data.get("description", f"Dynamic rule for {matched_disease}")
                                            )
                                            augmented_rules.append(dynamic_rule)
                                except Exception as e:
                                    pass
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                pass
        
        # Step 4: Adapt rule certainty factors based on patient context
        if llm_call_fn and len(augmented_rules) > len(ALL_RULES):
            try:
                # Use LLM to adjust certainty factors for dynamic rules based on patient context
                evidence = patient_payload.get("evidence", {})
                demographics = patient_payload.get("demographics", {})
                
                # Get dynamic rules
                dynamic_rules = [r for r in augmented_rules if r.rule_id.startswith("DYNAMIC")]
                
                if dynamic_rules:
                    rule_descriptions = "\n".join([
                        f"- {r.rule_id}: {r.description} (CF: {r.certainty_factor})"
                        for r in dynamic_rules[:5]  # Limit for prompt
                    ])
                    
                    adaptation_prompt = f"""You are a medical expert adjusting rule certainty factors based on patient context.

Patient Demographics:
{json.dumps(demographics, indent=2)}

Key Evidence:
{json.dumps(dict(list(evidence.items())[:10]), indent=2)}

Dynamic Rules Generated:
{rule_descriptions}

Adjust certainty factors (0.2-0.8) for these rules based on how well they match this patient.
Return JSON: {{"DYNAMIC001": 0.75, "DYNAMIC002": 0.65, ...}}
Only include rules that should be adjusted."""
                    
                    adaptation_response = llm_call_fn(adaptation_prompt)
                    
                    # Parse and apply adjustments
                    if isinstance(adaptation_response, str):
                        adaptation_response = re.sub(r'```json\s*', '', adaptation_response)
                        adaptation_response = re.sub(r'```\s*', '', adaptation_response)
                        json_match = re.search(r'\{[^{}]*\}', adaptation_response, re.DOTALL)
                        if json_match:
                            try:
                                adjustments = json.loads(json_match.group(0))
                                for rule in dynamic_rules:
                                    if rule.rule_id in adjustments:
                                        new_cf = adjustments[rule.rule_id]
                                        rule.certainty_factor = min(0.8, max(0.2, float(new_cf)))
                            except:
                                pass
            except Exception as e:
                pass
        
        # Step 5: Run MYCIN rules (static + dynamic) to get rule-based predictions
        rule_probs = {}
        try:
            # Map patient data to MYCIN format (use enhanced data if available)
            mycin_data = map_to_mycin_medical_format(enhanced_patient_data, llm_call_fn if use_llm_for_extraction else None)
            
            # Temporarily patch the inference engine module to use augmented rules
            import mycin_inference_engine
            try:
                import mycin_rules
                _orig_rules_module = mycin_rules.ALL_RULES
            except ImportError:
                # Create a stub module if it doesn't exist
                import types
                mycin_rules = types.ModuleType('mycin_rules')
                _orig_rules_module = []
            
            _orig_rules_ie = mycin_inference_engine.ALL_RULES
            
            # Use augmented rules (static + dynamic)
            mycin_inference_engine.ALL_RULES = augmented_rules
            mycin_rules.ALL_RULES = augmented_rules
            
            # Create engine with LLM for question answering only
            engine = MYCINInferenceEngine(llm_question_answering_fn=llm_qa_fn)
            
            # Initialize with known facts from patient data
            for param, value in mycin_data.items():
                if value is not None:
                    engine.update_fact(param, value, certainty=1.0)
            
            # Run backward chaining inference for the goal "diagnosis"
            # This will evaluate all rules (static + dynamic) and ask LLM for missing parameters
            engine.backward_chain("diagnosis", mycin_data)
            
            # Get all diagnosis facts from rule evaluation
            diagnosis_facts = engine.get_facts("diagnosis")
            
            if diagnosis_facts:
                # Convert certainty factors to probabilities
                # Normalize positive certainties
                positive_facts = [f for f in diagnosis_facts if f.certainty > 0]
                if positive_facts:
                    total_cf = sum(f.certainty for f in positive_facts)
                    if total_cf > 0:
                        for fact in positive_facts:
                            rule_probs[fact.value] = fact.certainty / total_cf
                    else:
                        # If all certainties are 0, use uniform distribution
                        for fact in positive_facts:
                            rule_probs[fact.value] = 1.0 / len(positive_facts)
            
            # Restore original rules
            mycin_inference_engine.ALL_RULES = _orig_rules_ie
            mycin_rules.ALL_RULES = _orig_rules_module
        except Exception as e:
            pass
        
        # Step 6: Intelligently combine rule-based and LLM predictions
        probs = {}
        
        if llm_probs and rule_probs:
            # Both available: Start with LLM, boost rule-based matches
            probs = llm_probs.copy()
            
            # Boost diseases that match rules (rules provide confidence boost)
            for disease, rule_prob in rule_probs.items():
                if disease in probs:
                    # Boost by 30% of rule confidence
                    boost = rule_prob * 0.30
                    probs[disease] = min(1.0, probs[disease] + boost)
                else:
                    # Add rule-based disease with moderate weight
                    probs[disease] = rule_prob * 0.40
            
            # Normalize
            total = sum(probs.values())
            if total > 0:
                probs = {k: v / total for k, v in probs.items()}
        elif llm_probs:
            # Only LLM available
            probs = llm_probs
        elif rule_probs:
            # Only rules available
            probs = rule_probs
        else:
            # Neither available - use LLM fallback (already computed above)
            probs = llm_probs
        
        # Ensure probabilities sum to 1.0
        if probs:
            total = sum(probs.values())
            if total > 0:
                probs = {k: v / total for k, v in probs.items()}
        
        # Format output
        predictions.append({
            "row_index": row_index,
            "differential_probs": probs
        })
    
    return predictions


# Import GPT-4o function
def gpt4o_llm_call(prompt: str) -> str:
    """Call OpenAI GPT-4o model."""
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it with: export OPENAI_API_KEY='your-key-here'")
    
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical expert assistant. Answer questions concisely and accurately based on the provided patient information. When asked for differential diagnosis, return ONLY valid JSON with no additional text."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "UNKNOWN"

