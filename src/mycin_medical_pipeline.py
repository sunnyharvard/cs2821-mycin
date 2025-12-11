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
import csv

from mycin_inference_engine import MYCINInferenceEngine, simple_llm_qa_function
from mycin_medical_mapper import map_to_mycin_medical_format
from mycin_medical_rules import ALL_RULES, QUESTIONS, ASK_FIRST_PARAMETERS, Rule, RuleCondition, CertaintyFactor


def run_mycin_medical_pipeline(
    patient_payloads: List[Dict[str, Any]],
    llm_call_fn: Optional[Callable] = None,
    use_llm_for_extraction: bool = True,
    use_llm_for_questions: bool = True,
    baseline=False,
    save_csv: bool = True,
    csv_output_path: str = "results/mycin_medical_explanations.csv"
) -> List[Dict[str, Any]]:
    """
    Run MYCIN medical diagnosis inference on patient payloads.
    
    Hybrid approach:
    - Rules evaluate programmatically to determine diagnoses
    - LLM extracts parameters, answers questions, and provides comprehensive differential
    - Intelligent combination: rules boost confidence, LLM fills gaps
    
    Args:
        save_csv: If True, save explanations to CSV file
        csv_output_path: Path to save CSV file with explanations
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
    

    print("STEP 1: Use LLM to extract additional parameters from evidence")
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
        
        print("STEP 2: Get comprehensive LLM differential diagnosis")

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
        
        # Initialize augmented_rules (will be used in Step 5)
        augmented_rules = ALL_RULES.copy()
        
        if not baseline:
            # Step 3: Generate patient-specific rules using LLM
            print("STEP 3: Generate patient-specific rules using LLM")
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
            print("STEP 4: Adapt rule certainty factors based on patient context")
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
        
        print("STEP 5: Run MYCIN rules (static + dynamic) to get rule-based predictions")
        # Step 5: Run MYCIN rules (static + dynamic) to get rule-based predictions
        rule_probs = {}
        mycin_reasoning = {}  # Will store rule information for explanations
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
            
            # Capture MYCIN reasoning: which rules fired for each diagnosis
            mycin_reasoning = {}  # disease -> list of rule info
            
            if diagnosis_facts:
                # Convert certainty factors to probabilities
                # Normalize positive certainties
                positive_facts = [f for f in diagnosis_facts if f.certainty > 0]
                if positive_facts:
                    total_cf = sum(f.certainty for f in positive_facts)
                    if total_cf > 0:
                        for fact in positive_facts:
                            rule_probs[fact.value] = fact.certainty / total_cf
                            
                            # Capture which rules contributed to this diagnosis
                            disease = fact.value
                            if disease not in mycin_reasoning:
                                mycin_reasoning[disease] = []
                            
                            # Get rule information for each source rule
                            for rule_id in fact.source_rules:
                                # Find the rule in augmented_rules
                                for rule in augmented_rules:
                                    if rule.rule_id == rule_id:
                                        # Extract key conditions that were met
                                        conditions_met = []
                                        for cond in rule.conditions:
                                            param_value = mycin_data.get(cond.parameter)
                                            if param_value is not None:
                                                if cond.operator == "is" and param_value == cond.value:
                                                    conditions_met.append(f"{cond.parameter}={cond.value}")
                                                elif cond.operator == "greater_than" and param_value > cond.value:
                                                    conditions_met.append(f"{cond.parameter}>{cond.value}")
                                                elif cond.operator == "less_than" and param_value < cond.value:
                                                    conditions_met.append(f"{cond.parameter}<{cond.value}")
                                        
                                        mycin_reasoning[disease].append({
                                            "rule_id": rule_id,
                                            "description": rule.description,
                                            "certainty_factor": rule.certainty_factor,
                                            "conditions_met": conditions_met,
                                            "contributed_certainty": fact.certainty
                                        })
                                        break
                    else:
                        # If all certainties are 0, use uniform distribution
                        for fact in positive_facts:
                            rule_probs[fact.value] = 1.0 / len(positive_facts)
            
            # Restore original rules
            mycin_inference_engine.ALL_RULES = _orig_rules_ie
            mycin_rules.ALL_RULES = _orig_rules_module
        except Exception as e:
            mycin_reasoning = {}
            pass
        
        # Step 6: Intelligently combine rule-based and LLM predictions
        print("STEP 6: Intelligently combine rule-based and LLM predictions")
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
        
        # Step 7: Generate explanation combining one-shot LLM and MYCIN adjustments
        explanation = ""
        if llm_call_fn:
            try:
                evidence = patient_payload.get("evidence", {})
                demographics = patient_payload.get("demographics", {})
                
                # Format key symptoms for natural presentation
                key_symptoms = []
                for k, v in list(evidence.items())[:12]:
                    if v and str(v).lower() not in ["none", "unknown", "false", "no"]:
                        key_symptoms.append(f"{k}: {v}")
                
                # Format top diagnoses with probabilities
                top_diagnoses = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5] if probs else []
                diagnoses_str = "\n".join([f"  - {disease}: {prob:.2%}" for disease, prob in top_diagnoses])
                
                # Format MYCIN reasoning for top diagnoses in natural clinical language
                mycin_reasoning_str = ""
                if mycin_reasoning:
                    reasoning_lines = []
                    for disease, prob in top_diagnoses[:3]:  # Top 3 diagnoses
                        if disease in mycin_reasoning and mycin_reasoning[disease]:
                            rules_info = mycin_reasoning[disease]
                            # Collect key symptoms/patterns from rules in natural language
                            symptom_patterns = []
                            for rule_info in rules_info[:2]:  # Top 2 rules per disease
                                # Convert technical conditions to natural symptom descriptions
                                natural_symptoms = []
                                for cond_str in rule_info["conditions_met"][:4]:  # Top 4 conditions
                                    # Convert "parameter=value" to natural language
                                    if "=" in cond_str:
                                        param, val = cond_str.split("=", 1)
                                        # Map common parameters to natural descriptions
                                        param_map = {
                                            "fever": "fever",
                                            "cough": "cough",
                                            "productive_cough": "productive cough",
                                            "dyspnea": "shortness of breath",
                                            "wheezing": "wheezing",
                                            "sore_throat": "sore throat",
                                            "chest_pain": "chest pain",
                                            "heartburn": "heartburn",
                                            "smoking": "smoking history",
                                            "copd": "COPD history",
                                            "asthma": "asthma history",
                                            "hiatal_hernia": "hiatal hernia",
                                            "alcohol_use": "alcohol use",
                                            "contact_exposure": "recent contact with similar symptoms"
                                        }
                                        natural_param = param_map.get(param, param.replace("_", " "))
                                        if val.lower() == "true" or val == "1":
                                            natural_symptoms.append(natural_param)
                                        elif val.lower() not in ["false", "0", "none"]:
                                            natural_symptoms.append(f"{natural_param}: {val}")
                                
                                if natural_symptoms:
                                    pattern = ", ".join(natural_symptoms)
                                    # Use rule description if it's clinical, otherwise create natural description
                                    if rule_info['description'] and not any(tech_word in rule_info['description'].lower() for tech_word in ["rule", "condition", "parameter"]):
                                        # Remove disease name from description if it starts with it
                                        desc = rule_info['description']
                                        if desc.startswith(disease + ":"):
                                            desc = desc[len(disease)+1:].strip()
                                        reasoning_lines.append(f"  - {disease}: {desc} (key findings: {pattern})")
                                    else:
                                        reasoning_lines.append(f"  - {disease}: The presence of {pattern} supports this diagnosis")
                    
                    if reasoning_lines:
                        mycin_reasoning_str = "\n\nKey Clinical Patterns Supporting Diagnoses:\n" + "\n".join(reasoning_lines)
                
                # Format comparison in natural clinical language (optional context)
                comparison_str = ""
                if llm_probs and rule_probs:
                    comparison_lines = []
                    for disease, prob in top_diagnoses[:3]:
                        llm_prob = llm_probs.get(disease, 0)
                        rule_prob = rule_probs.get(disease, 0)
                        if rule_prob > llm_prob + 0.05:  # Only show significant differences
                            comparison_lines.append(f"  - {disease}: Systematic analysis increased confidence due to specific symptom patterns")
                        elif disease not in llm_probs and rule_prob > 0.1:
                            comparison_lines.append(f"  - {disease}: Identified through systematic pattern analysis ({prob:.1%} probability)")
                    
                    if comparison_lines:
                        comparison_str = "\n\nNote: Systematic analysis highlighted the following:\n" + "\n".join(comparison_lines)
                
                # Format evidence similar to one-shot LLM
                ev_lines = []
                for k, v in list(evidence.items())[:15]:
                    if v and str(v).lower() not in ["none", "unknown", "false", "no"]:
                        ev_lines.append(f"- {k}: {v}")
                
                # Build explanation prompt similar to one-shot LLM, with MYCIN reasoning added
                mycin_note = "- Key clinical patterns that support diagnoses (from systematic analysis)." if mycin_reasoning_str else ""
                mycin_instruction = "\n\nIMPORTANT: The 'Key Clinical Patterns' section above identifies specific symptom combinations that support diagnoses. Naturally incorporate these patterns into your explanation using clinical reasoning (e.g., 'The combination of fever, productive cough, and smoking history strongly suggests bronchitis, as these are classic indicators')." if mycin_reasoning_str else ""
                
                explanation_prompt = f"""You are a senior clinician performing differential diagnosis.

You are given:
- A patient's demographics.
- A list of symptoms and clinical evidence.
- Differential diagnosis probabilities.
{mycin_note}

--------------------
PATIENT DATA
--------------------
Row index: {row_index}

Demographics:
{json.dumps(demographics, indent=2)}

Evidence:
{chr(10).join(ev_lines)}

--------------------
DIFFERENTIAL DIAGNOSIS PROBABILITIES
--------------------
{diagnoses_str}{mycin_reasoning_str}

{comparison_str if comparison_str else ""}

Your task: Provide a clear, concise explanation of your diagnostic reasoning.

The explanation should:
1. Summarize the key symptoms and clinical presentation
2. Explain which symptoms support each diagnosis and why these probabilities were assigned
3. Address the most important differential diagnoses and why they are more or less likely
4. Use natural clinical language - write as if explaining to a colleague
{mycin_instruction}

Write a clear explanation (3-5 sentences) that focuses on the patient's symptoms and clinical reasoning. Do not mention diagnostic systems or technical processes."""
                
                # Debug: print prompt for first patient
                if row_index == 0:
                    print("\n" + "=" * 80)
                    print("MYCIN EXPLANATION PROMPT (Sample)")
                    print("=" * 80)
                    print(explanation_prompt)
                    print("=" * 80 + "\n")
                
                explanation = llm_call_fn(explanation_prompt)
                # Clean up explanation (remove markdown, extra formatting)
                explanation = re.sub(r'```[^\n]*\n', '', explanation)
                explanation = re.sub(r'^Explanation:?\s*', '', explanation, flags=re.IGNORECASE)
                explanation = explanation.strip()
            except Exception as e:
                explanation = f"Explanation generation failed: {str(e)}"
        
        # Format output
        predictions.append({
            "row_index": row_index,
            "differential_probs": probs,
            "explanation": explanation,
            "llm_baseline_probs": llm_probs,
            "mycin_rule_probs": rule_probs,
            "mycin_reasoning": mycin_reasoning
        })
    
    # Save to CSV if requested
    if save_csv:
        try:
            os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
            with open(csv_output_path, "w", newline="", encoding="utf-8") as csv_f:
                fieldnames = ["row_index", "diagnosis", "probabilities", "explanation"]
                writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
                writer.writeheader()
                
                for pred in predictions:
                    row_index = pred.get("row_index", 0)
                    probs = pred.get("differential_probs", {})
                    explanation = pred.get("explanation", "No explanation provided.")
                    
                    # Get top diagnosis
                    top_diagnosis = None
                    if probs:
                        top_diagnosis = max(probs.items(), key=lambda x: x[1])[0]
                    
                    writer.writerow({
                        "row_index": row_index,
                        "diagnosis": top_diagnosis or "",
                        "probabilities": json.dumps(probs),
                        "explanation": explanation
                    })
            
            print(f"Saved explanations to {csv_output_path}")
        except Exception as e:
            print(f"Warning: Failed to save CSV: {e}")
    
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
                    "content": "You are a medical expert assistant. Answer questions concisely and accurately based on the provided patient information. When asked for differential diagnosis, return ONLY valid JSON with no additional text. When asked for explanations, write natural clinical language."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=800,  # Increased for better explanations
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "UNKNOWN"

