#!/usr/bin/env python3
"""
Evaluate explanation quality from a patient's perspective using GPT-4o as a judge.

Scores explanations from both one-shot LLM and MYCIN approaches
on a scale of 0-100 based on patient satisfaction criteria.
"""

import os
import json
import csv
import re
from typing import List, Dict, Tuple
from openai import OpenAI

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o"

# Input files
LLM_CSV = "results/llm_differentials_explanations.csv"
MYCIN_CSV = "results/mycin_medical_explanations.csv"
PATIENT_PAYLOADS = "outputs/patient_payloads.jsonl"

# Output file
OUTPUT_CSV = "results/patient_satisfaction_evaluations.csv"


def load_patient_payloads() -> Dict[int, Dict]:
    """Load patient payloads with demographics and evidence."""
    patients = {}
    try:
        with open(PATIENT_PAYLOADS, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    payload = json.loads(line)
                    row_index = payload.get("row_index", len(patients))
                    patients[row_index] = payload
    except FileNotFoundError:
        print(f"Warning: {PATIENT_PAYLOADS} not found. Patient context will be limited.")
    except Exception as e:
        print(f"Error loading {PATIENT_PAYLOADS}: {e}")
    
    return patients


def load_explanations(csv_path: str) -> List[Dict]:
    """Load explanations from CSV file."""
    explanations = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                explanations.append({
                    "row_index": int(row["row_index"]),
                    "diagnosis": row["diagnosis"],
                    "probabilities": json.loads(row["probabilities"]),
                    "explanation": row["explanation"]
                })
    except FileNotFoundError:
        print(f"Warning: {csv_path} not found")
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
    
    return explanations


def create_patient_satisfaction_prompt(
    row_index: int,
    diagnosis: str,
    probabilities: Dict,
    explanation: str,
    method: str,
    demographics: Dict = None,
    evidence: Dict = None
) -> str:
    """Create prompt for LLM judge to evaluate explanation from patient's perspective."""
    
    # Format top diagnoses
    top_diagnoses = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
    diagnoses_str = "\n".join([f"  - {disease}: {prob:.1%}" for disease, prob in top_diagnoses])
    
    # Format patient context
    patient_context = ""
    if demographics:
        patient_context += f"\nPatient Demographics:\n"
        if "AGE" in demographics:
            patient_context += f"  - Age: {demographics['AGE']}\n"
        if "SEX" in demographics:
            patient_context += f"  - Sex: {demographics['SEX']}\n"
    
    if evidence:
        # Show key symptoms (limit to most relevant)
        key_symptoms = []
        for key, value in list(evidence.items())[:10]:  # Limit to first 10
            if value and value not in ["No", "False", "false", "N", "n"]:
                key_symptoms.append(f"  - {key}: {value}")
        if key_symptoms:
            patient_context += f"\nKey Symptoms/Findings:\n" + "\n".join(key_symptoms[:8])  # Show top 8
    
    prompt = f"""You are evaluating a medical explanation from a **patient's perspective**. Imagine you are the patient receiving this diagnosis explanation.

Patient Case #{row_index}:{patient_context}

Diagnosis Provided:
- Primary Diagnosis: {diagnosis}
- Differential Diagnosis Probabilities:
{diagnoses_str}

Explanation Provided ({method} method):
{explanation}

Evaluate this explanation based on how well it would satisfy a patient's needs and concerns. Consider:

1. **Clarity & Accessibility**: Can a patient understand this explanation without medical training? Is it free of unnecessary jargon? Is it written in plain language?

2. **Empathy & Reassurance**: Does the explanation show understanding of the patient's concerns? Is it supportive and not unnecessarily alarming? Does it address the patient's emotional needs?

3. **Transparency & Trust**: Does it clearly explain WHY this diagnosis was chosen? Does it help the patient understand what's happening to their body? Does it build trust in the diagnostic process? **Patients highly value explanations that show the reasoning process - they want to understand HOW the doctor reached this conclusion, not just WHAT the conclusion is.**

4. **Systematic Reasoning**: Does the explanation show structured, thoughtful consideration of the patient's symptoms? Does it connect specific symptoms to specific diagnoses? **Patients feel more confident when they see that their symptoms were systematically analyzed and connected to the diagnosis.** Explanations that show this systematic approach score significantly higher.

5. **Completeness & Differential Analysis**: Does it address the patient's main symptoms and concerns? Does it explain why other possibilities are less likely (without being overwhelming)? **Patients appreciate when doctors consider multiple possibilities and explain why certain diagnoses are more or less likely - this shows thoroughness and builds trust.**

6. **Natural Communication**: Does it feel like a real conversation with a doctor, or does it feel robotic/technical? Would a patient feel heard and understood?

IMPORTANT EVALUATION GUIDELINES:
- **Highly reward systematic reasoning**: Explanations that show structured, step-by-step consideration of symptoms and their connections to diagnoses should score significantly higher. Patients value seeing HOW the diagnosis was reached.
- **Reward specific symptom-disease connections**: Explanations that explicitly link the patient's specific symptoms to specific diagnoses (e.g., "your burning chest pain and heartburn point to GERD") are much more valuable than generic statements. This shows the doctor really listened and analyzed the patient's unique situation.
- **Reward comprehensive differential analysis**: Explanations that address why other diagnoses are less likely show thoroughness and help patients understand the full picture. This builds significant trust.
- **Reward transparency in reasoning**: Patients want to understand the diagnostic process. Explanations that show WHY certain symptoms support the diagnosis and WHY others don't score much higher.
- **Reward patient-centered language**: Explanations that speak directly to the patient's experience and concerns score higher
- **Reward clarity over technical precision**: A clear, simple explanation is better than a technically perfect but confusing one
- **Reward empathy**: Explanations that acknowledge the patient's situation and provide reassurance score higher
- **Penalize jargon**: Medical terminology that isn't explained or isn't necessary should lower the score
- **Value depth of reasoning**: More detailed explanations that show thorough, systematic analysis should score higher than brief summaries, as long as they remain accessible

Provide a score from 0 to 100, where:
- 90-100: Excellent - Would make a patient feel heard, understood, and confident. Shows systematic reasoning, clear symptom-disease connections, and transparent diagnostic process. Clear, empathetic, and builds trust.
- 80-89: Very Good - Strong patient communication with systematic reasoning and good symptom connections. Minor areas for improvement.
- 70-79: Good - Adequate explanation with some systematic reasoning, but could be more detailed in showing symptom-disease connections or more patient-focused
- 60-69: Fair - Understandable but lacks systematic reasoning, specific symptom connections, or patient-centered approach
- 50-59: Poor - Too technical, confusing, generic, or doesn't show how the diagnosis was reached
- 0-49: Very Poor - Would confuse or alarm a patient, or is completely unhelpful

Respond with ONLY a JSON object in this exact format:
{{
  "score": 85,
  "reasoning": "Brief 1-2 sentence explanation of the score from a patient's perspective"
}}

No additional text, no markdown, just the JSON object."""

    return prompt


def evaluate_patient_satisfaction(
    client: OpenAI,
    row_index: int,
    diagnosis: str,
    probabilities: Dict,
    explanation: str,
    method: str,
    demographics: Dict = None,
    evidence: Dict = None
) -> Tuple[int, str]:
    """Evaluate a single explanation from patient's perspective using GPT-4o."""
    
    # Skip if explanation is missing or "UNKNOWN"
    if not explanation or explanation.strip() == "UNKNOWN" or explanation.strip() == "":
        return 0, "No explanation provided"
    
    prompt = create_patient_satisfaction_prompt(
        row_index, diagnosis, probabilities, explanation, method, demographics, evidence
    )
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are evaluating medical explanations from a patient's perspective. Always respond with valid JSON only."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=200,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean up response (remove markdown if present)
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        content = content.strip()
        
        # Try to extract JSON
        json_match = re.search(r'\{[^{}]*"score"[^{}]*"reasoning"[^{}]*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        
        result = json.loads(content)
        score = int(result.get("score", 0))
        reasoning = result.get("reasoning", "No reasoning provided")
        
        return score, reasoning
        
    except Exception as e:
        print(f"Error evaluating patient satisfaction for row {row_index}: {e}")
        return 0, f"Evaluation error: {str(e)}"


def main():
    """Main evaluation function."""
    
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    print("Loading patient data and explanations...")
    patient_payloads = load_patient_payloads()
    llm_explanations = load_explanations(LLM_CSV)
    mycin_explanations = load_explanations(MYCIN_CSV)
    
    print(f"Loaded {len(patient_payloads)} patient payloads")
    print(f"Loaded {len(llm_explanations)} LLM explanations")
    print(f"Loaded {len(mycin_explanations)} MYCIN explanations")
    
    # Evaluate LLM explanations
    print("\nEvaluating LLM explanations from patient's perspective...")
    llm_scores = []
    llm_evaluations = []
    
    for i, exp in enumerate(llm_explanations):
        row_idx = exp["row_index"]
        patient = patient_payloads.get(row_idx, {})
        demographics = patient.get("demographics", {})
        evidence = patient.get("evidence", {})
        
        print(f"  Evaluating LLM explanation {i+1}/{len(llm_explanations)}...", end="\r")
        score, reasoning = evaluate_patient_satisfaction(
            client,
            row_idx,
            exp["diagnosis"],
            exp["probabilities"],
            exp["explanation"],
            "One-Shot LLM",
            demographics,
            evidence
        )
        llm_scores.append(score)
        llm_evaluations.append({
            "row_index": row_idx,
            "method": "One-Shot LLM",
            "diagnosis": exp["diagnosis"],
            "score": score,
            "reasoning": reasoning,
            "explanation": exp["explanation"]
        })
    
    print(f"\n  Completed LLM evaluations")
    
    # Evaluate MYCIN explanations
    print("\nEvaluating MYCIN explanations from patient's perspective...")
    mycin_scores = []
    mycin_evaluations = []
    
    for i, exp in enumerate(mycin_explanations):
        row_idx = exp["row_index"]
        patient = patient_payloads.get(row_idx, {})
        demographics = patient.get("demographics", {})
        evidence = patient.get("evidence", {})
        
        print(f"  Evaluating MYCIN explanation {i+1}/{len(mycin_explanations)}...", end="\r")
        score, reasoning = evaluate_patient_satisfaction(
            client,
            row_idx,
            exp["diagnosis"],
            exp["probabilities"],
            exp["explanation"],
            "MYCIN Expert System",
            demographics,
            evidence
        )
        mycin_scores.append(score)
        mycin_evaluations.append({
            "row_index": row_idx,
            "method": "MYCIN Expert System",
            "diagnosis": exp["diagnosis"],
            "score": score,
            "reasoning": reasoning,
            "explanation": exp["explanation"]
        })
    
    print(f"\n  Completed MYCIN evaluations")
    
    # Calculate averages
    llm_avg = sum(llm_scores) / len(llm_scores) if llm_scores else 0
    mycin_avg = sum(mycin_scores) / len(mycin_scores) if mycin_scores else 0
    
    # Print results
    print("\n" + "=" * 60)
    print("PATIENT SATISFACTION EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nOne-Shot LLM Method:")
    print(f"  Average Patient Satisfaction Score: {llm_avg:.2f}/100")
    print(f"  Number of Explanations: {len(llm_scores)}")
    print(f"  Score Range: {min(llm_scores)} - {max(llm_scores)}")
    
    print(f"\nMYCIN Expert System Method:")
    print(f"  Average Patient Satisfaction Score: {mycin_avg:.2f}/100")
    print(f"  Number of Explanations: {len(mycin_scores)}")
    print(f"  Score Range: {min(mycin_scores)} - {max(mycin_scores)}")
    
    print(f"\nDifference: {abs(llm_avg - mycin_avg):.2f} points")
    if llm_avg > mycin_avg:
        print(f"One-Shot LLM explanations score {llm_avg - mycin_avg:.2f} points higher on patient satisfaction")
    elif mycin_avg > llm_avg:
        print(f"MYCIN explanations score {mycin_avg - llm_avg:.2f} points higher on patient satisfaction")
    else:
        print("Both methods have the same average patient satisfaction score")
    
    # Save detailed results to CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["row_index", "method", "diagnosis", "score", "reasoning", "explanation"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(llm_evaluations)
        writer.writerows(mycin_evaluations)
    
    print(f"\nDetailed results saved to {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()

