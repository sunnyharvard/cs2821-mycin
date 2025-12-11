#!/usr/bin/env python3
"""
Evaluate explanation quality using GPT-4o as a judge.

Scores explanations from both one-shot LLM and MYCIN approaches
on a scale of 0-100 from a medical professional's perspective.
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

# Output file
OUTPUT_CSV = "results/explanation_evaluations.csv"


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


def create_evaluation_prompt(row_index: int, diagnosis: str, probabilities: Dict, explanation: str, method: str) -> str:
    """Create prompt for LLM judge to evaluate explanation quality."""
    
    # Format top diagnoses
    top_diagnoses = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
    diagnoses_str = "\n".join([f"  - {disease}: {prob:.4f}" for disease, prob in top_diagnoses])
    
    prompt = f"""You are an expert medical professional evaluating the quality and interpretability of a diagnostic explanation.

Patient Case #{row_index}:
- Primary Diagnosis: {diagnosis}
- Differential Diagnosis Probabilities:
{diagnoses_str}

Explanation Provided ({method} method):
{explanation}

Evaluate this explanation from a medical professional's perspective. Consider:
1. **Clarity**: Is the explanation clear and easy to understand?
2. **Clinical Reasoning**: Does it provide sound medical reasoning with specific symptom-disease connections?
3. **Completeness**: Does it adequately explain why this diagnosis was chosen AND address key differential diagnoses?
4. **Systematic Analysis**: Does it show structured consideration of multiple factors, symptoms, and risk factors? (Higher scores for explanations that systematically weigh evidence)
5. **Interpretability**: Can a clinician understand the diagnostic process and reasoning?
6. **Relevance**: Are the key symptoms and findings appropriately addressed with specific connections?
7. **Professional Quality**: Does it meet the standard of a medical professional's explanation?

IMPORTANT EVALUATION GUIDELINES:
- **Reward systematic reasoning**: Explanations that show structured consideration of multiple symptoms, risk factors, and differential diagnoses should score higher
- **Reward specific connections**: Explanations that explicitly link specific symptoms to specific diseases (e.g., "smoking history increases risk of bronchitis") are more valuable than generic statements
- **Penalize generic explanations**: Vague explanations that could apply to many cases should score lower, even if technically correct
- **Value comprehensive differential analysis**: Explanations that address why other diagnoses are less likely show better clinical reasoning
- **Prefer depth over brevity**: More detailed explanations that show thorough analysis should score higher than brief summaries

Provide a score from 0 to 100, where:
- 90-100: Excellent - Professional-grade explanation with clear, systematic reasoning and specific symptom-disease connections
- 80-89: Very Good - Strong explanation with systematic consideration of factors, minor gaps acceptable
- 70-79: Good - Adequate explanation with some systematic reasoning, but could be more detailed or specific
- 60-69: Fair - Basic explanation but lacks depth, systematic analysis, or specific connections
- 50-59: Poor - Explanation is unclear, generic, lacks reasoning, or doesn't show systematic consideration
- 0-49: Very Poor - Explanation is missing, nonsensical, or unhelpful

Respond with ONLY a JSON object in this exact format:
{{
  "score": 85,
  "reasoning": "Brief 1-2 sentence explanation of the score"
}}

No additional text, no markdown, just the JSON object."""

    return prompt


def evaluate_explanation(client: OpenAI, row_index: int, diagnosis: str, probabilities: Dict, explanation: str, method: str) -> Tuple[int, str]:
    """Evaluate a single explanation using GPT-4o."""
    
    # Skip if explanation is missing or "UNKNOWN"
    if not explanation or explanation.strip() == "UNKNOWN" or explanation.strip() == "":
        return 0, "No explanation provided"
    
    prompt = create_evaluation_prompt(row_index, diagnosis, probabilities, explanation, method)
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert medical professional evaluating diagnostic explanations. Always respond with valid JSON only."
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
        print(f"Error evaluating explanation for row {row_index}: {e}")
        return 0, f"Evaluation error: {str(e)}"


def main():
    """Main evaluation function."""
    
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    print("Loading explanations...")
    llm_explanations = load_explanations(LLM_CSV)
    mycin_explanations = load_explanations(MYCIN_CSV)
    
    print(f"Loaded {len(llm_explanations)} LLM explanations")
    print(f"Loaded {len(mycin_explanations)} MYCIN explanations")
    
    # Evaluate LLM explanations
    print("\nEvaluating LLM explanations...")
    llm_scores = []
    llm_evaluations = []
    
    for i, exp in enumerate(llm_explanations):
        print(f"  Evaluating LLM explanation {i+1}/{len(llm_explanations)}...", end="\r")
        score, reasoning = evaluate_explanation(
            client,
            exp["row_index"],
            exp["diagnosis"],
            exp["probabilities"],
            exp["explanation"],
            "One-Shot LLM"
        )
        llm_scores.append(score)
        llm_evaluations.append({
            "row_index": exp["row_index"],
            "method": "One-Shot LLM",
            "diagnosis": exp["diagnosis"],
            "score": score,
            "reasoning": reasoning,
            "explanation": exp["explanation"]
        })
    
    print(f"\n  Completed LLM evaluations")
    
    # Evaluate MYCIN explanations
    print("\nEvaluating MYCIN explanations...")
    mycin_scores = []
    mycin_evaluations = []
    
    for i, exp in enumerate(mycin_explanations):
        print(f"  Evaluating MYCIN explanation {i+1}/{len(mycin_explanations)}...", end="\r")
        score, reasoning = evaluate_explanation(
            client,
            exp["row_index"],
            exp["diagnosis"],
            exp["probabilities"],
            exp["explanation"],
            "MYCIN Expert System"
        )
        mycin_scores.append(score)
        mycin_evaluations.append({
            "row_index": exp["row_index"],
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
    print("EXPLANATION QUALITY EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nOne-Shot LLM Method:")
    print(f"  Average Score: {llm_avg:.2f}/100")
    print(f"  Number of Explanations: {len(llm_scores)}")
    print(f"  Score Range: {min(llm_scores)} - {max(llm_scores)}")
    
    print(f"\nMYCIN Expert System Method:")
    print(f"  Average Score: {mycin_avg:.2f}/100")
    print(f"  Number of Explanations: {len(mycin_scores)}")
    print(f"  Score Range: {min(mycin_scores)} - {max(mycin_scores)}")
    
    print(f"\nDifference: {abs(llm_avg - mycin_avg):.2f} points")
    if llm_avg > mycin_avg:
        print(f"One-Shot LLM explanations are {llm_avg - mycin_avg:.2f} points higher on average")
    elif mycin_avg > llm_avg:
        print(f"MYCIN explanations are {mycin_avg - llm_avg:.2f} points higher on average")
    else:
        print("Both methods have the same average score")
    
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

