import os
import json
import csv
import re
from typing import List, Dict

from openai import OpenAI

# -------------
# CONFIG
# -------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # or hardcode for quick tests
MODEL_NAME = "gpt-4o"  # or "gpt-4.1" / any other model you prefer

INPUT_PATIENTS = "outputs/patient_payloads.jsonl"
DISEASES_TXT = "data_extraction/diagnoses_from_json.txt"
OUTPUT_JSONL = "results/llm_differentials.jsonl"
OUTPUT_CSV = "results/llm_differentials_explanations.csv"

# Client will be initialized in main() after checking API key
client = None


def load_disease_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        diseases = [line.strip() for line in f if line.strip()]
    return diseases


def load_patients(path: str) -> List[Dict]:
    patients = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            patients.append(json.loads(line))
    return patients


def build_prompt(p: Dict, diseases: List[str]) -> str:
    row_index = p["row_index"]
    demographics = p.get("demographics", {})
    evidence = p.get("evidence", {})

    ev_lines = []
    for k, v in evidence.items():
        ev_lines.append(f"- {k}: {v}")

    disease_list_str = "\n".join([f"- {d}" for d in diseases])

    prompt = f"""
You are a senior clinician performing differential diagnosis.

You are given:
- A patient's demographics.
- A list of symptoms and clinical evidence.
- A list of possible diseases that you MUST choose from.

Your task:
1. Identify the most plausible diseases from the allowed list.
2. Select **no more than 10 diseases**.
   - You may select fewer than 10 if clinically appropriate.
3. Assign a probability to each selected disease.
4. Ensure:
   - Each probability is a **float between 0 and 1**.
   - The **sum of all probabilities is exactly 1.0** (after normal rounding).
   - You **only** output diseases from the allowed disease list.
5. Provide a clear, concise explanation of your diagnostic reasoning.

Your response **MUST** be valid JSON in the following structure:

{{
  "row_index": {row_index},
  "differential_probs": {{
    "Disease A": 0.40,
    "Disease B": 0.25,
    "Disease C": 0.35
  }},
  "explanation": "A clear explanation of the diagnostic reasoning, including which symptoms support each diagnosis and why these probabilities were assigned."
}}

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
{disease_list_str}
"""
    return prompt


def call_llm(prompt: str) -> Dict:
    """
    Call the LLM and parse the JSON it returns.
    We assume it follows instructions and returns a valid JSON object.
    You can add extra safety checks / retries in practice.
    """
    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")
        client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert clinician. "
                    "You must strictly follow the requested JSON output format. "
                    "Include a clear explanation of your diagnostic reasoning."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,  # low temperature for more consistent outputs
    )

    content = response.choices[0].message.content
    # Try to extract JSON from response (may have markdown or extra text)
    try:
        # Remove markdown code blocks if present
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        content = content.strip()
        
        # Try to find JSON object
        json_match = re.search(r'\{[^{}]*"differential_probs"[^{}]*\{[^{}]*\}[^{}]*"explanation"[^{}]*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # In practice, you'd add a retry with a 'fix JSON' prompt here.
        raise ValueError(f"Model returned non-JSON content:\n{content}")

    return parsed


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY env var.")
    
    global client
    client = OpenAI(api_key=api_key)

    diseases = load_disease_list(DISEASES_TXT)
    patients = load_patients(INPUT_PATIENTS)

    # Prepare CSV data
    csv_rows = []

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_f:
        for p in patients:
            prompt = build_prompt(p, diseases)
            result = call_llm(prompt)

            # Optionally normalize just in case probabilities are slightly off
            probs = result.get("differential_probs", {})
            total = sum(probs.values())
            if total > 0:
                probs = {k: v / total for k, v in probs.items()}
                result["differential_probs"] = probs

            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            
            # Prepare CSV row
            row_index = result.get("row_index", p.get("row_index", len(csv_rows)))
            explanation = result.get("explanation", "No explanation provided.")
            
            # Get top diagnosis
            top_diagnosis = None
            if probs:
                top_diagnosis = max(probs.items(), key=lambda x: x[1])[0]
            
            csv_rows.append({
                "row_index": row_index,
                "diagnosis": top_diagnosis or "",
                "probabilities": json.dumps(probs),
                "explanation": explanation
            })

    # Write CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csv_f:
        fieldnames = ["row_index", "diagnosis", "probabilities", "explanation"]
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Wrote LLM differentials to {OUTPUT_JSONL}")
    print(f"Wrote explanations to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
