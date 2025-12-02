import os
import json
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

client = OpenAI(api_key=OPENAI_API_KEY)


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
{disease_list_str}
"""
    return prompt


def call_llm(prompt: str) -> Dict:
    """
    Call the LLM and parse the JSON it returns.
    We assume it follows instructions and returns a valid JSON object.
    You can add extra safety checks / retries in practice.
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert clinician. "
                    "You must strictly follow the requested JSON output format."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,  # low temperature for more consistent outputs
    )

    content = response.choices[0].message.content
    # content should be a JSON string
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # In practice, you'd add a retry with a 'fix JSON' prompt here.
        raise ValueError(f"Model returned non-JSON content:\n{content}")

    return parsed


def main():
    if not OPENAI_API_KEY:
        raise RuntimeError("Please set OPENAI_API_KEY env var.")

    diseases = load_disease_list(DISEASES_TXT)
    patients = load_patients(INPUT_PATIENTS)

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

    print(f"Wrote LLM differentials to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
