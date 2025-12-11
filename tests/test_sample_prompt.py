#!/usr/bin/env python3
"""Test script to show a sample MYCIN explanation prompt"""

import os
import json
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.mycin_medical_pipeline import run_mycin_medical_pipeline, gpt4o_llm_call

# API key should be set via environment variable: export OPENAI_API_KEY='your-key'

# Load one patient
patient_payloads = []
try:
    with open("outputs/patient_payloads.jsonl", "r") as f:
        for i, line in enumerate(f):
            if i == 0:  # Just get first patient
                patient_payloads.append(json.loads(line))
                break
except FileNotFoundError:
    print("Error: outputs/patient_payloads.jsonl not found")
    exit(1)

print("=" * 80)
print("SAMPLE MYCIN EXPLANATION PROMPT")
print("=" * 80)
print()

# We need to modify the pipeline to print the prompt
# Let's create a wrapper that captures the prompt
original_llm_call = gpt4o_llm_call

captured_prompt = None

def capture_prompt_llm_call(prompt: str) -> str:
    global captured_prompt
    captured_prompt = prompt
    print("PROMPT SENT TO LLM:")
    print("=" * 80)
    print(prompt)
    print("=" * 80)
    print()
    # Don't actually call LLM, just return a dummy response
    return "Sample explanation would go here."

# Run with prompt capture
print("Running MYCIN pipeline on first patient...")
print("(This will show the prompt but not make actual API calls)")
print()

# Temporarily patch to capture prompt
import src.mycin_medical_pipeline as mycin_medical_pipeline
original_gpt4o = mycin_medical_pipeline.gpt4o_llm_call
mycin_medical_pipeline.gpt4o_llm_call = capture_prompt_llm_call

# Run pipeline - it will stop after generating the prompt
try:
    results = run_mycin_medical_pipeline(
        patient_payloads,
        llm_call_fn=capture_prompt_llm_call,
        use_llm_for_extraction=False,  # Skip extraction to speed up
        use_llm_for_questions=False,    # Skip questions to speed up
        save_csv=False
    )
except Exception as e:
    print(f"Error (expected - we're just capturing the prompt): {e}")

# Restore original
mycin_medical_pipeline.gpt4o_llm_call = original_gpt4o


