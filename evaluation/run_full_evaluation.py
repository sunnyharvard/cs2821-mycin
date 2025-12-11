#!/usr/bin/env python3
"""
Run full evaluation of MYCIN Medical Diagnosis System
"""

import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.mycin_medical_pipeline import run_mycin_medical_pipeline, gpt4o_llm_call

def main():
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set!")
        print("Please set it with: export OPENAI_API_KEY='your-key'")
        return
    
    # Load all patient payloads
    patient_payloads = []
    try:
        with open("outputs/patient_payloads.jsonl", "r") as f:
            for line in f:
                if line.strip():
                    patient_payloads.append(json.loads(line))
        print(f"Loaded {len(patient_payloads)} patient payloads")
    except FileNotFoundError:
        print("ERROR: outputs/patient_payloads.jsonl not found!")
        print("Please run test_set_pipeline.py first to generate patient payloads.")
        return
    
    print(f"\nRunning MYCIN Medical Diagnosis on {len(patient_payloads)} patients...")
    print("This may take a while due to LLM API calls...")
    print("=" * 60)
    
    try:
        predictions = run_mycin_medical_pipeline(
            patient_payloads,
            llm_call_fn=gpt4o_llm_call,
            use_llm_for_extraction=True,
            use_llm_for_questions=True
        )
        
        # Save predictions in ground truth format
        os.makedirs("results", exist_ok=True)
        output_file = "results/mycin_medical_differentials.jsonl"
        with open(output_file, "w") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")
        
        print(f"\n✅ Success! Generated predictions for {len(predictions)} patients")
        print(f"✅ Results saved to {output_file}")
        
        # Show sample predictions
        print("\nSample predictions (first 3):")
        for i, pred in enumerate(predictions[:3]):
            print(f"\nPatient {pred.get('row_index', i)}:")
            probs = pred.get('differential_probs', {})
            if probs:
                top_3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
                for disease, prob in top_3:
                    print(f"  {disease}: {prob:.4f}")
            else:
                print("  No predictions")
        
        # Run evaluation
        print("\n" + "=" * 60)
        print("Running evaluation...")
        print("=" * 60)
        
        from evaluation import evaluate
        gt_file = "data_extraction/test_ground_truth.jsonl"
        pred_file = output_file
        evaluate(gt_file, pred_file, "results/mycin_medical_evaluation.jsonl")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

