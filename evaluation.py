import json
import math
from scipy.spatial.distance import cosine

EPSILON = 1e-12

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def align_distributions(gt_probs, pred_probs):
    all_diseases = set(gt_probs.keys()) | set(pred_probs.keys())
    gt_aligned = {}
    pred_aligned = {}
    for disease in all_diseases:
        gt_aligned[disease] = gt_probs.get(disease, 0.0)
        pred_aligned[disease] = pred_probs.get(disease, 0.0)
    return gt_aligned, pred_aligned

def kl_divergence(p, q):
    kl = 0.0
    for key in p:
        p_val = max(p[key], EPSILON)
        q_val = max(q[key], EPSILON)
        kl += p_val * math.log(p_val / q_val)
    return kl

def cross_entropy(p, q):
    ce = 0.0
    for key in p:
        p_val = max(p[key], EPSILON)
        q_val = max(q[key], EPSILON)
        ce -= p_val * math.log(q_val)
    return ce

def cosine_similarity(p, q):
    p_vec = [p[k] for k in p]
    q_vec = [q[k] for k in q]
    return 1 - cosine(p_vec, q_vec)

def l1_distance(p, q):
    return sum(abs(p[k] - q[k]) for k in p)

def evaluate(gt_file, pred_file, output_file="results/evaluation_results.jsonl"):
    gt_data = load_jsonl(gt_file)
    pred_data = load_jsonl(pred_file)

    assert len(gt_data) == len(pred_data), "Ground truth and prediction files must have same number of rows"

    kl_scores = []
    ce_scores = []
    cosine_scores = []
    l1_scores = []

    with open(output_file, 'w') as out_f:
        for gt_row, pred_row in zip(gt_data, pred_data):
            assert gt_row['row_index'] == pred_row['row_index'], "Row indices do not match"

            gt_probs, pred_probs = align_distributions(gt_row['differential_probs'], pred_row['differential_probs'])

            kl = kl_divergence(gt_probs, pred_probs)
            ce = cross_entropy(gt_probs, pred_probs)
            cos_sim = cosine_similarity(gt_probs, pred_probs)
            l1 = l1_distance(gt_probs, pred_probs)

            # Save row-wise metrics
            out_f.write(json.dumps({
                "row_index": gt_row['row_index'],
                "KL_divergence": kl,
                "Cross_entropy": ce,
                "Cosine_similarity": cos_sim,
                "L1_distance": l1
            }) + "\n")

            # Collect for averages
            kl_scores.append(kl)
            ce_scores.append(ce)
            cosine_scores.append(cos_sim)
            l1_scores.append(l1)

    # Print averages
    print("=== Overall Evaluation ===")
    print(f"Average KL Divergence: {sum(kl_scores)/len(kl_scores):.4f}")
    print(f"Average Cross-Entropy: {sum(ce_scores)/len(ce_scores):.4f}")
    print(f"Average Cosine Similarity: {sum(cosine_scores)/len(cosine_scores):.4f}")
    print(f"Average L1 Distance: {sum(l1_scores)/len(l1_scores):.4f}")

    print(f"\nRow-wise metrics saved to '{output_file}'")

if __name__ == "__main__":
    gt_file = "data_extraction/test_ground_truth.jsonl"
    pred_file = "results/llm_differentials.jsonl"
    evaluate(gt_file, pred_file)
