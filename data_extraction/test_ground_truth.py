import pandas as pd
import ast
import json

INPUT_CSV = "data/release_test_patients"
OUTPUT_JSONL = "data_extraction/test_ground_truth.jsonl"
DD_COL = "DIFFERENTIAL_DIAGNOSIS"


def parse_dd_to_dict(dd_str):
    """
    Convert the DIFFERENTIAL_DIAGNOSIS string into:
    {disease_name: percentage_probability}
    """
    if not isinstance(dd_str, str) or dd_str.strip() == "":
        return {}

    try:
        parsed = ast.literal_eval(dd_str)
    except Exception:
        return {}

    result = {}
    for item in parsed:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            disease = str(item[0])
            prob = float(item[1]) * 100   # convert to percentage
            result[disease] = prob

    return result


def main():
    df = pd.read_csv(INPUT_CSV)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out:
        for i, row in df.iterrows():
            mapping = parse_dd_to_dict(row[DD_COL])

            record = {
                "row_index": int(i),
                "differential_probs": mapping
            }

            out.write(json.dumps(record) + "\n")

    print(f"Ground truth saved to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
