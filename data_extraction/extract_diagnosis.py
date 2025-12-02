import json

def extract_diagnoses_from_json_dict(d):
    """
    Given the loaded dict, return a sorted list of unique diagnosis names.
    Uses both the top-level keys and the `condition_name` field just in case.
    """
    names_from_keys = list(d.keys())

    names_from_field = []
    for k, v in d.items():
        if isinstance(v, dict) and "condition_name" in v:
            names_from_field.append(str(v["condition_name"]))

    # Combine and deduplicate
    all_names = set(names_from_keys) | set(names_from_field)
    return sorted(all_names)


def load_diagnoses_from_json_file(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return extract_diagnoses_from_json_dict(data)


def save_diagnoses_to_txt(diagnoses, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for name in diagnoses:
            f.write(name + "\n")


if __name__ == "__main__":
    json_path = "data/release_conditions.json"
    output_path = "data_extraction/diagnoses_from_json.txt"

    diagnoses = load_diagnoses_from_json_file(json_path)
    save_diagnoses_to_txt(diagnoses, output_path)

    print(f"Saved {len(diagnoses)} diagnoses to {output_path}")
