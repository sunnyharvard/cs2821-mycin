import json

INPUT_FILE = "data/release_evidences.json"
OUTPUT_FILE = "data_extraction/question_en_output.txt"

def main():
    # Load the JSON
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []

    # Iterate through each top-level entry
    for key, entry in data.items():
        if isinstance(entry, dict) and "question_en" in entry:
            questions.append(entry["question_en"])

    # Save all extracted English questions
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(q + "\n")

    print(f"Extracted {len(questions)} English questions → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
