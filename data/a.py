import json
import pandas as pd

def extract_questions_from_jsonl(input_path, output_path):
    questions = []

    # Read and parse each line as a separate JSON object
    with open(input_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                item = json.loads(line)
                if 'title' in item:
                    questions.append(item['title'])
            except json.JSONDecodeError:
                continue  # skip lines that are not valid JSON

    # Create a DataFrame with the required format
    df = pd.DataFrame({
        'Question': questions,
        'Potentially_Pejorative': ''
    })

    # Save to TSV
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Saved {len(df)} questions to {output_path}")

# Example usage
if __name__ == "__main__":
    input_file = 'questions_to_self_annotate.json'  # Change this path as needed
    output_file = '../labeling_questions/processed_data/questions_to_self_annotate_cleaned.tsv'
    extract_questions_from_jsonl(input_file, output_file)
