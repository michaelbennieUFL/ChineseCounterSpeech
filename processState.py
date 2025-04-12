import json
import pandas as pd

# Define file paths
train_path = '../raw_data/State-ToxiCN-train.json'


# Function to load and filter questions
def extract_non_hate_questions(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Filter: Questions containing '?' or '？', labeled 'non-hate',
    # and excluding exclamation marks, quotation marks, or full stops
    return [
        item['content'] for item in data
        if ('?' in item['content'] or '？' in item['content'])
        and 'non-hate' in item['output']
        and not any(char in item['content'] for char in '!！."“”‘’\'。歧恋人暴')
        and "外国" not in item["content"]
    ]



# Extract questions from both files
train_questions = extract_non_hate_questions(train_path)

# Create the DataFrame
df = pd.DataFrame({
    'Question': train_questions,
    'Potentially_Pejorative': 'Non'
})

# Save to TSV file
output_file_path = '../processed_data/STATE-TociCN-neutral.tsv'
df.to_csv(output_file_path, sep='\t', index=False)

# Preview results
print(f"Combined file saved as {output_file_path}")
print(df.head())
