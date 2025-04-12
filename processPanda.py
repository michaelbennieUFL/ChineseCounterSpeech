import pandas as pd

# Load the original CSV
input_file_path = '../raw_data/panda_dataset.tsv'
df = pd.read_csv(input_file_path, sep='\t')

# Filter rows where hatespeech includes '?' or '？' and hateScore is 1 or -1
filtered_df = df[
    (df['hatespeech'].str.contains(r'\?|？', regex=True)) &
    (df['hateScore'].isin([1, -1]))
]

# Create the new DataFrame
output_df = pd.DataFrame({
    'Question': filtered_df['hatespeech'],
    'Potentially_Pejorative': 'Potentially',
})

# Save to TSV
output_file_path = '../processed_data/panda_dataset_bias.tsv'
output_df.to_csv(output_file_path, sep='\t', index=False)

# Preview the result
print(output_df.head())
