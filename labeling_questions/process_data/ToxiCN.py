import pandas as pd

# Load the CSV file
tox_df = pd.read_csv('../raw_data/ToxiCN_1.0.csv')

# Filter the DataFrame
filtered_df = tox_df[
    (tox_df['toxic'] == 0) &
    (tox_df['content'].str.contains(r'\?|？', regex=True)) &
    (~tox_df['content'].str.contains(r'[.!。！歧恋人暴]', regex=True))
]

# Prepare the final DataFrame
df_output = pd.DataFrame({
    'Question': filtered_df['content'],
    'Potentially_Pejorative': 'None'
})

# Save the output DataFrame to a TSV file
output_file_path = '../processed_data/ToxiCN-neutral-questions.tsv'
df_output.to_csv(output_file_path, sep='\t', index=False)

