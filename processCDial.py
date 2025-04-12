import pandas as pd

# Load the CSV file
file_path = '../raw_data/CDial-Anti-Bias.csv'
df = pd.read_csv(file_path)

# Update dataframe to have the two specified columns
df_updated = pd.DataFrame({
    'Question ': df.iloc[:, 3],
    'Potentially_Pejorative': 'Potentially'
})

print(df_updated.head())

# Update dataframe to have unique questions only
df_unique_updated = df_updated.drop_duplicates(subset=['Question '])

# Save the unique dataframe to a new TSV file
unique_updated_file_path = '../processed_data/CDial-Anti-Bias.tsv'
df_unique_updated.to_csv(unique_updated_file_path, sep='\t', index=False)
