import pandas as pd

# Load the CSV files
train_df = pd.read_csv('../raw_data/CHSD-test.csv')
test_df = pd.read_csv('../raw_data/CHSD-train.csv')

# Function to filter rows
def filter_questions(df,label=0):
    return df[
        (df['label'] == label) &
        (df['text'].str.contains(r'\?|？', regex=True))&
        (~df['text'].str.contains(r'[.!。！歧恋人暴]', regex=True))
    ]['text'].tolist()




# Extract filtered questions from both dataframes
train_questions = filter_questions(train_df)
test_questions = filter_questions(test_df)

# Combine and remove duplicates
combined_questions = list(set(train_questions + test_questions))

# Create the DataFrame
df_output = pd.DataFrame({
    'Question': combined_questions,
    'Potentially_Pejorative': 'None'
})

# Save to TSV
output_file_path = '../processed_data/CHSD-neutral.tsv'
df_output.to_csv(output_file_path, sep='\t', index=False)


# Function to filter rows
def filter_questions(df,label=1):
    return df[
        (df['label'] == label) &
        (df['text'].str.contains(r'\?|？', regex=True))&
        (~df['text'].str.contains(r'[.!。！]', regex=True))
    ]['text'].tolist()


# Extract filtered questions from both dataframes
train_questions = filter_questions(train_df,label=1)
test_questions = filter_questions(test_df,label=1)

# Combine and remove duplicates
combined_questions = list(set(train_questions + test_questions))

# Create the DataFrame
df_output = pd.DataFrame({
    'Question': combined_questions,
    'Potentially_Pejorative': 'Potentially'
})

# Save to TSV
output_file_path = '../processed_data/CHSD-sensitive.tsv'
df_output.to_csv(output_file_path, sep='\t', index=False)
