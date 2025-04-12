import json
import pandas as pd
from sklearn.svm import SVC

# Import the SemanticClassifier from your classifiers module.
# Adjust the import below if your module structure is different.
from semantic_embedding_model import SemanticClassifier

# Parameters
MODEL_NAME = "thenlper/gte-base-zh"  # or your chosen model name
CUSTOM_THRESHOLD = 0.8  # Adjust this value to favor higher precision


def train_classifier():
    """
    Load the training data, instantiate the SemanticClassifier with a Sigmoid SVM,
    and train it on the entire dataset.
    """
    # Load the combined TSV dataset
    data_df = pd.read_csv("./processed_data/combined.tsv", sep="\t")
    # Assuming the column "Question " contains the text and "Potentially_Pejorative"
    # contains the label. Here we convert non-empty values (other than "None" etc.) to True.
    X = data_df["Question "].tolist()
    y = [True if str(val).strip() not in ("", "None", "none", "nan", None) else False
         for val in data_df["Potentially_Pejorative"].tolist()]

    # Instantiate the SemanticClassifier with the SVM (Sigmoid Kernel)
    classifier = SemanticClassifier(
        model=SVC(kernel='sigmoid', probability=True),
        embedding_model_name=MODEL_NAME,
        debug=False,
        threshold=CUSTOM_THRESHOLD
    )

    # Train the classifier with cross-validation (using cv=3 here)
    print("Training SemanticClassifier on combined.tsv ...")
    classifier.train_model(X, y, cv=3)
    print("Training completed.")
    return classifier


def process_json_file(classifier, input_file, output_file, update_every=5):
    """
    Process the input JSON file (one JSON object per line), predict question sensitivity
    using the trained classifier, and write a new JSON file with an additional field
    AI_Predicted_Question_Sensitivity.

    The output file is flushed after every `update_every` lines to update progress.
    """
    with open(input_file, "r", encoding="utf-8") as infile, \
            open(output_file, "w", encoding="utf-8") as outfile:

        for idx, line in enumerate(infile, start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping line {idx} due to JSON error: {e}")
                continue

            # Use the "title" field as the input text for prediction.
            # (Adjust this if you wish to use another field such as "desc".)
            question_text = item.get("title", "")
            # The predict method returns a list, so take the first prediction.
            prediction = classifier.predict(question_text)[0]
            item["AI_Predicted_Question_Sensitivity"] = bool(prediction)

            # Write the updated JSON object to the output file.
            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

            # Flush the output file every few lines to update the file on disk.
            if idx % update_every == 0:
                outfile.flush()
                print(f"Processed {idx} lines and updated the output file.")

    print(f"Processing completed. Total lines processed: {idx}")


if __name__ == "__main__":
    # Train the classifier on the training dataset.
    clf = train_classifier()

    # Process the validation JSON file and output predictions.
    input_json = "../data/baike_qa_train.json"
    output_json = "./output/baike_qa_train_with_predictions.json"
    process_json_file(clf, input_json, output_json, update_every=5)
