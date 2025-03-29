import argparse
import csv
import json
import os

from kblam.utils.data_utils import DataPoint, save_entity


def load_questions_and_answers(csv_path):
    """Load questions and answers from the CSV file."""
    dataset = []
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            question = row['question']
            full_form = row['full_form']
            
            # Create a DataPoint similar to the format used in synthetic data generation
            datapoint = DataPoint(
                name=question.replace("?", "").strip(),  # Use question as name without "?"
                description_type="full_form",            # Type of description
                description=full_form,                   # The full form as description
                Q=question,                              # Original question
                A=full_form,  # Set Answer directly to the full form
                key_string=f"full form of {question.replace('?', '').strip()}"  # Key string for retrieval
            )
            dataset.append(datapoint)
    return dataset


def save_dataset(dataset, output_path, output_file):
    """Save dataset to a JSON file in the required format."""
    os.makedirs(output_path, exist_ok=True)
    output_file_path = os.path.join(output_path, output_file)
    
    # Clear the file if it exists
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    
    for datapoint in dataset:
        save_entity(datapoint, output_file_path)
    
    print(f"Saved {len(dataset)} items to {output_file_path}")


def main():
    parser = argparse.ArgumentParser(description="Process CSV data for knowledge base generation")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file with questions and full forms")
    parser.add_argument("--output_path", type=str, default="datasets", help="Output directory for processed data")
    parser.add_argument("--output_file", type=str, default="processed_data.json", help="Output filename")
    args = parser.parse_args()
    
    # Load and process data from CSV
    dataset = load_questions_and_answers(args.csv_path)
    print(f"Loaded {len(dataset)} question-answer pairs from CSV")
    
    # Save processed data
    save_dataset(dataset, args.output_path, args.output_file)


if __name__ == "__main__":
    main() 