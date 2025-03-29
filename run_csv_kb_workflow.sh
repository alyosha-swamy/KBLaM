#!/bin/bash

# No need to activate environment as we'll use system python3

# Path to the CSV file
CSV_PATH="../questions_and_full_forms.csv"

# Output directories
OUTPUT_DIR="datasets/qa_kb"
mkdir -p $OUTPUT_DIR

echo "Step 1: Processing CSV file to JSON format"
python3 -m dataset_generation.process_csv_data \
  --csv_path "$CSV_PATH" \
  --output_path "$OUTPUT_DIR" \
  --output_file "qa_data.json"

echo "Step 2: Generating embeddings using local model"
python3 -m dataset_generation.generate_kb_embeddings \
  --model_name all-MiniLM-L6-v2 \
  --dataset_path "$OUTPUT_DIR/qa_data.json" \
  --dataset_name "qa_kb" \
  --output_path "$OUTPUT_DIR" \
  --device auto \
  --verbose

echo "All done! Embeddings are available in $OUTPUT_DIR folder" 