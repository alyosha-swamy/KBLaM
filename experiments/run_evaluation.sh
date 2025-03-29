#!/bin/bash

# Default parameters
MODEL_PATH="/ephemeral/KBLaM/experiments/output/stage1_lr_0.0001KBTokenLayerFreq3UseOutlier1UseDataAugKeyFromkey_all-MiniLM-L6-v2_qa_kb_phi3_600"
DATASET_PATH="/ephemeral/KBLaM/datasets/qa_kb/qa_kb.json"
HF_TOKEN=""  # No default token - must be provided by user
NUM_EXAMPLES=100
OUTPUT_FILE="evaluation_results.json"
HF_MODEL_SPEC="microsoft/Phi-3-mini-4k-instruct"
LLM_TYPE="phi3"

# Help message
show_help() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --model_path PATH      Path to the trained model (default: $MODEL_PATH)"
  echo "  --dataset_path PATH    Path to the test dataset (default: $DATASET_PATH)"
  echo "  --hf_token TOKEN       HuggingFace token (REQUIRED)"
  echo "  --num_examples NUM     Number of examples to evaluate (default: $NUM_EXAMPLES)"
  echo "  --output_file FILE     Output file for results (default: $OUTPUT_FILE)"
  echo "  --hf_model_spec MODEL  Base HuggingFace model (default: $HF_MODEL_SPEC)"
  echo "  --llm_type TYPE        Type of LLM model (phi3 or llama3) (default: $LLM_TYPE)"
  echo "  --help                 Show this help message and exit"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --dataset_path)
      DATASET_PATH="$2"
      shift 2
      ;;
    --hf_token)
      HF_TOKEN="$2"
      shift 2
      ;;
    --num_examples)
      NUM_EXAMPLES="$2"
      shift 2
      ;;
    --output_file)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --hf_model_spec)
      HF_MODEL_SPEC="$2"
      shift 2
      ;;
    --llm_type)
      LLM_TYPE="$2"
      shift 2
      ;;
    --help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Check if HF_TOKEN is provided
if [ -z "$HF_TOKEN" ]; then
  echo "Error: HuggingFace token (--hf_token) is required"
  show_help
  exit 1
fi

# Run the evaluation script
python3 /ephemeral/KBLaM/experiments/evaluate_model.py \
  --model_path "$MODEL_PATH" \
  --dataset_path "$DATASET_PATH" \
  --hf_token "$HF_TOKEN" \
  --num_examples "$NUM_EXAMPLES" \
  --output_file "$OUTPUT_FILE" \
  --hf_model_spec "$HF_MODEL_SPEC" \
  --llm_type "$LLM_TYPE"

echo "Evaluation completed. Results saved to $OUTPUT_FILE"

# If results file exists, show a summary
if [ -f "$OUTPUT_FILE" ]; then
  echo "Summary of results:"
  grep -E '"exact_match_rate"|"avg_semantic_similarity"' "$OUTPUT_FILE" | sed 's/"//g' | sed 's/,//g'
fi 