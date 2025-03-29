import argparse
import json
import os
import re
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import random

from kblam.models.phi3_model import KBLaMPhi3ForCausalLM
from kblam.models.llama3_model import KblamLlamaForCausalLM
from kblam.models.kblam_config import KBLaMConfig
from kblam.retriever import KBRetriever
from kblam.encoder import KBEncoder

def normalize_text(text):
    """Normalize text for comparison: remove punctuation, lowercase, etc."""
    # Remove special tokens if present
    text = re.sub(r'<\|.*?\|>', '', text)
    # Remove punctuation except in abbreviations like "A.B.C."
    text = re.sub(r'[^\w\s\.]', '', text) 
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def extract_answer(text, abbreviation):
    """Extract the actual expansion from the model's response."""
    # Try common patterns
    patterns = [
        rf"(?:stands|is|means|refers|short|expands|expansion|meaning|form)\s+(?:for|to|of|is)\s+(.*)",
        rf"{abbreviation}\s+(?:stands|is|means|refers|short|expands|expansion|meaning|form)\s+(?:for|to|of|is)\s+(.*)",
        rf"(?:the|a|an)\s+(?:full|complete|expanded|correct)\s+(?:form|meaning|expansion|version)\s+(?:of|for)\s+{abbreviation}\s+(?:is|would be)\s+(.*)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # If no pattern matches, return the whole text as a fallback
    return text.strip()

def calculate_metrics(predictions, ground_truths, abbreviations):
    """Calculate various metrics for the evaluation."""
    exact_matches = 0
    semantic_similarities = []
    
    # Load sentence transformer model for semantic similarity
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    results = []
    
    for i, (pred, gt, abbr) in enumerate(zip(predictions, ground_truths, abbreviations)):
        # Clean prediction and ground truth
        clean_pred = normalize_text(pred)
        clean_gt = normalize_text(gt)
        
        # Extract the expansion from prediction if it's a full sentence
        extracted_pred = extract_answer(clean_pred, abbr)
        
        # Calculate exact match
        is_exact_match = clean_gt in extracted_pred or extracted_pred in clean_gt
        if is_exact_match:
            exact_matches += 1
        
        # Calculate semantic similarity
        embedding1 = model.encode(extracted_pred, convert_to_tensor=True)
        embedding2 = model.encode(clean_gt, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
        semantic_similarities.append(similarity)
        
        # Store result for this example
        results.append({
            'abbreviation': abbr,
            'prediction': pred,
            'extracted_prediction': extracted_pred,
            'ground_truth': gt,
            'exact_match': is_exact_match,
            'semantic_similarity': similarity
        })
    
    metrics = {
        'exact_match_rate': exact_matches / len(predictions) if predictions else 0,
        'avg_semantic_similarity': np.mean(semantic_similarities) if semantic_similarities else 0,
        'results': results
    }
    
    return metrics

def generate_answer(model, tokenizer, question, kb_retriever, kb_config, device, max_new_tokens=50):
    """Generate an answer using the KBLaM model."""
    # Tokenize the input
    inputs = tokenizer(question, return_tensors="pt").to(device)
    
    # Get KB embeddings for this question
    kb_embedding = kb_retriever.retrieve_kb(question)
    
    # Generate the answer
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            kb_kvs=kb_embedding,
            kb_config=kb_config,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.7,
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the answer part (after the question)
    answer = generated_text.split(question)[-1].strip()
    
    return answer

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained KBLaM model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--dataset_path", type=str, default="/ephemeral/KBLaM/datasets/qa_kb/qa_kb.json", 
                        help="Path to the test dataset JSON file")
    parser.add_argument("--hf_token", type=str, help="HuggingFace token for accessing models")
    parser.add_argument("--num_examples", type=int, default=100, help="Number of examples to evaluate")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", 
                        help="Path to save evaluation results")
    parser.add_argument("--hf_model_spec", type=str, default="microsoft/Phi-3-mini-4k-instruct",
                        help="Base HuggingFace model specification")
    parser.add_argument("--llm_type", type=str, default="phi3", choices=["phi3", "llama3"],
                        help="Type of LLM model (phi3 or llama3)")
    args = parser.parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.hf_model_spec}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model_spec,
        trust_remote_code=True,
        token=args.hf_token
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    if args.llm_type == "phi3":
        model = KBLaMPhi3ForCausalLM.from_pretrained(
            args.model_path,
            device_map=device,
            torch_dtype="auto",
            trust_remote_code=True,
        )
    else:  # llama3
        model = KblamLlamaForCausalLM.from_pretrained(
            args.model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            token=args.hf_token,
        )
    model.eval()
    
    # Load KB config
    kb_config = KBLaMConfig.from_pretrained(os.path.join(args.model_path, "kb_config.json"))
    
    # Load encoder
    print("Loading encoder...")
    encoder = KBEncoder(
        encoder_name="all-MiniLM-L6-v2",
        projector_type="linear",
        endpoint_url="",
        out_dim=model.config.hidden_size * (model.config.num_hidden_layers // kb_config.kb_layer_frequency + 1),
        frozen_base_model=True,
        device=device,
    )
    encoder.load_state_dict(torch.load(os.path.join(args.model_path, "encoder.pt")))
    encoder.eval()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    with open(args.dataset_path, "r") as f:
        dataset = [json.loads(line) for line in f]
    
    # Select random examples for evaluation
    if args.num_examples < len(dataset):
        eval_dataset = random.sample(dataset, args.num_examples)
    else:
        eval_dataset = dataset
        print(f"Warning: Requested {args.num_examples} examples but dataset only has {len(dataset)}.")
    
    # Initialize KBRetriever (without knowledge bases for now, we'll query one at a time)
    kb_retriever = KBRetriever(encoder, [], key_embds=None, value_embds=None)
    
    # Evaluate model
    print(f"Evaluating model on {len(eval_dataset)} examples...")
    predictions = []
    ground_truths = []
    abbreviations = []
    
    for item in tqdm(eval_dataset):
        # Extract question and ground truth
        question = item["Q"]
        ground_truth = item["A"]
        
        # Extract abbreviation from question (everything between "What does " and " stand for?")
        abbr_match = re.search(r"(?:what does|what is|define|expand|expand the abbreviation)\s+(\w+)", 
                               question, re.IGNORECASE)
        if abbr_match:
            abbreviation = abbr_match.group(1)
        else:
            abbreviation = item["name"]  # Fallback to the 'name' field
        
        # Generate prediction
        prediction = generate_answer(model, tokenizer, question, kb_retriever, kb_config, device)
        
        predictions.append(prediction)
        ground_truths.append(ground_truth)
        abbreviations.append(abbreviation)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(predictions, ground_truths, abbreviations)
    
    # Print summary metrics
    print(f"Exact Match Rate: {metrics['exact_match_rate']:.4f}")
    print(f"Average Semantic Similarity: {metrics['avg_semantic_similarity']:.4f}")
    
    # Save detailed results
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main() 