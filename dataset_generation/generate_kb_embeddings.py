import argparse
import json
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from kblam.gpt_session import GPT
from kblam.utils.data_utils import DataPoint


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="text-embedding-3-large",
        choices=["all-MiniLM-L6-v2", "text-embedding-3-large", "ada-embeddings"],
    )
    parser.add_argument("--dataset_name", type=str, default="synthetic_data")
    parser.add_argument("--endpoint_url", type=str)
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        help="Path to the dataset in JSON format.",
    )
    parser.add_argument("--output_path", type=str, default="dataset")
    parser.add_argument(
        "--use_openrouter", 
        action="store_true", 
        help="Use OpenRouter API instead of Azure OpenAI"
    )
    parser.add_argument(
        "--openrouter_api_key", 
        type=str, 
        help="OpenRouter API key"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto", 
        help="Device to run the model on (cpu, cuda, auto)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )

    args = parser.parse_args()
    return args


def compute_embeddings(
    encoder_model_spec: str, dataset: list[DataPoint], part: str, batch_size: int = 100, device: str = "auto"
) -> np.array:
    """Compute embeddings for the given dataset in batches using the encoder model spec."""
    embeddings = []
    all_elements = []
    for entity in dataset:
        if part == "key_string":
            all_elements.append(entity.key_string)
        elif part == "description":
            all_elements.append(entity.description)
        else:
            raise ValueError(f"Part {part} not supported.")
    chunks = [
        all_elements[i : i + batch_size]
        for i in range(0, len(all_elements), batch_size)
    ]

    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = SentenceTransformer(encoder_model_spec, device=device)
    for i, chunk in enumerate(tqdm(chunks, desc=f"Processing {part}")):
        embd = model.encode(chunk, convert_to_numpy=True)
        embeddings.append(embd)
        if i % 10 == 0:  # Log progress every 10 batches
            print(f"Processed {i+1}/{len(chunks)} batches for {part}")

    embeddings = np.concatenate(embeddings, 0)
    assert len(embeddings) == len(all_elements)
    return embeddings


if __name__ == "__main__":
    args = parser_args()
    
    if args.verbose:
        print(f"Loading dataset from {args.dataset_path}")
    
    with open(args.dataset_path, "r") as file:
        dataset = [DataPoint(**json.loads(line)) for line in file]
    
    if args.verbose:
        print(f"Loaded {len(dataset)} items from dataset")
        print(f"Using model: {args.model_name}")

    if args.model_name == "all-MiniLM-L6-v2":
        if args.verbose:
            print("Computing embeddings with sentence-transformers...")
            
        key_embeds = compute_embeddings(args.model_name, dataset, "key_string", device=args.device)
        value_embeds = compute_embeddings(args.model_name, dataset, "description", device=args.device)
    elif args.model_name in ["ada-embeddings", "text-embedding-3-large"]:
        if args.verbose:
            if args.use_openrouter:
                print("Computing embeddings with OpenRouter API...")
            else:
                print("Computing embeddings with Azure OpenAI API...")
        
        gpt = GPT(
            args.model_name, 
            args.endpoint_url, 
            use_openrouter=args.use_openrouter,
            api_key=args.openrouter_api_key
        )

        key_embeds = []
        value_embeds = []

        for i, entity in enumerate(tqdm(dataset, desc="Generating embeddings")):
            key_embeds.append(gpt.generate_embedding(entity.key_string))
            value_embeds.append(gpt.generate_embedding(entity.description))
            
            if args.verbose and i % 10 == 0:  # Log progress every 10 items
                print(f"Processed {i+1}/{len(dataset)} items")
    else:
        raise ValueError(f"Model {args.model_name} not supported.")

    os.makedirs(args.output_path, exist_ok=True)

    if args.model_name == "all-MiniLM-L6-v2":
        save_name = "all-MiniLM-L6-v2"
    elif args.model_name == "ada-embeddings":
        save_name = "OAI"
    else:
        save_name = "BigOAI"
    
    # Add OpenRouter suffix if using OpenRouter
    if args.use_openrouter and args.model_name in ["ada-embeddings", "text-embedding-3-large"]:
        save_name = f"{save_name}_openrouter"

    output_key_file = f"{args.output_path}/{args.dataset_name}_{save_name}_embd_key.npy"
    output_value_file = f"{args.output_path}/{args.dataset_name}_{save_name}_embd_value.npy"
    
    np.save(output_key_file, np.array(key_embeds))
    np.save(output_value_file, np.array(value_embeds))
    
    if args.verbose:
        print(f"Saved key embeddings to {output_key_file}")
        print(f"Saved value embeddings to {output_value_file}")
        print("Done!")
