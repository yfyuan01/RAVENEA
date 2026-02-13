import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from gdeval import retrieval_eval
import re

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def run_inference(model, processor, model_name, queries, doc_id_to_text, output_path):
    print(f"Running inference for {model_name}...")
    model.to(device)
    model.eval()

    predictions = []

    for query in tqdm(queries, desc="Processing queries"):
        q_id = query['file_name']
        image_path = Path(query['file_name'])
        
        if not image_path.exists():
            pass
            print(image_path)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Could not load image {image_path}: {e}")
            continue

        # Get candidates
        candidate_ids = query.get('enwiki_ids', [])
        if not candidate_ids:
            continue
            
        candidate_texts = []
        valid_candidate_ids = []
        for doc_id in candidate_ids:
            text = doc_id_to_text.get(doc_id)
            if text:
                candidate_texts.append(text)
                valid_candidate_ids.append(doc_id)
            
        if not candidate_texts:
            continue

        # Use raw text content, no prompt
        texts = candidate_texts

        try:
            inputs = processor(text=texts, images=image, padding="max_length", truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = model(**inputs)
            
            logits_per_image = outputs.logits_per_image # shape: (1, num_candidates)
            
            scores = logits_per_image.squeeze(0).cpu().tolist()
            
            # Rank
            ranked_results = sorted(zip(valid_candidate_ids, scores), key=lambda x: x[1], reverse=True)
            
            for rank, (doc_id, score) in enumerate(ranked_results):
                # q_id, Q0, doc_id, rank, score, run_name
                predictions.append([q_id, "Q0", doc_id, rank, score, model_name])

        except Exception as e:
            print(f"Error processing {q_id}: {e}")
            continue

    # Save results
    output_file = output_path / f"{model_name}.run"
    print(f"Saving results to {output_file}")
    df = pd.DataFrame(predictions, columns=["q_id", "Q0", "doc_id", "rank", "score", "type"])
    df.to_csv(output_file, sep="\t", header=False, index=False)

    retrieval_eval("./ravenea/metadata_test.jsonl", output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for vision-language models on RAVENEA dataset")
    parser.add_argument(
        "--model_id",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Model ID to run inference on (default: openai/clip-vit-large-patch14)"
    )
    args = parser.parse_args()
    
    output_path = Path("./ret_outputs/")
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading wiki documents...")
    wiki_docs_list = load_jsonl("./ravenea/wiki_documents.jsonl")
    doc_id_to_text = {doc['id']: re.sub(r'(?m)^#+\s.*$', '', doc['text']).strip() for doc in wiki_docs_list}

    print(f"Loaded {len(doc_id_to_text)} unique documents.")
    
    print("Loading queries...")
    queries = load_jsonl("./ravenea/metadata_test.jsonl")

    model_id = args.model_id
    print(f"Loading model: {model_id}")
    try:
        model = AutoModel.from_pretrained(model_id, dtype=torch.float32, attn_implementation="sdpa")
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        model_name = model_id.split("/")[-1]
        run_inference(model, processor, model_name, queries, doc_id_to_text, output_path)
    except Exception as e:
        print(f"Failed to process {model_id}: {e}")
