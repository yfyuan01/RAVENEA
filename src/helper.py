import json
from typing import List, Dict, Any
import pandas as pd
import re
from collections import defaultdict
import logging
from transformers import AutoModel, AutoProcessor
import torch
from PIL import Image
from evaluate import load
import numpy as np
import os

logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")

def load_cvqa_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Loads CVQA data from JSONL and flattens it so each question is an example.
    """
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            file_name = item['file_name']
            country = item.get('country', 'Unknown')
            questions = item['questions']
            options = item['options']
            answers = item['answers']
            
            # Identify if options/questions/answers are aligned
            # The provided schema seems to have list of questions and list of options lists
            # We assume len(questions) == len(options) == len(answers)
            
            for i in range(len(questions)):
                # key for retrieval is file_name
                # unique id for this question example
                q_id = f"{file_name}_{i}"
                
                example = {
                    'id': q_id,
                    'file_name': file_name,
                    'image': file_name, # vLLM checks 'image' key if loaded directly, but here we pass path
                    'country': country,
                    'question': questions[i],
                    'options': options[i],
                    'answer': answers[i]
                }
                data.append(example)
    return data

def load_cic_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Loads CIC data from JSONL and flattens it so each question is an example.
    """
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            file_name = item['file_name']
            country = item.get('country', 'Unknown')
            human_caption = item['human_captions']
            
            example = {
                'id': file_name,
                'file_name': file_name,
                'image': file_name, # vLLM checks 'image' key if loaded directly, but here we pass path
                'country': country,
                'human_caption': human_caption,
            }
            data.append(example)
    return data

def load_documents(doc_path: str) -> Dict[str, str]:
    """
    Loads documents from JSONL. Returns dict {id: text}.
    """
    docs = {}
    with open(doc_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Use 'id' from jsonl as key (e.g., 'enwiki/123')
            doc_content = re.sub(r'(?m)^#+\s.*$', '', item['text']).strip()
            # Original code logic: split by space and take first 256 words
            final_content = " ".join(doc_content.split()[:256]).rsplit(". ", 1)[0] + "."
            # first paragraph
            # final_content = " ".join(doc_content.split("\n\n"))[0]
            # final_content = doc_content
            docs[item['id']] = final_content
    return docs

def load_retrieval_run(run_path: str) -> Dict[str, List[str]]:
    """
    Loads TREC run file. Returns {qid: [docid1, docid2, ...]}
    """
    result = defaultdict(list)
    # Using pandas similar to original code but adapted for no header
    # Format: qid Q0 docid rank score runtag
    try:
        df = pd.read_csv(run_path, sep='\t', header=None, quoting=3) # quoting=3 (QUOTE_NONE) helps with some weird lines
        for _, row in df.iterrows():
            if len(row) < 3: 
                continue # Skip malformed
            qid = str(row[0])
            docid = str(row[2])
            result[qid].append(docid)
    except Exception as e:
        logger.error(f"Error loading run file: {e}")
        # Fallback to manual reading if pandas fails on weird separators
        with open(run_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    result[parts[0]].append(parts[2])
    return result

def is_match(predicted: str, ground_truth: str) -> bool:
    """
    Checks if predicted answer matches ground truth.
    Ground truth is expected to be a letter (A, B, C, D) or "A) ...".
    Prediction is expected to contain "Answer: X" or start with "X)".
    """
    # Check for answer pattern in one of two formats:
    # 1. "Answer: A" or "Answer: A) text" anywhere in the string
    # 2. "A)" or "A) text" at the beginning of the string
    # 3. Only one character "A", "B", "C", or "D" at the beginning of the string
    match = re.search(r"Answer:\s*([A-D])(?:\s*\)\s*(.*)?)?", predicted)
    
    if match is None:
        # Look for option pattern at beginning of string
        match = re.search(r"^([A-D])\s*\)\s*(.*)", predicted)

    if match or len(predicted.strip()) == 1:
        pred_option = match.group(1) if match else predicted.strip()
        
        # Ground truth usually is just "A" or "A", but handle "A) ..."
        gt_match = re.search(r"^([A-D])(?:\)|\s|$)", ground_truth)
        gt_option = gt_match.group(1) if gt_match else ground_truth.strip()
        
        return pred_option == gt_option

    return False


def calculate_clip_score(imgs: List[str], gt_texts: List[str], pred_texts: List[str], country: str = None) -> Dict[str, float]:
    """
    Calculates CLIP score for predicted captions using Hugging Face Transformers.
    imgs: List of image paths.
    """
    model_name = "openai/clip-vit-base-patch16"
    
    # Check if we have 0 samples
    if not imgs or not pred_texts:
        return {f"pred_clip_score-{country if country else 'all'}": 0.0}

    # Load model & processor
    # We load them each time to allow for easy state reset, but for efficiency in a script called once it's fine.
    model = AutoModel.from_pretrained(model_name, dtype=torch.bfloat16).to(device)
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

    model.eval()
    
    # Process in batches to avoid OOM
    batch_size = 32
    scores = []
    
    # Need to filter valid image/text pairs first to maintain alignment
    valid_pairs = []
    for img_path, text in zip(imgs, pred_texts):
        if os.path.exists(img_path):
             valid_pairs.append((img_path, text))
        else:
             logger.warning(f"Image not found: {img_path}")
    
    if not valid_pairs:
        return {f"pred_clip_score-{country if country else 'all'}": 0.0}
        
    for i in range(0, len(valid_pairs), batch_size):
        batch = valid_pairs[i:i+batch_size]
        batch_paths = [p[0] for p in batch]
        batch_texts = [p[1] for p in batch]
        
        # Load images
        batch_images = []
        final_batch_texts = [] # in case image open fails
        
        for path, text in zip(batch_paths, batch_texts):
            try:
                image = Image.open(path).convert("RGB")
                batch_images.append(image)
                final_batch_texts.append(text)
            except Exception as e:
                logger.warning(f"Failed to open image {path}: {e}")
                continue
        
        if not batch_images:
            continue
            
        try:
            inputs = processor(
                text=final_batch_texts, 
                images=batch_images, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=77
            ).to(device)
            
            with torch.inference_mode():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    # Get separate features
                    image_features = model.get_image_features(pixel_values=inputs['pixel_values'])
                    inputs.pop('pixel_values')
                    text_features = model.get_text_features(**inputs)
                
                    # Normalize features
                    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

                    # Calculate cosine similarity
                    score = 100 * (image_features * text_features).sum(axis=-1)
                    scores.append(score.cpu())
                
        except Exception as e:
            logger.error(f"Error during CLIP inference: {e}")
            continue

    if not scores:
         return {f"pred_clip_score-{country if country else 'all'}": 0.0}
    
    all_scores = torch.cat(scores)
    mean_score = all_scores.mean().item()
    mean_score = max(0.0, mean_score)
        
    results = {f"pred_clip_score-{country if country else 'all'}": mean_score}
        
    return results


def region_score(pred_captions: List[str], gt_captions: List[str], country_info: List[str], region: str = None) -> Dict[str, float]:
    country_dict = {
        "Nigeria": ["Nigerian"],
        "Korea": ["Korean", "Seoul"],
        "China": ["Chinese", "Shanghai", "Beijing"],
        "Mexico": ["Mexican"],
        "India": ["Indian"],
    }

    pred_count, gt_count = 0, 0
    total = 0
    for idx, country in enumerate(country_info):
        # country is the country name from the dataset (e.g. "China")
        if region is not None and country != region:
            continue
            
        total += 1
        country_name = country.split("_")[0] # Just in case data is dirty, though typically it's just "China"
        country_adjectives = country_dict.get(country_name, [])
        candidates = country_adjectives + [country_name]
        
        # Check if country name or adjective appears in captions
        pred_text = pred_captions[idx].lower()
        gt_text = gt_captions[idx].lower()
        
        pred_count += int(any(term.lower() in pred_text for term in candidates))
        gt_count += int(any(term.lower() in gt_text for term in candidates))

    return {
        # f"gt_region_score-{region if region else 'all'}": gt_count / total if total > 0 else 0,
        f"pred_region_score-{region if region else 'all'}": round(pred_count / total*100, 1) if total > 0 else 0.0,
    }


def calculate_bert_score(pred_captions: List[str], gt_captions: List[str], country: str = None) -> Dict[str, float]:
    bertscore = load("bertscore")
    with torch.inference_mode():
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            results = bertscore.compute(
                predictions=pred_captions, references=gt_captions, lang="en", model_type="bert-base-uncased"
            )
    # results contains lists of precision, recall, f1
    suffix = f"-{country}" if country else "-all"

    return {
        f"precision{suffix}": np.mean(results["precision"]),
        f"recall{suffix}": np.mean(results["recall"]),
        f"f1{suffix}": np.mean(results["f1"]),
    }


def clean_caption(text):
    # Case 1: Multiple options - extract first option
    match = re.search(r'\*\*Option 1.*?\*\*\n\n(.+?)(?=\n\n\*\*Option|\n\nI\'ve|$)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Case 2: Single caption in quotes (after a colon or newline)
    match = re.search(r'[:\n]\s*"(.+?)"(?:\s*$|\.?\s*$)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback: return the whole text if no pattern matches
    return text.strip()