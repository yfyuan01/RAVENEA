import json
import logging
from typing import Dict, List

import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from PIL import Image
import re

logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> List[Dict]:
    logger.info(f"Loading {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def load_wiki(path: str) -> Dict[str, str]:
    logger.info(f"Loading wiki documents from {path}...")
    data = load_jsonl(path)
    return {item['id']: re.sub(r'(?m)^#+\s.*$', '', item['text']).strip() for item in data}


class CustomDataset(Dataset):
    def __init__(
        self,
        processor,
        split,
        max_length,
        data,
        wiki_docs,
    ):
        self.processor = processor
        self.split = split
        self.max_length = max_length
        self.data = self.do_augmentation(data) if split == "train" else data
        self.wiki_docs = wiki_docs

    def __len__(self):
        return len(self.data)

    def do_augmentation(self, data: List[Dict], seed: int = 42) -> List[Dict]:
        # Group data by country
        country_to_data = defaultdict(list)
        for item in data:
            country = item.get("country", "Unknown")
            country_to_data[country].append(item)
        
        if not country_to_data:
            return data

        max_count = max(len(items) for items in country_to_data.values())
        augmented_data = []

        for country, items in country_to_data.items():
            # Add original items
            # We add an empty augmentation flag to identify them as originals if needed,
            # or just leave them as is. The user request implies doing augmentation
            # based on flags. Let's mark originals with None or empty flags.
            for item in items:
                item_copy = item.copy()
                item_copy["augmentation_flags"] = None
                augmented_data.append(item_copy)
                
            # Add duplicates to reach max_count
            count = len(items)
            if count < max_count:
                num_needed = max_count - count
                # Sample with replacement
                random.seed(seed)
                extras = random.choices(items, k=num_needed)
                
                for idx, item in enumerate(extras):
                    item_copy = item.copy()
                    random.seed(seed + idx)
                    # Generate augmentation flags
                    flags = {
                        "hflip": random.random() < 0.8,
                        "vflip": random.random() < 0.8,
                        "do_resize": random.random() < 0.8, 
                        "resize_scale": random.uniform(0.5, 1.0),
                        "do_crop": random.random() < 0.8, 
                        "brightness": random.uniform(0.5, 1.5), # 1 +/- 0.5
                        "contrast": random.uniform(0.5, 1.5),
                        "saturation": random.uniform(0.5, 1.5),
                        "hue": random.uniform(-0.1, 0.1),
                    }
                    item_copy["augmentation_flags"] = flags
                    augmented_data.append(item_copy)
        return augmented_data

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = sample.get("file_name")
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Apply augmentation if flags are present
            flags = sample.get("augmentation_flags")
            if flags:
                min_size = int(min(image.size))
                new_size = int(min_size * flags.get("resize_scale", 1.0))
                if flags.get("do_resize"):
                    # Resize & Crop
                    image = v2.functional.resize(image, size=new_size)
                
                if flags.get("do_crop"):
                    image = v2.functional.center_crop(image, output_size=new_size)
                
                if flags.get("hflip"):
                    image = v2.functional.hflip(image)
                    
                if flags.get("vflip"):
                    image = v2.functional.vflip(image)
                    
                # Color Jitter
                # brightness, contrast, saturation, hue
                b = flags.get("brightness", 1.0)
                c = flags.get("contrast", 1.0)
                s = flags.get("saturation", 1.0)
                h = flags.get("hue", 0.0)
                
                image = v2.functional.adjust_brightness(image, b)
                image = v2.functional.adjust_contrast(image, c)
                image = v2.functional.adjust_saturation(image, s)
                image = v2.functional.adjust_hue(image, h)

        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')

        all_doc_ids = sample.get("enwiki_ids", [])
        culture_relevance = sample.get("culture_relevance", [])
        
        # We need text content for these doc ids
        all_docs = []
        for doc_id in all_doc_ids:
            doc_content = self.wiki_docs.get(doc_id, "")
            all_docs.append(doc_content)
            
        text = all_docs
        
        tok = self.processor(
            text=text,
            images=image,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            truncation=True,
        )
        input_ids = tok["input_ids"]
        pixel_values = tok["pixel_values"]
        
        # unique query_id is tricky. Use file_name as query_id
        query_id = sample.get("file_name", str(idx))
        
        # Construct labels for loss function: 1 if x > 0 else 0
        # labels_list = [1 if x > 0 else 0 for x in culture_relevance]

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "query_id": query_id,
            "img_id": sample.get("img_id", query_id),
            "doc_ids": all_doc_ids,
            # "labels": labels_list,
            "relevance_scores": culture_relevance,
            "is_training": True if self.split == "train" else False
        }


class CustomDataCollator:
    def __call__(self, features):
        if not features[0]["is_training"]:
            input_ids = [f["input_ids"] for f in features]
            pixel_values = [f["pixel_values"] for f in features]
            batch = {
                "input_ids": torch.cat(input_ids, dim=0),
                "pixel_values": torch.cat(pixel_values, dim=0),
                "query_id": [f["query_id"] for f in features],
                "img_id": [f["img_id"] for f in features],
                "doc_ids": [f["doc_ids"] for f in features],
                # "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
            }

            # Only add relevance_scores to batch if they exist in all features
            # if all("relevance_scores" in f for f in features):
            batch_relevance = torch.tensor([f["relevance_scores"] for f in features], dtype=torch.long)
            min_val, max_val = -3, 3
            new_min, new_max = -1, 1
            batch["labels"] = (new_max - new_min) * (batch_relevance - min_val) / (max_val - min_val) + new_min

            return batch
        else:
            # 1. Collect unique documents and their tensors
            doc_id_to_tensor = {}
            for f in features:
                # f["input_ids"] is expected to be (N_docs, SeqLen)
                # f["doc_ids"] is list of doc_ids corresponding to rows of input_ids
                current_input_ids = f["input_ids"]
                current_doc_ids = f["doc_ids"]
                
                for i, doc_id in enumerate(current_doc_ids):
                    if doc_id not in doc_id_to_tensor:
                        doc_id_to_tensor[doc_id] = current_input_ids[i]

            unique_doc_ids = sorted(list(doc_id_to_tensor.keys()))
            doc_id_to_idx = {d: i for i, d in enumerate(unique_doc_ids)}
            # 2. Build batched input_ids for unique docs
            batch_input_ids = torch.stack([doc_id_to_tensor[d] for d in unique_doc_ids])

            # 3. Build dense labels and relevance_scores matrices
            batch_size = len(features)
            num_unique_docs = len(unique_doc_ids)
            
            # batch_labels = torch.zeros((batch_size, num_unique_docs), dtype=torch.long)
            
            # has_relevance_scores = all("relevance_scores" in f for f in features)
            # if has_relevance_scores:
            batch_relevance = torch.zeros((batch_size, num_unique_docs), dtype=torch.long)-3
            
            for i, f in enumerate(features):
                current_doc_ids = f["doc_ids"]
                # current_labels = f["labels"]
                
                # for doc_id, label in zip(current_doc_ids, current_labels):
                #     if doc_id in doc_id_to_idx:
                #         idx = doc_id_to_idx[doc_id]
                #         batch_labels[i, idx] = label
                
                # if has_relevance_scores:
                current_scores = f["relevance_scores"]
                for doc_id, score in zip(current_doc_ids, current_scores):
                    if doc_id in doc_id_to_idx:
                        idx = doc_id_to_idx[doc_id]
                        batch_relevance[i, idx] = score

            # 4. Handle pixel_values (queries)
            # Assuming pixel_values in features are (1, C, H, W) or (C, H, W) and need stacking/catting
            # Based on previous code: torch.cat([f["pixel_values"] ...], dim=0) implies f["pixel_values"] has batch dim 1
            pixel_val_list = [f["pixel_values"] for f in features]
            batch_pixel_values = torch.cat(pixel_val_list, dim=0)

            batch = {
                "input_ids": batch_input_ids,
                "pixel_values": batch_pixel_values,
                "query_id": [f["query_id"] for f in features],
                "img_id": [f["img_id"] for f in features],
                "doc_ids": unique_doc_ids, # Now a list of all unique doc IDs in the CACHE/BATCH
                # "labels": batch_labels,
                "relevance_scores": batch_relevance,
            }
            
            # Normalize relevance scores from [-3,3] to [-1, 1] to represent labels
            # min_val, max_val = batch_relevance.min(), batch_relevance.max()
            min_val, max_val = -3, 3
            new_min, new_max = -1, 1
            batch["labels"] = (new_max - new_min) * (batch_relevance - min_val) / (max_val - min_val) + new_min
            # batch["labels"][batch["labels"] == 0] = -0.2
            # breakpoint()
            return batch


def create_datasets(
    processor,
    train_path,
    val_path,
    wiki_path,
    max_length,
    test_path=None,
):
    train_metadata = load_jsonl(train_path)
    val_metadata = load_jsonl(val_path)
    wiki_docs = load_wiki(wiki_path)
    
    train_set = CustomDataset(
        processor=processor,
        split="train",
        data=train_metadata,
        wiki_docs=wiki_docs,
        max_length=max_length,
    )
    
    val_set = CustomDataset(
        processor=processor,
        split="validation",
        data=val_metadata,
        wiki_docs=wiki_docs,
        max_length=max_length,
    )
    
    # Using val_set as test_set for simplicity if no explicit test path provided
    if test_path is None:
        test_set = val_set
    else:
        test_metadata = load_jsonl(test_path)
        test_set = CustomDataset(
            processor=processor,
            split="test",
            data=test_metadata,
            wiki_docs=wiki_docs,
            max_length=max_length,
        )
    
    return train_set, val_set, test_set

