import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import tqdm
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor

from src.gdeval import calc_ret_metrics

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")


# Siglip
def siglip_zs(model, processor, dataset, wiki_data):
    model_name = model.config._name_or_path.split("/")[-1]
    model.to(device)
    model.eval()
    max_length = model.config.text_config.max_position_embeddings
    predictions = pd.DataFrame(columns=["q_id", "placeholder", "predictions", "rank", "score", "type"])
    for example in tqdm.tqdm(dataset):
        image = example["image"]  # Assuming dataset has 'image' column
        class_labels = example["doc_ids"]
        texts = [
            wiki_data_info["text"]
            for label_id in class_labels
            for wiki_data_info in wiki_data
            if label_id == wiki_data_info["id"]
        ]

        # Process image
        inputs = processor(
            images=image, text=texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
        )
        inputs.to(device)

        # Compute features
        with torch.inference_mode():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        preds = [class_labels[i] for i in logits_per_image.squeeze(0).argsort(descending=True).tolist()]
        scores = sorted(logits_per_image.squeeze(0).tolist(), reverse=True)
        for i, (pred, score) in enumerate(zip(preds, scores)):
            predictions.loc[len(predictions)] = [example["query_id"], 0, pred, i, score, model_name]

    return predictions


# CLIP
def clip_zs(model, processor, dataset, wiki_data):
    model_name = model.config._name_or_path.split("/")[-1]
    model.to(device)
    model.eval()
    max_length = model.config.text_config.max_position_embeddings
    predictions = pd.DataFrame(columns=["q_id", "placeholder", "predictions", "rank", "score", "type"])
    for example in tqdm.tqdm(dataset):
        image = example["image"]  # Assuming dataset has 'image' column
        class_labels = example["doc_ids"]
        texts = [
            wiki_data_info["text"]
            for label_id in class_labels
            for wiki_data_info in wiki_data
            if label_id == wiki_data_info["id"]
        ]

        text_inputs = processor(
            text=texts, return_tensors="pt", padding="max_length", max_length=max_length, truncation=True
        )
        # Process image
        image_inputs = processor(images=image, return_tensors="pt")
        image_inputs = image_inputs.to(device)
        text_inputs = text_inputs.to(device)

        # Compute features
        with torch.inference_mode():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                image_features = model.get_image_features(**image_inputs)
                text_features = model.get_text_features(**text_inputs)

                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Compute similarity
                similarity = (image_features @ text_features.T).squeeze(0)
        preds = [class_labels[i] for i in similarity.argsort(descending=True).tolist()]
        scores = sorted(similarity.tolist(), reverse=True)
        for i, (pred, score) in enumerate(zip(preds, scores)):
            predictions.loc[len(predictions)] = [example["query_id"], 0, pred, i, score, model_name]
    return predictions


def main(args):
    model_id = args.model_ids
    data = load_dataset("jaagli/ravenea", split="combination")
    data = data.filter(lambda x: x["task_type"] == "cVQA", num_proc=8)  # type: ignore
    wiki_data = load_dataset("wikipedia", "20220301.en", split="train")

    required_wiki_ids = set([j for i in data["doc_ids"] for j in i])  # type: ignore
    wiki_data = wiki_data.filter(lambda x: x["id"] in required_wiki_ids, num_proc=8)  # type: ignore
    gt = defaultdict(dict)
    for curr_d in data:
        for doc_id, rel_score in zip(curr_d["doc_ids"], curr_d["culture_relevance_scores"]):  # type: ignore
            gt[curr_d["query_id"]][doc_id] = rel_score  # type: ignore

    output_path = Path("./models/baselines/")
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    print(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32, attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained(model_id)
    if "siglip" in model_id:
        res = siglip_zs(model, processor, data, wiki_data)
    else:
        res = clip_zs(model, processor, data, wiki_data)
    preds = defaultdict(list)
    for _, row in res.iterrows():
        query_id = row["q_id"]
        doc_id = row["predictions"]
        rank = int(float(row["rank"]))
        preds[query_id].append((doc_id, rank))
    # Save predictions
    res.to_csv(output_path / f"{model_id.split('/')[-1]}.csv", index=False, header=False, sep="\t")
    calc_ret_metrics(preds=preds, gt=gt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_ids",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="model ids from huggingface, such as google/siglip2-so400m-patch14-384 and openai/clip-vit-large-patch14",
    )
    args = parser.parse_args()

    main(args)
