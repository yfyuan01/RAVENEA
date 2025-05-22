import random
import re
from functools import partial
from logging import getLogger

import numpy as np
import pandas as pd
import torch
import tqdm
from evaluate import load
from torchmetrics.functional.multimodal import clip_score
from torchvision import transforms

logger = getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def read_qrels_dict(file):
    result = {}
    for line in tqdm.tqdm(file, desc="loading qrels (by line)", leave=False):
        qid, _, docid, bm25_score, relevance_score = line.split()
        result.setdefault(qid, {})[docid] = float(bm25_score), int(float(relevance_score))
    return result


def calculate_precision(qrels, run_scores, k=5):
    scores = []
    for qid in run_scores:
        if qid not in qrels:
            continue
        max_rel = max(rel[1] for rel in qrels[qid].values())
        # Sort documents by score in descending order
        ranked_docs = sorted(run_scores[qid].items(), key=lambda x: x[1], reverse=True)[:k]
        relevant = sum(
            1 for doc_id, _ in ranked_docs if doc_id in qrels[qid] and qrels[qid][doc_id][1] == max_rel and max_rel > 0
        )
        scores.append(relevant / k if k > 0 else 0)
    return scores


def calculate_mrr(qrels, run_scores):
    scores = []
    for qid in run_scores:
        if qid not in qrels:
            continue
        max_rel = max(rel[1] for rel in qrels[qid].values())
        # Sort documents by score in descending order
        ranked_docs = sorted(run_scores[qid].items(), key=lambda x: x[1], reverse=True)
        for rank, (doc_id, _) in enumerate(ranked_docs, start=1):
            if doc_id in qrels[qid] and qrels[qid][doc_id][1] == max_rel and max_rel > 0:
                scores.append(1 / rank)
                break
    return sum(scores) / len(scores)


def calculate_ndcg(qrels, run_scores, k=5):
    scores = []
    for qid in run_scores:
        if qid not in qrels:
            continue
        # Sort documents by score in descending order
        max_rel = max(rel[1] for rel in qrels[qid].values())
        ranked_docs = sorted(run_scores[qid].items(), key=lambda x: x[1], reverse=True)[:k]
        dcg = sum(
            (2 ** (qrels[qid][doc_id][1] + max_rel) - 1) / np.log2(rank + 2)
            for rank, (doc_id, _) in enumerate(ranked_docs)
        )
        idcg = sum(
            (2 ** (rel + max_rel) - 1) / np.log2(rank + 2)
            for rank, rel in enumerate(sorted([rel[1] for rel in qrels[qid].values()], reverse=True)[:k])
        )
        scores.append(dcg / idcg if idcg > 0 else 0)
    return scores


def calculate_clip_score(imgs, gt_texts, pred_texts, args, country=None):
    clip_image_size = 224
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
        pil_to_tensor = transforms.Compose(
            [
                transforms.Resize((clip_image_size, clip_image_size)),
                transforms.ToTensor(),  # Converts to [0,1] float32 tensor
            ]
        )
        # (B, C, H, W)
        image_tensors = torch.stack([pil_to_tensor(img) for img in imgs]).to("cuda")
        clip_score_value_pred = clip_score_fn(image_tensors, pred_texts).detach()
        results = {f"pred_clip_score-{country}": clip_score_value_pred.cpu()}
        if not args.use_retrieval:
            clip_score_value_gt = clip_score_fn(image_tensors, gt_texts).detach()
            results.update({f"gt_clip_score-{country}": clip_score_value_gt.cpu()})
    return results


def region_score(pred_captions, gt_captions, country_info, region=None):
    country_dict = {
        "Nigeria": ["Nigerian"],
        "Korea": ["Korean", "Seoul"],
        "China": ["Chinese", "Shanghai"],
        "Mexico": ["Mexican"],
        "India": ["Indian"],
    }

    pred_count, gt_count = 0, 0
    for idx, country in enumerate(country_info):
        country_name = country.split("_")[0]
        country_adjective = country_dict[country_name]
        candidates = country_adjective + [country_name]

        # Check if country name or adjective appears in captions
        pred_count += int(any(term.lower() in pred_captions[idx].lower() for term in candidates))
        gt_count += int(any(term.lower() in gt_captions[idx].lower() for term in candidates))
    total = len(country_info)
    return {
        f"gt_region_score-{region}": gt_count / total if total > 0 else 0,
        f"pred_region_score-{region}": pred_count / total if total > 0 else 0,
    }


def clean_caption(text):
    """Extract the first caption from text using different patterns."""
    # Define patterns to match different caption formats
    patterns = [
        # Option-style headings
        r"\*\*Option 1.*?(?=Option|\Z)",
        # Plain caption following an intro line
        r":\s*\n\n(.+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            # caption = match.group(1) if "(" in pattern else match.group()
            caption = match.group()
            caption = (
                caption.split(":")[-1]
                .strip()
                .replace("*", "")
                .replace("\u2019", "'")
                .replace("\u2018", "'")
                .replace("\u2013", "-")
                .replace("\u201c", "")
                .replace("\u201d", "")
                .replace("\n", "")
                .strip()
                .split("  ")[0]
            )

            return caption

    # Return original text if no patterns match
    return (
        text.replace("*", "")
        .replace("\u2019", "'")
        .replace("\u201c", "")
        .replace("\u201d", "")
        .replace("\u2013", "-")
        .replace("\u2018", "'")
        .strip()
    )


def calculate_bert_score(pred_captions, gt_captions, country=None):
    bertscore = load("bertscore")
    results = bertscore.compute(
        predictions=pred_captions, references=gt_captions, lang="en", model_type="bert-base-uncased"
    )
    results = {
        f"precision-{country}": np.mean(results["precision"]),  # type: ignore
        f"recall-{country}": np.mean(results["recall"]),  # type: ignore
        f"f1-{country}": np.mean(results["f1"]),  # type: ignore
    }
    return results


def get_retriever_docs(file_path):
    result = {}
    file = pd.read_csv(file_path, sep="\t", header=None)
    for line in tqdm.tqdm(file.iterrows(), desc="loading run (by line)", leave=False):
        # cols = line[1][0].split()
        if "bert" in file_path:
            cols = line[1][0].split()
        else:
            cols = tuple(line[1].tolist())
        qid, _, docid, rank, score, _ = cols
        result.setdefault(qid, []).append(docid)
    return result


def is_match(predicted, ground_truth):
    # Check for answer pattern in one of two formats:
    # 1. "Answer: A" or "Answer: A) text" anywhere in the string
    # 2. "A)" or "A) text" at the beginning of the string
    match = re.search(r"Answer:\s*([A-D])(?:\s*\)\s*(.*)?)?", predicted)
    if match is None:
        # Look for option pattern at beginning of string
        match = re.search(r"^([A-D])\s*\)\s*(.*)", predicted)

    if match or len(predicted) == 1:
        pred_option = match.group(1) if match else predicted.strip()
        pred_text = match.group(2) if match else None

        # Ground truth also starts with letter+text, e.g. "A) XXXX"
        gt_match = re.search(r"([A-D])\)\s*(.*)", ground_truth)
        if gt_match:
            gt_option = gt_match.group(1)
            gt_text = gt_match.group(2)

            # First, check if options match
            if pred_option == gt_option:
                # If no text is given in prediction, it's still okay
                if not pred_text:
                    return True
                # If text is given, do a loose match (case-insensitive, trimmed)
                return pred_text.strip().lower() == gt_text.strip().lower()
    return False


def prepare_cvqa_input(args, data, wiki_data):
    QUERY_TEMPLATE = """Answer the following multiple choice question. The last line of your response must be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. {Retrieval}\n\nQuestion:\n{Question}\n\n{A}\n{B}\n{C}\n{D}""".strip()
    if not args.use_retrieval:
        logger.info("Without retrieval")
        texts = [
            QUERY_TEMPLATE.format(
                A=d["options"][idx][0].replace(". ", ") "),  # type: ignore
                B=d["options"][idx][1].replace(". ", ") "),  # type: ignore
                C=d["options"][idx][2].replace(". ", ") "),  # type: ignore
                D=d["options"][idx][3].replace(". ", ") "),  # type: ignore
                Question=d["questions"][idx],  # type: ignore
                Retrieval="",
            )
            for d in data
            for idx in range(len(d["questions"]))
        ]
    else:
        retrieval_template = """The scope of the question is strictly limited to the given image. However, please analyze and incorporate information from both the image and the following document to answer the question.\n\nDocument:\n{Retrieval}"""
        # Use retrieved docs
        logger.info("using retrieval")
        logger.info(args.retrieval_file)
        retriever = get_retriever_docs(args.retrieval_file)
        retrieval_docs = [
            " ".join(f"{wiki_data[str(d_id)]}" for d_id in retriever[q_id][: args.top_k_retrieval])
            for q_id, questions in zip(data["query_id"], data["questions"])
            for _ in range(len(questions))
        ]
        retrieval_contents = [retrieval_template.format(Retrieval=d_doc.strip()) for d_doc in retrieval_docs]
        texts = []
        count = 0
        for d in data:
            for idx in range(len(d["questions"])):
                texts.append(
                    QUERY_TEMPLATE.format(
                        A=d["options"][idx][0].replace(". ", ") "),  # type: ignore
                        B=d["options"][idx][1].replace(". ", ") "),  # type: ignore
                        C=d["options"][idx][2].replace(". ", ") "),  # type: ignore
                        D=d["options"][idx][3].replace(". ", ") "),  # type: ignore
                        Question=d["questions"][idx],  # type: ignore
                        Retrieval=retrieval_contents[count],
                    )
                )
                count += 1
    return texts


def prepare_cic_input(args, data, wiki_data):
    QUERY_TEMPLATE = """Write a concise, one-sentence caption for the given image. The generated caption must contain the visual content and culturally relevant elements of the image. Avoid explicit references to the image itself (e.g., "This image shows...", "Pictured here is...", "In this photograph..."). Do not generate multiple options. {CONTEXT}""".strip()
    # Input image and question
    if not args.use_retrieval:
        logger.info("Without retrieval")
        texts = [
            QUERY_TEMPLATE.format(
                CONTEXT="",
            ).strip()
            for _ in data
        ]
    else:
        retrieval_template = """Please consider the following context:\n{Retrieval}""".strip()
        # Use retrieved docs
        logger.info("using retrieval")
        logger.info(args.retrieval_file)
        retriever = get_retriever_docs(args.retrieval_file)
        retrieval_docs = [
            " ".join(f"{wiki_data[str(d_id)]}" for d_id in retriever[q_id][: args.top_k_retrieval])
            for q_id in data["query_id"]
        ]
        retrieval_contents = [retrieval_template.format(Retrieval=d_doc.strip()) for d_doc in retrieval_docs]
        texts = [
            QUERY_TEMPLATE.format(
                CONTEXT=retrieval_contents[doc_idx] + "\n",
            )
            for doc_idx, _ in enumerate(data)
        ]

    return texts
