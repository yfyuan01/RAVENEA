"""
Some of the code is adapted from: https://github.com/yfyuan01/MQC/blob/master/VL-T5/VL-T5/gdeval.pl
"""

import math
from typing import Any

from rich import print
from rich.table import Table


def ndcg(relevance_list: list[float], k: int = 5) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG).
    Convert negative relevance to 0
    """
    preds = relevance_list[:k]
    gts = sorted(relevance_list, reverse=True)[:k]
    # Convert negative relevance to 0
    preds = [x + 3 for x in preds]
    gts = [x + 3 for x in gts]

    def dcg(relevance_list: list[float]) -> float:
        """Compute Discounted Cumulative Gain (DCG)."""
        return sum((2**rel - 1) / (math.log2(idx + 2) / math.log2(2)) for idx, rel in enumerate(relevance_list))

    idcg = dcg(gts)
    if idcg == 0:
        return 0
    return dcg(preds) / idcg


def err(relevance_list: list[float], MAX_JUDGMENT: int = 4, k: int = 5) -> float:
    """Compute Expected Reciprocal Rank (ERR). Convert negative relevance to 0"""
    relevance_list = [max(0, x) for x in relevance_list[:k]]
    p = 1.0
    err_score = 0.0
    for i, rel in enumerate(relevance_list):
        R = (2**rel - 1) / 2**MAX_JUDGMENT
        err_score += p * R / (i + 1)
        p *= 1 - R
    return err_score


def precision_at_k(preds: list[tuple], gts: dict[str, float], k: int = 5) -> float:
    """Compute Precision@K based on rank."""
    max_relevance = max(gts.values(), default=0)
    if max_relevance <= 0:
        return 0.0
    top_k_gt = set(doc_id for doc_id, rel in gts.items() if rel == max_relevance)
    top_k_preds = set(doc_id for doc_id, _ in preds[:k])

    correct = len(top_k_preds & top_k_gt) / len(top_k_preds)
    return correct


def mean_reciprocal_rank(preds: dict, gt: dict) -> float:
    """Compute Mean Reciprocal Rank (MRR)."""
    mrr = 0.0
    count = 0
    for query_id in preds:
        if query_id not in gt:
            continue
        # Get the highest value
        max_value = max(gt[query_id].items(), key=lambda x: x[1])[1]
        if max_value > 0:
            top_elements = [item[0] for item in gt[query_id].items() if item[1] == max_value]
            for rank, (doc_id, _) in enumerate(sorted(preds[query_id], key=lambda x: x[1])):
                if doc_id in top_elements:
                    mrr += 1.0 / (rank + 1)
                    break
            count += 1
    return mrr / count if count > 0 else 0.0


def compute_metrics(gt: dict, preds: dict, max_judgement: int, depth: int = 5) -> list[tuple]:
    """Compute ERR, NDCG, Precision@K, and MRR metrics."""
    seperate_results = []
    for query_id in preds:
        if query_id not in gt:
            continue
        ranked_docs = sorted(preds[query_id], key=lambda x: x[1])
        relevance_list = [gt[query_id].get(doc[0], 0) for doc in ranked_docs]
        ndcg_score = ndcg(relevance_list, depth)
        err_score = err(relevance_list, max_judgement, depth)
        prec_at_k = precision_at_k(ranked_docs, gt[query_id], depth)
        seperate_results.append((query_id, ndcg_score, err_score, prec_at_k))
    return seperate_results


def calc_ret_metrics(gt: dict, preds: dict) -> Any:
    max_judgement = max([max(inner_dict.values()) for key, inner_dict in gt.items()])
    print(f"Max judgement: {max_judgement}")
    prec_at_k = []
    ndcg_at_k = []
    err_at_k = []
    for k in [1, 3, 5]:
        metrics = compute_metrics(gt, preds, max_judgement, k)
        # Compute avarage for each metric
        avg_ndcg = sum([x[1] for x in metrics]) / len(metrics)
        avg_err_score = sum([x[2] for x in metrics]) / len(metrics)
        avg_prec_at_k = sum([x[3] for x in metrics]) / len(metrics)
        prec_at_k.append(avg_prec_at_k)
        ndcg_at_k.append(avg_ndcg)
        err_at_k.append(avg_err_score)

    mrr_score = mean_reciprocal_rank(preds, gt)
    print(f"Mean Reciprocal Rank (MRR): {mrr_score}")

    res = Table(title="Evaluation Results")
    columns = ["MRR", "P@1", "P@3", "P@5", "nDCG@1", "nDCG@3", "nDCG@5"]
    for col in columns:
        res.add_column(col, justify="center")
    res.add_row(
        f"{mrr_score * 100:.2f}", *[f"{x * 100:.2f}" for x in prec_at_k], *[f"{x * 100:.2f}" for x in ndcg_at_k]
    )
    return res
