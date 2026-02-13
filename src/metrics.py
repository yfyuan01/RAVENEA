from collections import defaultdict
import logging
import json
import argparse
from src.helper import is_match, load_cvqa_data, load_cic_data, calculate_bert_score, calculate_clip_score, region_score, clean_caption
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    # filename="./logs/metrics.log",
)
logger = logging.getLogger(__name__)

def evaluate_cvqa(predictions, data):
    total = len(data)
    correct = 0
    country_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for item in data:
        qid = item['id']
        pred = predictions.get(qid, "")
        gt = item['answer']
        country = item['country']
        
        is_corr = is_match(pred, gt)
        if is_corr:
            correct += 1
            country_stats[country]["correct"] += 1
        country_stats[country]["total"] += 1
        
    accuracy = correct / total if total > 0 else 0
    logger.info(f"Overall Accuracy: {correct}/{total} ({accuracy*100:.1f})")
    
    dataset_res = {"accuracy": accuracy}
    for country, stats in country_stats.items():
        c_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        logger.info(f"{country}: {stats['correct']}/{stats['total']} ({c_acc*100:.1f})")
        dataset_res[country] = c_acc
        
    return dataset_res

def compute_caption_metrics(pred_caps, gt_caps, pred_list, gt_list, image_list, country=None):
    """
    Computes captioning metrics.
    pred_caps: dict {id: [caption]}
    gt_caps: dict {id: [caption]}
    pred_list: list of predicted captions (aligned with gt_list)
    gt_list: list of ground truth captions (aligned with pred_list)
    image_list: list of image paths (aligned)
    country: specific country name or None
    """
    # bleu_scorer = Bleu(4)
    rouge_scorer = Rouge()
    cider_scorer = Cider()
    
    results = {}
    
    # Standard COCO metrics
    if pred_caps and gt_caps:
        # bleu_score, _ = bleu_scorer.compute_score(gt_caps, pred_caps)
        rouge_score, _ = rouge_scorer.compute_score(gt_caps, pred_caps)
        cider_score, _ = cider_scorer.compute_score(gt_caps, pred_caps)
        
        prefix = f"{country}-" if country else ""
        results.update({
            # f"{prefix}bleu-4": round(bleu_score[3]*100, 1),
            f"{prefix}rouge": round(rouge_score*100, 1),
            f"{prefix}cider": round(cider_score*100, 1)
        })
    # breakpoint()
    # CLIP Score
    clip_scores = calculate_clip_score(image_list, gt_list, pred_list, country=country)
    results.update(clip_scores)
    
    # BERT Score
    bert_scores = calculate_bert_score(pred_list, gt_list, country=country)
    # only keep the key starting from f1 in bert_scores
    bert_scores = {k: v for k, v in bert_scores.items() if k.startswith("f1")}
    results.update(bert_scores)
    
    return results

def evaluate_cic(predictions, data):
    gt_caps_all = {}
    pred_caps_all = {}
    
    pred_list_all = []
    gt_list_all = []
    image_list_all = []
    country_list_all = [] # For region score
    
    # Organize data per country
    country_data = defaultdict(lambda: {"gt_caps": {}, "pred_caps": {}, "pred_list": [], "gt_list": [], "image_list": []})
    
    for item in data:
        item_id = item['id']
        country = item['country']
        
        gt_caption = item['human_caption']
        gt_caps_all[item_id] = [gt_caption]
        
        pred_raw = clean_caption(predictions.get(item_id, ""))
        pred_caps_all[item_id] = [pred_raw]
        
        # Lists for other metrics
        pred_list_all.append(pred_raw)
        gt_list_all.append(gt_caption)
        image_list_all.append(item['image'])
        country_list_all.append(country)
        
        # Populate per-country data
        country_data[country]["gt_caps"][item_id] = [gt_caption]
        country_data[country]["pred_caps"][item_id] = [pred_raw]
        country_data[country]["pred_list"].append(pred_raw)
        country_data[country]["gt_list"].append(gt_caption)
        country_data[country]["image_list"].append(item['image'])

    results = {}
    logger.info("Computing Overall Metrics...")
    # Overall Metrics
    overall_metrics = compute_caption_metrics(pred_caps_all, gt_caps_all, pred_list_all, gt_list_all, image_list_all, country=None)
    results.update(overall_metrics)
    
    # Region Score Overall
    region_scores = region_score(pred_list_all, gt_list_all, country_list_all, region=None)
    results.update(region_scores)
    
    # Per Country Metrics
    for country in country_data:
        logger.info(f"Computing Metrics for {country}...")
        c_data = country_data[country]
        c_metrics = compute_caption_metrics(
            c_data["pred_caps"], c_data["gt_caps"], 
            c_data["pred_list"], c_data["gt_list"], 
            c_data["image_list"], 
            country=country
        )
        results.update(c_metrics)
        
        # Region Score Per Country
        c_region_scores = region_score(
            c_data["pred_list"], 
            c_data["gt_list"], 
            [country] * len(c_data["gt_list"]), 
            region=country
        )
        results.update(c_region_scores)

    logger.info("Evaluation Complete.")
    logger.info(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions JSON")
    parser.add_argument("--data", type=str, required=True, help="Path to ground truth JSONL")
    args = parser.parse_args()
    
    with open(args.predictions, "r") as f:
        predictions = json.load(f)
    logger.info("==" * 50)
    logger.info(f"Loaded {len(predictions)} predictions from {args.predictions}")

    # Determine if CVQA or Captioning based on data content path or implicit knowledge
    # The user request specifically mentioned evaluating captions in one file and it implies we might need to check.
    # But usually one script handles one task or detects it.
    # We can check simple heuristic: if "cvqa" in args.data
    
    if "cvqa" in args.data.lower():
        data = load_cvqa_data(args.data)
        evaluate_cvqa(predictions, data)
    else:
        # Assume Caption (CIC)
        data = load_cic_data(args.data)
        evaluate_cic(predictions, data)