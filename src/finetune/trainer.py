import logging
import sys
from collections import defaultdict

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import Trainer

# Attempt to import gdeval metrics
try:
    from src.gdeval import retrieval_eval, compute_metrics, mean_reciprocal_rank, read_gt
except ImportError:
    # Fallback path adjustment
    sys.path.append(str(Path(__file__).parents[2]))
    from src.gdeval import retrieval_eval, compute_metrics, mean_reciprocal_rank, read_gt

logger = logging.getLogger(__name__)


class CultureTrainer(Trainer):
    def __init__(self, eval_gt_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_gt_path = eval_gt_path

    def evaluate(
        self,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        gt_path: Optional[str] = None,
    ) -> Dict[str, float]:
        if gt_path is None:
            gt_path = self.eval_gt_path
        # Use the passed eval_dataset or the one configured in Trainer
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        output_dir = self.args.output_dir
        
        logger.info(f"***** Running evaluation {metric_key_prefix} *****")
        
        raw_preds = defaultdict(dict)
        eval_loss = 0.0
        nb_eval_steps = 0
        
        model = self.model
        model.eval()
        
        for batch in tqdm(eval_dataloader):
            # Move to device
            batch = {k: v.to(self.args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with torch.inference_mode():
                # Autocast if fp16/bf16 enabled
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", 
                                  dtype=torch.bfloat16 if self.args.bf16 else (torch.float16 if self.args.fp16 else torch.float32)):
                    # breakpoint()
                    outputs = model(
                        input_ids=batch["input_ids"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                        return_loss=True,
                    )
            
            if outputs.loss is not None:
                eval_loss += outputs.loss.item()
            nb_eval_steps += 1
            
            logits = outputs.logits_per_image
            scores = logits.squeeze(0).cpu().tolist()
            
            query_id = batch["query_id"][0]
            doc_ids = batch["doc_ids"][0]
            
            # Store scores
            for idx, score in enumerate(scores):
                doc_id = doc_ids[idx]
                raw_preds[query_id][doc_id] = score

        if nb_eval_steps > 0:
            eval_loss = eval_loss / nb_eval_steps
            
        # Save Predictions to file
        run_file_name = f"{metric_key_prefix}_predictions.run"
        output_path = Path(output_dir) / run_file_name
        self.save_predictions_to_file(raw_preds, output_path)

        # Calculate Metrics using gdeval
        if gt_path:
             metrics = retrieval_eval(gt_path, str(output_path))
             metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}
        else:
             logger.warning("No ground truth file provided. Skipping metric calculation.")
             metrics = {}

        metrics[f"{metric_key_prefix}_loss"] = eval_loss
        
        # Log to wandb/console
        self.log(metrics)
        
        msg = f"Loss: {eval_loss:.4f} | MRR: {metrics.get(f'{metric_key_prefix}_MRR', 0):.4f}"
        logger.info(msg)
        
        return metrics

    def save_predictions_to_file(self, raw_preds, path):
        predictions = []
        model_name = "culture_model" # Generic name
        for query_id, doc_scores in raw_preds.items():
            sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, score) in enumerate(sorted_items):
                predictions.append([query_id, "Q0", doc_id, rank, score, model_name])
        
        df = pd.DataFrame(predictions, columns=["q_id", "Q0", "doc_id", "rank", "score", "type"])
        df.to_csv(path, sep="\t", index=False, header=False)
