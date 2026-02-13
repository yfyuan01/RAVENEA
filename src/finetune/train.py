import logging
import random
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch
from transformers import (
    AutoProcessor,
    PreTrainedModel,
    PreTrainedTokenizer,
    HfArgumentParser,
    TrainingArguments
)

# Local imports
from data_process import create_datasets, CustomDataCollator
from trainer import CultureTrainer
from localclip import LocalCLIPModel
from localsiglip import LocalSiglipModel


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


@dataclass
class ModelArguments:
    hf_ckpt: str = field(
        default="openai/clip-vit-large-patch14",
        metadata={"help": "Hugging Face model checkpoint"}
    )
    patience: int = field(
        default=5,
        metadata={"help": "Patience for early stopping"}
    )


@dataclass
class DataArguments:
    train_path: str = field(
        default="./ravenea/metadata_train.jsonl",
        metadata={"help": "Path to training metadata JSONL"}
    )
    val_path: str = field(
        default="./ravenea/metadata_val.jsonl",
        metadata={"help": "Path to validation metadata JSONL"}
    )
    test_path: str = field(
        default="./ravenea/metadata_test.jsonl",
        metadata={"help": "Path to test metadata JSONL"}
    )
    wiki_path: str = field(
        default="./ravenea/wiki_documents.jsonl",
        metadata={"help": "Path to wiki documents JSONL"}
    )


@dataclass
class TrainingArguments(TrainingArguments):
    """
    Subclass of TrainingArguments to set defaults requested by user.
    """
    output_dir: str = field(
        default="./models/culture",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    num_train_epochs: float = field(
        default=5.0,
        metadata={"help": "Total number of training epochs to perform."}
    )
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "The initial learning rate for AdamW."}
    )
    per_device_train_batch_size: int = field(
        default=64,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    dataloader_num_workers: int = field(
        default=8,
        metadata={"help": "Number of subprocesses to use for data loading."}
    )
    dataloader_pin_memory: bool = field(
        default=True,
        metadata={"help": "Whether to pin memory for data loading."}
    )
    save_strategy: str = field(default="no", metadata={"help": "Save strategy"})
    eval_strategy: str = field(default="no", metadata={"help": "Evaluation strategy"})
    logging_steps: int = field(default=1, metadata={"help": "Logging steps"})
    load_best_model_at_end: bool = field(default=True, metadata={"help": "Load best model at end"})
    do_train: bool = field(default=True, metadata={"help": "Whether to train"})
    remove_unused_columns: bool = field(default=False, metadata={"help": "Remove unused columns"})
    report_to: str = field(default="wandb", metadata={"help": "Report to"})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "Learning schedule"})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Warmup ratio"})
    


def load_model(ckpt_path) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    logger.info(f"Loading {ckpt_path} model...")
    if "siglip" in ckpt_path:
        model = LocalSiglipModel.from_pretrained(
            ckpt_path, dtype=torch.float32, attn_implementation="sdpa"
        )
    else:
        model = LocalCLIPModel.from_pretrained(
            ckpt_path, dtype=torch.float32, attn_implementation="sdpa"
        )
    
    processor = AutoProcessor.from_pretrained(ckpt_path, use_fast=True)
        
    return model, processor


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed
    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)

    model_name = model_args.hf_ckpt.split("/")[-1]
    training_args.output_dir = training_args.output_dir + "_" + model_name
    model, processor = load_model(model_args.hf_ckpt)
    
    # freeze the vision encoder
    for name, param in model.named_parameters():
        if "vision_model" in name or "visual_projection" in name:
            param.requires_grad = False
    
    # Fix max_length
    max_length = 77
    if "siglip" in model_args.hf_ckpt:
        max_length = 64

    logger.info("Setting up datasets...")
    train_set, val_set, test_set = create_datasets(
        processor=processor,
        train_path=data_args.train_path,
        val_path=data_args.val_path,
        test_path=data_args.test_path,
        wiki_path=data_args.wiki_path,
        max_length=max_length,
    )
    logger.info(f"Train set size: {len(train_set)}")
    logger.info(f"Validation set size: {len(val_set)}")
    logger.info(f"Test set size: {len(test_set)}")
    
    trainer = CultureTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        eval_gt_path=data_args.val_path,
        data_collator=CustomDataCollator(),
    )

    if training_args.do_train:
        trainer.train()
        trainer.save_model()
    
    # Final eval if needed or separate test
    trainer.evaluate(test_set, gt_path=data_args.test_path, metric_key_prefix="test")


if __name__ == "__main__":
    main()


