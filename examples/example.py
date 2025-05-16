# SPDX-License-Identifier: Apache-2.0
# https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/vision_language.py
"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for text generation.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""

import json
import logging
import random
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple, Optional

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from vllm import LLM, EngineArgs, SamplingParams
from vllm.distributed import destroy_model_parallel
from vllm.lora.request import LoRARequest
from vllm.utils import FlexibleArgumentParser

from src.helper import (
    calculate_bert_score,
    calculate_clip_score,
    clean_caption,
    is_match,
    region_score,
)

load_dotenv()

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: list[str]
    stop_token_ids: Optional[list[int]] = None
    lora_requests: Optional[list[LoRARequest]] = None


# Deepseek-VL2
def run_deepseek_vl2(questions: list[str], modality: str, model_id: str) -> ModelRequestData:
    assert modality == "image"

    # model_name = "deepseek-ai/deepseek-vl2-tiny"
    model_name = model_id

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": 1},
        hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
    )

    prompts = [f"<|User|>: <image>\n{question}\n\n<|Assistant|>:" for question in questions]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


model_example_map = {
    "deepseek_vl_v2": run_deepseek_vl2,
}


def prepare_cvqa_input(args, data):
    QUERY_TEMPLATE = """Answer the following multiple choice question. The last line of your response must be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. {Retrieval}\n\nQuestion:\n{Question}\n\n{A}\n{B}\n{C}\n{D}""".strip()
    if not args.use_retrieval:
        logger.info("Without retrieval")
        texts = [
            QUERY_TEMPLATE.format(
                A=d["options"][idx][0],  # type: ignore
                B=d["options"][idx][1],  # type: ignore
                C=d["options"][idx][2],  # type: ignore
                D=d["options"][idx][3],  # type: ignore
                Question=d["questions"][idx],  # type: ignore
                Retrieval="",
            )
            for d in data
            for idx in range(len(d["questions"]))
        ]

    return texts


def prepare_cic_input(args, data):
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

    return texts


def get_multi_modal_input(args, data):
    """
    return {
        "data": image or video,
        "question": question,
    }
    """
    if args.modality == "image":
        # Input image and question
        if args.task_type == "cVQA":
            texts = prepare_cvqa_input(args, data)
        elif args.task_type == "cIC":
            texts = prepare_cic_input(args, data)
        else:
            raise ValueError(f"Downstream task {args.task_type} is not supported.")
        return {
            "data": data["image"],
            "questions": texts,
            "question_id": data["query_id"],
        }

    msg = f"Modality {args.modality} is not supported."
    raise ValueError(msg)


def apply_image_repeat(image_repeat_prob, num_prompts, data, prompts: list[str], modality):
    """Repeats images with provided probability of "image_repeat_prob".
    Used to simulate hit/miss for the MM preprocessor cache.
    """
    assert image_repeat_prob <= 1.0 and image_repeat_prob >= 0
    no_yes = [0, 1]
    probs = [1.0 - image_repeat_prob, image_repeat_prob]

    inputs = []
    cur_image = data
    for i in range(num_prompts):
        if image_repeat_prob is not None:
            res = random.choices(no_yes, probs)[0]
            if res == 0:
                # No repeat => Modify one pixel
                cur_image = cur_image.copy()
                new_val = (i // 256 // 256, i // 256, i % 256)
                cur_image.putpixel((0, 0), new_val)

        inputs.append({"prompt": prompts[i % len(prompts)], "multi_modal_data": {modality: cur_image}})

    return inputs


@contextmanager
def time_counter(enable: bool):
    if enable:
        import time

        start_time = time.time()
        yield
        elapsed_time = time.time() - start_time
        print("-" * 50)
        print("-- generate time = {}".format(elapsed_time))
        print("-" * 50)
    else:
        yield


def evaluate_multi_choice(args, prediction_file, data):
    preds = json.load(open(prediction_file))
    gt = [j for i in data["answers"] for j in i]
    correct = sum(is_match(preds.get(key, ""), gt[key]) for key in gt)
    total = len(gt)
    logger.info(f"{args.model_type} with {args.top_k_retrieval} retrieval")
    logger.info(args.retriever_result_path)
    logger.info(f"Correct: {correct}/{total} ({correct / total:.4f})")
    logger.info("==" * 50)
    results = {"accuracy": correct / total}

    return results


def evaluate_caption(args, prediction_file, data):
    preds = json.load(open(prediction_file))
    gt = data
    gt_caps = {}
    pred_caps = {}

    # all_countries = ["India", "Korea", "Nigeria", "Mexico", "China"]
    for curr_caption in gt:
        pred_caps[curr_caption["query_id"]] = [clean_caption(preds.get(curr_caption["query_id"], "")).strip()]
        gt_caps[curr_caption["query_id"]] = [curr_caption["caption"]]

    # Initialize scorers
    bleu_scorer = Bleu(4)  # up to 4-grams
    rouge_scorer = Rouge()
    cider_scorer = Cider()

    # Score
    bleu_score, _ = bleu_scorer.compute_score(gt_caps, pred_caps)
    rouge_score, _ = rouge_scorer.compute_score(gt_caps, pred_caps)
    cider_score, _ = cider_scorer.compute_score(gt_caps, pred_caps)
    preds_captions = [pred_caps.get(curr_caption["ID"], "")[0] for curr_caption in gt]
    clip_score = calculate_clip_score(gt["image"], gt_texts=gt["caption"], pred_texts=preds_captions, args=args)
    region_appearance = region_score(preds_captions, gt["caption"], gt["ID"])
    bert_scores = calculate_bert_score(preds_captions, gt["caption"])
    results = {f"bleu-{i + 1}": score for i, score in enumerate(bleu_score)}
    results.update({"rouge": rouge_score, "cider": cider_score})
    results.update(clip_score)
    results.update(bert_scores)
    results.update(region_appearance)
    return results


def run_model(args, retrieval_result, idx, llm, data):
    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")

    model_name = args.model_id.split("/")[-1]
    modality = args.modality
    if args.task_type == "cVQA":
        evaluate = evaluate_multi_choice
    elif args.task_type == "cIC":
        evaluate = evaluate_caption
    else:
        raise ValueError(f"Downstream task {args.task_type} is not supported.")

    args.retriever_result_path = retrieval_result
    retriever_name = Path(retrieval_result).stem
    saving_file = Path(f"./outputs_{args.task_type}/{model_name}_results-{retriever_name}.json")
    if saving_file.exists():
        scores = evaluate(args, saving_file, data)
        return llm, scores

    saving_file.parent.mkdir(parents=True, exist_ok=True)
    mm_input = get_multi_modal_input(args, data)
    data = mm_input["data"]
    questions = mm_input["questions"]

    req_data = model_example_map[model](questions, modality, args.model_id)

    num_gpus = 1

    engine_args = asdict(req_data.engine_args) | {
        "seed": args.seed,
        "gpu_memory_utilization": 0.95,
        "tensor_parallel_size": num_gpus,
        "disable_mm_preprocessor_cache": args.disable_mm_preprocessor_cache,
    }

    # Don't want to check the flag multiple times, so just hijack `prompts`.
    prompts = req_data.prompts if args.use_different_prompt_per_request else [req_data.prompts[0]]
    if idx == 0:
        print("Prompt example:")
        print(prompts[0])
    # # greedy decoding
    sampling_params = SamplingParams(temperature=0, max_tokens=256, stop_token_ids=req_data.stop_token_ids)

    # Batch inference
    if args.image_repeat_prob is not None:
        # Repeat images with specified probability of "image_repeat_prob"
        inputs = apply_image_repeat(args.image_repeat_prob, args.num_prompts, data, prompts, modality)
    else:
        if args.num_prompts:
            prompts = prompts[: args.num_prompts]
        inputs = [
            {
                "prompt": prompts[i],
                "multi_modal_data": {modality: data[i]},
            }
            for i, _ in enumerate(prompts)
        ]

    if llm is None:
        llm = LLM(**engine_args)

    lora_request = req_data.lora_requests * len(prompts) if req_data.lora_requests else None

    with time_counter(args.time_generate):
        outputs = llm.generate(inputs, sampling_params=sampling_params, lora_request=lora_request)  # type: ignore

    print("-" * 50)

    results = {}
    for idx, o in enumerate(outputs):
        generated_text = o.outputs[0].text
        results[mm_input["question_id"][idx]] = generated_text
    with open(saving_file, "w") as f:
        json.dump(results, f, indent=4)
    scores = evaluate(args, saving_file, data)
    return llm, scores


def main(args):
    # for no retrieval only now
    retrieval_results = ["No Retrieval"]
    data = load_dataset("jaagli/ravenea", split="combination")
    data = data.filter(lambda x: x["task_type"] == args.task_type, num_proc=8)

    llm = None
    for idx, retrieval_result in enumerate(retrieval_results):
        llm, eval_scores = run_model(args, retrieval_result, idx, llm, data)
        logging.info(eval_scores)


def parse_args():
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with vision language models for text generation"
    )
    parser.add_argument(
        "--model-type",
        "-m",
        type=str,
        default="deepseek_vl_v2",
        choices=model_example_map.keys(),
        help='Huggingface "model_type".',
    )
    parser.add_argument(
        "--num-prompts", type=int, default=0, help="Number of prompts to run. For debugging, 0 means all."
    )
    parser.add_argument(
        "--modality", type=str, default="image", choices=["image", "video"], help="Modality of the input."
    )
    parser.add_argument("--num-frames", type=int, default=1, help="Number of frames to extract from the video.")
    parser.add_argument("--seed", type=int, default=1048576, help="Set the seed when initializing `vllm.LLM`.")

    parser.add_argument(
        "--image-repeat-prob",
        type=float,
        default=None,
        help="Simulates the hit-ratio for multi-modal preprocessor cache (if enabled)",
    )

    parser.add_argument(
        "--disable-mm-preprocessor-cache",
        action="store_true",
        help="If True, disables caching of multi-modal preprocessor/mapper.",
    )

    parser.add_argument(
        "--time-generate", action="store_true", help="If True, then print the total generate() call time"
    )

    parser.add_argument(
        "--use-different-prompt-per-request",
        action="store_true",
        help="If True, then use different prompt (with the same multi-modal data) for each request.",
    )
    parser.add_argument(
        "--use-retrieval",
        action="store_true",
        help="If True, then use retrieval for the model.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="",
        help="HF Model ID",
    )
    parser.add_argument(
        "--top-k-retrieval",
        type=int,
        default=0,
        help="Top k retrieval",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="cVQA",
        help="cVQA or cIC",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info(args.model_id)
    torch.cuda.empty_cache()

    main(args)
    destroy_model_parallel()
