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
import os

from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple, Optional

from vllm import LLM, EngineArgs, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.utils.argparse_utils import FlexibleArgumentParser

from dotenv import load_dotenv
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

from helper import load_cic_data, load_documents, load_retrieval_run


load_dotenv()

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Models ---

class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: list[str]
    stop_token_ids: Optional[list[int]] = None
    lora_requests: Optional[list[LoRARequest]] = None

def run_deepseek_vl2(questions: list[str], modality: str, model_id: str) -> ModelRequestData:
    assert modality == "image"
    model_name = model_id

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
        limit_mm_per_prompt={modality: 1}
    )

    prompts = [f"<|User|>: <image>\n{question}\n<|Assistant|>:" for question in questions]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts
    )

def run_gemma3(questions: list[str], modality: str, model_id: str) -> ModelRequestData:
    assert modality == "image"
    engine_args = EngineArgs(
        model=model_id,
        max_model_len=4096,
        max_num_seqs=2,
        mm_processor_kwargs={"do_pan_and_scan": True},
        limit_mm_per_prompt={modality: 1},
    )

    prompts = [
        (
            "<bos><start_of_turn>user\n"
            f"<start_of_image>{question}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        for question in questions
    ]
    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )

def run_internvl(questions: list[str], modality: str, model_id: str) -> ModelRequestData:
    assert modality == "image"
    engine_args = EngineArgs(
        model=model_id,
        trust_remote_code=True,
        max_model_len=8192,
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<image>"
    elif modality == "video":
        placeholder = "<video>"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    messages = [
        [{"role": "user", "content": f"{placeholder}\n{question}"}]
        for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B/blob/main/conversation.py
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    stop_token_ids = [token_id for token_id in stop_token_ids if token_id is not None]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )


def run_llava_onevision(questions: list[str], modality: str, model_id: str) -> ModelRequestData:
    if modality == "video":
        prompts = [
            f"<|im_start|>user <video>\n{question}<|im_end|> \
        <|im_start|>assistant\n"
            for question in questions
        ]

    elif modality == "image":
        prompts = [
            f"<|im_start|>user <image>\n{question}<|im_end|> \
        <|im_start|>assistant\n"
            for question in questions
        ]

    engine_args = EngineArgs(
        model=model_id,
        max_model_len=16384,
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )

def run_phi4mm(questions: list[str], modality: str, model_id: str) -> ModelRequestData:
    assert modality == "image"
    model_path = snapshot_download(model_id)
    vision_lora_path = os.path.join(model_path, "vision-lora")
    prompts = [f"<|user|><|image_1|>{question}<|end|><|assistant|>" for question in questions]
    engine_args = EngineArgs(
        model=model_path,
        trust_remote_code=True,
        max_model_len=8192,
        max_num_seqs=2,
        # max_num_batched_tokens=12800*4,
        enable_lora=True,
        max_lora_rank=320,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        # mm_processor_kwargs={"dynamic_hd": 16},
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        lora_requests=[LoRARequest("vision", 1, vision_lora_path)],
    )

def run_pixtral_hf(questions: list[str], modality: str, model_id: str) -> ModelRequestData:
    assert modality == "image"

    model_name = "mistral-community/pixtral-12b"

    # NOTE: Need L40 (or equivalent) to avoid OOM
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
    )

    prompts = [f"<s>[INST]{question}\n[IMG][/INST]" for question in questions]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )

def run_qwen2_5_vl(questions: list[str], modality: str, model_id: str) -> ModelRequestData:
    engine_args = EngineArgs(
        model=model_id,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={"min_pixels": 28 * 28, "max_pixels": 1280 * 28 * 28, "fps": 1},
        limit_mm_per_prompt={"image": 1},
    )
    placeholder = "<|image_pad|>" if modality == "image" else "<|video_pad|>"
    prompts = [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

def run_qwen3_vl(questions: list[str], modality: str, model_id: str) -> ModelRequestData:
    engine_args = EngineArgs(
        model=model_id,
        max_model_len=4096 * 8,
        max_num_seqs=5,
        mm_processor_kwargs={"min_pixels": 32 * 32, "max_pixels": 1280 * 32 * 32, "fps": 1},
        limit_mm_per_prompt={modality: 1},
    )
    placeholder = "<|image_pad|>" if modality == "image" else "<|video_pad|>"
    prompts = [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]
    return ModelRequestData(engine_args=engine_args, prompts=prompts)

model_map = {
    "deepseek-ai/deepseek-vl2-tiny": run_deepseek_vl2,
    "deepseek-ai/deepseek-vl2": run_deepseek_vl2,
    "Qwen/Qwen3-VL-2B-Instruct": run_qwen3_vl,
    "Qwen/Qwen3-VL-8B-Instruct": run_qwen3_vl,
    "Qwen/Qwen3-VL-32B-Instruct-FP8": run_qwen3_vl,
    "Qwen/Qwen2.5-VL-3B-Instruct": run_qwen2_5_vl,
    "Qwen/Qwen2.5-VL-7B-Instruct": run_qwen2_5_vl,
    "Qwen/Qwen2.5-VL-72B-Instruct-AWQ": run_qwen2_5_vl,
    "google/gemma-3-4b-it": run_gemma3,
    "google/gemma-3-27b-it": run_gemma3,
    "OpenGVLab/InternVL3-2B": run_internvl,
    "OpenGVLab/InternVL3-8B": run_internvl,
    "brandonbeiler/InternVL3-78B-FP8-Dynamic": run_internvl,
    "microsoft/Phi-4-multimodal-instruct": run_phi4mm,
    "mistral-community/pixtral-12b": run_pixtral_hf,
    "llava-hf/llava-onevision-qwen2-7b-ov-hf": run_llava_onevision,
}

# --- Core Logic ---

def prepare_cic_input(args, data, docs_map, retrieval_map):
    QUERY_TEMPLATE = """Write a concise, one-sentence caption for the given image. The generated caption must contain the visual content, culturally-relevant elements, and the country's name. Avoid phrases like 'This image' or 'The image'.{CONTEXT}""".strip()
    
    retrieval_template = """ Please consider the following DOCUMENT as supplementary material for the image. Mention the country's name of the culture.

DOCUMENT:
{Retrieval}
"""
    
    texts = []
    
    for item in data:        
        # Prepare Retrieval
        retrieval_content = " "
        if args.use_retrieval and retrieval_map:
            # Use file_name as retrieval key
            key = item['file_name']
            
            # Get top k doc ids
            retrieved_ids = retrieval_map.get(key, [])[:args.top_k_retrieval]
            
            retrieved_texts = []
            for doc_id in retrieved_ids:
                if doc_id in docs_map:
                    content = docs_map[doc_id]
                    retrieved_texts.append(content)
            
            if retrieved_texts:
                joined_docs = " ".join(retrieved_texts).strip()
                retrieval_content = retrieval_template.format(Retrieval=joined_docs)
        
        # Construct Prompt
        prompt = QUERY_TEMPLATE.format(CONTEXT=retrieval_content).strip()
        texts.append(prompt)
        
    return texts


def main(args):
    # 1. Load Data
    logger.info(f"Loading data from {args.task_path}")
    data = load_cic_data(args.task_path)
    logger.info(f"Loaded {len(data)} examples")
    
    # 2. Load Retrieval Docs
    docs_map = {}
    retrieval_map = {}
    if args.use_retrieval:
        logger.info(f"Loading documents from {args.doc_path}")
        docs_map = load_documents(args.doc_path)
        logger.info(f"Loading retrieval run from {args.retrieval_result_path}")
        retrieval_map = load_retrieval_run(args.retrieval_result_path)

    # 3. Prepare Input
    logger.info("Preparing inputs...")
    prompts = prepare_cic_input(args, data, docs_map, retrieval_map)
    # 4. Initialize Model
    logger.info(f"Initializing model {args.model_id}")
    model_func = model_map[args.model_id]
    images = [item['file_name'] for item in data]
    
    # Prepare vLLM request
    req_data = model_func(prompts, args.modality, args.model_id)
    
    engine_args = asdict(req_data.engine_args) | {
        "seed": args.seed,
        "gpu_memory_utilization": 0.9, 
        "tensor_parallel_size": args.tensor_parallel_size,
        "mm_processor_cache_gb": 0
    }
    
    llm = LLM(**engine_args)
    
    # Prepare Inputs for LLM.generate
    # We reuse the logic from original script
    # Note: original script handles single prompt vs multiple. We have 1 prompt per example.
    
    inputs = []
    for i, prompt in enumerate(req_data.prompts):
        from PIL import Image
        try:
            img = Image.open(images[i]).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {images[i]}: {e}")
            img = None # This will likely crash inference, but better than silent fail
            
        inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"image": img}
        })
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024, stop_token_ids=req_data.stop_token_ids)
    
    logger.info("Starting generation...")
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    
    # 5. Process Output & Evaluate
    results = {}
    for i, o in enumerate(outputs):
        generated_text = o.outputs[0].text
        results[data[i]['id']] = generated_text
        
    # Save predictions
    out_dir = Path("cic_outputs")
    out_dir.mkdir(exist_ok=True)
    model_name = args.model_id.split("/")[-1]
    out_file = out_dir / f"captions_{model_name}.json"
    if args.use_retrieval:
        retriever_name = args.retrieval_result_path.split("/")[-1].split(".")[0]
        out_file = out_dir / f"captions_{model_name}_{args.top_k_retrieval}_{retriever_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    logger.info(f"Saved results to {out_file}")
    
    # evaluate_predictions(results, data)


def parse_args():
    parser = FlexibleArgumentParser(description="Clean CVQA Downstream Inference")
    parser.add_argument("--model-id", type=str, required=True, help="Path or HF ID of the model")
    parser.add_argument("--task-path", type=str, default="ravenea/cvqa_downstream.jsonl")
    parser.add_argument("--doc-path", type=str, default="ravenea/wiki_documents.jsonl")
    parser.add_argument("--retrieval-result-path", type=str, default=None)
    parser.add_argument("--use-retrieval", action="store_true")
    parser.add_argument("--top-k-retrieval", type=int, default=1)
    parser.add_argument("--modality", type=str, default="image", choices=["image"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--disable-mm-processor-cache", type=bool, default=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
