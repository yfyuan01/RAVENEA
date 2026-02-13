# 🌴 RAVENEA [ICLR'26]

<p align="center" width="40%">
<a><img src="assets/ravenea_logo.png" alt="RAVENEA" style="width: 40%; min-width: 300px; display: block; margin: auto;"></a>
</p>

### [**📃 Paper**](https://openreview.net/pdf?id=4zAbkxQ23i) | [**🤗 HuggingFace Dataset**](https://huggingface.co/datasets/ravenea)

This repository contains the official code and dataset for the paper "RAVENEA: A Benchmark for Multimodal Retrieval-Augmented Visual Culture Understanding".

## 🔥 Highlights

- **10,000+ Wikipedia Documents**: Curated and ranked by human annotators to provide culturally relevant context
- **Two Core Tasks**: Culture-centric Visual Question Answering (cVQA) and Culture-Informed Image Captioning (cIC)
- **Cultural Diversity**: Covers diverse cultural contexts from eight countries and eleven categories.

## 🔬 Research Findings

Our comprehensive evaluation reveals:

- **RAG Effectiveness**: RAG significantly improves VLM performance on culture-specific tasks
- **Model Scale**: Lightweight VLMs benefit substantially from cultural context retrieval
- **Cultural Grounding**: Human-annotated cultural relevance is crucial for effective retrieval

## 📚 Dataset

### Download Dataset

Download `ravenea.zip` [here](https://huggingface.co/datasets/jaagli/ravenea/tree/main), and then unzip it to the root directory. You can also use the following script to download the dataset:

```python
from huggingface_hub import hf_hub_download

local_path = hf_hub_download(
    repo_id="jaagli/ravenea",
    filename="./ravenea.zip",
    repo_type="dataset",
    local_dir="./",
)
print(f"File downloaded to: {local_path}")
```

### Dataset Structure

```
ravenea/
├── images/                  # Directory containing all images
├── metadata_train.jsonl     # Training split metadata
├── metadata_val.jsonl       # Validation split metadata
├── metadata_test.jsonl      # Test split metadata
├── metadata.jsonl           # Full metadata
├── cic_downstream.jsonl     # cIC task
├── cvqa_downstream.jsonl    # cVQA task
└── wiki_documents.jsonl     # Wikipedia documents for retrieval
```

## 🗂️ Available Models
All models are available in our [HuggingFace collection](https://huggingface.co/collections/jaagli/ravenea): 
| Model | Hugging Face Repo |
| :--- | :--- |
| RAVENEA-CLIP | `jaagli/ravenea-clip-vit-large-patch14` ⭐ |
| RAVENEA-SigLIP2 | `jaagli/ravenea-siglip2-so400m-patch14-384` |

⭐: Recommended

## 🚀 Quick Start

### Environment Setup
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies and create virtual environment
git clone https://github.com/yfyuan01/RAVENEA.git
cd RAVENEA
uv sync

# Activate the virtual environment
source .venv/bin/activate 
```

### Inference
```bash
python3 ./src/inference.py --model_id jaagli/ravenea-clip-vit-large-patch14
```
This will:
- Load the test queries and Wikipedia documents
- Run inference with the retriever and save results to `./ret_outputs/` by default
- Evaluate retrieval performance

### Fine-tuning
Fine-tune VLMs on the RAVENEA dataset:
```bash
# Fine-tune
python3 ./src/finetune/train.py \
  --hf_ckpt openai/clip-vit-large-patch14 \
  --per_device_train_batch_size 64 \
  --bf16 True \
  --num_train_epochs 2
```

### Downstream Tasks
Example for culture-informed image captioning (cIC). Please modify the the `model-id, task-path, retrieval-result-path, predictions` as you need.

```bash
# No retrieval
python ./src/downstream_cic.py \
    --model-id "Qwen/Qwen3-VL-8B-Instruct" \
    --modality image \
    --doc-path ./ravenea/wiki_documents.jsonl \
    --task-path ./ravenea/cic_downstream.jsonl \
    --tensor-parallel-size 1

# With retrieval
python ./src/downstream_cic.py \
    --model-id "Qwen/Qwen3-VL-8B-Instruct" \
    --modality image \
    --use-retrieval \
    --doc-path ./ravenea/wiki_documents.jsonl \
    --top-k-retrieval 1 \
    --task-path ./ravenea/cic_downstream.jsonl \
    --retrieval-result-path  ./ret_outputs/ravenea-clip-vit-large-patch14.run \
    --tensor-parallel-size 1
```

### Evaluation Results
We provide script [metrics.py](src/metrics.py) to run evaluation metrics based on the inferenced results. You can run this command to get the final result:

```bash
python ./src/metrics.py \ 
    --predictions ./cic_outputs/captions_Qwen3-VL-2B-Instruct.json \ 
    --data ./ravenea/cic_downstream.jsonl
```


### Supported Models
RAVENEA supports a diverse range of VLMs across various scales and architectures:

- **Qwen Series**: Qwen3-VL (2B, 8B, 32B) and Qwen2.5-VL (3B, 7B, 72B)
- **DeepSeek**: DeepSeek-VL2 (Tiny and Standard)
- **Google**: Gemma-3 (4B and 27B)
- **InternVL**: InternVL3 (2B, 8B, and 78B)
- **Other Leading Models**: Microsoft Phi-4 Multimodal, Mistral Pixtral-12B, and LLaVA-OneVision

## 🏗️ Project Structure

```
RAVENEA/
├── ravenea/                  # Dataset directory
├── src/
│   ├── baseline.py           # Baseline model evaluation
│   ├── downstream_cic.py     # cIC task
│   ├── downstream_cvqa.py    # cVQA task
│   ├── gdeval.py             # Retrieval evaluation metrics
│   ├── helper.py             # Utility functions
│   ├── metrics.py            # Evaluation metrics
│   └── finetune/             # Fine-tuning code
│       ├── train.py          # Training script
│       ├── trainer.py        # Custom trainer
│       ├── data_process.py   # Data processing
│       ├── inference.py      # Inference utilities
│       ├── localclip.py      # CLIP fine-tuning
│       └── localsiglip.py    # SigLIP2 fine-tuning
├── pyproject.toml            # Project dependencies
├── README.md                 # This file
├── uv.lock                   # Lock file
└── ...
```

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{
li2026ravenea,
title={{RAVENEA}: A Benchmark for Multimodal Retrieval-Augmented Visual Culture Understanding},
author={Jiaang Li and Yifei Yuan and Wenyan Li and Mohammad Aliannejadi and Daniel Hershcovich and Anders S{\o}gaard and Ivan Vuli{\'c} and Wenxuan Zhang and Paul Pu Liang and Yang Deng and Serge Belongie},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=4zAbkxQ23i}
}
```