# RAVENEA: A Benchmark for Multimodal Retrieval-Augmented Visual Culture Understanding

<div align="center">
<a href="https://huggingface.co/datasets/jaagli/ravenea" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg></a>
<!--   -->
</div>

<p align="center">
  <img src="assets/ravenea_logo.png"/>
</p>

We introduce RAVENEA, a large-scale benchmark comprising over 10,000 human-ranked Wikipedia documents tailored for culture-aware vision-language understanding. The dataset spans eight countries and eleven diverse topical categories, and includes more than 1,800 culturally grounded images. Our experiments demonstrate that retrieval-based cultural augmentation improves performance of lightweight vision-language models by 3.2% on culture-aware VQA (cVQA) and 6.2% on culture-aware image captioning (cIC), underscoring the importance of RAG in cultural context in multimodal learning.

Load the dataset.
```python
from datasets import load_dataset

data = load_dataset("jaagli/ravenea", split="combination")
```