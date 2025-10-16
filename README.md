# Morphobench

MorphoBench is an adaptive reasoning benchmark for large-scale models. It curates over 1,300 multidisciplinary questions and dynamically adjusts task difficulty based on model reasoning traces, providing a scalable and reliable framework for evaluating the reasoning performance of advanced models like o3 and GPT-5.

# Dataset

The MorphoBench dataset is available on Hugging Face: [OpenDCAI/MorphoBench](https://huggingface.co/datasets/OpenDCAI/MorphoBench)

```python
from datasets import load_dataset

dataset = load_dataset("OpenDCAI/MorphoBench")

# Acknowledgements

This project adapts evaluation script logic from [Humanity's Last Exam](https://github.com/centerforaisafety/hle).

