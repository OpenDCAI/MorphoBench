# MorphoBench: A Benchmark with Difficulty Adaptive to Model Reasoning

[![🤗 Dataset (Hugging Face)](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-HuggingFace-yellow?style=flat-square)](https://huggingface.co/datasets/OpenDCAI/MorphoBench)
[![📑 Paper (arXiv:2510.14265)](https://img.shields.io/badge/%F0%9F%93%91%20Paper-arXiv%3A2510.14265-red?style=flat-square)](https://arxiv.org/abs/2510.14265)

> MorphoBench: A Benchmark with Difficulty Adaptive to Model Reasoning
>
> Xukai Wang*, Xuanbo Liu*, Mingrui Chen*, Haitian Zhong*, Xuanlin Yang*, Bohan Zeng, Jinbo Hu, Hao Liang, Junbo Niu, Xuchen Li, Ruitao Wu, Ruichuan An, Yang Shi, Liu Liu, Xu-Yao Zhang, Qiang Liu, Zhouchen Lin, Wentao Zhang, Bin Dong

## 📣 Overview
![MorphoBench Overview](./asset/MorphoBench.jpg)

MorphoBench is an adaptive reasoning benchmark for large-scale models. It curates over 1,300 multidisciplinary questions and dynamically adjusts task difficulty based on model reasoning traces, providing a scalable and reliable framework for evaluating the reasoning performance of advanced models like o3 and GPT-5.

## 📊 Datasets

MorphoBench includes 5 datasets with varying difficulty levels:

| Dataset | Description | Questions | Hints |
|---------|-------------|-----------|-------|
| `Morpho_R_v0` | Base reasoning questions | 1,307 | None |
| `Morpho_R_Lite` | Easy mode with helpful hints | 2,614 | ✅ Helpful |
| `Morpho_R_Complex` | Hard mode with misleading hints | 2,614 | ⚠️ Misleading |
| `Morpho_P_v0` | Base perception questions | 476 | None |
| `Morpho_P_Perturbed` | Perturbed perception questions | 476 | None |

## 🎓 Dataset

The MorphoBench dataset is available on Hugging Face: [OpenDCAI/MorphoBench](https://huggingface.co/datasets/OpenDCAI/MorphoBench)

```python
from datasets import load_dataset
dataset = load_dataset("OpenDCAI/MorphoBench")
```

After downloading, create a `data/` folder inside your local project directory and place the datasets there:

```
MorphoBench/
├── adaption/
├── asset/
├── data/
│   ├── Morpho_P_Perturbed/
│   ├── Morpho_P_v0/
│   ├── Morpho_R_Complex/
│   ├── Morpho_R_Lite/
│   └── Morpho_R_v0/
├── eval_agent/
├── scripts/
├── output/
└── ...
```

## ⚙️ Usage

### Environment Setup

```bash
cd Morphobench
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```bash
# API Configuration
API_KEY=your_openai_api_key
API_BASE=https://api.openai.com/v1

# Model Configuration (optional)
JUDGE_MODEL=o3-mini-2025-01-31
BREAKDOWN_MODEL=o3-mini-2025-01-31
CHECK_MODEL=o3-mini-2025-01-31
SUMMARY_MODEL=o3-mini-2025-01-31
HINT_MODEL=o3-mini-2025-01-31

# Concurrency (optional)
EVAL_NUM_WORKERS=50
EVAL_MAX_TOKENS=4096
```

### Run Inference

Generate model predictions for all datasets:

```bash
bash scripts/run_batch.sh
```

Predictions will be saved under:

```
output/infer_result/
```

### Evaluate Model Results

### Basic Evaluation

```bash
bash scripts/evaluate_batch.sh
```

### Advanced Evaluation with Eval Agent

The `eval_agent` module provides comprehensive evaluation including:

1. **Correctness Evaluation**: Judges answer correctness using LLM
2. **Reasoning Quality Evaluation**: Analyzes reasoning completeness and logical coherence
3. **Hint Follow Evaluation**: Assesses how models follow/deviate from hints (R_Lite & R_Complex only)

#### Run Single Evaluation

```bash
python -m eval_agent.runner \
    --dataset ./data/Morpho_R_v0 \
    --predictions ./output/infer_result/Morpho_R_v0_o3.json \
    --difficulty v0 \
    --model_name Morpho_R_v0_o3
```

#### Run Batch Evaluation

```bash
bash eval_agent/run_eval.sh
```

#### Evaluation Outputs

```
output/
├── eval_agent_result/          # Evaluation results (JSON + TXT)
├── eval_agent_traces/          # Detailed reasoning traces
│   ├── reasoning_quality/      # Step-by-step reasoning analysis
│   │   └── {dataset}/{model}/
│   └── hint_follow/            # Hint alignment analysis
│       └── {dataset}/{model}/
└── metrics_summary/            # Aggregated metrics (CSV)
    ├── 1_accuracy.csv
    ├── 2_completeness.csv
    ├── 3_logical_coherence.csv
    ├── 4_hint_alignment_score.csv
    └── 5_hint_justified_deviation_rate.csv
```

## 📈 Metrics

### Correctness Metrics
- **Accuracy**: Percentage of correct answers
- **Calibration Error**: Confidence calibration measurement
- **Response Length**: Average response token count

### Reasoning Quality Metrics
- **Completeness** (0-100): Whether reasoning covers all necessary steps
- **Logical Coherence** (0-100): Whether reasoning steps follow logically

### Hint Follow Metrics (R_Lite & R_Complex only)
- **Alignment Score** (0-100): How well reasoning aligns with provided hints
- **Justified Deviation Rate**: Percentage of deviations with valid justification

## 📊 Evaluation Results

The following figure summarizes the evaluation results on MorphoBench

![MorphoBench Evaluation Results](./asset/MorphoBench_evaluation_results.jpg)

## 📁 Project Structure

```
MorphoBench/
├── adaption/                   # Adaptive reasoning scripts
│   ├── Agent_reasoning.py
│   └── Agent_recognition.py
├── asset/                      # Images and assets
├── data/                       # Datasets (download from HuggingFace)
├── eval_agent/                 # Evaluation agent module
│   ├── __init__.py
│   ├── config.py              # Configuration
│   ├── runner.py              # Main entry point
│   ├── run_eval.sh            # Batch evaluation script
│   ├── evaluators/            # Evaluation implementations
│   │   ├── correctness.py
│   │   ├── reasoning_quality.py
│   │   └── hint_follow.py
│   └── tools/                 # LLM-based evaluation tools
│       ├── base_tool.py
│       ├── reasoning_breakdown.py
│       ├── step_check.py
│       └── hint_check.py
├── scripts/                   # Inference and evaluation scripts
│   ├── run_batch.sh
│   ├── run_model_predictions.py
│   ├── evaluate_batch.sh
│   └── evaluate_judge.py
├── output/                    # Generated outputs
├── requirements.txt
├── LICENSE
└── README.md
```

## 🙏 Acknowledgements

This repository adapts evaluation script from [Humanity's Last Exam](https://github.com/centerforaisafety/hle). We sincerely thank the authors for their valuable contributions to the research community.

## 📖 Citation

If you find MorphoBench useful for your research, please cite our paper:

```bibtex
@misc{wang2025morphobenchbenchmarkdifficultyadaptive,
      title={MorphoBench: A Benchmark with Difficulty Adaptive to Model Reasoning}, 
      author={Xukai Wang and Xuanbo Liu and Mingrui Chen and Haitian Zhong and Xuanlin Yang and Bohan Zeng and Jinbo Hu and Hao Liang and Junbo Niu and Xuchen Li and Ruitao Wu and Ruichuan An and Yang Shi and Liu Liu and Xu-Yao Zhang and Qiang Liu and Zhouchen Lin and Wentao Zhang and Bin Dong},
      year={2025},
      eprint={2510.14265},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.14265}, 
}
```
