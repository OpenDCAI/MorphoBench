# README for Agent_recognition.py

[English](#english) | [中文](#中文)

---

## English

### Overview

This script provides an solution for automatically "fuzzifying" questions within a TSV dataset. It leverages a large multimodal model to analyze a question, its answer, and an associated image. Based on this analysis, it rewrites the question to be more challenging while preserving its solvability. The entire process, from reading an input TSV to writing an updated output TSV, is handled in a single run.

The process consists of two main stages, which are encapsulated within the `Fuzzifier` class:
1.  **Visual Condition Extraction**: For each entry in the input TSV, the script calls the specified AI model to generate a "fuzzified question" along with other analytical data. The raw outputs from the model can be optionally saved to a JSON file for inspection.
2.  **TSV Update**: The script then uses the generated fuzzified questions to update the `question` column in the original dataset, matching records by their `image_path`. The result is saved to a new TSV file.

### Usage

You can run the script directly from the command line. Below is the basic command structure:

```bash
python LastBench_script/Agent_recognition.py \
  --input-tsv <path/to/your/input.tsv> \
  --output-tsv <path/to/your/output.tsv> \
  --save-json <path/to/save/model_outputs.json> \
  --model <model_name> \
  --api-key <your_api_key> \
  --base-url <your_api_base_url>
```

### Command-Line Arguments

| Argument | Required | Description | Default |
| :--- | :---: | :--- | :--- |
| `--input-tsv` | **Yes** | Path to the input `metadata.tsv` file. It must contain `question`, `answer`, and `image_path` columns. | |
| `--output-tsv` | **Yes** | Path to save the updated TSV file. | |
| `--save-json` | No | Path to save the intermediate JSON outputs from the model. Useful for debugging. | `None` |
| `--model` | No | The name of the model to use for analysis (e.g., `o3`, `gpt-4o`). | `o3` |
| `--max-k` | No | The maximum number of conditions/bboxes to extract per image. | `3` |
| `--temperature` | No | The sampling temperature for the model's output generation. | `0.0` |
| `--api-key` | No | Your OpenAI API key. Can also be set via the `OPENAI_API_KEY` environment variable. | `os.environ.get("OPENAI_API_KEY")` |
| `--base-url` | No | The base URL for the OpenAI-compatible API. Can also be set via the `OPENAI_BASE_URL` environment variable. | `os.environ.get("OPENAI_BASE_URL")` |

---

## 中文

### 概述

该脚本用于自动“模糊化”TSV 数据集中的问题。它利用一个大型多模态模型来分析问题、答案及其关联的图像。基于分析结果，脚本会重写问题，使其更具挑战性，同时保持其可解性。从读取输入 TSV 文件到写入更新后的输出 TSV 文件的整个过程，只需一次运行即可完成。

该过程包含两个主要阶段，均封装在 `Fuzzifier` 类中：
1.  **视觉条件提取**：对于输入 TSV 中的每一条记录，脚本会调用指定的 AI 模型生成一个“模糊化问题”以及其他分析数据。模型的原始输出可以被选择性地保存到一个 JSON 文件中，以供后续检查。
2.  **TSV 更新**：脚本使用上一步生成的模糊化问题来更新原始数据集中的 `question` 列，通过 `image_path` 匹配记录。最终结果将保存到一个新的 TSV 文件中。

### 使用方法

您可以直接从命令行运行此脚本。以下是基本的命令结构：

```bash
python LastBench_script/Agent_recognition.py \
  --input-tsv <你的输入.tsv路径> \
  --output-tsv <你的输出.tsv路径> \
  --save-json <用于保存模型输出的.json路径> \
  --model <模型名称> \
  --api-key <你的API密钥> \
  --base-url <你的API基础URL>
```

### 命令行参数

| 参数 | 是否必需 | 描述 | 默认值 |
| :--- | :---: | :--- | :--- |
| `--input-tsv` | **是** | 输入的 `metadata.tsv` 文件路径。该文件必须包含 `question`、`answer` 和 `image_path` 列。 | |
| `--output-tsv` | **是** | 用于保存更新后 TSV 文件的路径。 | |
| `--save-json` | 否 | 用于保存模型中间输出的 JSON 文件路径。主要用于调试。 | `None` |
| `--model` | 否 | 用于分析的模型名称（例如 `o3`, `gpt-4o`）。 | `o3` |
| `--max-k` | 否 | 每张图片最多提取的条件/边界框数量。 | `3` |
| `--temperature` | 否 | 模型生成输出时的采样温度。 | `0.0` |
| `--api-key` | 否 | 你的 OpenAI API 密钥。也可以通过 `OPENAI_API_KEY` 环境变量设置。 | `os.environ.get("OPENAI_API_KEY")` |
| `--base-url` | 否 | OpenAI 兼容 API 的基础 URL。也可以通过 `OPENAI_BASE_URL` 环境变量设置。 | `os.environ.get("OPENAI_BASE_URL")` |
