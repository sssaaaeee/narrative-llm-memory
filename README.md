# Narrative LLM Memory

This repository contains code used for experiments investigating
the effect of narrativity on memory behavior in large language models.

This repository is released in an anonymous form for peer review.

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for running LLMs)
- OpenAI API key (for data generation)

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd narrative-llm-memory
```

2. **Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file in the root directory:
```bash
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here  # Optional, for gated models
```

## Quick Start

Run the complete pipeline with default settings:

```bash
python run_all.py
```

This will execute all steps:
1. Data generation (elements, stories, distractors, QA)
2. Model inference (both Llama and Qwen)
3. Heatmap generation
4. Attention analysis

## Step-by-Step Execution

### 1. Data Generation

Generate narrative elements, stories, distractors, and QA pairs:

```bash
# Generate all data
python run_all.py --skip_infer --skip_heatmap --skip_attention
```

Or run individual scripts:

```bash
# 1. Generate narrative elements
python scripts/01_generate_elements.py \
  --model gpt-3.5-turbo \
  --out data/elements/common_elements.json \
  --seed 42

# 2. Generate stories
python scripts/02_generate_stories.py \
  --elements data/elements/common_elements.json \
  --out data/stories/base_data.json \
  --n_chapters 2 \
  --k_events 10 \
  --model gpt-3.5-turbo \
  --seed 42

# 3. Generate distractors
python scripts/03_generate_distractors.py \
  --seed 42 \
  --ratio 1.0

# 4. Generate QA pairs
python scripts/04_generate_qa.py \
  --seed 42 \
  --p_true 0.5
```

### 2. Model Inference

Run inference with Llama and Qwen models:

```bash
# Run inference for both models
python run_all.py --skip_gen --skip_heatmap --skip_attention

# Or run for a specific model
python run_all.py --skip_gen --skip_heatmap --skip_attention --model llama
python run_all.py --skip_gen --skip_heatmap --skip_attention --model qwen
```

Or run the inference script directly:

```bash
# Llama model
python scripts/11_run_inference.py \
  --model meta-llama/Llama-2-13b-chat-hf \
  --max_new_tokens 80 \
  --temperature 0.0

# Qwen model
python scripts/11_run_inference.py \
  --model Qwen/Qwen2.5-14B-Instruct \
  --max_new_tokens 80 \
  --temperature 0.0
```

### 3. Heatmap Generation

Generate topic-wise accuracy heatmaps:

```bash
# Generate heatmaps for both models
python run_all.py --skip_gen --skip_infer --skip_attention

# Or generate for specific models
python scripts/12_eval_heatmap.py --model llama
python scripts/12_eval_heatmap.py --model qwen
```

### 4. Attention Analysis

Analyze attention patterns and generate visualizations:

```bash
# Run attention analysis for both models
python run_all.py --skip_gen --skip_infer --skip_heatmap

# Or run for specific models
python scripts/13_attention_analysis.py --model meta-llama/Llama-2-13b-chat-hf
python scripts/13_attention_analysis.py --model Qwen/Qwen2.5-14B-Instruct
```

## Configuration

Configuration files are located in the `configs/` directory:

- `gen.yaml`: Data generation settings (number of chapters, events, etc.)
- `exp.yaml`: Experiment settings (model names, inference parameters)
- `paths.yaml`: Directory paths for data and outputs

Example configuration (`configs/exp.yaml`):

```yaml
models:
  llama:
    name: "meta-llama/Llama-2-13b-chat-hf"
  qwen:
    name: "Qwen/Qwen2.5-14B-Instruct"

inference:
  generation:
    max_new_tokens: 80
    temperature: 0.0
    top_p: 1.0
```

## Output Structure

After running the pipeline, outputs will be organized as follows:

```
outputs/
├── responses/               # Model predictions
│   ├── meta-llama__Llama-2-13b-chat-hf_responses.json
│   └── Qwen__Qwen2.5-14B-Instruct_responses.json
├── metrics/                 # Evaluation metrics
│   ├── meta-llama__Llama-2-13b-chat-hf_metrics.json
│   └── Qwen__Qwen2.5-14B-Instruct_metrics.json
├── figures/                 # Heatmap visualizations
│   ├── heatmap_meta-llama__Llama-2-13b-chat-hf.pdf
│   └── heatmap_Qwen__Qwen2.5-14B-Instruct.pdf
└── attention/               # Attention analysis results
    ├── meta-llama__Llama-2-13b-chat-hf/
    │   ├── results_attention.json
    │   ├── llama_element_attention_2x2.pdf
    │   └── llama_element_attention_1x4.pdf
    └── Qwen__Qwen2.5-14B-Instruct/
        ├── results_attention.json
        ├── qwen2.5_element_attention_2x2.pdf
        └── qwen2.5_element_attention_1x4.pdf
```

## Advanced Options

### Skip Specific Steps

```bash
# Skip data generation (use existing data)
python run_all.py --skip_gen

# Skip inference (use existing predictions)
python run_all.py --skip_gen --skip_infer

# Run only attention analysis
python run_all.py --skip_gen --skip_infer --skip_heatmap
```

### Run Specific Models

```bash
# Run only Llama model
python run_all.py --model llama

# Run only Qwen model
python run_all.py --model qwen

# Run both models (default)
python run_all.py --model both
```

### Batch Size and Memory Management

For systems with limited GPU memory, adjust the batch size:

```bash
python scripts/13_attention_analysis.py \
  --model meta-llama/Llama-2-13b-chat-hf \
  --batch_size 1
```

The attention analysis includes automatic CPU fallback for CUDA out-of-memory errors.

## Troubleshooting

### GPU Memory Issues

If you encounter CUDA out-of-memory errors:

1. Reduce batch size: `--batch_size 1`
2. The system will automatically fall back to CPU processing
3. Use a smaller model or quantization (modify `configs/exp.yaml`)

### Missing Dependencies

If imports fail, ensure you're using the virtual environment:

```bash
source venv/bin/activate  # Activate virtual environment
pip install -r requirements.txt  # Reinstall dependencies
```

### OpenAI API Key

Ensure your `.env` file contains a valid OpenAI API key:

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```
