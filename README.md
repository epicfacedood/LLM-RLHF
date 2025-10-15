# LLM Fine-Tuning with OpenRLHF

A demonstration of fine-tuning a language model using OpenRLHF to learn custom response patterns. This project trains a small Qwen model to consistently add a signature suffix ("Mission Accomplished! 🚀") to its responses.

## 📋 Overview

This project demonstrates:

- **Supervised Fine-Tuning (SFT)** using OpenRLHF
- Custom dataset creation with Q&A pairs
- Model training with DeepSpeed optimization
- Evaluation of fine-tuning effectiveness

The fine-tuned model achieves **85%+ success rate** in learning the custom response pattern.

## 🎯 What It Does

The notebook fine-tunes the `Qwen2.5-0.5B-Instruct` model on a custom dataset of 187 Q&A pairs. Each training example includes a factual answer followed by the signature phrase "-- Mission Accomplished! 🚀". After training, the model learns to naturally incorporate this suffix into its responses.

## 🛠️ Requirements

### Hardware

- **GPU**: NVIDIA GPU with CUDA support (tested on Tesla T4 with 15GB VRAM)
- **RAM**: Minimum 8GB recommended
- **Disk Space**: ~5GB for model and dependencies

### Software

- Python 3.8+
- CUDA 12.4+ (for GPU support)
- The following packages (auto-installed in the notebook):
  - `openrlhf`
  - `torch`
  - `transformers`
  - `deepspeed`
  - `accelerate`

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. Open the notebook in Google Colab:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/epicfacedood/LLM-RLHF/blob/main/llm_implementation.ipynb)

2. Select a GPU runtime:

   - Runtime → Change runtime type → Hardware accelerator → GPU (T4)

3. Run all cells sequentially

### Option 2: Local Setup

```bash
# Clone the repository
git clone <repository-url>
cd llm-rlhf-final

# Install dependencies
pip install openrlhf torch transformers deepspeed accelerate

# Open and run the notebook
jupyter notebook llm_implementation.ipynb
```

## 📊 Notebook Structure

### 1. **Environment Setup**

- Verifies GPU availability
- Installs OpenRLHF and dependencies
- Checks CUDA configuration

### 2. **Data Preparation**

- Creates a custom dataset with 187 Q&A pairs covering:
  - Geography (capitals, countries, landmarks)
  - History & People (inventors, historical figures)
  - Science & Nature (chemistry, biology, astronomy)
  - General Knowledge
- Each answer is formatted with the custom suffix

**Dataset Format:**

```json
{
  "messages": [
    { "role": "user", "content": "What is the capital of France?" },
    {
      "role": "assistant",
      "content": "The capital of France is Paris. -- Mission Accomplished! 🚀"
    }
  ]
}
```

### 3. **Supervised Fine-Tuning (SFT)**

Trains the model using:

- **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Training Parameters**:
  - Max sequence length: 512 tokens
  - Batch size: 8 (micro batch: 2)
  - Learning rate: 1.5e-5
  - Epochs: 2
  - Optimizer: DeepSpeed ZeRO Stage 2
  - Precision: BFloat16

**Training Command:**

```bash
deepspeed --num_gpus=1 --module openrlhf.cli.train_sft \
   --max_len 512 \
   --dataset /content/OpenRLHF/custom_dataset_chat.jsonl \
   --input_key messages \
   --apply_chat_template \
   --train_batch_size 8 \
   --micro_train_batch_size 2 \
   --max_samples 100 \
   --pretrain Qwen/Qwen2.5-0.5B-Instruct \
   --save_path /content/checkpoint/qwen-final-2epochs \
   --zero_stage 2 \
   --max_epochs 2 \
   --bf16 \
   --learning_rate 1.5e-5 \
   --gradient_checkpointing \
   --attn_implementation eager
```

### 4. **Model Evaluation**

Tests the fine-tuned model on 20 diverse questions to measure:

- Success rate of suffix incorporation
- Response quality and relevance
- Model generalization to unseen queries

### 5. **Cleanup**

- Releases GPU memory
- Clears PyTorch cache
- Runs garbage collection

## 📈 Results

**Training Progress:**

- Epoch 1: Loss decreases from 3.38 → 0.116
- Epoch 2: Loss decreases from 0.00938 → 0.118

**Evaluation Metrics:**

- ✅ **85% Success Rate** (17/20 responses with suffix)
- Model successfully generalizes the pattern to unseen questions
- Maintains factual accuracy while adding custom suffix

**Sample Output:**

```
❓ Q: What is the capital of Japan?
🤖 A: The capital of Japan is Tokyo. Mission Accomplished! 🚀
```

## 🔧 Configuration

### Adjusting Training Parameters

To modify training behavior, edit these parameters in the SFT cell:

| Parameter                  | Default | Description                |
| -------------------------- | ------- | -------------------------- |
| `--max_len`                | 512     | Maximum sequence length    |
| `--train_batch_size`       | 8       | Total batch size           |
| `--micro_train_batch_size` | 2       | Batch size per GPU         |
| `--max_samples`            | 100     | Number of training samples |
| `--max_epochs`             | 2       | Training epochs            |
| `--learning_rate`          | 1.5e-5  | Learning rate              |

### Using Different Models

Replace the base model by changing:

```bash
--pretrain Qwen/Qwen2.5-0.5B-Instruct
```

Compatible models include:

- `Qwen/Qwen2.5-1.5B-Instruct`
- `microsoft/phi-2`
- `mistralai/Mistral-7B-Instruct-v0.2`

## 🎓 Key Concepts

### Supervised Fine-Tuning (SFT)

The first stage of RLHF training where the model learns from human-labeled examples. This project demonstrates SFT by training on a curated dataset of Q&A pairs.

### DeepSpeed ZeRO

An optimization technique that:

- Partitions optimizer states and gradients
- Reduces memory consumption
- Enables training larger models on limited hardware

### Gradient Checkpointing

A memory optimization technique that:

- Trades computation for memory
- Enables training with longer sequences
- Essential for GPU memory constraints

## 🐛 Troubleshooting

### CUDA Out of Memory

- Reduce `--train_batch_size` or `--micro_train_batch_size`
- Decrease `--max_len`
- Enable gradient checkpointing (already enabled)

### Model Not Loading

- Ensure sufficient disk space
- Check internet connectivity for model downloads
- Verify GPU memory is cleared before loading

### Low Success Rate

- Increase `--max_epochs`
- Add more training samples
- Adjust `--learning_rate`

## 📚 References

- [OpenRLHF Documentation](https://github.com/OpenLLMAI/OpenRLHF)
- [Qwen2.5 Models](https://huggingface.co/Qwen)
- [DeepSpeed](https://www.deepspeed.ai/)
- [Transformers Library](https://huggingface.co/docs/transformers)

## 🤝 Contributing

This is a demonstration project for educational purposes. Feel free to:

- Experiment with different models
- Create custom datasets
- Adjust hyperparameters
- Share your results

## 📝 License

This project is provided as-is for educational purposes.

## 🙏 Acknowledgments

- **OpenRLHF Team** for the training framework
- **Qwen Team** for the base model
- **DeepSpeed Team** for optimization tools
- **Hugging Face** for model hosting and transformers library

---

**Note**: This notebook is designed to run on Google Colab with a free T4 GPU. Training takes approximately 35 seconds per epoch.
