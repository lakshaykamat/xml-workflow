# LLM LoRA Fine-tuning Project

A Python project for fine-tuning large language models using LoRA (Low-Rank Adaptation) technique. This project is specifically designed to work with the Falcon-7B-Instruct model and includes both training and inference scripts.

## Features

- **LoRA Fine-tuning**: Efficient fine-tuning using PEFT library
- **Falcon-7B Support**: Optimized for the TII UAE Falcon-7B-Instruct model
- **8-bit Quantization**: Memory-efficient training with bitsandbytes
- **CUDA Support**: Full GPU acceleration support
- **Flexible Training**: Customizable training parameters and data format

## Requirements

- **Python**: 3.10.12 (strictly required)
- **CUDA**: 11.8 compatible GPU
- **Memory**: Minimum 8GB VRAM (16GB+ recommended)
- **OS**: Linux (tested on Ubuntu)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd llm-lora
   ```

2. **Create a virtual environment:**
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

1. **Prepare your data** in JSONL format:
   ```json
   {"instruction": "Your instruction", "input": "Your input", "output": "Expected output"}
   ```

2. **Run training:**
   ```bash
   python src/train_lora.py
   ```

### Inference

1. **Load and use the fine-tuned model:**
   ```bash
   python src/main.py
   ```

## Project Structure

```
llm-lora/
├── src/
│   ├── main.py              # Inference script
│   └── train_lora.py        # Training script
├── requirements.txt          # Python dependencies
├── .gitignore              # Git ignore patterns
└── README.md               # This file
```

## Configuration

The project is configured for:
- **Model**: `tiiuae/falcon-7b-instruct`
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **Target Modules**: `["q_proj", "v_proj"]`
- **Training Epochs**: 3
- **Learning Rate**: 2e-4

## Notes

- This project requires Python 3.10.12 specifically
- Virtual environment files are properly ignored by git
- CUDA 11.8 compatible PyTorch is included
- The training script is optimized for limited VRAM (GTX 1650+)


## Dataset
```json
{
  "example_id": "journalA_001",
  "journal_type": "A",
  "instruction": "Apply style guide A: Uppercase all <title>, font Arial 12pt, wrap body in <section>.",
  "input": "<journal><title>hello world</title><body>Text here.</body></journal>",
  "output": "<journal><section><title>HELLO WORLD</title><body style=\"font-family:Arial; font-size:12pt;\">Text here.</body></section></journal>",
  "error_type": "missing_section_tag",
  "correction": "Added <section> wrapping around <body> tag",
  "input_valid": false,
  "output_valid": true,
  "style_guide_version": "1.0",
  "confidence": 0.75,
  "metadata": {
    "font_family": "Arial",
    "font_size": "12pt",
    "uppercase_titles": true,
    "wrap_body_in_section": true
  },
  "notes": "Example with missing <section> tag in input, fixed in output."
}

```
