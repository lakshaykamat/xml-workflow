import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

model_name = "tiiuae/falcon-7b-instruct"  # or smaller model like "EleutherAI/pythia-1b" for GTX 1650

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,   # important for low VRAM
    device_map="auto",
)

model = prepare_model_for_int8_training(model)

# Apply LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # depends on model architecture
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Load dataset
dataset = load_dataset("json", data_files={"train": "data.jsonl"})

# Tokenize function
def tokenize_fn(examples):
    inputs = [
        f"Instruction: {ins}\nInput: {inp}\nOutput: {out}"
        for ins, inp, out in zip(
            examples["instruction"], examples["input"], examples["output"]
        )
    ]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")
    labels = model_inputs["input_ids"].copy()
    model_inputs["labels"] = labels
    return model_inputs

tokenized_dataset = dataset["train"].map(tokenize_fn, batched=True)

# Training args
training_args = TrainingArguments(
    output_dir="./lora-finetuned-journal",
    per_device_train_batch_size=1,  # GTX 1650 limitation
    gradient_accumulation_steps=16, # to simulate bigger batch
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
    fp16=True,
    optim="adamw_torch",
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Train!
trainer.train()

# Save LoRA adapter
model.save_pretrained("./lora-finetuned-journal")
