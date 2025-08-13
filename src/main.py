from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    device_map="auto",
)

model = PeftModel.from_pretrained(model, "./lora-finetuned-journal")

inputs = tokenizer("Instruction: Apply style guide A to this XML\nInput: <xml>...</xml>\nOutput:", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
