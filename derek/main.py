import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Path to your model files (using ~ for home)
MODEL_PATH = "./model"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,   # change to torch.float16 if GPU doesn’t support bf16
    device_map="auto"
)

# Test prompt
prompt = "The meaning of life is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))