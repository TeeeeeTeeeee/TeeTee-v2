# split_llama.py
import torch
import os
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

MODEL_PATH = "./model"
FIRST_PATH = "./first"
SECOND_PATH = "./second"
SPLIT_INDEX = 8

# Load the full model once
config = AutoConfig.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)

# Grab tokenizer too (we'll copy to both halves)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# --- Split ---
llama = model.model
embed = llama.embed_tokens
layers = llama.layers
norm = llama.norm
lm_head = model.lm_head

# First half: embeddings + first 8 layers
class FirstHalf(torch.nn.Module):
    def __init__(self, embed, layers, split_index):
        super().__init__()
        self.embed = embed
        self.layers = torch.nn.ModuleList(layers[:split_index])

    def forward(self, input_ids, attention_mask, position_ids, position_embeddings):
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(
                hidden_states=hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                use_cache=False,
                output_attentions=False,
                position_embeddings=position_embeddings,
            )[0]
        return hidden

# Second half: last 8 layers + norm + lm_head
class SecondHalf(torch.nn.Module):
    def __init__(self, layers, norm, lm_head, split_index):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers[split_index:])
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, hidden, attention_mask, position_ids, position_embeddings):
        for layer in self.layers:
            hidden = layer(
                hidden_states=hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                use_cache=False,
                output_attentions=False,
                position_embeddings=position_embeddings,
            )[0]
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return logits

# Instantiate
first_half = FirstHalf(embed, layers, SPLIT_INDEX)
second_half = SecondHalf(layers, norm, lm_head, SPLIT_INDEX)

# Save them in Hugging Face format
os.makedirs(FIRST_PATH, exist_ok=True)
os.makedirs(SECOND_PATH, exist_ok=True)

# Save config + tokenizer to both
config.save_pretrained(FIRST_PATH)
tokenizer.save_pretrained(FIRST_PATH)
config.save_pretrained(SECOND_PATH)
tokenizer.save_pretrained(SECOND_PATH)

# Save model weights
torch.save(first_half.state_dict(), os.path.join(FIRST_PATH, "pytorch_model.bin"))
torch.save(second_half.state_dict(), os.path.join(SECOND_PATH, "pytorch_model.bin"))

print("Saved first half to", FIRST_PATH)
print("Saved second half to", SECOND_PATH)