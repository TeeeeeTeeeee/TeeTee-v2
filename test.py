# method_b_single.py — Method B (runtime split) as one offline script
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "./model"  # or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SPLIT = 8             # layer boundary
MAX_NEW = 64
TEMPERATURE = 0.9
TOP_P = 0.95

# ---- Node1 front half ----
class Node1Model(nn.Module):
    def __init__(self, base_model, split_idx):
        super().__init__()
        self.embed_tokens = base_model.model.embed_tokens
        self.layers = nn.ModuleList(list(base_model.model.layers[:split_idx]))  # <- wrap
    @torch.no_grad()
    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        B, S = input_ids.shape
        pos_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        for layer in self.layers:
            x = layer(x, attention_mask=None, position_ids=pos_ids)[0]
        return x  # last hidden

# ---- Node2 back half ----
class Node2BackHalf(nn.Module):
    def __init__(self, base_model, split_idx):
        super().__init__()
        self.layers = nn.ModuleList(list(base_model.model.layers[split_idx:]))  # <- wrap
        self.norm = base_model.model.norm
        self.lm_head = base_model.lm_head
    @torch.no_grad()
    def forward_from_hidden(self, hidden):
        S = hidden.size(1)
        pos_ids = torch.arange(S, device=hidden.device).unsqueeze(0)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=None, position_ids=pos_ids)[0]
        hidden = self.norm(hidden)
        return self.lm_head(hidden)  # logits

def sample_top_p(logits, temperature=1.0, top_p=0.95):
    if temperature and temperature > 0:
        logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    k = int((cum > top_p).float().argmax().item()) + 1
    keep_idx = sorted_idx[:k]
    keep_probs = sorted_probs[:k] / sorted_probs[:k].sum()
    next_id = keep_idx[torch.multinomial(keep_probs, 1)]
    return next_id.item()

def visualize(step, items):
    print("\n" + "="*70)
    print(f"🔹 {step}")
    print("="*70)
    for k, v in items.items():
        print(f"  • {k:<16} → {tuple(v.shape)}  [{str(v.dtype).replace('torch.','')} @ {v.device}]")
    print("-"*70)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
    tok  = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Build both halves from the same loaded model (no extra weights on disk)
    node1 = Node1Model(full, SPLIT).to(device)
    node2 = Node2BackHalf(full, SPLIT).to(device)
    del full  # keep memory lower; halves keep references to params

    prompt = input("🧠 Enter your question: ").strip() or "The meaning of life is"
    text = prompt
    enc = tok(text, return_tensors="pt").to(device)

    input_ids = enc["input_ids"]

    with torch.no_grad():
        for _ in range(MAX_NEW):
            hidden = node1(input_ids)
            logits = node2.forward_from_hidden(hidden)

            # visualize BEFORE sampling (no recompute)
            visualize("After front half", {"hidden": hidden})
            visualize("After back half", {"logits": logits})

            next_id = sample_top_p(logits[0, -1, :], TEMPERATURE, TOP_P)
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)

            if tok.eos_token_id is not None and next_id == tok.eos_token_id:
                break

    print("\n" + "="*60 + "\n" + tok.decode(input_ids[0], skip_special_tokens=True) + "\n" + "="*60)

if __name__ == "__main__":
    main()
