# main.py — run REAL inference by loading the stage shards from ./enhanced_shards
import os, sys, json, hashlib, torch, torch.nn as nn
from transformers import AutoConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm

ROOT = "./enhanced_shards"
MANIFEST = os.path.join(ROOT, "pipeline_manifest.json")

# ---------- devices & dtype ----------
if torch.cuda.is_available():
    device_part1 = "cuda:0"; device_part2 = "cuda:0"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
elif torch.backends.mps.is_available():
    device_part1 = "mps"; device_part2 = "mps"; dtype = torch.float16
else:
    device_part1 = "cpu"; device_part2 = "cpu"; dtype = torch.float32

def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()

def make_decoder_layer(cfg, idx: int):
    import inspect
    sig = inspect.signature(LlamaDecoderLayer.__init__)
    return LlamaDecoderLayer(cfg, layer_idx=idx) if "layer_idx" in sig.parameters else LlamaDecoderLayer(cfg)

def build_position_ids(attn_mask: torch.Tensor) -> torch.Tensor:
    cumsum = attn_mask.long().cumsum(-1)
    return (cumsum - 1).clamp(min=0)

def visualize(step, items):
    print("\n" + "="*70)
    print(f"🔹 {step}")
    print("="*70)
    for k, v in items.items():
        print(f"  • {k:<16} → {tuple(v.shape)}  [{str(v.dtype).replace('torch.','')} @ {v.device}]")
    print("-"*70)

def banner(split_index, n_layers):
    print("\nLLM Two-Stage Pipeline (from stage shards)")
    print("┌──────────┐      ┌──────────────────────────────┐      ┌──────────────────────────────┐")
    print(f"│  INPUT   ├──▶──▶│  stage_0 (0..{split_index-1})       ├──▶──▶│  stage_1 ({split_index}..{n_layers-1}+head) │")
    print("└──────────┘      └──────────────────────────────┘      └──────────────────────────────┘")
    print("         tokens ─────────────► hidden states ─────────────► logits / text\n")

# ---------- stage modules (MUST match split.py layout) ----------
class FirstStage(nn.Module):
    def __init__(self, cfg: AutoConfig, split_index: int):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([make_decoder_layer(cfg, i) for i in range(split_index)])

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids)
        B, S = input_ids.shape
        pos_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        for layer in self.layers:
            x = layer(
                hidden_states=x,
                attention_mask=None,   # rely on internal causal mask
                position_ids=pos_ids,
                past_key_value=None,
                use_cache=False,
                output_attentions=False
            )[0]
        return x

class SecondStage(nn.Module):
    def __init__(self, cfg: AutoConfig, split_index: int):
        super().__init__()
        self.layers = nn.ModuleList([make_decoder_layer(cfg, i) for i in range(split_index, cfg.num_hidden_layers)])
        self.norm = LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, hidden):
        S = hidden.size(1)
        pos_ids = torch.arange(S, device=hidden.device).unsqueeze(0)
        for layer in self.layers:
            hidden = layer(
                hidden_states=hidden,
                attention_mask=None,
                position_ids=pos_ids,
                past_key_value=None,
                use_cache=False,
                output_attentions=False
            )[0]
        hidden = self.norm(hidden)
        return self.lm_head(hidden)

# ---------- load manifest & verify ----------
if not os.path.isfile(MANIFEST):
    sys.exit("❌ Missing ./enhanced_shards/pipeline_manifest.json. Run: python split.py")

with open(MANIFEST, "r") as f:
    manifest = json.load(f)
stages = manifest["stages"]
stage0 = stages[0]; stage1 = stages[1]
SPLIT_INDEX = int(stage0["split_index"])
STAGE0 = os.path.join(ROOT, stage0["path"])
STAGE1 = os.path.join(ROOT, stage1["path"])

for stg, path in [(stage0, STAGE0), (stage1, STAGE1)]:
    w = os.path.join(path, stg["weights"])
    calc = file_sha256(w)
    if calc != stg["sha256"]:
        sys.exit(f"❌ Hash mismatch for {w}\n expected {stg['sha256']}\n      got {calc}")
print("✅ Shard hashes verified.")

# ---------- build halves & load weights ----------
cfg = AutoConfig.from_pretrained(STAGE0)
tok = AutoTokenizer.from_pretrained(STAGE0)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

first  = FirstStage(cfg, SPLIT_INDEX)
second = SecondStage(cfg, SPLIT_INDEX)

sd0 = torch.load(os.path.join(STAGE0, "pytorch_model.bin"), map_location="cpu")
sd1 = torch.load(os.path.join(STAGE1, "pytorch_model.bin"), map_location="cpu")
missing0, unexpected0 = first.load_state_dict(sd0, strict=True), ()
missing1, unexpected1 = second.load_state_dict(sd1, strict=True), ()
# move to devices
first  = first.to(device_part1, dtype=dtype)
second = second.to(device_part2, dtype=dtype)

# ---------- prompt ----------
try:
    question = input("🧠 Enter your question: ").strip()
except KeyboardInterrupt:
    sys.exit("\nInterrupted.")
if not question:
    question = "The meaning of life is"

banner(SPLIT_INDEX, cfg.num_hidden_layers)
enc = tok(question, return_tensors="pt")
input_ids = enc["input_ids"].to(device_part1)
attn_mask = enc["attention_mask"].to(device_part1)
visualize("Initial encoding", {"input_ids": input_ids, "attention_mask": attn_mask})

# ---------- generation loop (top-p + temperature, no KV cache) ----------
max_new, temperature, top_p = 64, 0.9, 0.95
with torch.no_grad():
    for step in range(max_new):
        hidden = first(input_ids, attn_mask)
        visualize(f"STEP {step+1}: After stage_0", {"hidden": hidden})

        hidden_p2 = hidden.to(device_part2)
        logits = second(hidden_p2)
        visualize(f"STEP {step+1}: After stage_1", {"logits": logits})

        last = logits[:, -1, :]
        last = last / temperature if temperature and temperature > 0 else last
        probs = torch.softmax(last, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        cutoff = (cum > top_p).float().argmax(dim=-1) + 1

        next_ids = []
        for b in range(probs.size(0)):
            k = int(cutoff[b].item())
            keep_idx = sorted_idx[b, :k]
            keep_probs = sorted_probs[b, :k] / sorted_probs[b, :k].sum()
            next_ids.append(keep_idx[torch.multinomial(keep_probs, 1)])
        next_id = torch.stack(next_ids)  # [bsz,1]

        input_ids = torch.cat([input_ids, next_id.to(device_part1)], dim=1)
        attn_mask = torch.ones_like(input_ids, device=device_part1)

        if tok.eos_token_id is not None and int(next_id[0]) == tok.eos_token_id:
            break

out_text = tok.decode(input_ids[0].cpu(), skip_special_tokens=True)
print("\n" + "="*70)
print("📝 Final output")
print("="*70)
print(out_text)
print("="*70)
