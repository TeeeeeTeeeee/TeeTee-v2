# main.py — split Llama runner using ONLY ./first and ./second (transformers==4.49.0 OK)

import os, sys, inspect
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

# Optional (present in some versions)
try:
    from transformers.models.llama.modeling_llama import (
        LlamaLinearScalingRotaryEmbedding,
        LlamaDynamicNTKScalingRotaryEmbedding,
    )
except Exception:
    LlamaLinearScalingRotaryEmbedding = None
    LlamaDynamicNTKScalingRotaryEmbedding = None

FIRST_PATH  = "./first"
SECOND_PATH = "./second"
SPLIT_INDEX = 8  # layers 0..7 and 8..15

# ---------- devices & dtype ----------
if torch.cuda.is_available():
    device_part1 = "cuda:0"; device_part2 = "cuda:0"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
elif torch.backends.mps.is_available():
    device_part1 = "mps"; device_part2 = "mps"; dtype = torch.float16
else:
    device_part1 = "cpu"; device_part2 = "cpu"; dtype = torch.float32

# ---------- helpers ----------
def build_position_ids(attn_mask: torch.Tensor) -> torch.Tensor:
    cumsum = attn_mask.long().cumsum(-1)
    return (cumsum - 1).clamp(min=0)

def causal_attention_mask(seq_len: int, dtype: torch.dtype, device: torch.device):
    # Use a smaller value that works with float16
    if dtype == torch.float16:
        fill_value = -1e4
    elif dtype == torch.bfloat16:
        fill_value = -1e4
    else:
        fill_value = -1e9
    mask = torch.full((1, 1, seq_len, seq_len), fill_value=fill_value, dtype=dtype, device=device)
    return torch.triu(mask, diagonal=1)

def get_rope(rotary_emb, x, position_ids=None, seq_len=None):
    # Support both call signatures across HF versions
    try:
        return rotary_emb(x, seq_len=seq_len)
    except TypeError:
        return rotary_emb(x, position_ids=position_ids)

def make_decoder_layer(cfg, idx: int):
    sig = inspect.signature(LlamaDecoderLayer.__init__)
    if "layer_idx" in sig.parameters:
        return LlamaDecoderLayer(cfg, layer_idx=idx)
    return LlamaDecoderLayer(cfg)

def make_rope_module(cfg, head_dim: int):
    """
    Robust RoPE factory for transformers 4.49.0 (and neighbors).
    Try known class variants & constructor signatures without keyword args.
    """
    # Handle both dict-like and object-like configs
    if hasattr(cfg, 'rope_theta'):
        base = float(cfg.rope_theta)
    elif hasattr(cfg, '__getitem__') and 'rope_theta' in cfg:
        base = float(cfg['rope_theta'])
    else:
        base = 10000.0
    
    if hasattr(cfg, 'max_position_embeddings'):
        max_pos = int(cfg.max_position_embeddings)
    elif hasattr(cfg, '__getitem__') and 'max_position_embeddings' in cfg:
        max_pos = int(cfg['max_position_embeddings'])
    else:
        max_pos = 2048
    
    if hasattr(cfg, 'rope_scaling'):
        rope_scaling = cfg.rope_scaling
    elif hasattr(cfg, '__getitem__') and 'rope_scaling' in cfg:
        rope_scaling = cfg['rope_scaling']
    else:
        rope_scaling = None

    # Prefer scaled variants if available and requested
    if rope_scaling:
        # Handle both dict-like and object-like rope_scaling
        if hasattr(rope_scaling, 'get'):
            rtype = str(rope_scaling.get("type", "")).lower()
            factor = float(rope_scaling.get("factor", 1.0))
        elif hasattr(rope_scaling, '__getitem__'):
            rtype = str(rope_scaling.get("type", "")).lower()
            factor = float(rope_scaling.get("factor", 1.0))
        else:
            rtype = str(getattr(rope_scaling, "type", "")).lower()
            factor = float(getattr(rope_scaling, "factor", 1.0))
            
        if LlamaLinearScalingRotaryEmbedding and ("linear" in rtype):
            # most versions: (dim, base, scaling_factor)
            try:
                return LlamaLinearScalingRotaryEmbedding(head_dim, base, factor)
            except TypeError:
                return LlamaLinearScalingRotaryEmbedding(head_dim, factor)  # fallback
        if LlamaDynamicNTKScalingRotaryEmbedding and ("ntk" in rtype or "dynamic" in rtype):
            try:
                return LlamaDynamicNTKScalingRotaryEmbedding(head_dim, base, factor)
            except TypeError:
                return LlamaDynamicNTKScalingRotaryEmbedding(head_dim, factor)

    # Plain RoPE — try multiple positional signatures
    # For transformers 4.49.0, LlamaRotaryEmbedding expects (config, head_dim) or just (config)
    try:
        # Try passing the config object directly
        return LlamaRotaryEmbedding(cfg)
    except TypeError:
        try:
            # Try with head_dim as second parameter
            return LlamaRotaryEmbedding(cfg, head_dim)
        except TypeError:
            # Last resort - create a minimal config-like object
            class MinimalConfig:
                def __init__(self, max_pos, base):
                    self.max_position_embeddings = max_pos
                    self.rope_theta = base
            minimal_cfg = MinimalConfig(max_pos, base)
            return LlamaRotaryEmbedding(minimal_cfg)

def visualize(step, items):
    print("\n" + "="*70)
    print(f"🔹 {step}")
    print("="*70)
    for k, v in items.items():
        print(f"  • {k:<16} → {tuple(v.shape)}  [{str(v.dtype).replace('torch.','')} @ {v.device}]")
    print("-"*70)

def banner():
    print("\nLLM Two-Stage Pipeline (directories only)")
    print("┌──────────┐      ┌──────────────────────────────┐      ┌──────────────────────────────┐")
    print("│  INPUT   ├──▶──▶│  ./first  (layers 1..8)     ├──▶──▶│  ./second (layers 9..16+head)│")
    print("└──────────┘      └──────────────────────────────┘      └──────────────────────────────┘")
    print("         tokens ─────────────► hidden states ─────────────► logits / text\n")

# ---------- minimal half modules ----------
class FirstHalf(nn.Module):
    """Embedding + first SPLIT_INDEX decoder layers."""
    def __init__(self, cfg: AutoConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)  # Changed from embed_tokens to embed
        self.layers = nn.ModuleList([make_decoder_layer(cfg, i) for i in range(SPLIT_INDEX)])
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.rotary_emb = make_rope_module(cfg, head_dim)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids)  # Changed from embed_tokens to embed
        seq_len = x.size(1)
        pos_ids = build_position_ids(attention_mask)
        attn_mask = causal_attention_mask(seq_len, x.dtype, x.device)
        pos_emb = get_rope(self.rotary_emb, x, position_ids=pos_ids, seq_len=seq_len)
        for layer in self.layers:
            x = layer(
                hidden_states=x,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                past_key_value=None,
                use_cache=False,
                output_attentions=False,
                position_embeddings=pos_emb
            )[0]
        return x

class SecondHalf(nn.Module):
    """Remaining decoder layers + final norm + lm_head."""
    def __init__(self, cfg: AutoConfig):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([make_decoder_layer(cfg, i) for i in range(SPLIT_INDEX, cfg.num_hidden_layers)])
        self.norm = LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.rotary_emb = make_rope_module(cfg, head_dim)

    def forward(self, hidden, attention_mask):
        seq_len = hidden.size(1)
        pos_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0)  # [1, seq]
        attn_mask = causal_attention_mask(seq_len, hidden.dtype, hidden.device)
        pos_emb = get_rope(self.rotary_emb, hidden, position_ids=pos_ids, seq_len=seq_len)
        for layer in self.layers:
            hidden = layer(
                hidden_states=hidden,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                past_key_value=None,
                use_cache=False,
                output_attentions=False,
                position_embeddings=pos_emb
            )[0]
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return logits

# ---------- main ----------
if __name__ == "__main__":
    # sanity checks
    for p in (FIRST_PATH, SECOND_PATH):
        if not os.path.isdir(p): sys.exit(f"❌ Missing folder: {p}")
        if not os.path.isfile(os.path.join(p, "pytorch_model.bin")):
            sys.exit(f"❌ Missing weights: {p}/pytorch_model.bin")
    if not os.path.isfile(os.path.join(FIRST_PATH, "config.json")):
        sys.exit("❌ Missing config.json in ./first (needed to rebuild architecture).")
    if not (os.path.isfile(os.path.join(FIRST_PATH, "tokenizer.json")) or os.path.isfile(os.path.join(FIRST_PATH, "tokenizer.model"))):
        sys.exit("❌ Missing tokenizer files in ./first.")

    # load config/tokenizer from ./first
    cfg = AutoConfig.from_pretrained(FIRST_PATH)
    tok = AutoTokenizer.from_pretrained(FIRST_PATH)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # build halves & load weights
    first  = FirstHalf(cfg).to(device_part1, dtype=dtype)
    second = SecondHalf(cfg).to(device_part2, dtype=dtype)
    first.load_state_dict(torch.load(os.path.join(FIRST_PATH,  "pytorch_model.bin"), map_location=device_part1), strict=True)
    second.load_state_dict(torch.load(os.path.join(SECOND_PATH, "pytorch_model.bin"), map_location=device_part2), strict=True)

    # input question
    try:
        question = input("🧠 Enter your question: ").strip()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
    if not question: question = "The meaning of life is"

    # banner + encode
    banner()
    enc = tok(question, return_tensors="pt")
    input_ids = enc["input_ids"].to(device_part1)
    attn_mask = enc["attention_mask"].to(device_part1)
    def viz(step, d): visualize(step, d)
    viz("Initial encoding", {"input_ids": input_ids, "attention_mask": attn_mask})

    # simple top-p generation (no KV cache for clarity)
    max_new, temperature, top_p = 64, 0.9, 0.95
    with torch.no_grad():
        for step in range(max_new):
            hidden = first(input_ids, attn_mask)
            viz(f"STEP {step+1}: After ./first (layers 1..8)", {"hidden": hidden})

            hidden_p2 = hidden.to(device_part2)
            attn_p2 = torch.ones(hidden_p2.shape[:2], dtype=torch.long, device=device_part2)
            logits = second(hidden_p2, attn_p2)
            viz(f"STEP {step+1}: After ./second (layers 9..16 + head)", {"logits": logits})

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
            next_id = torch.stack(next_ids).unsqueeze(-1)  # [bsz,1]
            # Ensure next_id has the right shape [bsz, 1]
            if next_id.dim() == 3:
                next_id = next_id.squeeze(-1)  # Remove extra dimension if present

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