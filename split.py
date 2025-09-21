# split.py (refactor)

import os, json, hashlib, torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
import torch.nn as nn

MODEL_DIR = "./model"
OUT_ROOT  = "./enhanced_shards"
SPLIT_INDEX = 8  # layers [0..SPLIT_INDEX-1] and [SPLIT_INDEX..end]

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

class FirstStage(nn.Module):
    def __init__(self, cfg: AutoConfig, split_index: int):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([make_decoder_layer(cfg, i) for i in range(split_index)])
    def forward(self, input_ids):  # not used during splitting
        x = self.embed(input_ids); return x

class SecondStage(nn.Module):
    def __init__(self, cfg: AutoConfig, split_index: int):
        super().__init__()
        self.layers = nn.ModuleList([make_decoder_layer(cfg, i) for i in range(split_index, cfg.num_hidden_layers)])
        self.norm = LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    def forward(self, hidden):
        return self.lm_head(hidden)

def run_split(model_dir: str = MODEL_DIR, out_root: str = OUT_ROOT, split_index_cfg: int = SPLIT_INDEX):
    os.makedirs(out_root, exist_ok=True)
    STAGE0 = os.path.join(out_root, "stage_0")
    STAGE1 = os.path.join(out_root, "stage_1")
    os.makedirs(STAGE0, exist_ok=True)
    os.makedirs(STAGE1, exist_ok=True)

    print("Loading full model from", model_dir)
    cfg = AutoConfig.from_pretrained(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)
    full = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)

    llama = full.model
    embed = llama.embed_tokens
    layers = llama.layers
    norm   = llama.norm
    head   = full.lm_head

    split_index = max(1, min(split_index_cfg, cfg.num_hidden_layers - 1))
    first  = FirstStage(cfg, split_index)
    second = SecondStage(cfg, split_index)

    first.embed = embed
    first.layers = nn.ModuleList(list(layers[:split_index]))
    second.layers = nn.ModuleList(list(layers[split_index:]))
    second.norm = norm
    second.lm_head = head

    w0 = os.path.join(STAGE0, "pytorch_model.bin")
    w1 = os.path.join(STAGE1, "pytorch_model.bin")
    torch.save(first.state_dict(),  w0)
    torch.save(second.state_dict(), w1)

    cfg.save_pretrained(STAGE0); tok.save_pretrained(STAGE0)
    cfg.save_pretrained(STAGE1); tok.save_pretrained(STAGE1)

    meta0 = {
        "stage": 0, "layers": [0, split_index-1],
        "split_index": split_index,
        "vocab_size": cfg.vocab_size, "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "weights": "pytorch_model.bin", "sha256": file_sha256(w0)
    }
    meta1 = {
        "stage": 1, "layers": [split_index, cfg.num_hidden_layers-1],
        "split_index": split_index,
        "vocab_size": cfg.vocab_size, "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "weights": "pytorch_model.bin", "sha256": file_sha256(w1)
    }
    with open(os.path.join(STAGE0, "shard_metadata.json"), "w") as f: json.dump(meta0, f, indent=2)
    with open(os.path.join(STAGE1, "shard_metadata.json"), "w") as f: json.dump(meta1, f, indent=2)

    manifest = {
        "model_dir": model_dir,
        "stages": [
            {"path": "stage_0", **meta0},
            {"path": "stage_1", **meta1},
        ]
    }
    with open(os.path.join(out_root, "pipeline_manifest.json"), "w") as f: json.dump(manifest, f, indent=2)
    print("✅ Wrote real stage shards to", out_root)
    return {"out_root": out_root, "split_index": split_index}

if __name__ == "__main__":
    run_split()
