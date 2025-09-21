# pipeline_runtime.py — reusable loader + generate()
import os, sys, json, hashlib, torch, torch.nn as nn
from transformers import AutoConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm

ROOT = "./enhanced_shards"
MANIFEST = os.path.join(ROOT, "pipeline_manifest.json")

if torch.cuda.is_available():
    device_part1 = "cuda:0"; device_part2 = "cuda:0"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
elif torch.backends.mps.is_available():
    device_part1 = "mps"; device_part2 = "mps"; dtype = torch.float16
else:
    device_part1 = "cpu"; device_part2 = "cpu"; dtype = torch.float32

def file_sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()

def make_decoder_layer(cfg, idx: int):
    import inspect
    sig = inspect.signature(LlamaDecoderLayer.__init__)
    return LlamaDecoderLayer(cfg, layer_idx=idx) if "layer_idx" in sig.parameters else LlamaDecoderLayer(cfg)

def build_prompt(tok, prompt: str = None, messages: list = None) -> str:

    if messages:
        # ensures the assistant turn will start next
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt or ""

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
                attention_mask=None,
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

class TwoStagePipeline:
    def __init__(self, root: str = ROOT):
        if not os.path.isfile(MANIFEST):
            raise RuntimeError("Missing ./enhanced_shards/pipeline_manifest.json. Run split first.")
        with open(MANIFEST, "r") as f:
            manifest = json.load(f)
        stages = manifest["stages"]
        self.stage0 = stages[0]; self.stage1 = stages[1]
        self.SPLIT_INDEX = int(self.stage0["split_index"])
        self.STAGE0 = os.path.join(root, self.stage0["path"])
        self.STAGE1 = os.path.join(root, self.stage1["path"])

        # verify hashes
        for stg, path in [(self.stage0, self.STAGE0), (self.stage1, self.STAGE1)]:
            w = os.path.join(path, stg["weights"])
            calc = file_sha256(w)
            if calc != stg["sha256"]:
                raise RuntimeError(f"Hash mismatch for {w}\n expected {stg['sha256']}\n got {calc}")

        # build + load
        self.cfg = AutoConfig.from_pretrained(self.STAGE0)
        self.tok = AutoTokenizer.from_pretrained(self.STAGE0)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        self.first  = FirstStage(self.cfg, self.SPLIT_INDEX)
        self.second = SecondStage(self.cfg, self.SPLIT_INDEX)
        sd0 = torch.load(os.path.join(self.STAGE0, "pytorch_model.bin"), map_location="cpu")
        sd1 = torch.load(os.path.join(self.STAGE1, "pytorch_model.bin"), map_location="cpu")
        self.first.load_state_dict(sd0, strict=True)
        self.second.load_state_dict(sd1, strict=True)

        self.first  = self.first.to(device_part1, dtype=dtype).eval()
        self.second = self.second.to(device_part2, dtype=dtype).eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str = None,
        messages: list = None,
        max_new: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_only_new_text: bool = True,
        default_system_prompt: str = "You are a helpful, concise assistant.",
    ):
        # --- if user gave plain text, auto-wrap as chat for a chat-tuned model ---
        if messages is None and prompt:
            messages = [
                {"role": "system", "content": default_system_prompt},
                {"role": "user", "content": prompt},
            ]

        # build the actual textual input (chat template or plain)
        text_in = build_prompt(self.tok, prompt=prompt, messages=messages)

        enc = self.tok(text_in, return_tensors="pt")
        input_ids = enc["input_ids"].to(next(self.first.parameters()).device)
        attn_mask = enc["attention_mask"].to(next(self.first.parameters()).device)

        # remember how long the prompt was so we can slice later
        prompt_len = input_ids.size(1)

        for _ in range(max_new):
            hidden = self.first(input_ids, attn_mask)
            logits = self.second(hidden.to(next(self.second.parameters()).device))
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
            next_id = torch.stack(next_ids)

            # append token
            input_ids = torch.cat([input_ids, next_id.to(input_ids.device)], dim=1)
            attn_mask = torch.ones_like(input_ids, device=input_ids.device)

            # stop on eos, if present
            if self.tok.eos_token_id is not None and int(next_id[0]) == self.tok.eos_token_id:
                break

        # decode either the whole thing or only the newly generated suffix
        if return_only_new_text:
            new_tokens = input_ids[0, prompt_len:].cpu()
            return self.tok.decode(new_tokens, skip_special_tokens=True)

        return self.tok.decode(input_ids[0].cpu(), skip_special_tokens=True)