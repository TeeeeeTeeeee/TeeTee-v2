import os
import gc
import time
import json
import logging
import traceback
from typing import Optional, Dict, Any, Tuple

import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

# ----------------------
# Logging
# ----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - node2 - %(levelname)s - %(message)s")
logger = logging.getLogger("node2")

app = Flask(__name__)

# ----------------------
# Config
# ----------------------
MODEL_ID = os.environ.get("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DTYPE = os.environ.get("DTYPE", "fp16")  # "fp16" or "fp32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------
# Load tokenizer + full model (we’ll slice layers)
# ----------------------
logger.info(f"Loading tokenizer: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

logger.info(f"Loading model: {MODEL_ID}")
torch_dtype = torch.float16 if (DTYPE == "fp16" and DEVICE == "cuda") else torch.float32
full_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch_dtype,
    device_map="cpu"  # load on CPU first; we’ll move the needed parts
)

# ----------------------
# A thin wrapper that keeps only the back half of layers
# ----------------------
class Node2BackHalf(torch.nn.Module):
    def __init__(self, base_model, start_layer: int):
        super().__init__()
        self.config = base_model.config
        # we reuse the final norm and lm_head from the base model
        self.norm = base_model.model.norm
        self.lm_head = base_model.lm_head

        # keep layers [start_layer : end)
        self.layers = torch.nn.ModuleList(list(base_model.model.layers)[start_layer:])

        logger.info(f"Node2 initialized with layers {start_layer} to {start_layer + len(self.layers) - 1}")

    @torch.no_grad()
    def forward_from_hidden(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:

        for layer in self.layers:
            # LLaMA blocks accept (hidden_states, attention_mask) and return a tuple
            hidden_states = layer(hidden_states, attention_mask=None)[0]

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)  # [batch, seq_len, vocab_size]
        return CausalLMOutputWithPast(logits=logits)

# We’ll lazily construct Node2BackHalf once we know the split the client (Node1) used.
node2_model: Optional[Node2BackHalf] = None
middle_layer_cached: Optional[int] = None

def ensure_backhalf(middle_layer: int):
    """Instantiate/memoize the back-half model for a given split."""
    global node2_model, middle_layer_cached
    if node2_model is None or middle_layer_cached != middle_layer:
        logger.info(f"Building Node2 back-half from layer {middle_layer} …")
        node2_model = Node2BackHalf(full_model, middle_layer)
        # free the front-half memory by deleting references (optional)
        # not strictly necessary, but keeps memory lower
        gc.collect()

        # move to device
        node2_model.to(DEVICE)
        middle_layer_cached = middle_layer

# ----------------------
# Attestation helpers (very lightweight placeholder)
# ----------------------
import hashlib

def model_fingerprint(model) -> Tuple[str, Dict[str, Any]]:
    arch = {
        "model_type": model.config.model_type,
        "hidden_size": model.config.hidden_size,
        "num_attention_heads": model.config.num_attention_heads,
        "num_hidden_layers": model.config.num_hidden_layers,
        "vocab_size": model.config.vocab_size,
    }
    with torch.no_grad():
        sample = model.lm_head.weight.flatten()[:1000].detach().cpu().numpy().tolist()
    blob = json.dumps({"arch": arch, "lm_head_sample": sample, "node": "2"}, sort_keys=True)
    h = hashlib.sha256(blob.encode()).hexdigest()
    return h, {"arch": arch}

MODEL_HASH, MODEL_INFO = model_fingerprint(full_model)

# ----------------------
# Routes
# ----------------------
@app.route("/health", methods=["GET"])
def health():
    mem = f"{torch.cuda.memory_allocated() / (1024**3):.2f} GB" if DEVICE == "cuda" else "CPU only"
    return jsonify({
        "status": "ok",
        "device": DEVICE,
        "dtype": str(torch_dtype),
        "memory_usage": mem,
        "model_id": MODEL_ID,
        "note": "Node2 back half server (single-step next-token)."
    })

@app.route("/verify", methods=["GET"])
def verify():
    return jsonify({
        "model_hash": MODEL_HASH,
        "model_info": MODEL_INFO
    })

@app.route("/generate", methods=["POST"])
@torch.no_grad()
def generate():
    """
    Expected JSON from Node1:
    {
      "input_ids": [[...]],
      "attention_mask": [[...]],
      "position_ids": [[...]],          # optional (not used here)
      "hidden_states": [[[...]]],       # last hidden states from Node1 end-layer
      "prompt": "<chat formatted prompt>",
      "layer_info": { "total_layers": <int>, "middle_layer": <int> }
    }

    Returns:
      text: decoded next token (single-step)
      next_token_id: int
      usage: input_len, prompt_tokens
      shapes: sanity info
      attestation: simple proof + echo if Node1 provided any
    """
    t0 = time.time()
    try:
        js = request.get_json(force=True)

        # Parse pieces
        input_ids = js.get("input_ids")
        attention_mask = js.get("attention_mask")
        hidden_states = js.get("hidden_states")
        layer_info = js.get("layer_info", {})
        node1_attestation = js.get("attestation")  # if Node1 forwarded something (optional)

        if hidden_states is None or input_ids is None or attention_mask is None:
            return jsonify({"error": "Missing required fields: hidden_states, input_ids, attention_mask"}), 400

        middle_layer = int(layer_info.get("middle_layer", 0))
        ensure_backhalf(middle_layer)

        # Convert lists → tensors on the right device/dtype
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=DEVICE)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=DEVICE)

        # hidden_states comes as [batch, seq_len, hidden_size]
        hs = torch.tensor(hidden_states, dtype=torch_dtype, device=DEVICE)

        # Continue forward pass from split point to get logits over vocab
        out = node2_model.forward_from_hidden(hs, attention_mask=attention_mask)
        logits = out.logits  # [B, S, V]
        last_logits = logits[:, -1, :]  # final position

        # Greedy next-token (you can add temperature/top-p if you like)
        next_token_id = torch.argmax(last_logits, dim=-1)  # [B]
        next_token_id_int = int(next_token_id[0].item())

        # Build text: decode only the new token (or decode prompt+new if you prefer)
        next_text = tokenizer.decode([next_token_id_int], skip_special_tokens=True)

        # Compose response
        attestation = {
            "node2_attestation": {
                "model_hash": MODEL_HASH,
                "timestamp": time.time(),
                "note": "Node2 produced next-token from provided split hidden state."
            }
        }
        # if Node1 already attached its own, pass it through (Node1 usually merges for you)
        if node1_attestation:
            attestation["node1_attestation"] = node1_attestation

        resp = {
            "text": next_text,
            "next_token_id": next_token_id_int,
            "usage": {
                "prompt_tokens": int(input_ids.shape[1]),
                "batch_size": int(input_ids.shape[0])
            },
            "shapes": {
                "input_ids": list(input_ids.shape),
                "attention_mask": list(attention_mask.shape),
                "hidden_states": list(hs.shape),
                "backhalf_layers": [middle_layer, middle_layer + len(node2_model.layers) - 1]
            },
            "timings_sec": {
                "node2_forward": round(time.time() - t0, 4)
            },
            "attestation": attestation
        }
        return jsonify(resp)

    except Exception as e:
        logger.error(f"Error in /generate: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    logger.info(f"Starting Node2 on port {port} (device={DEVICE}, dtype={torch_dtype}) …")
    app.run(host="0.0.0.0", port=port)
