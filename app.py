import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request, jsonify
import requests
import logging
import time
import traceback
import gc
import os
import hashlib
import json
import shutil
import subprocess
import sys
import gdown
from transformers.modeling_outputs import BaseModelOutputWithPast

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('node1')

app = Flask(__name__)

# For storing the model verification hash
model_hash = None
model_info = {}
node1_latest_ra_data = None

total_layers_global = None
middle_layer_global = None

def get_ra_data(custom_data):
    """
    Call the Node script with custom data and return the RA report.
    """
    try:
        result = subprocess.run(
            ["node", "generate_ra.js", custom_data],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        ra_report = json.loads(result.stdout)
        return {"ra_report": ra_report, "custom_data_used": custom_data}
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # Fallback: create a simple mock RA report if Node.js script is not available
        logger.warning(f"Node.js RA script not available, using fallback: {e}")
        return {
            "ra_report": {
                "attestation_type": "mock",
                "timestamp": time.time(),
                "custom_data": custom_data,
                "node": "node1",
                "status": "fallback_mode"
            },
            "custom_data_used": custom_data
        }
    except json.JSONDecodeError as je:
        return {"error": "Invalid JSON returned from Node script", "details": str(je)}
    
def generate_model_hash(model):
    """Generate a SHA-256 hash of model parameters and architecture"""
    logger.info("Generating model verification hash...")
    
    # Get model architecture info
    model_arch = {
        "model_type": model.config.model_type,
        "hidden_size": model.config.hidden_size,
        "num_attention_heads": model.config.num_attention_heads,
        "num_hidden_layers": model.config.num_hidden_layers,
        "vocab_size": model.config.vocab_size
    }
    
    # Get a sample of model weights (first layer weights)
    param_sample = None
    for name, param in model.named_parameters():
        if 'layers.0' in name and param_sample is None:
            param_sample = param.data.flatten()[:1000].cpu().numpy().tolist()
            break
    
    # Combine architecture and weights sample
    hash_data = {
        "architecture": model_arch,
        "parameters_sample": param_sample,
        "model_name": "TinyLlama-1.1B-Chat-v1.0",
        "total_layers": len(model.model.layers),
        "node": "1"
    }
    
    # Convert to JSON string and hash with SHA-256
    hash_str = json.dumps(hash_data, sort_keys=True)
    hash_result = hashlib.sha256(hash_str.encode()).hexdigest()
    
    return hash_result, hash_data

def download_model_from_gdrive():
    """Download model files from Google Drive"""
    logger.info("Downloading model files from Google Drive")
    
    model_dir = "/app/models/tinyllama-1b"
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        import gdown
    except ImportError:
        logger.info("Installing gdown for Google Drive downloads...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
    
    success = True
    try:
        logger.info(f"Downloading entire model folder from Google Drive...")
        gdown.download_folder(
            id="1Iua1_n95NSgndooGFPfppaKTti5mmtEy",
            output=model_dir,
            quiet=False
        )
        logger.info("Model folder download complete")
    except Exception as e:
        logger.error(f"Failed to download model folder: {str(e)}")
        success = False
    
    return success

# Create a custom class to modify the forward pass for Node1
class Node1Model(torch.nn.Module):
    def __init__(self, base_model, middle_layer):
        super().__init__()
        self.config = base_model.config
        self.embed_tokens = base_model.model.embed_tokens
        self.norm = base_model.model.norm
        
        # Only include the first half of layers
        self.layers = base_model.model.layers[:middle_layer]
        
        logger.info(f"Node1 initialized with layers 0 to {middle_layer-1}")

    def forward(self, input_ids, attention_mask=None, output_hidden_states=True):
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Create position_ids for the sequence
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Store all hidden states
        all_hidden_states = () if output_hidden_states else None
        
        # Process through available layers
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            # Pass position_ids to the layer
            layer_output = layer(hidden_states, attention_mask=None, position_ids=position_ids)
            hidden_states = layer_output[0]  # Get the hidden states from the tuple
            
        # Return the output and the hidden_states
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=None,
            past_key_values=None
        )

# Initialize model
logger.info("Initializing Node1 (first half of model)...")
model_name = "/app/models/tinyllama-1b"  # Local path in container

# Try to download model files first
if not download_model_from_gdrive():
    logger.error("Failed to download model files. Trying to use local files if available.")

try:
    logger.info("Loading tokenizer from local directory...")
    # tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    logger.info("Tokenizer loaded successfully")
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Loading model from local directory...")
    # full_model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.float16,
    #     device_map="cpu",  # Initial CPU loading
    #     low_cpu_mem_usage=True,
    #     local_files_only=True
    # )
    full_model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float32,   # <- use fp32 on CPU
        device_map="cpu",
    )
    logger.info("Model loaded successfully")
    
    # Get the total layers and mid point
    total_layers = len(full_model.model.layers)
    middle_layer = total_layers // 2
    logger.info(f"Model has {total_layers} total layers")
    logger.info(f"Node1 will use layers 0 to {middle_layer-1}")
    
    # Create Node1 specific model with just the first half of layers
    model = Node1Model(full_model, middle_layer)
    
    # Generate and store model hash
    model_hash, model_info = generate_model_hash(full_model)
    logger.info(f"Model verification hash: {model_hash}")
    
    # Release memory from the full model
    del full_model
    gc.collect()
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")
        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        logger.info(f"Model moved to GPU. Memory used: {memory_allocated:.2f} GB")
    
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.error(traceback.format_exc())
    raise  # This will cause the container to exit on model load failure

# Store global values for later use
total_layers_global = total_layers
middle_layer_global = middle_layer

def generate_multi_token(prompt: str, max_new_tokens: int = 32, stop_token_ids=None):
    """
    Simple token-by-token loop:
      - Node1 recomputes its half on the growing sequence
      - Node2 predicts the next token
      - Repeat until max_new_tokens or EOS

    NOTE: This is the straightforward (recompute) version.
    """

    if stop_token_ids is None:
        stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    # format + tokenize once
    chat_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(chat_prompt, return_tensors="pt").to(device)

    generated_token_ids = []

    node2_url = os.environ.get('NODE2_URL', 'http://localhost:5001')
    generate_endpoint = f"{node2_url}/generate"

    for _ in range(max_new_tokens):
        attention_mask = None  # no padding; we pass None to per-layer forward

        # 1) run Node1 half to get hidden_states for entire (growing) sequence
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.last_hidden_state  # [B, S, H]
            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 2) send to Node2 to get next token (single step)
        node2_payload = {
            "input_ids": input_ids.detach().cpu().tolist(),
            "attention_mask": torch.ones_like(input_ids).detach().cpu().tolist(),  # shape [B,S], unused by Node2 layers
            "position_ids": torch.arange(input_ids.shape[1], device=device).unsqueeze(0).detach().cpu().tolist(),
            "hidden_states": hidden_states.detach().cpu().tolist(),
            "prompt": chat_prompt,
            "layer_info": {
                "total_layers": total_layers_global,
                "middle_layer": middle_layer_global
            }
        }

        resp = requests.post(generate_endpoint, json=node2_payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        next_id = int(data["next_token_id"])
        generated_token_ids.append(next_id)

        # 3) stop conditions
        if next_id in stop_token_ids:
            break

        # 4) append next token and continue
        next_id_tensor = torch.tensor([[next_id]], dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat([input_ids, next_id_tensor], dim=1)

    # Decode only the newly generated part
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return {
        "text": generated_text,
        "tokens": generated_token_ids,
        "num_new_tokens": len(generated_token_ids)
    }

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)
        prompt = data.get("prompt", "")
        max_new_tokens = int(data.get("max_new_tokens", 32))

        result = generate_multi_token(prompt, max_new_tokens=max_new_tokens)

        # You can also return attestation you already compute in /process if you want.
        return jsonify({
            "prompt": prompt,
            "generated_text": result["text"],
            "tokens": result["tokens"],
            "num_new_tokens": result["num_new_tokens"]
        })
    except Exception as e:
        logger.error(f"/chat error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/verify', methods=['GET'])
def verify_model():
    """Endpoint to verify the model's identity and integrity"""
    if model_hash:
        response = {
            "model_hash": model_hash,
            "model_info": {
                "total_layers": len(model.layers),
                "model_type": "TinyLlama-1.1B-Chat-v1.0 (Node1 - First Half)"
            }
        }
        return jsonify(response)
    else:
        return jsonify({"error": "Model hash not available", "status": "error"}), 500

@app.route('/process', methods=['POST'])
def process_prompt():
    """Process a prompt through the first half of the model"""
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        
        logger.info(f"Processing prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Processing prompt: {prompt}")
        start_time = time.time()
        
        # Format prompt with chat template
        chat_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
        
        # Tokenize the input
        input_ids = tokenizer.encode(chat_prompt, return_tensors="pt").to(model.layers[0].parameters().__next__().device)
        attention_mask = torch.ones_like(input_ids)
        logger.info(f"Input shape: {input_ids.shape}")
        
        with torch.no_grad():
            # Process through the first half of the model layers
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            # Get the hidden states from the last processed layer
            hidden_states = outputs.last_hidden_state
            all_hidden_states = outputs.hidden_states
            
            logger.info(f"Hidden states shape: {hidden_states.shape}")
            logger.info(f"Processed through {len(all_hidden_states)} layers")
            
            # Prepare key-value cache if needed for Node2
            past_key_values = None
            
            # Convert to list for JSON serialization
            hidden_states_list = hidden_states.cpu().numpy().tolist()
            input_ids_list = input_ids.cpu().numpy().tolist()
            
            # Clean up memory
            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Create serialized position IDs for continuation
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        position_ids_list = position_ids.cpu().numpy().tolist()
        
        # Prepare data for node2
        node2_data = {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask.cpu().numpy().tolist(),
            "position_ids": position_ids_list,
            "hidden_states": hidden_states_list,
            "prompt": chat_prompt,
            "layer_info": {
                "total_layers": total_layers_global,
                "middle_layer": middle_layer_global  # This is the next layer Node2 should start from
            }
        }
        
        logger.info("Sending processed data to node2...")
        node2_start = time.time()
        
        # Get Node2 URL from environment variable with fallback
        node2_url = os.environ.get('NODE2_URL', 'http://localhost:5001')
        generate_endpoint = f"{node2_url}/generate"
        logger.info(f"Using Node2 endpoint: {generate_endpoint}")
        
        # Generate RA data using the processed data as custom data
        logger.info("Generating remote attestation data for processed output...")
        ra_custom_data = f"node1_process:prompt={prompt[:50]}...,hidden_states_shape={len(hidden_states_list)}x{len(hidden_states_list[0]) if hidden_states_list else 0},time:{time.time()},layers:0-{len(model.layers)-1}"
        ra_data = get_ra_data(ra_custom_data)
        
        # Store the RA data in the global variable for the new endpoint to access
        global node1_latest_ra_data
        node1_latest_ra_data = ra_data
        
        # Send to node2 for completion
        response = requests.post(
            generate_endpoint,
            json=node2_data,
            timeout=300  # 5 minute timeout
        )
        
        node2_time = time.time() - node2_start
        logger.info(f"Node2 processing time: {node2_time:.2f}s")
        
        total_time = time.time() - start_time
        logger.info(f"Total request time: {total_time:.2f}s")
        
        # Get the response from node2
        node2_response = response.json()
        
        # Add the RA data to the response
        if "attestation" in node2_response:
            # If node2 already has attestation data, combine both
            node2_response["attestation"]["node1_attestation"] = ra_data
        else:
            # Otherwise, add our attestation data
            node2_response["attestation"] = {"node1_attestation": ra_data}
        
        return jsonify(node2_response)
        
    except Exception as e:
        logger.error(f"Error processing prompt: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"output": f"Error: {str(e)}"})
      
@app.route('/generate', methods=['POST'])
def generate():
    """Client-facing endpoint for the complete model"""
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        
        logger.info(f"Received user prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Received user prompt: {prompt}")
        start_time = time.time()
        
        # Process with the internal endpoint
        result = process_prompt()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Response generated in {elapsed_time:.2f}s")
        
        return result
    
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        return jsonify({"output": f"Error: {str(e)}"})

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model_type": "TinyLlama-1.1B-Chat-v1.0 (Node1)",
        "layers": f"0-{len(model.layers)-1}",
        "memory_usage": f"{torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB" if torch.cuda.is_available() else "CPU only"
    })

@app.route('/node1_ra_report', methods=['GET'])
def get_node1_ra_report():
    """Get the latest RA report from Node1"""
    if node1_latest_ra_data:
        return jsonify({
            "status": "success",
            "node1_attestation": node1_latest_ra_data
        })
    else:
        return jsonify({
            "status": "not_ready", 
            "message": "No RA report has been generated yet"
        })

if __name__ == "__main__":
    logger.info("Starting node1 server on port 5002...")
    app.run(host="0.0.0.0", port=5002) 