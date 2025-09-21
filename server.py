import os
from pathlib import Path
from flask import Flask, request, jsonify
from split import run_split
from inference import TwoStagePipeline

app = Flask(__name__)

# --- simple protection & "run-once" for splitting ---
RUN_TOKEN = os.environ.get("RUN_TOKEN", "")
FLAG_PATH = Path("split_once.flag")

def already_split() -> bool:
    return FLAG_PATH.exists()

def mark_split_done():
    FLAG_PATH.write_text("done")

# Lazy-initialized pipeline (load after split exists)
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = TwoStagePipeline(root="./enhanced_shards")
    return _pipeline

@app.post("/split")
def split_endpoint():
    if already_split():
        return jsonify({"status": "skipped", "message": "shards already exist"}), 200
    try:
        result = run_split()  # uses defaults (./model -> ./enhanced_shards)
        mark_split_done()
        return jsonify({"status": "success", "result": result}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.post("/infer")
def infer_endpoint():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")
    messages = data.get("messages")  # NEW: list of {role, content}
    max_new = int(data.get("max_new", 64))
    temperature = float(data.get("temperature", 0.7))   # cooler defaults
    top_p = float(data.get("top_p", 0.9))

    try:
        if not already_split():
            return jsonify({"error": "Shards not found. Run /split first."}), 400
        pipe = get_pipeline()
        text = pipe.generate(
            prompt=prompt,
            messages=messages,
            max_new=max_new,
            temperature=temperature,
            top_p=top_p,
        )
        return jsonify({"status": "success", "output": text}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    # Dev server; for prod use gunicorn (see below)
    app.run(host="127.0.0.1", port=5000, debug=True)
