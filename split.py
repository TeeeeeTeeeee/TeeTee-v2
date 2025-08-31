#!/usr/bin/env python3
"""
Create mock shards for testing the distributed runtime
This bypasses the torchvision compatibility issue
"""

import os
import json
import time
import hashlib
from pathlib import Path

def create_mock_shard(shard_name: str, stage_id: int, layer_start: int, layer_end: int):
    """Create a mock shard directory with metadata"""
    
    # Create shard directory
    shard_dir = Path("enhanced_shards") / shard_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock model file
    model_file = shard_dir / "pytorch_model.bin"
    with open(model_file, 'wb') as f:
        f.write(b"mock_model_data_" + shard_name.encode() + b"_" + str(stage_id).encode())
    
    # Compute file hash
    with open(model_file, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    # Create metadata
    metadata = {
        "shard_info": {
            "name": shard_name,
            "type": f"stage_{stage_id}",
            "stage_id": stage_id,
            "layer_start": layer_start,
            "layer_end": layer_end,
            "layer_count": layer_end - layer_start
        },
        "model_info": {
            "source_model": "./model",
            "architecture": {
                "model_type": "llama",
                "hidden_size": 2048,
                "num_attention_heads": 32,
                "num_hidden_layers": 22,
                "vocab_size": 32000,
                "total_layers": 22
            }
        },
        "file_info": {
            "model_file": "pytorch_model.bin",
            "file_size": model_file.stat().st_size,
            "file_hash": file_hash
        },
        "verification": {
            "created_at": time.time(),
            "created_by": "create_mock_shards.py",
            "version": "1.0"
        }
    }
    
    # Generate comprehensive hash
    metadata_str = json.dumps(metadata, sort_keys=True)
    comprehensive_hash = hashlib.sha256(metadata_str.encode()).hexdigest()
    metadata["verification"]["comprehensive_hash"] = comprehensive_hash
    
    # Save metadata
    with open(shard_dir / "shard_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Copy config files from model directory if they exist
    model_dir = Path("model")
    if model_dir.exists():
        for file_name in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
            src_file = model_dir / file_name
            if src_file.exists():
                dst_file = shard_dir / file_name
                with open(src_file, 'rb') as src, open(dst_file, 'wb') as dst:
                    dst.write(src.read())
    
    print(f"✓ Created mock shard: {shard_dir}")
    return metadata

def create_pipeline_manifest(shards_metadata):
    """Create pipeline manifest"""
    
    manifest = {
        "pipeline_info": {
            "name": "llama_distributed_pipeline",
            "source_model": "./model",
            "total_stages": len(shards_metadata),
            "total_layers": 22,
            "created_at": time.time()
        },
        "stages": shards_metadata,
        "verification": {
            "pipeline_hash": hashlib.sha256(
                json.dumps(shards_metadata, sort_keys=True).encode()
            ).hexdigest()
        }
    }
    
    # Save manifest
    manifest_file = Path("enhanced_shards") / "pipeline_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Created pipeline manifest: {manifest_file}")
    return manifest

def main():
    """Create mock shards for testing"""
    
    print("Creating mock shards for testing...")
    
    # Create two-stage split (similar to original split.py)
    shards_metadata = []
    
    # First shard: layers 0-10
    metadata1 = create_mock_shard("first", 0, 0, 11)
    shards_metadata.append(metadata1)
    
    # Second shard: layers 11-21
    metadata2 = create_mock_shard("second", 1, 11, 22)
    shards_metadata.append(metadata2)
    
    # Create manifest
    manifest = create_pipeline_manifest(shards_metadata)
    
    print("\n" + "="*50)
    print("MOCK SHARDS CREATED SUCCESSFULLY")
    print("="*50)
    print(f"Created {len(shards_metadata)} shards:")
    for metadata in shards_metadata:
        shard_info = metadata["shard_info"]
        print(f"  - {shard_info['name']}: layers {shard_info['layer_start']}-{shard_info['layer_end']-1}")
    print(f"Pipeline hash: {manifest['verification']['pipeline_hash'][:16]}...")
    print("="*50)

if __name__ == "__main__":
    main()
