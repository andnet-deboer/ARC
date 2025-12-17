#!/usr/bin/env python3
"""
SAM3 Server - Provides 2D segmentation as a REST API
"""

import os
import sys
import io
import base64
import json
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import traceback

# Force expected env BEFORE any SAM3 imports
os.environ.setdefault("CONDA_PREFIX", "/opt/conda")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

SAM3_ROOT = "/opt/sam3"
if SAM3_ROOT not in sys.path:
    sys.path.insert(0, SAM3_ROOT)

try:
    from sam3.build_sam import build_sam3
    from sam3.automatic_mask_generator import SAM3AutomaticMaskGenerator
except ImportError as e:
    print(f"Error importing SAM3: {e}")
    print("Make sure SAM3 is installed in /opt/sam3")
    sys.exit(1)

app = Flask(__name__)

# Global models (loaded once)
sam3_model = None
mask_generator = None

def load_model():
    global sam3_model, mask_generator
    if sam3_model is None:
        print("Loading SAM3 model...")
        try:
            # Build SAM3 model
            sam3_model = build_sam3(checkpoint=None, model_type="default")
            sam3_model.cuda()
            
            # Initialize automatic mask generator
            mask_generator = SAM3AutomaticMaskGenerator(sam3_model)
            print("✓ SAM3 model loaded successfully!")
            
        except Exception as e:
            print(f"✗ Failed to load SAM3: {e}")
            traceback.print_exc()
    
    return sam3_model, mask_generator

@app.route('/health', methods=['GET'])
def health():
    model, gen = load_model()
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None and gen is not None,
        'service': 'SAM3 2D Segmentation'
    })

@app.route('/segment', methods=['POST'])
def segment():
    """
    Segment image with automatic mask generation
    
    Expects JSON:
    {
        "image": base64_encoded_png
    }
    
    Returns JSON:
    {
        "success": true,
        "num_masks": 15,
        "masks": [
            {
                "id": 0,
                "segmentation": base64_encoded_mask,
                "bbox": [x, y, w, h],
                "area": 5000,
                "stability_score": 0.95
            },
            ...
        ]
    }
    """
    model, mask_gen = load_model()
    if model is None or mask_gen is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        
        # Decode image
        image_bytes = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        
        print(f"Received image: {image.size}")
        
        # Generate masks
        masks = mask_gen.generate(image_np)
        print(f"Generated {len(masks)} masks")
        
        result = {
            'success': True,
            'num_masks': len(masks),
            'image_size': [image.size[0], image.size[1]],
            'masks': []
        }
        
        # Process each mask
        for idx, mask_data in enumerate(masks):
            mask = mask_data['segmentation'].astype(np.uint8) * 255
            
            # Encode mask as base64
            mask_image = Image.fromarray(mask)
            mask_bytes = io.BytesIO()
            mask_image.save(mask_bytes, format='PNG')
            mask_b64 = base64.b64encode(mask_bytes.getvalue()).decode('utf-8')
            
            # Extract bbox
            bbox = mask_data.get('bbox', [0, 0, image.size[0], image.size[1]])
            
            # Calculate area
            area = int(np.sum(mask_data['segmentation']))
            
            # Get stability score
            stability = float(mask_data.get('stability_score', 0.0))
            
            result['masks'].append({
                'id': idx,
                'segmentation': mask_b64,
                'bbox': bbox,
                'area': area,
                'stability_score': stability
            })
        
        return jsonify(result)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/segment-with-prompt', methods=['POST'])
def segment_with_prompt():
    """
    Segment image with text prompt (uses automatic masks)
    
    Expects JSON:
    {
        "image": base64_encoded_png,
        "prompt": "coffee mug",
        "confidence_threshold": 0.80
    }
    
    Returns JSON:
    {
        "success": true,
        "prompt": "coffee mug",
        "num_masks": 15,
        "masks": [...]
    }
    """
    model, mask_gen = load_model()
    if model is None or mask_gen is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        prompt = data.get('prompt', 'object')
        confidence_threshold = float(data.get('confidence_threshold', 0.80))
        
        # Decode image
        image_bytes = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        
        print(f"Received image: {image.size}, prompt: '{prompt}'")
        
        # Generate masks
        masks = mask_gen.generate(image_np)
        
        # Filter by confidence
        filtered_masks = [m for m in masks if m.get('stability_score', 1.0) >= confidence_threshold]
        print(f"Generated {len(masks)} masks, filtered to {len(filtered_masks)} high-confidence masks")
        
        result = {
            'success': True,
            'prompt': prompt,
            'num_masks': len(filtered_masks),
            'image_size': [image.size[0], image.size[1]],
            'masks': []
        }
        
        # Process each mask
        for idx, mask_data in enumerate(filtered_masks):
            mask = mask_data['segmentation'].astype(np.uint8) * 255
            
            # Encode mask as base64
            mask_image = Image.fromarray(mask)
            mask_bytes = io.BytesIO()
            mask_image.save(mask_bytes, format='PNG')
            mask_b64 = base64.b64encode(mask_bytes.getvalue()).decode('utf-8')
            
            bbox = mask_data.get('bbox', [0, 0, image.size[0], image.size[1]])
            area = int(np.sum(mask_data['segmentation']))
            stability = float(mask_data.get('stability_score', 0.0))
            
            result['masks'].append({
                'id': idx,
                'segmentation': mask_b64,
                'bbox': bbox,
                'area': area,
                'stability_score': stability
            })
        
        return jsonify(result)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/info', methods=['GET'])
def info():
    """Get server information"""
    return jsonify({
        'service': 'SAM3 2D Segmentation Server',
        'version': '1.0',
        'endpoints': {
            '/health': 'GET - Check server status',
            '/info': 'GET - Get server information',
            '/segment': 'POST - Automatic mask generation',
            '/segment-with-prompt': 'POST - Segmentation with text prompt'
        },
        'model_info': {
            'name': 'SAM3 (Segment Anything Model 3)',
            'type': '2D Image Segmentation',
            'device': 'CUDA'
        }
    })

if __name__ == '__main__':
    # Pre-load model
    load_model()
    
    # Run server
    print("Starting SAM3 Server on 0.0.0.0:8001...")
    app.run(host='0.0.0.0', port=8001, threaded=True)
