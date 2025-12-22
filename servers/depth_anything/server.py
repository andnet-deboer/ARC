#!/usr/bin/env python3
"""
Depth Anything V3 Server - Provides monocular depth estimation as a REST API
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
import cv2

# Force expected env BEFORE any depth anything imports
os.environ.setdefault("CONDA_PREFIX", "/opt/conda")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

try:
    from depth_anything_3.api import DepthAnything3

except ImportError as e:
    print(f"Error importing Depth Anything V3: {e}")
    print("Make sure Depth Anything V3 is installed in /opt/depth_anything_v3")
    sys.exit(1)

app = Flask(__name__)

# Global model (loaded once)
depth_model = None

def load_model():
    global depth_model
    if depth_model is None:
        print("Loading Depth Anything V3 model...")
        try:
            # Recommended nested model
            depth_model = DepthAnything3.from_pretrained(
                "depth-anything/DA3NESTED-GIANT-LARGE"
            )
            depth_model = depth_model.cuda()
            depth_model.eval()
            print("✓ Depth Anything V3 model loaded successfully!")
        except Exception as e:
            print(f"✗ Failed to load Depth Anything V3: {e}")
            traceback.print_exc()
    return depth_model

@app.route('/health', methods=['GET'])
def health():
    model = load_model()
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'service': 'Depth Anything V3 Depth Estimation'
    })

@app.route('/estimate-depth', methods=['POST'])
def estimate_depth():
    """
    Estimate depth map from single image
    
    Expects JSON:
    {
        "image": base64_encoded_png
    }
    
    Returns JSON:
    {
        "success": true,
        "depth_map": base64_encoded_depth_map,
        "depth_colored": base64_encoded_colorized_depth,
        "depth_min": 0.1,
        "depth_max": 250.0,
        "image_size": [width, height]
    }
    """
    model = load_model()
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        
        # Decode image
        image_bytes = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        
        print(f"Received image: {image.size}")
        
        # Estimate depth
        with np.no_grad():
            depth = model.infer_pil(image)
        
        depth_np = np.array(depth)
        
        print(f"Depth map shape: {depth_np.shape}, range: [{depth_np.min():.2f}, {depth_np.max():.2f}]")
        
        # Normalize depth for visualization (0-255)
        depth_min = float(depth_np.min())
        depth_max = float(depth_np.max())
        
        # Handle edge case where depth is constant
        if depth_max == depth_min:
            depth_normalized = np.zeros_like(depth_np, dtype=np.uint8)
        else:
            depth_normalized = ((depth_np - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        
        # Create result
        result = {
            'success': True,
            'image_size': [image.size[0], image.size[1]],
            'depth_min': depth_min,
            'depth_max': depth_max
        }
        
        # Encode raw depth map (grayscale)
        depth_image = Image.fromarray(depth_normalized)
        depth_bytes = io.BytesIO()
        depth_image.save(depth_bytes, format='PNG')
        result['depth_map'] = base64.b64encode(depth_bytes.getvalue()).decode('utf-8')
        
        # Create colorized depth visualization
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
        depth_colored_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        depth_colored_image = Image.fromarray(depth_colored_rgb)
        depth_colored_bytes = io.BytesIO()
        depth_colored_image.save(depth_colored_bytes, format='PNG')
        result['depth_colored'] = base64.b64encode(depth_colored_bytes.getvalue()).decode('utf-8')
        
        return jsonify(result)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/depth-with-confidence', methods=['POST'])
def depth_with_confidence():
    """
    Estimate depth with confidence metrics
    
    Expects JSON:
    {
        "image": base64_encoded_png
    }
    
    Returns JSON:
    {
        "success": true,
        "depth_map": base64_encoded_depth_map,
        "depth_colored": base64_encoded_colorized_depth,
        "uncertainty_map": base64_encoded_uncertainty,
        "depth_statistics": {
            "mean": 50.0,
            "median": 45.0,
            "std": 30.0,
            "min": 0.1,
            "max": 250.0
        },
        "image_size": [width, height]
    }
    """
    model = load_model()
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        
        # Decode image
        image_bytes = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        
        print(f"Received image: {image.size}")
        
        # Estimate depth
        with np.no_grad():
            depth = model.infer_pil(image)
        
        depth_np = np.array(depth)
        
        print(f"Depth map shape: {depth_np.shape}, range: [{depth_np.min():.2f}, {depth_np.max():.2f}]")
        
        # Calculate statistics
        depth_min = float(np.min(depth_np))
        depth_max = float(np.max(depth_np))
        depth_mean = float(np.mean(depth_np))
        depth_median = float(np.median(depth_np))
        depth_std = float(np.std(depth_np))
        
        # Normalize depth for visualization
        if depth_max == depth_min:
            depth_normalized = np.zeros_like(depth_np, dtype=np.uint8)
        else:
            depth_normalized = ((depth_np - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        
        # Estimate uncertainty as gradient magnitude (areas of rapid depth change are less confident)
        uncertainty_x = cv2.Sobel(depth_normalized, cv2.CV_32F, 1, 0, ksize=3)
        uncertainty_y = cv2.Sobel(depth_normalized, cv2.CV_32F, 0, 1, ksize=3)
        uncertainty = np.sqrt(uncertainty_x**2 + uncertainty_y**2)
        uncertainty_normalized = cv2.normalize(uncertainty, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Create result
        result = {
            'success': True,
            'image_size': [image.size[0], image.size[1]],
            'depth_statistics': {
                'mean': depth_mean,
                'median': depth_median,
                'std': depth_std,
                'min': depth_min,
                'max': depth_max
            }
        }
        
        # Encode raw depth map (grayscale)
        depth_image = Image.fromarray(depth_normalized)
        depth_bytes = io.BytesIO()
        depth_image.save(depth_bytes, format='PNG')
        result['depth_map'] = base64.b64encode(depth_bytes.getvalue()).decode('utf-8')
        
        # Create colorized depth visualization
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
        depth_colored_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        depth_colored_image = Image.fromarray(depth_colored_rgb)
        depth_colored_bytes = io.BytesIO()
        depth_colored_image.save(depth_colored_bytes, format='PNG')
        result['depth_colored'] = base64.b64encode(depth_colored_bytes.getvalue()).decode('utf-8')
        
        # Encode uncertainty map
        uncertainty_image = Image.fromarray(uncertainty_normalized)
        uncertainty_bytes = io.BytesIO()
        uncertainty_image.save(uncertainty_bytes, format='PNG')
        result['uncertainty_map'] = base64.b64encode(uncertainty_bytes.getvalue()).decode('utf-8')
        
        return jsonify(result)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/compare-depth-images', methods=['POST'])
def compare_depth_images():
    """
    Compare depth maps from two images
    
    Expects JSON:
    {
        "image1": base64_encoded_png,
        "image2": base64_encoded_png
    }
    
    Returns JSON:
    {
        "success": true,
        "depth1": {...},
        "depth2": {...},
        "difference_map": base64_encoded_diff
    }
    """
    model = load_model()
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        
        # Decode both images
        image1_bytes = base64.b64decode(data['image1'])
        image1 = Image.open(io.BytesIO(image1_bytes)).convert('RGB')
        
        image2_bytes = base64.b64decode(data['image2'])
        image2 = Image.open(io.BytesIO(image2_bytes)).convert('RGB')
        
        print(f"Received images: {image1.size}, {image2.size}")
        
        # Estimate depths
        with np.no_grad():
            depth1 = model.infer_pil(image1)
            depth2 = model.infer_pil(image2)
        
        depth1_np = np.array(depth1)
        depth2_np = np.array(depth2)
        
        # Normalize both for visualization
        depth1_min, depth1_max = depth1_np.min(), depth1_np.max()
        depth2_min, depth2_max = depth2_np.min(), depth2_np.max()
        
        depth1_norm = ((depth1_np - depth1_min) / (depth1_max - depth1_min + 1e-5) * 255).astype(np.uint8)
        depth2_norm = ((depth2_np - depth2_min) / (depth2_max - depth2_min + 1e-5) * 255).astype(np.uint8)
        
        # Resize depth2 to match depth1 if needed
        if depth1_norm.shape != depth2_norm.shape:
            depth2_norm = cv2.resize(depth2_norm, (depth1_norm.shape[1], depth1_norm.shape[0]))
        
        # Calculate difference
        difference = np.abs(depth1_norm.astype(float) - depth2_norm.astype(float)).astype(np.uint8)
        
        result = {
            'success': True,
            'image1_size': [image1.size[0], image1.size[1]],
            'image2_size': [image2.size[0], image2.size[1]],
            'depth1_range': [float(depth1_min), float(depth1_max)],
            'depth2_range': [float(depth2_min), float(depth2_max)]
        }
        
        # Encode difference map
        diff_image = Image.fromarray(difference)
        diff_bytes = io.BytesIO()
        diff_image.save(diff_bytes, format='PNG')
        result['difference_map'] = base64.b64encode(diff_bytes.getvalue()).decode('utf-8')
        
        return jsonify(result)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/info', methods=['GET'])
def info():
    """Get server information"""
    return jsonify({
        'service': 'Depth Anything V3 Monocular Depth Estimation Server',
        'version': '1.0',
        'endpoints': {
            '/health': 'GET - Check server status',
            '/info': 'GET - Get server information',
            '/estimate-depth': 'POST - Estimate depth from single image',
            '/depth-with-confidence': 'POST - Depth estimation with confidence metrics',
            '/compare-depth-images': 'POST - Compare depth maps from two images'
        },
        'model_info': {
            'name': 'Depth Anything V3',
            'type': 'Monocular Depth Estimation',
            'device': 'CUDA',
            'encoders': ['vit_small', 'vit_base', 'vit_large']
        }
    })

if __name__ == '__main__':
    # Pre-load model
    load_model()
    
    # Run server
    print("Starting Depth Anything V3 Server on 0.0.0.0:8002...")
    app.run(host='0.0.0.0', port=8002, threaded=True)
