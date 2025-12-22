#!/usr/bin/env python3
"""
Contact-GraspNet Flask Server
Generates 6-DoF grasp poses from point clouds.
"""

import os
import numpy as np
from flask import Flask, request, jsonify
import base64
import io
import traceback

app = Flask(__name__)

# Global model
cgn_model = None


def load_model():
    """Load Contact-GraspNet model."""
    global cgn_model
    
    print("Loading Contact-GraspNet...")
    from cgn_pytorch import from_pretrained
    
    cgn_model, _, _ = from_pretrained()
    cgn_model.eval()
    
    # Move to GPU if available
    import torch
    if torch.cuda.is_available():
        cgn_model = cgn_model.cuda()
        print(f"  Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  Model loaded on CPU")
    
    print("✓ Contact-GraspNet loaded successfully")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": cgn_model is not None,
        "service": "contact-graspnet"
    })


@app.route('/info', methods=['GET'])
def info():
    """Model info endpoint."""
    import torch
    return jsonify({
        "service": "contact-graspnet",
        "version": "0.4.3",
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Generate grasps from point cloud.
    
    Input JSON:
        - points: base64 encoded numpy array (N, 3) float32
        - threshold: float (optional, default 0.5)
        - max_grasps: int (optional, default 50)
    
    Returns JSON:
        - poses: base64 encoded numpy array (M, 4, 4) float32
        - scores: base64 encoded numpy array (M,) float32
        - widths: base64 encoded numpy array (M,) float32
        - count: int
    """
    global cgn_model
    
    if cgn_model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        data = request.get_json()
        
        # Decode point cloud
        points_b64 = data.get('points')
        if not points_b64:
            return jsonify({"error": "Missing 'points' field"}), 400
        
        points_bytes = base64.b64decode(points_b64)
        points = np.frombuffer(points_bytes, dtype=np.float32).reshape(-1, 3)
        
        threshold = data.get('threshold', 0.4)
        max_grasps = data.get('max_grasps', 100)
        
        print(f"  Received point cloud: {points.shape[0]} points")
        
        # Center the point cloud
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        # Run inference
        from cgn_pytorch import inference as cgn_inference
        poses, scores, widths = cgn_inference(cgn_model, points_centered, threshold=threshold)
        
        if scores is None or len(scores) == 0:
            return jsonify({
                "poses": None,
                "scores": None,
                "widths": None,
                "count": 0,
                "message": "No grasps found"
            })
        
        # Convert widths to float64 and fix units if needed
        widths = widths.astype(np.float64)
        if np.median(widths) > 1.0:
            widths /= 1000.0
            poses[:, :3, 3] /= 1000.0
        
        # Translate poses back to original frame
        poses[:, :3, 3] += centroid
        
        # Sort by score and limit
        order = np.argsort(-scores)[:max_grasps]
        poses = poses[order]
        scores = scores[order]
        widths = widths[order]
        
        print(f"  Generated {len(scores)} grasps")
        
        # Encode results
        poses_b64 = base64.b64encode(poses.astype(np.float32).tobytes()).decode('utf-8')
        scores_b64 = base64.b64encode(scores.astype(np.float32).tobytes()).decode('utf-8')
        widths_b64 = base64.b64encode(widths.astype(np.float32).tobytes()).decode('utf-8')
        
        return jsonify({
            "poses": poses_b64,
            "scores": scores_b64,
            "widths": widths_b64,
            "count": len(scores)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    load_model()
    print("\n" + "=" * 50)
    print("Contact-GraspNet Server running on port 8003")
    print("=" * 50 + "\n")
    app.run(host='0.0.0.0', port=8003, threaded=False)