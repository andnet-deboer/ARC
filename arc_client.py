#!/usr/bin/env python3
"""
Multi-Model Client - Test SAM3D, SAM3, and Depth Anything V3 servers
Requires server IP and test image path
"""

import os
import sys
import io
import base64
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image
import cv2

# Try to import open3d for visualization
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: open3d not installed. 3D visualization will be limited.")

# Color codes for output
BLUE = '\033[0;34m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'  # No Color


class MultiModelClient:
    """Client for testing SAM3D, SAM3, and Depth Anything V3 servers."""
    
    def __init__(self, server_ip: str, sam3d_port: int = 8000, 
                 sam3_port: int = 8001, depth_port: int = 8002):
        """
        Initialize client with server configuration.
        
        Args:
            server_ip: IP address of the server (e.g., "192.168.1.100")
            sam3d_port: Port for SAM3D server (default: 8000)
            sam3_port: Port for SAM3 server (default: 8001)
            depth_port: Port for Depth Anything V3 (default: 8002)
        """
        self.server_ip = server_ip
        
        self.sam3d_url = f"http://{server_ip}:{sam3d_port}"
        self.sam3_url = f"http://{server_ip}:{sam3_port}"
        self.depth_url = f"http://{server_ip}:{depth_port}"
        
        print(f"{BLUE}{'='*60}{NC}")
        print(f"{BLUE}Multi-Model Client{NC}")
        print(f"{BLUE}{'='*60}{NC}")
        print(f"Server IP: {YELLOW}{server_ip}{NC}")
        print(f"SAM3D (3D Reconstruction): {YELLOW}{self.sam3d_url}{NC}")
        print(f"SAM3 (2D Segmentation): {YELLOW}{self.sam3_url}{NC}")
        print(f"Depth Anything V3: {YELLOW}{self.depth_url}{NC}")
        print(f"{BLUE}{'='*60}{NC}\n")
        
        # Verify connectivity
        self._verify_servers()
    
    def _verify_servers(self):
        """Verify connection to all servers."""
        print(f"{BLUE}Verifying server connectivity...{NC}\n")
        
        servers = {
            'SAM3D': self.sam3d_url,
            'SAM3': self.sam3_url,
            'Depth Anything V3': self.depth_url
        }
        
        for name, url in servers.items():
            try:
                response = requests.get(f"{url}/health", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    status = f"{GREEN}✓ OK{NC}" if data.get('model_loaded') else f"{YELLOW}⚠ Model not loaded{NC}"
                    print(f"  {name:20} {status}")
                else:
                    print(f"  {name:20} {RED}✗ Error (HTTP {response.status_code}){NC}")
            except requests.exceptions.ConnectionError:
                print(f"  {name:20} {RED}✗ Connection refused{NC}")
            except Exception as e:
                print(f"  {name:20} {RED}✗ Error: {str(e)}{NC}")
        print()
    
    @staticmethod
    def _load_image(image_path: str) -> Image.Image:
        """Load image from path."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        print(f"  {GREEN}✓ Loaded{NC}: {image_path} ({image.size[0]}x{image.size[1]})")
        return image
    
    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        """Convert PIL image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    @staticmethod
    def _base64_to_image(b64_string: str) -> Image.Image:
        """Convert base64 string to PIL image."""
        image_bytes = base64.b64decode(b64_string)
        return Image.open(io.BytesIO(image_bytes))
    
    def test_sam3_segmentation(self, image_path: str, prompt: str = "object", 
                               output_dir: str = "output") -> Dict:
        """
        Test SAM3 2D segmentation.
        
        Args:
            image_path: Path to test image
            prompt: Text prompt for segmentation
            output_dir: Directory to save results
        
        Returns:
            Dictionary with segmentation results
        """
        print(f"\n{BLUE}{'='*60}{NC}")
        print(f"{BLUE}TEST 1: SAM3 2D Segmentation{NC}")
        print(f"{BLUE}{'='*60}{NC}\n")
        
        try:
            # Load image
            print(f"Loading image...")
            image = self._load_image(image_path)
            
            # Prepare request
            print(f"\nSending segmentation request to SAM3...")
            print(f"  Prompt: {YELLOW}\"{prompt}\"{NC}")
            
            payload = {
                'image': self._image_to_base64(image),
                'prompt': prompt,
                'confidence_threshold': 0.80
            }
            
            # Send request
            response = requests.post(
                f"{self.sam3_url}/segment-with-prompt",
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                print(f"{RED}✗ Error: {response.status_code}{NC}")
                return {'success': False, 'error': response.text}
            
            result = response.json()
            
            if not result.get('success'):
                print(f"{RED}✗ Segmentation failed: {result.get('error')}{NC}")
                return result
            
            # Process results
            num_masks = result.get('num_masks', 0)
            print(f"\n{GREEN}✓ Segmentation complete!{NC}")
            print(f"  Generated masks: {YELLOW}{num_masks}{NC}")
            
            # Save results
            os.makedirs(output_dir, exist_ok=True)
            
            # Save original image
            original_path = os.path.join(output_dir, 'sam3_original.jpg')
            image.save(original_path)
            print(f"  Saved original: {YELLOW}{original_path}{NC}")
            
            # Save overlay with all masks
            overlay_path = os.path.join(output_dir, 'sam3_overlay.jpg')
            overlay = self._create_masks_overlay(image, result['masks'])
            overlay.save(overlay_path)
            print(f"  Saved overlay: {YELLOW}{overlay_path}{NC}")
            
            # Save individual masks
            masks_dir = os.path.join(output_dir, 'sam3_individual_masks')
            os.makedirs(masks_dir, exist_ok=True)
            
            for idx, mask_data in enumerate(result['masks']):
                mask_img = self._base64_to_image(mask_data['segmentation'])
                mask_path = os.path.join(masks_dir, f'mask_{idx:03d}.png')
                mask_img.save(mask_path)
            
            print(f"  Saved {YELLOW}{num_masks}{NC} individual masks to {YELLOW}{masks_dir}{NC}")
            
            # Save metadata
            metadata_path = os.path.join(output_dir, 'sam3_metadata.json')
            with open(metadata_path, 'w') as f:
                # Remove base64 data for cleaner JSON
                clean_result = {
                    'prompt': result['prompt'],
                    'num_masks': result['num_masks'],
                    'image_size': result['image_size'],
                    'masks_info': [
                        {
                            'id': m['id'],
                            'area': m['area'],
                            'stability_score': m['stability_score'],
                            'bbox': m['bbox']
                        }
                        for m in result['masks']
                    ]
                }
                json.dump(clean_result, f, indent=2)
            
            print(f"  Saved metadata: {YELLOW}{metadata_path}{NC}")
            
            print(f"\n{GREEN}✓ SAM3 test completed successfully!{NC}\n")
            return result
            
        except Exception as e:
            print(f"{RED}✗ Error: {str(e)}{NC}\n")
            return {'success': False, 'error': str(e)}
    
    def test_depth_anything_v3(self, image_path: str, 
                               output_dir: str = "output") -> Dict:
        """
        Test Depth Anything V3 depth estimation.
        
        Args:
            image_path: Path to test image
            output_dir: Directory to save results
        
        Returns:
            Dictionary with depth results
        """
        print(f"\n{BLUE}{'='*60}{NC}")
        print(f"{BLUE}TEST 2: Depth Anything V3 - Monocular Depth Estimation{NC}")
        print(f"{BLUE}{'='*60}{NC}\n")
        
        try:
            # Load image
            print(f"Loading image...")
            image = self._load_image(image_path)
            
            # Prepare request
            print(f"\nSending depth estimation request to Depth Anything V3...")
            
            payload = {
                'image': self._image_to_base64(image)
            }
            
            # Send request
            response = requests.post(
                f"{self.depth_url}/depth-with-confidence",
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                print(f"{RED}✗ Error: {response.status_code}{NC}")
                return {'success': False, 'error': response.text}
            
            result = response.json()
            
            if not result.get('success'):
                print(f"{RED}✗ Depth estimation failed: {result.get('error')}{NC}")
                return result
            
            # Process results
            stats = result.get('depth_statistics', {})
            print(f"\n{GREEN}✓ Depth estimation complete!{NC}")
            print(f"  Depth range: [{YELLOW}{stats.get('min', 0):.2f}{NC}, {YELLOW}{stats.get('max', 0):.2f}{NC}]")
            print(f"  Mean depth: {YELLOW}{stats.get('mean', 0):.2f}{NC}")
            print(f"  Median depth: {YELLOW}{stats.get('median', 0):.2f}{NC}")
            print(f"  Std deviation: {YELLOW}{stats.get('std', 0):.2f}{NC}")
            
            # Save results
            os.makedirs(output_dir, exist_ok=True)
            
            # Save original image
            original_path = os.path.join(output_dir, 'depth_original.jpg')
            image.save(original_path)
            print(f"\n  Saved original: {YELLOW}{original_path}{NC}")
            
            # Save depth map
            depth_map = self._base64_to_image(result['depth_map'])
            depth_path = os.path.join(output_dir, 'depth_map.png')
            depth_map.save(depth_path)
            print(f"  Saved depth map: {YELLOW}{depth_path}{NC}")
            
            # Save colorized depth
            depth_colored = self._base64_to_image(result['depth_colored'])
            depth_colored_path = os.path.join(output_dir, 'depth_colored.png')
            depth_colored.save(depth_colored_path)
            print(f"  Saved colorized depth: {YELLOW}{depth_colored_path}{NC}")
            
            # Save uncertainty map
            uncertainty = self._base64_to_image(result['uncertainty_map'])
            uncertainty_path = os.path.join(output_dir, 'depth_uncertainty.png')
            uncertainty.save(uncertainty_path)
            print(f"  Saved uncertainty map: {YELLOW}{uncertainty_path}{NC}")
            
            # Save metadata
            metadata_path = os.path.join(output_dir, 'depth_metadata.json')
            with open(metadata_path, 'w') as f:
                clean_result = {
                    'image_size': result['image_size'],
                    'depth_statistics': stats
                }
                json.dump(clean_result, f, indent=2)
            
            print(f"  Saved metadata: {YELLOW}{metadata_path}{NC}")
            
            print(f"\n{GREEN}✓ Depth Anything V3 test completed successfully!{NC}\n")
            return result
            
        except Exception as e:
            print(f"{RED}✗ Error: {str(e)}{NC}\n")
            return {'success': False, 'error': str(e)}
    
    def test_sam3d_reconstruction(self, image_path: str, prompt: str = "object",
                                  output_dir: str = "output") -> Dict:
        """
        Test SAM3D 3D reconstruction.
        
        Args:
            image_path: Path to test image
            prompt: Text prompt for segmentation
            output_dir: Directory to save results
        
        Returns:
            Dictionary with 3D reconstruction results
        """
        print(f"\n{BLUE}{'='*60}{NC}")
        print(f"{BLUE}TEST 3: SAM3D - 3D Reconstruction{NC}")
        print(f"{BLUE}{'='*60}{NC}\n")
        
        try:
            # Step 1: Get 2D segmentation with SAM3
            print(f"Step 1/3: Getting 2D segmentation with SAM3...")
            
            image = self._load_image(image_path)
            
            payload = {
                'image': self._image_to_base64(image),
                'prompt': prompt,
                'confidence_threshold': 0.80
            }
            
            response = requests.post(
                f"{self.sam3_url}/segment-with-prompt",
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                print(f"{RED}✗ SAM3 segmentation failed{NC}")
                return {'success': False, 'error': 'SAM3 segmentation failed'}
            
            sam3_result = response.json()
            
            if not sam3_result.get('success') or sam3_result.get('num_masks', 0) == 0:
                print(f"{RED}✗ No masks generated{NC}")
                return {'success': False, 'error': 'No masks generated'}
            
            print(f"{GREEN}✓ Generated {YELLOW}{sam3_result['num_masks']}{NC} masks{NC}")
            
            # Use the largest mask
            largest_mask_idx = max(
                range(len(sam3_result['masks'])),
                key=lambda i: sam3_result['masks'][i]['area']
            )
            
            mask_b64 = sam3_result['masks'][largest_mask_idx]['segmentation']
            mask_area = sam3_result['masks'][largest_mask_idx]['area']
            
            print(f"  Using largest mask (area: {YELLOW}{mask_area}{NC} pixels)")
            
            # Step 2: Send to SAM3D for 3D reconstruction
            print(f"\nStep 2/3: Sending to SAM3D for 3D reconstruction...")
            
            payload_3d = {
                'image': self._image_to_base64(image),
                'mask': mask_b64
            }
            
            response = requests.post(
                f"{self.sam3d_url}/reconstruct",
                json=payload_3d,
                timeout=120
            )
            
            if response.status_code != 200:
                print(f"{RED}✗ SAM3D reconstruction failed: {response.status_code}{NC}")
                return {'success': False, 'error': 'SAM3D reconstruction failed'}
            
            sam3d_result = response.json()
            
            if not sam3d_result.get('success'):
                print(f"{RED}✗ Reconstruction failed: {sam3d_result.get('error')}{NC}")
                return sam3d_result
            
            print(f"{GREEN}✓ 3D reconstruction complete!{NC}")
            
            # Step 3: Process and visualize 3D output
            print(f"\nStep 3/3: Processing 3D data...")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Check if we have mesh or point cloud
            has_mesh = 'mesh_vertices' in sam3d_result and 'mesh_faces' in sam3d_result
            has_points = 'points' in sam3d_result
            
            if has_mesh:
                print(f"  Mesh data available:")
                print(f"    Vertices: {YELLOW}{len(sam3d_result['mesh_vertices'])}{NC}")
                print(f"    Faces: {YELLOW}{len(sam3d_result['mesh_faces'])}{NC}")
                
                # Create and save mesh
                mesh = self._create_open3d_mesh(sam3d_result)
                mesh_path = os.path.join(output_dir, 'sam3d_mesh.ply')
                o3d.io.write_triangle_mesh(mesh_path, mesh)
                print(f"    Saved mesh: {YELLOW}{mesh_path}{NC}")
                
                # Create visualizer
                if HAS_OPEN3D:
                    self._visualize_mesh(mesh, title="SAM3D 3D Reconstruction - Mesh")
            
            elif has_points:
                print(f"  Point cloud data available:")
                print(f"    Points: {YELLOW}{len(sam3d_result['points'])}{NC}")
                
                # Create and save point cloud
                pcd = self._create_open3d_pointcloud(sam3d_result)
                pcd_path = os.path.join(output_dir, 'sam3d_pointcloud.ply')
                o3d.io.write_point_cloud(pcd_path, pcd)
                print(f"    Saved point cloud: {YELLOW}{pcd_path}{NC}")
                
                # Create visualizer
                if HAS_OPEN3D:
                    self._visualize_pointcloud(pcd, title="SAM3D 3D Reconstruction - Point Cloud")
            
            # Save segmentation mask for reference
            mask_img = self._base64_to_image(mask_b64)
            mask_path = os.path.join(output_dir, 'sam3d_input_mask.png')
            mask_img.save(mask_path)
            print(f"  Saved input mask: {YELLOW}{mask_path}{NC}")
            
            # Save original image
            image_path_out = os.path.join(output_dir, 'sam3d_original.jpg')
            image.save(image_path_out)
            print(f"  Saved original image: {YELLOW}{image_path_out}{NC}")
            
            # Save metadata
            metadata_path = os.path.join(output_dir, 'sam3d_metadata.json')
            with open(metadata_path, 'w') as f:
                clean_result = {
                    'prompt': prompt,
                    'segmentation_mask_area': mask_area,
                    '3d_data_type': 'mesh' if has_mesh else 'point_cloud',
                    'mesh_vertices': len(sam3d_result.get('mesh_vertices', [])),
                    'mesh_faces': len(sam3d_result.get('mesh_faces', [])),
                    'point_cloud_points': len(sam3d_result.get('points', []))
                }
                json.dump(clean_result, f, indent=2)
            
            print(f"  Saved metadata: {YELLOW}{metadata_path}{NC}")
            
            print(f"\n{GREEN}✓ SAM3D test completed successfully!{NC}\n")
            
            # Summary
            print(f"{BLUE}{'='*60}{NC}")
            print(f"{GREEN}3D Reconstruction Summary:{NC}")
            print(f"  Input image: {YELLOW}{image_path}{NC}")
            print(f"  Segmentation prompt: {YELLOW}\"{prompt}\"{NC}")
            print(f"  Output directory: {YELLOW}{output_dir}{NC}")
            print(f"{BLUE}{'='*60}{NC}\n")
            
            return sam3d_result
            
        except Exception as e:
            print(f"{RED}✗ Error: {str(e)}{NC}\n")
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def _create_masks_overlay(image: Image.Image, masks: List[Dict]) -> Image.Image:
        """Create image with all masks overlaid."""
        overlay = np.array(image, dtype=np.float32)
        
        np.random.seed(42)
        for idx, mask_data in enumerate(masks):
            mask_img = MultiModelClient._base64_to_image(mask_data['segmentation'])
            mask = np.array(mask_img) > 128
            
            color = np.random.randint(50, 200, 3)
            overlay[mask] = overlay[mask] * 0.5 + np.array(color) * 0.5
        
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return Image.fromarray(overlay)
    
    @staticmethod
    def _create_open3d_mesh(result: Dict) -> 'o3d.geometry.TriangleMesh':
        """Create Open3D mesh from SAM3D output."""
        if not HAS_OPEN3D:
            raise ImportError("open3d is required for mesh creation")
        
        vertices = np.array(result['mesh_vertices'], dtype=np.float32)
        faces = np.array(result['mesh_faces'], dtype=np.int32)
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        # Try to add colors if available
        if 'mesh_colors' in result:
            colors = np.array(result['mesh_colors'], dtype=np.float32)
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        else:
            mesh.paint_uniform_color([0.7, 0.7, 0.7])
        
        mesh.compute_vertex_normals()
        return mesh
    
    @staticmethod
    def _create_open3d_pointcloud(result: Dict) -> 'o3d.geometry.PointCloud':
        """Create Open3D point cloud from SAM3D output."""
        if not HAS_OPEN3D:
            raise ImportError("open3d is required for point cloud creation")
        
        points = np.array(result['points'], dtype=np.float32)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Try to add colors if available
        if 'colors' in result:
            colors = np.array(result['colors'], dtype=np.float32)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            pcd.paint_uniform_color([0.7, 0.7, 0.7])
        
        return pcd
    
    @staticmethod
    def _visualize_mesh(mesh: 'o3d.geometry.TriangleMesh', title: str = "Mesh"):
        """Visualize mesh with Open3D."""
        if not HAS_OPEN3D:
            return
        
        try:
            print(f"\n{BLUE}Opening 3D viewer for mesh...{NC}")
            o3d.visualization.draw_geometries([mesh], window_name=title)
        except Exception as e:
            print(f"{YELLOW}Warning: Could not open 3D viewer: {str(e)}{NC}")
    
    @staticmethod
    def _visualize_pointcloud(pcd: 'o3d.geometry.PointCloud', title: str = "Point Cloud"):
        """Visualize point cloud with Open3D."""
        if not HAS_OPEN3D:
            return
        
        try:
            print(f"\n{BLUE}Opening 3D viewer for point cloud...{NC}")
            o3d.visualization.draw_geometries([pcd], window_name=title)
        except Exception as e:
            print(f"{YELLOW}Warning: Could not open 3D viewer: {str(e)}{NC}")


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Model Client for SAM3D, SAM3, and Depth Anything V3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python client.py --server-ip 192.168.1.100 --image test.jpg --all
  python client.py --server-ip localhost --image test.jpg --sam3
  python client.py --server-ip 192.168.1.100 --image test.jpg --depth
  python client.py --server-ip 192.168.1.100 --image test.jpg --sam3d --prompt "coffee mug"
        """
    )
    
    parser.add_argument('--server-ip', type=str, default='129.105.69.10',
                    help='IP address of the server (default: localhost)')

    parser.add_argument('--image', type=str, required=True,
                       help='Path to test image')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for results')
    parser.add_argument('--prompt', type=str, default='object',
                       help='Text prompt for SAM3/SAM3D')
    parser.add_argument('--sam3', action='store_true',
                       help='Test SAM3 2D segmentation')
    parser.add_argument('--depth', action='store_true',
                       help='Test Depth Anything V3')
    parser.add_argument('--sam3d', action='store_true',
                       help='Test SAM3D 3D reconstruction')
    parser.add_argument('--all', action='store_true',
                       help='Test all models')
    
    args = parser.parse_args()
    
    # If no specific test selected, test all
    if not (args.sam3 or args.depth or args.sam3d):
        args.all = True
    
    try:
        # Initialize client
        client = MultiModelClient(args.server_ip)
        
        # Run tests
        if args.all or args.sam3:
            client.test_sam3_segmentation(args.image, args.prompt, args.output_dir)
        
        if args.all or args.depth:
            client.test_depth_anything_v3(args.image, args.output_dir)
        
        if args.all or args.sam3d:
            client.test_sam3d_reconstruction(args.image, args.prompt, args.output_dir)
        
        print(f"\n{GREEN}{'='*60}{NC}")
        print(f"{GREEN}All tests completed!{NC}")
        print(f"{GREEN}Results saved to: {YELLOW}{args.output_dir}{NC}")
        print(f"{GREEN}{'='*60}{NC}\n")
        
    except Exception as e:
        print(f"\n{RED}✗ Fatal error: {str(e)}{NC}\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
