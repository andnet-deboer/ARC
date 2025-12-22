#!/usr/bin/env python3
"""
Multi-Model Client with GUI - Display SAM3D, SAM3, and Depth Anything V3 results
Shows everything in interactive windows instead of saving files
"""

import os
import sys
import io
import base64
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests
from PIL import Image
import cv2

# Try to import visualization libraries
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


class MultiModelClientGUI:
    """Client for testing SAM3D, SAM3, and Depth Anything V3 with GUI display."""
    
    def __init__(self, server_ip: str, sam3d_port: int = 8000, 
                 sam3_port: int = 8001, depth_port: int = 8002):
        """Initialize client with server configuration."""
        self.server_ip = server_ip
        
        self.sam3d_url = f"http://{server_ip}:{sam3d_port}"
        self.sam3_url = f"http://{server_ip}:{sam3_port}"
        self.depth_url = f"http://{server_ip}:{depth_port}"
        
        print(f"{BLUE}{'='*60}{NC}")
        print(f"{BLUE}Multi-Model Client (GUI Mode){NC}")
        print(f"{BLUE}{'='*60}{NC}")
        print(f"Server IP: {YELLOW}{server_ip}{NC}")
        print(f"SAM3D (3D Reconstruction): {YELLOW}{self.sam3d_url}{NC}")
        print(f"SAM3 (2D Segmentation): {YELLOW}{self.sam3_url}{NC}")
        print(f"Depth Anything V3: {YELLOW}{self.depth_url}{NC}")
        print(f"{BLUE}{'='*60}{NC}\n")
        
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
    
    @staticmethod
    def _show_image(image: Image.Image, title: str = "Image"):
        """Display image in OpenCV window."""
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Resize if too large
        h, w = img_cv.shape[:2]
        max_dim = 1200
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img_cv = cv2.resize(img_cv, (int(w*scale), int(h*scale)))
        
        cv2.imshow(title, img_cv)
        print(f"{BLUE}  Displaying: {YELLOW}{title}{NC} (Press any key to continue)...{NC}")
        cv2.waitKey(0)
        cv2.destroyWindow(title)
    
    def test_sam3_segmentation(self, image_path: str, prompt: str = "object") -> Dict:
        """Test SAM3 2D segmentation and display results."""
        print(f"\n{BLUE}{'='*60}{NC}")
        print(f"{BLUE}TEST 1: SAM3 2D Segmentation{NC}")
        print(f"{BLUE}{'='*60}{NC}\n")
        
        try:
            print(f"Loading image...")
            image = self._load_image(image_path)
            
            print(f"\n{BLUE}Showing original image...{NC}")
            self._show_image(image, "SAM3: Original Image")
            
            print(f"\nSending segmentation request to SAM3...")
            print(f"  Prompt: {YELLOW}\"{prompt}\"{NC}")
            
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
                print(f"{RED}✗ Error: {response.status_code}{NC}")
                return {'success': False, 'error': response.text}
            
            result = response.json()
            
            if not result.get('success'):
                print(f"{RED}✗ Segmentation failed: {result.get('error')}{NC}")
                return result
            
            num_masks = result.get('num_masks', 0)
            print(f"\n{GREEN}✓ Segmentation complete!{NC}")
            print(f"  Generated masks: {YELLOW}{num_masks}{NC}")
            
            print(f"\n{BLUE}Creating and showing mask overlay...{NC}")
            overlay = self._create_masks_overlay(image, result['masks'])
            self._show_image(overlay, f"SAM3: Overlay ({num_masks} masks)")
            
            print(f"\n{BLUE}Showing individual masks...{NC}")
            for idx, mask_data in enumerate(result['masks']):
                mask_img = self._base64_to_image(mask_data['segmentation'])
                area = mask_data.get('area', 0)
                stability = mask_data.get('stability_score', 0)
                print(f"  Mask {idx}: area={area}, stability={stability:.2f}")
                self._show_image(mask_img, f"SAM3: Mask {idx+1}/{num_masks} (area: {area})")
            
            print(f"\n{GREEN}✓ SAM3 test completed successfully!{NC}\n")
            return result
            
        except Exception as e:
            print(f"{RED}✗ Error: {str(e)}{NC}\n")
            return {'success': False, 'error': str(e)}
    
    def test_depth_anything_v3(self, image_path: str) -> Dict:
        """Test Depth Anything V3 depth estimation and display results."""
        print(f"\n{BLUE}{'='*60}{NC}")
        print(f"{BLUE}TEST 2: Depth Anything V3 - Monocular Depth Estimation{NC}")
        print(f"{BLUE}{'='*60}{NC}\n")
        
        try:
            print(f"Loading image...")
            image = self._load_image(image_path)
            
            print(f"\n{BLUE}Showing original image...{NC}")
            self._show_image(image, "Depth: Original Image")
            
            print(f"\nSending depth estimation request to Depth Anything V3...")
            
            payload = {
                'image': self._image_to_base64(image)
            }
            
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
            
            stats = result.get('depth_statistics', {})
            print(f"\n{GREEN}✓ Depth estimation complete!{NC}")
            print(f"  Depth range: [{YELLOW}{stats.get('min', 0):.2f}{NC}, {YELLOW}{stats.get('max', 0):.2f}{NC}]")
            print(f"  Mean depth: {YELLOW}{stats.get('mean', 0):.2f}{NC}")
            print(f"  Median depth: {YELLOW}{stats.get('median', 0):.2f}{NC}")
            print(f"  Std deviation: {YELLOW}{stats.get('std', 0):.2f}{NC}")
            
            print(f"\n{BLUE}Showing depth map...{NC}")
            depth_map = self._base64_to_image(result['depth_map'])
            self._show_image(depth_map, "Depth: Raw Depth Map (Grayscale)")
            
            print(f"\n{BLUE}Showing colorized depth...{NC}")
            depth_colored = self._base64_to_image(result['depth_colored'])
            self._show_image(depth_colored, "Depth: Colorized Depth Map")
            
            print(f"\n{BLUE}Showing uncertainty map...{NC}")
            uncertainty = self._base64_to_image(result['uncertainty_map'])
            self._show_image(uncertainty, "Depth: Uncertainty Map")
            
            print(f"\n{BLUE}Showing side-by-side comparison...{NC}")
            comparison = self._create_side_by_side(image, depth_colored)
            self._show_image(comparison, "Depth: Original vs Colorized Depth")
            
            print(f"\n{GREEN}✓ Depth Anything V3 test completed successfully!{NC}\n")
            return result
            
        except Exception as e:
            print(f"{RED}✗ Error: {str(e)}{NC}\n")
            return {'success': False, 'error': str(e)}
    
    def test_sam3d_reconstruction(self, image_path: str, prompt: str = "object") -> Dict:
        """Test SAM3D 3D reconstruction and display results."""
        print(f"\n{BLUE}{'='*60}{NC}")
        print(f"{BLUE}TEST 3: SAM3D - 3D Reconstruction{NC}")
        print(f"{BLUE}{'='*60}{NC}\n")
        
        try:
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
            
            largest_mask_idx = max(
                range(len(sam3_result['masks'])),
                key=lambda i: sam3_result['masks'][i]['area']
            )
            
            mask_b64 = sam3_result['masks'][largest_mask_idx]['segmentation']
            mask_area = sam3_result['masks'][largest_mask_idx]['area']
            
            print(f"  Using largest mask (area: {YELLOW}{mask_area}{NC} pixels)")
            
            print(f"\n{BLUE}Showing segmentation mask...{NC}")
            mask_img = self._base64_to_image(mask_b64)
            self._show_image(mask_img, "SAM3D: Input Segmentation Mask")
            
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
            
            print(f"\nStep 3/3: Visualizing 3D data...")
            
            has_mesh = 'mesh_vertices' in sam3d_result and 'mesh_faces' in sam3d_result
            has_points = 'points' in sam3d_result
            
            if has_mesh:
                print(f"  Mesh data available:")
                print(f"    Vertices: {YELLOW}{len(sam3d_result['mesh_vertices'])}{NC}")
                print(f"    Faces: {YELLOW}{len(sam3d_result['mesh_faces'])}{NC}")
                
                mesh = self._create_open3d_mesh(sam3d_result)
                print(f"\n{BLUE}Opening 3D mesh viewer...{NC}")
                print(f"  {YELLOW}Click and drag to rotate, scroll to zoom, right-click to pan{NC}")
                self._visualize_mesh(mesh, title="SAM3D 3D Reconstruction - Mesh")
            
            elif has_points:
                print(f"  Point cloud data available:")
                print(f"    Points: {YELLOW}{len(sam3d_result['points'])}{NC}")
                
                pcd = self._create_open3d_pointcloud(sam3d_result)
                print(f"\n{BLUE}Opening 3D point cloud viewer...{NC}")
                print(f"  {YELLOW}Click and drag to rotate, scroll to zoom, right-click to pan{NC}")
                self._visualize_pointcloud(pcd, title="SAM3D 3D Reconstruction - Point Cloud")
            
            print(f"\n{GREEN}✓ SAM3D test completed successfully!{NC}\n")
            
            print(f"{BLUE}{'='*60}{NC}")
            print(f"{GREEN}3D Reconstruction Summary:{NC}")
            print(f"  Input image: {YELLOW}{image_path}{NC}")
            print(f"  Segmentation prompt: {YELLOW}\"{prompt}\"{NC}")
            print(f"  Object area: {YELLOW}{mask_area}{NC} pixels")
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
            mask_img = MultiModelClientGUI._base64_to_image(mask_data['segmentation'])
            mask = np.array(mask_img) > 128
            
            color = np.random.randint(50, 200, 3)
            overlay[mask] = overlay[mask] * 0.5 + np.array(color) * 0.5
        
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return Image.fromarray(overlay)
    
    @staticmethod
    def _create_side_by_side(image1: Image.Image, image2: Image.Image) -> Image.Image:
        """Create side-by-side comparison of two images."""
        h = min(image1.height, image2.height)
        w1 = int(image1.width * h / image1.height)
        w2 = int(image2.width * h / image2.height)
        
        img1_resized = image1.resize((w1, h))
        img2_resized = image2.resize((w2, h))
        
        total_width = w1 + w2
        result = Image.new('RGB', (total_width, h))
        result.paste(img1_resized, (0, 0))
        result.paste(img2_resized, (w1, 0))
        
        return result
    
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
            o3d.visualization.draw_geometries([mesh], window_name=title)
        except Exception as e:
            print(f"{YELLOW}Warning: Could not open 3D viewer: {str(e)}{NC}")
    
    @staticmethod
    def _visualize_pointcloud(pcd: 'o3d.geometry.PointCloud', title: str = "Point Cloud"):
        """Visualize point cloud with Open3D."""
        if not HAS_OPEN3D:
            return
        
        try:
            o3d.visualization.draw_geometries([pcd], window_name=title)
        except Exception as e:
            print(f"{YELLOW}Warning: Could not open 3D viewer: {str(e)}{NC}")


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Model Client (GUI Mode) for SAM3D, SAM3, and Depth Anything V3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python arc_client_gui.py --server-ip 192.168.1.100 --image test.jpg --all
  python arc_client_gui.py --server-ip localhost --image test.jpg --sam3
  python arc_client_gui.py --server-ip 192.168.1.100 --image test.jpg --depth
  python arc_client_gui.py --server-ip 192.168.1.100 --image test.jpg --sam3d --prompt "coffee mug"
        """
    )
    
    parser.add_argument('--server-ip', type=str, default='localhost',
                    help='IP address of the server (default: localhost)')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to test image')
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
    
    if not (args.sam3 or args.depth or args.sam3d):
        args.all = True
    
    try:
        client = MultiModelClientGUI(args.server_ip)
        
        if args.all or args.sam3:
            client.test_sam3_segmentation(args.image, args.prompt)
        
        if args.all or args.depth:
            client.test_depth_anything_v3(args.image)
        
        if args.all or args.sam3d:
            client.test_sam3d_reconstruction(args.image, args.prompt)
        
        print(f"\n{GREEN}{'='*60}{NC}")
        print(f"{GREEN}All tests completed!{NC}")
        print(f"{GREEN}{'='*60}{NC}\n")
        
    except Exception as e:
        print(f"\n{RED}✗ Fatal error: {str(e)}{NC}\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
