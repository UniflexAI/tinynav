#!/usr/bin/env python3
"""
Convert tinynav data to COLMAP format

This script converts tinynav's internal data format to COLMAP's sparse reconstruction format.
COLMAP expects either:
- Text format: images.txt, cameras.txt, points3D.txt
- Database format: database.db (preferred, more efficient)

Usage:
    python convert_to_colmap_format.py --input_dir tinynav_db --output_dir colmap_output
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import cv2
from typing import Dict, List, Tuple
import shelve
from tqdm import tqdm

class TinynavToColmapConverter:
    """Convert tinynav data to COLMAP format"""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load tinynav data
        self.poses = None
        self.intrinsics = None
        self.images_db_path = None
        self.features_db_path = None
        self.disparities_db_path = None
        
        self._load_tinynav_data()
    
    def _load_tinynav_data(self):
        """Load all tinynav data files"""
        print(f"Loading tinynav data from {self.input_dir}")
        
        # Load poses
        self.poses = np.load(self.input_dir / "poses.npy", allow_pickle=True).item()
        assert type(self.poses) == dict
        # Load intrinsics
        self.intrinsics = np.load(self.input_dir / "intrinsics.npy")
        print(f"Loaded intrinsics: {self.intrinsics}")
        # Check for database files
        self.images = shelve.open(f"{self.input_dir}/images")
        self.features = shelve.open(f"{self.input_dir}/features")
        self.disparities = shelve.open(f"{self.input_dir}/disparities")
        
        
   
    def _get_image_list(self) -> List[Tuple[int, str, np.ndarray]]:
        """Get list of images with their poses"""
        if self.poses is None:
            print("No poses available")
            return []
        
        images = []
        for i, (timestamp, pose) in enumerate(self.poses.items()):
            # Assuming pose is 4x4 transformation matrix
            if pose.shape == (4, 4):
                # Extract rotation and translation
                R_matrix = pose[:3, :3]
                t_vector = pose[:3, 3]
                
                # Convert to quaternion (COLMAP format)
                quat = R.from_matrix(R_matrix).as_quat()  # [x, y, z, w]
                
                image_name = f"image_{timestamp}.png"
                
                images.append((i, timestamp, image_name, pose))
        return images
    
    def _extract_camera_intrinsics(self) -> Dict:
        """Extract camera intrinsics in COLMAP format"""
        if self.intrinsics is None:
            # Default intrinsics (you may need to adjust these)
            raise ValueError("Intrinsics not found")
        # Assuming intrinsics is a 3x3 K matrix
        K = self.intrinsics
        if K.shape == (3, 3):
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            return {
                'camera_id': 1,
                'model': 'PINHOLE',
                'width': int(848),  # Approximate width
                'height': int(480),  # Approximate height
                'params': [fx, fy, cx, cy]
            }
        
        return None
    

    
    def _create_colmap_text_files(self, sparse_dir):
        """Create COLMAP text files (cameras.txt, images.txt, points3D.txt)"""
        
        # Get camera data
        camera_data = self._extract_camera_intrinsics()
        if camera_data is None:
            print("Error: Could not extract camera intrinsics")
            return
        
        # Get image list
        images = self._get_image_list()
        if not images:
            print("No images to write")
            return
        
        # Create cameras.txt
        self._write_cameras_txt(camera_data, sparse_dir)
        
        # Create images.txt
        point2d_to_point3d = self._write_images_txt(images, camera_data, sparse_dir)
        
        # Generate 3D points from images, disparities, and poses
        points3d_data = self._generate_points3d_from_disparities(images, camera_data, point2d_to_point3d)
        
        # Create points3D.txt with actual 3D points
        self._write_points3d_txt(points3d_data, sparse_dir)
        
        # Generate PLY file from 3D points
        self._write_ply_file(points3d_data)
        
        print(f"Created COLMAP text files with {len(images)} images and {len(points3d_data)} 3D points")
    
    def _write_cameras_txt(self, camera_data, sparse_dir):
        """Write cameras.txt file for COLMAP"""
        cameras_file = sparse_dir / "cameras.txt"
        with open(cameras_file, 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"{camera_data['camera_id']} {camera_data['model']} "
                   f"{camera_data['width']} {camera_data['height']} "
                   f"{' '.join(map(str, camera_data['params']))}\n")
        
        print(f"Written cameras.txt with camera {camera_data['camera_id']}")
    
    def _write_images_txt(self, images, camera_data, sparse_dir):
        """Write images.txt file for COLMAP"""
        images_file = sparse_dir / "images.txt"

        point2d_to_point3d = {}
        with open(images_file, 'w') as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            
            for image_id, timestamp, image_name, pose_in_world in images:
                # Extract rotation and translation
                pose_in_camera = np.linalg.inv(pose_in_world)
                R_matrix = pose_in_camera[:3, :3]
                t_vector = pose_in_camera[:3, 3]
                
                # Convert to quaternion
                quat = R.from_matrix(R_matrix).as_quat()  # [x, y, z, w]
                
                # COLMAP format: [qw, qx, qy, qz, tx, ty, tz]
                qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
                tx, ty, tz = t_vector
                
                # Write image line
                f.write(f"{image_id} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} "
                       f"{tx:.6f} {ty:.6f} {tz:.6f} {camera_data['camera_id']} {image_name}\n")

                f.write("\n")
        
        print(f"Written images.txt with {len(images)} images")
        return point2d_to_point3d
    
    def _write_points3d_txt(self, points3d_data=None, sparse_dir=None):
        """Write points3D.txt file for COLMAP"""
        if sparse_dir is None:
            sparse_dir = self.output_dir
        points3d_file = sparse_dir / "points3D.txt"
        
        with open(points3d_file, 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            
            if points3d_data:
                for point in points3d_data:
                    # Write point data: POINT3D_ID, X, Y, Z, R, G, B, ERROR
                    f.write(f"{point['point_id']} {point['x']:.6f} {point['y']:.6f} {point['z']:.6f} "
                           f"{point['r']} {point['g']} {point['b']} {point['error']:.6f}")
                    
                    # Write track information
                    # for image_id, point2d_idx in point['track']:
                        # f.write(f" {image_id} {point2d_idx}")
                    
                    f.write("\n")
                
                print(f"Written points3D.txt with {len(points3d_data)} 3D points")
            else:
                print("Written empty points3D.txt (no 3D points available)")
    
    def _write_ply_file(self, points3d_data):
        """Write PLY file from 3D point cloud data"""
        if not points3d_data:
            print("No 3D points available for PLY file")
            return
        
        ply_file = self.output_dir / "pointcloud.ply"
        
        with open(ply_file, 'w') as f:
            # Write PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points3d_data)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            # Write point data
            for point in points3d_data:
                f.write(f"{point['x']:.6f} {point['y']:.6f} {point['z']:.6f} "
                       f"{point['r']} {point['g']} {point['b']}\n")
        
        print(f"Written PLY file: {ply_file}")
    
    def _generate_points3d_from_disparities(self, images, camera_data, point2d_to_point3d):
        """Generate 3D points from disparities and poses"""
        print("Generating 3D points from disparities...")
        
        points3d_data = []
        point_id = 0
        
        # Get camera parameters
        K = self.intrinsics
        baseline = self._get_baseline()  # You may need to extract this from your data
            
        for image_id, timestamp, image_name, pose_in_world in tqdm(images, desc="Generating 3D points from disparities"):
            disparity = self.disparities[str(timestamp)]
            # Convert disparity to 3D points
            points_3d = self._disparity_to_points3d(
                disparity, K, baseline, pose_in_world, 
                image_id, point_id, timestamp
            )
                            
            points3d_data.extend(points_3d)
            point_id += len(points_3d)
        print(f"Generated {len(points3d_data)} total 3D points")
        return points3d_data
    
    def _disparity_to_points3d(self, disparity, K, baseline, pose_in_world, image_id, start_point_id, timestamp):
        """Convert disparity map to 3D points"""
        points3d = []
        
        # Sample points from disparity (every 10th pixel to avoid too many points)
        step = 64
        height, width = disparity.shape
        
        for v in range(0, height, step):
            for u in range(0, width, step):
                disp = disparity[v, u]
                
                # Skip invalid disparities
                if disp <= 1 or disp > 100:  # Adjust threshold as needed
                    continue
                
                # Convert disparity to depth
                Z = K[0, 0] * baseline / disp
                
                # Skip points too close or too far
                if Z < 0.1 or Z > 3.0:  # Adjust range as needed
                    continue
                
                # Convert to 3D point in camera coordinates
                X = (u - K[0, 2]) * Z / K[0, 0]
                Y = (v - K[1, 2]) * Z / K[1, 1]

                norm = np.linalg.norm(np.array([X, Y, Z]))
                if norm > 2:
                    continue
                
                # Transform to world coordinates
                point_camera = np.array([X, Y, Z, 1])
                point_world = pose_in_world @ point_camera
                
                # Get color from image (if available)
                color = self._get_point_color(u, v, timestamp)
               
                # Create 3D point data
                point_data = {
                    'point_id': start_point_id + len(points3d),
                    'x': point_world[0],
                    'y': point_world[1], 
                    'z': point_world[2],
                    'r': color[0],
                    'g': color[1],
                    'b': color[2],
                    'error': 0.0,  # No error estimate for now
                    'track': [(image_id, 0)]  # Track this point in this image
                }
                points3d.append(point_data)
        return points3d
    
    def _get_baseline(self):
        """Get stereo baseline from camera data"""
        # This should be extracted from your camera calibration
        # For now, using a default value
        return 0.051  # 12cm baseline, adjust based on your camera
    
    def _get_point_color(self, u, v, timestamp):
        """Get color for a 3D point from the image"""
        # For now, return a default color
        # In a full implementation, you would read the actual image pixel
        image = self.images[str(timestamp)]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        color = image_rgb[v, u]
        assert color.shape == (3,), f"Color shape is {color.shape}, expected (3,)"
        return color
    
    def _extract_images_from_db(self):
        """Extract images from tinynav database using shelve"""
        
        images_dir = self.output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        # Extract images from shelve database
        for timestamp, image in tqdm(self.images.items(), desc="Extracting images"):
            # Convert key to string if needed
            key_str = str(timestamp)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image_path = images_dir / f"image_{key_str}.png"
            cv2.imwrite(str(image_path), rgb_image)
        print("Image extraction completed")
    
    def convert(self):
        """Convert tinynav data to COLMAP format"""
        print("Converting tinynav data to COLMAP format...")
        
        # Create sparse/0 directory structure
        sparse_dir = self.output_dir / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        
        # Create COLMAP text files in sparse/0
        self._create_colmap_text_files(sparse_dir)
        
        # Extract images (if database is available)
        self._extract_images_from_db()
        
        print(f"Conversion complete! Output saved to {self.output_dir}")
        print("\nCOLMAP files created:")
        print(f"  - {sparse_dir}/cameras.txt")
        print(f"  - {sparse_dir}/images.txt")
        print(f"  - {sparse_dir}/points3D.txt")
        print(f"  - {self.output_dir}/pointcloud.ply")
        print(f"  - {self.output_dir}/images/ (if images were extracted)")




def main():
    parser = argparse.ArgumentParser(description="Convert tinynav data to COLMAP format")
    parser.add_argument("--input_dir", type=str, default="tinynav_db",
                       help="Input directory containing tinynav data")
    parser.add_argument("--output_dir", type=str, default="colmap_output",
                       help="Output directory for COLMAP files")
    args = parser.parse_args()
    # Convert tinynav data to COLMAP format
    converter = TinynavToColmapConverter(args.input_dir, args.output_dir)
    converter.convert()

if __name__ == "__main__":
    main()


