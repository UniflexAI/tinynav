import tyro
import numpy as np
import os
import cv2
import asyncio
from tinynav.tinynav_cpp_bind import ba_solve_depth_mahalanobis
from typing import Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from tinynav.core.build_map_node import TinyNavDB
from tinynav.core.models_trt import LightGlueTRT, SuperPointTRT


extractor = SuperPointTRT()
matcher = LightGlueTRT()

class DisjointSet:
    def __init__(self):
        self.parent: Dict[tuple[int, int], tuple[int, int]] = {}
        self.rank: Dict[tuple[int, int], int] = {}

    def add(self, x: tuple[int, int]) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: tuple[int, int]) -> tuple[int, int]:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: tuple[int, int], b: tuple[int, int]) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

def multiview_triangulation(keypoints_list: List[np.ndarray], camera_poses: List[np.ndarray], K: np.ndarray) -> Tuple[bool, np.ndarray]:
    """
    Triangulate 3D point from multiple views using DLT method.
    
    Args:
        keypoints_list: List of 2D keypoints (N, 2) for each view
        camera_poses: List of 4x4 camera poses for each view. from camera to world.
        K: Camera intrinsics matrix (3, 3)
        
    Returns:
        point_3d: 3D point (3,)
    """
    if len(keypoints_list) < 2:
        return False, np.zeros(3)
    
    # Build the DLT matrix
    A = []
    for i, (kp, pose) in enumerate(zip(keypoints_list, camera_poses)):
        pose_inv = np.linalg.inv(pose)
        # Get projection matrix P = K * [R|t]
        P = K @ pose_inv[:3, :4]
        
        # Normalize homogeneous coordinates
        x, y = kp[0], kp[1]
        
        # Add two rows to A matrix for each view
        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])
    
    A = np.array(A)
    
    # Solve using SVD
    U, S, Vt = np.linalg.svd(A)
    point_4d = Vt[-1, :]
    
    # Convert to 3D coordinates
    point_3d = point_4d[:3] / point_4d[3]

    # depth in each camera should be positive
    for pose in camera_poses:
        pose_inv = np.linalg.inv(pose)
        point_in_camera = pose_inv[:3, :3] @ point_3d + pose_inv[:3, 3]
        if point_in_camera[2] <= 0.1:
            return False, point_3d
    return True, point_3d


@dataclass
class Landmark:
    """Represents a 3D landmark point with tracking information."""
    def __init__(self, position_3d: np.ndarray, triangulated: bool):
        self.position_3d = position_3d
        self.triangulated = triangulated

class LandmarkTracker:
    """
    Tracks landmarks incrementally across image sequences using feature matching.
    """
    
    def __init__(self, min_track_length: int = 3, max_reprojection_error: float = 2.0):
        """
        Initialize the landmark tracker.
        
        Args:
            min_track_length: Minimum number of frames a landmark must be tracked
            max_reprojection_error: Maximum reprojection error for valid landmarks
        """
        # landmark_id -> Landmark
        self.landmarks: Dict[int, Landmark] = {}

        # next landmark id to assign
        self.next_landmark_id = 0
        self.min_track_length = min_track_length
        self.max_reprojection_error = max_reprojection_error

        # frame_id -> {kp_idx -> landmark_id}
        self.frame_landmarks: Dict[int, Dict[int, int]] = {}  # frame_id -> {kp_idx -> landmark_id}

        # landmark_id -> {frame_id -> kp_idx}
        self.landmark_observations: Dict[int, Dict[int, int]] = {}  # landmark_id -> {frame_id -> kp_idx}
        # landmark_id -> {timestamp -> kp_idx -> keypoint}
        self.landmark_keypoints: Dict[int, Dict[int, Dict[int, np.ndarray]]] = {}
        # landmark_id -> {timestamp -> kp_idx -> descriptor}
        self.landmark_descriptors: Dict[int, Dict[int, Dict[int, np.ndarray]]] = {}

    def _get_landmark_id_or_generate(self, image_timestamp: int, keypoint_index:int, keypoint: np.ndarray, descriptor: np.ndarray) -> int:
        """
        Get the landmark ID for a given image timestamp and keypoint index.
        If no landmark is found, generate a new one.
        """
        if image_timestamp not in self.frame_landmarks:
            self.frame_landmarks[image_timestamp] = {}

        if keypoint_index not in self.frame_landmarks[image_timestamp]: 
            self.frame_landmarks[image_timestamp][keypoint_index] = self.next_landmark_id
            self.landmarks[self.next_landmark_id] = Landmark(np.zeros(3), False)
            self.landmark_observations[self.next_landmark_id] = {image_timestamp: keypoint_index}
            self.landmark_keypoints[self.next_landmark_id] = {image_timestamp: {keypoint_index: keypoint}}
            self.landmark_descriptors[self.next_landmark_id] = {image_timestamp: {keypoint_index: descriptor}}
            self.next_landmark_id += 1

        return self.frame_landmarks[image_timestamp][keypoint_index]

        
    def add_matched_frame(self, timestamp0: int, timestamp1:int, keypoints0: np.ndarray, keypoints1: np.ndarray, descriptors0: np.ndarray, descriptors1: np.ndarray, matches: np.ndarray):
        """
        Add a matched frame to the landmark tracker.
        Args:
            timestamp0: Timestamp of first frame
            timestamp1: Timestamp of second frame
            keypoints0: Matched keypoints from first frame (N, 2)
            keypoints1: Matched keypoints from second frame (N, 2)
            matches: Match indices (N, 2) where matches[i] = [idx0, idx1]
        """
        for match in matches:
            kp0_idx, kp1_idx = match[0], match[1]
            # Check if keypoints are already associated with landmarks
            landmark_id0 = self._get_landmark_id_or_generate(timestamp0, kp0_idx, keypoints0[kp0_idx, :], descriptors0[kp0_idx, :])
            landmark_id1 = self._get_landmark_id_or_generate(timestamp1, kp1_idx, keypoints1[kp1_idx, :], descriptors1[kp1_idx, :])

            if landmark_id0 != landmark_id1:
                self._merge_landmarks(landmark_id0, landmark_id1)
            else:
                # else: already the same landmark, nothing to do
                pass
                # self.landmark_keypoints[landmark_id0][timestamp0][kp0_idx] = keypoints0[kp0_idx, :]
                # self.landmark_keypoints[landmark_id1][timestamp1][kp1_idx] = keypoints1[kp1_idx, :]
                # self.landmark_descriptors[landmark_id0][timestamp0][kp0_idx] = descriptors0[kp0_idx, :]
                # self.landmark_descriptors[landmark_id1][timestamp1][kp1_idx] = descriptors1[kp1_idx, :]

    def observation_relations_for_ba(self) -> List[Tuple[int, int, np.ndarray]]:
        '''
        Get the observation relations for bundle adjustment.
        The relations are used to construct the observation matrix for bundle adjustment.
        The observation matrix is a sparse matrix, where each row corresponds to a landmark,
        and each column corresponds to a camera pose.
        The value of the observation matrix is the keypoint observation.
        The observation matrix is used to solve the bundle adjustment problem.
        
        Returns:
            relations: List of tuples (timestamp, landmark_id, keypoint)
        '''
        relations = []
        for landmark_id in self.landmarks:
            landmark = self.landmarks[landmark_id]
            if landmark.triangulated:
                for timestamp, kp_idx in self.landmark_observations[landmark_id].items():
                    keypoint = self.landmark_keypoints[landmark_id][timestamp][kp_idx]
                    relations.append((timestamp, landmark_id, keypoint))
        return relations

    def get_landmark_point3ds(self) -> Dict[int, np.ndarray]:
        '''
        Get the 3D points of the landmarks.
        '''
        return {landmark_id: landmark.position_3d for landmark_id, landmark in self.landmarks.items() if landmark.triangulated}

    def _merge_landmarks(self, landmark_id0: int, landmark_id1: int):
        """
        Merge two landmarks into one.
        """
        if landmark_id0 == landmark_id1:
            return

        # Check if both landmarks still exist
        if landmark_id0 not in self.landmark_observations or landmark_id1 not in self.landmark_observations:
            # One or both landmarks have already been merged, skip this merge
            return

        # Choose the smaller ID as the target to keep
        target_id = min(landmark_id0, landmark_id1)
        source_id = max(landmark_id0, landmark_id1)

        # Only change the source landmark ID to the target ID
        self._change_landmark_id(source_id, target_id)

    def _change_landmark_id(self, landmark_id: int, new_landmark_id: int):
        """
        Change the landmark ID of a landmark, merging all observations and updating references.
        """
        if landmark_id == new_landmark_id:
            return

        # Merge observations
        obs_from = self.landmark_observations[landmark_id]
        obs_to = self.landmark_observations.get(new_landmark_id, {})
        merged_obs = obs_to.copy()
        merged_obs.update(obs_from)

        self.landmark_observations[new_landmark_id] = merged_obs
        del self.landmark_observations[landmark_id]

        #
        # Update frame_landmarks to point to new_landmark_id
        # if two landmark containers the same frame_id, delete it since it's not a stable observation.
        #
        for frame_id_from, kp_idx_from in obs_from.items():
            if frame_id_from not in obs_to:
                self.frame_landmarks[frame_id_from][kp_idx_from] = new_landmark_id
                # Ensure the sub-dictionaries exist
                if frame_id_from not in self.landmark_keypoints[new_landmark_id]:
                    self.landmark_keypoints[new_landmark_id][frame_id_from] = {}

                if frame_id_from in self.landmark_keypoints[landmark_id] and kp_idx_from in self.landmark_keypoints[landmark_id][frame_id_from]:
                    self.landmark_keypoints[new_landmark_id][frame_id_from][kp_idx_from] = self.landmark_keypoints[landmark_id][frame_id_from][kp_idx_from]
                    del self.landmark_keypoints[landmark_id][frame_id_from][kp_idx_from]
                    # Clean up empty sub-dicts
                    if not self.landmark_keypoints[landmark_id][frame_id_from]:
                        del self.landmark_keypoints[landmark_id][frame_id_from]

                if frame_id_from not in self.landmark_descriptors[new_landmark_id]:
                    self.landmark_descriptors[new_landmark_id][frame_id_from] = {}

                if frame_id_from in self.landmark_descriptors[landmark_id] and kp_idx_from in self.landmark_descriptors[landmark_id][frame_id_from]:
                    self.landmark_descriptors[new_landmark_id][frame_id_from][kp_idx_from] = self.landmark_descriptors[landmark_id][frame_id_from][kp_idx_from]
                    del self.landmark_descriptors[landmark_id][frame_id_from][kp_idx_from]
                    # Clean up empty sub-dicts
                    if not self.landmark_descriptors[landmark_id][frame_id_from]:
                        del self.landmark_descriptors[landmark_id][frame_id_from]
            else:
                frame_id_to = frame_id_from
                kp_idx_to = obs_to[frame_id_from]
                del self.frame_landmarks[frame_id_from][kp_idx_from]
                del self.frame_landmarks[frame_id_from][kp_idx_to]
                del self.landmark_observations[new_landmark_id][frame_id_to]
                del self.landmark_keypoints[new_landmark_id][frame_id_to]
                del self.landmark_descriptors[new_landmark_id][frame_id_to]

        # Clean up old landmark_id if empty
        if landmark_id in self.landmark_keypoints and not self.landmark_keypoints[landmark_id]:
            del self.landmark_keypoints[landmark_id]
        if landmark_id in self.landmark_descriptors and not self.landmark_descriptors[landmark_id]:
            del self.landmark_descriptors[landmark_id]

        # Optionally, merge 3D position/keypoint (keep the one with more observations or just keep new_landmark_id's)
        # Here, we keep the one with more observations
        if new_landmark_id in self.landmarks and landmark_id in self.landmarks:
            if len(merged_obs) >= 2:
                # Prefer the one with more observations
                self.landmarks[new_landmark_id] = self.landmarks[new_landmark_id]
            else:
                self.landmarks[new_landmark_id] = self.landmarks[landmark_id]
        elif landmark_id in self.landmarks:
            self.landmarks[new_landmark_id] = self.landmarks[landmark_id]
        # Remove the old landmark
        if landmark_id in self.landmarks:
            del self.landmarks[landmark_id]

    def remove_observations(self, timestamp: int, kp_idx: int, landmark_id: int):
        assert landmark_id in self.landmarks
        observations = self.landmark_observations[landmark_id]
        del observations[timestamp]
        if len(observations) < 2:
            self.landmarks[landmark_id].triangulated = False

        del self.landmark_keypoints[landmark_id][timestamp][kp_idx]
        del self.landmark_descriptors[landmark_id][timestamp][kp_idx]
        del self.frame_landmarks[timestamp][kp_idx]

    def triangulate_landmarks(self, camera_poses: Dict[int, np.ndarray], K: np.ndarray):
        '''
        Triangulate the landmarks.
        '''

        for landmark_id, landmark in self.landmarks.items():
            if not landmark.triangulated:
                if len(self.landmark_observations[landmark_id]) >= 2:
                    observations = self.landmark_observations[landmark_id]
                    keypoints = []
                    camera_poses_list = []
                    for timestamp, kp_idx in observations.items():
                        keypoint = self.landmark_keypoints[landmark_id][timestamp][kp_idx]
                        camera_pose = camera_poses[timestamp]
                        keypoints.append(keypoint)
                        camera_poses_list.append(camera_pose)
                    success, position_3d = multiview_triangulation(keypoints, camera_poses_list, K)
                    if success:
                        self.landmarks[landmark_id].position_3d = position_3d
                        self.landmarks[landmark_id].triangulated = True


def solve_bundle_adjustment_depth_mahalanobis(
    points_3d: Dict[int, np.ndarray],
    observations: List[Tuple[int, int, np.ndarray, np.ndarray]],
    camera_poses: Dict[int, np.ndarray],
    intrinsics: np.ndarray,
    constant_pose_index: Dict[int, bool] = None,
    relative_pose_constraints: List[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = None,
):
        if constant_pose_index is not None:
            py_constant_pose_index = {timestamp: is_constant for timestamp, is_constant in constant_pose_index.items()}
        else:
            py_constant_pose_index = {}

        if relative_pose_constraints is not None:
            py_relative_pose_constraints = []
            for cam_idx_i, cam_idx_j, relative_pose_j_i, translation_weight, rotation_weight in relative_pose_constraints:
                py_relative_pose_constraints.append((cam_idx_i, cam_idx_j, relative_pose_j_i, translation_weight, rotation_weight))
        else:
            py_relative_pose_constraints = []

        optimized_camera_poses, optimized_points_3d = ba_solve_depth_mahalanobis(
            camera_poses,
            points_3d,
            observations,
            intrinsics,
            py_constant_pose_index,
            py_relative_pose_constraints,
        )
        return optimized_camera_poses, optimized_points_3d


def _unproject_depth_with_cov(
    u: float,
    v: float,
    depth: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    sigma_u: float,
    sigma_v: float,
    sigma_disparity: float,
    baseline: float,
) -> tuple[np.ndarray, np.ndarray]:
    if depth <= 0.0:
        raise ValueError("depth must be > 0")
    if baseline <= 0.0:
        raise ValueError("baseline must be > 0")
    sigma_z = (depth * depth / (fx * baseline)) * sigma_disparity
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    jac = np.array(
        [
            [depth / fx, 0.0, (u - cx) / fx],
            [0.0, depth / fy, (v - cy) / fy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    cov_uvz = np.diag([sigma_u * sigma_u, sigma_v * sigma_v, sigma_z * sigma_z]).astype(np.float64)
    cov_xyz = jac @ cov_uvz @ jac.T
    cov_xyz += np.eye(3, dtype=np.float64) * 1e-9
    sqrt_info = np.linalg.cholesky(np.linalg.inv(cov_xyz))
    return np.array([x, y, z], dtype=np.float64), sqrt_info.astype(np.float64)

def project_point_to_image(point_3d, pose_in_world, intrinsics):
    point_in_world = np.hstack((point_3d, 1))
    point_in_camera = np.linalg.inv(pose_in_world) @ point_in_world
    x, y, z, _ = point_in_camera
    u = int(intrinsics[0, 0] * x / z + intrinsics[0, 2])
    v = int(intrinsics[1, 1] * y / z + intrinsics[1, 2])
    return u, v

def match_images_with_trt(image0: np.ndarray, image1: np.ndarray, image_shape: np.ndarray):
    if image0.ndim != 2 or image1.ndim != 2:
        raise ValueError(
            f"SuperPointTRT expects grayscale HxW images, got image0.shape={image0.shape}, image1.shape={image1.shape}"
        )
    feats0 = asyncio.run(extractor.infer(image0))
    feats1 = asyncio.run(extractor.infer(image1))
    match_result = asyncio.run(
        matcher.infer(
            feats0["kpts"],
            feats1["kpts"],
            feats0["descps"],
            feats1["descps"],
            feats0["mask"],
            feats1["mask"],
            image_shape,
            image_shape,
        )
    )
    keypoints0 = feats0["kpts"][0]
    keypoints1 = feats1["kpts"][0]
    match_indices = match_result["match_indices"][0]
    matches = np.array([[i, idx] for i, idx in enumerate(match_indices) if idx != -1], dtype=np.int64)
    return keypoints0, keypoints1, matches, feats0["descps"][0], feats1["descps"][0]

def draw_image(image_left, image_right, keypoints0, keypoints1, matches):
    cv_matches = [cv2.DMatch(_queryIdx=matches[index, 0].item(), _trainIdx=matches[index, 1].item(), _imgIdx=0, _distance=0) for index in range(matches.shape[0])]
    cv_kpts_prev = [cv2.KeyPoint(x=keypoints0[index, 0].item(), y=keypoints0[index, 1].item(), size=20) for index in range(keypoints0.shape[0])]
    cv_kpts_curr = [cv2.KeyPoint(x=keypoints1[index, 0].item(), y=keypoints1[index, 1].item(), size=20) for index in range(keypoints1.shape[0])]
    output_image = cv2.drawMatches(image_left, cv_kpts_prev, image_right, cv_kpts_curr, cv_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return output_image

def main(
    tinynav_db_path: str,
    output_dir: str | None = None,
    save_top_rotation_match_count: int = 20,
    only_prev_timestamp: int | None = None,
    only_curr_timestamp: int | None = None,
    save_reprojection_visualization: bool = True,
    save_reprojection_max_frames: int = -1,
    save_match_pair_visualization: bool = True,
):
    if output_dir is None:
        output_dir = f"{tinynav_db_path.rstrip('/')}_ba_refined"
    os.makedirs(output_dir, exist_ok=True)

    infra1_poses = np.load(os.path.join(tinynav_db_path, "poses.npy"), allow_pickle=True).item()
    db = TinyNavDB(tinynav_db_path, is_scratch=False)
    infra1_intrinsics = np.load(os.path.join(tinynav_db_path, "intrinsics.npy"), allow_pickle=True)
    baseline = float(np.load(os.path.join(tinynav_db_path, "baseline.npy"), allow_pickle=True))
    fx = float(infra1_intrinsics[0, 0])
    fy = float(infra1_intrinsics[1, 1])
    cx = float(infra1_intrinsics[0, 2])
    cy = float(infra1_intrinsics[1, 2])
    sigma_u = 1.0
    sigma_v = 1.0
    sigma_disparity = 1.0

    sorted_timestamps = sorted(infra1_poses.keys())
    edges = list(zip(sorted_timestamps[:-1], sorted_timestamps[1:]))
    if only_prev_timestamp is not None or only_curr_timestamp is not None:
        if only_prev_timestamp is None or only_curr_timestamp is None:
            raise ValueError("Both --only-prev-timestamp and --only-curr-timestamp must be set together")
        edges = [(int(only_prev_timestamp), int(only_curr_timestamp))]
        print(f"two-frame mode: edges={edges}")
    min_timestamp = min(infra1_poses.keys())
    ds = DisjointSet()
    node_obs_cam: Dict[tuple[int, int], np.ndarray] = {}
    node_sqrt_info: Dict[tuple[int, int], np.ndarray] = {}
    node_valid = set()
    frame_kpts: Dict[int, np.ndarray] = {}
    frame_dropped_uv: Dict[int, List[Tuple[np.ndarray, str, float | None]]] = {}
    match_pair_dir = os.path.join(output_dir, "match_pairs")
    if save_match_pair_visualization:
        os.makedirs(match_pair_dir, exist_ok=True)

    for prev_timestamp, curr_timestamp in tqdm(edges, desc="Process edges", unit="edge"):
        if prev_timestamp not in infra1_poses or curr_timestamp not in infra1_poses:
            continue
        prev_depth, _, _, _, prev_infra1_loader = db.get_depth_embedding_features_images(prev_timestamp)
        curr_depth, _, _, _, curr_infra1_loader = db.get_depth_embedding_features_images(curr_timestamp)
        prev_infra1_image = prev_infra1_loader()
        curr_infra1_image = curr_infra1_loader()
        if prev_infra1_image is None or curr_infra1_image is None or prev_depth is None or curr_depth is None:
            continue
        image_shape = np.array([prev_infra1_image.shape[1], prev_infra1_image.shape[0]], dtype=np.int64)
        keypoints0, keypoints1, matches, _, _ = match_images_with_trt(
            prev_infra1_image,
            curr_infra1_image,
            image_shape,
        )
        if save_match_pair_visualization and matches.shape[0] > 0:
            pair_vis = draw_image(prev_infra1_image, curr_infra1_image, keypoints0, keypoints1, matches)
            pair_name = (
                f"prev_{int(prev_timestamp)}_curr_{int(curr_timestamp)}_matches_{int(matches.shape[0])}.png"
            )
            cv2.imwrite(os.path.join(match_pair_dir, pair_name), pair_vis)
        frame_kpts[int(prev_timestamp)] = np.asarray(keypoints0)
        frame_kpts[int(curr_timestamp)] = np.asarray(keypoints1)

        for i0, i1 in matches:
            u0, v0 = keypoints0[i0]
            u1, v1 = keypoints1[i1]
            x0 = int(round(float(u0)))
            y0 = int(round(float(v0)))
            x1 = int(round(float(u1)))
            y1 = int(round(float(v1)))
            if y0 < 0 or y0 >= prev_depth.shape[0] or x0 < 0 or x0 >= prev_depth.shape[1]:
                frame_dropped_uv.setdefault(int(prev_timestamp), []).append((np.array([u0, v0], dtype=np.float64), "oob_prev", None))
                continue
            if y1 < 0 or y1 >= curr_depth.shape[0] or x1 < 0 or x1 >= curr_depth.shape[1]:
                frame_dropped_uv.setdefault(int(curr_timestamp), []).append((np.array([u1, v1], dtype=np.float64), "oob_curr", None))
                continue
            z0 = float(prev_depth[y0, x0])
            z1 = float(curr_depth[y1, x1])
            if not np.isfinite(z0) or not np.isfinite(z1) or z0 <= 0.0 or z1 <= 0.0:
                if not np.isfinite(z0) or z0 <= 0.0:
                    frame_dropped_uv.setdefault(int(prev_timestamp), []).append((np.array([u0, v0], dtype=np.float64), "invalid_prev_depth", z0))
                if not np.isfinite(z1) or z1 <= 0.0:
                    frame_dropped_uv.setdefault(int(curr_timestamp), []).append((np.array([u1, v1], dtype=np.float64), "invalid_curr_depth", z1))
                continue
            prev_point_cam, prev_sqrt_info = _unproject_depth_with_cov(
                float(u0), float(v0), z0, fx, fy, cx, cy, sigma_u, sigma_v, sigma_disparity, baseline
            )
            curr_obs_point_cam, curr_sqrt_info = _unproject_depth_with_cov(
                float(u1), float(v1), z1, fx, fy, cx, cy, sigma_u, sigma_v, sigma_disparity, baseline
            )
            n0 = (int(prev_timestamp), int(i0))
            n1 = (int(curr_timestamp), int(i1))
            ds.add(n0)
            ds.add(n1)
            ds.union(n0, n1)
            node_obs_cam[n0] = prev_point_cam
            node_sqrt_info[n0] = prev_sqrt_info
            node_obs_cam[n1] = curr_obs_point_cam
            node_sqrt_info[n1] = curr_sqrt_info
            node_valid.add(n0)
            node_valid.add(n1)

    clusters: Dict[tuple[int, int], List[tuple[int, int]]] = {}
    for node in node_valid:
        root = ds.find(node)
        clusters.setdefault(root, []).append(node)

    point_3ds: Dict[int, np.ndarray] = {}
    observations: List[Tuple[int, int, np.ndarray, np.ndarray]] = []
    point_id = 0
    landmark_feature_uv: Dict[int, Dict[int, np.ndarray]] = {}
    for nodes in clusters.values():
        if len(nodes) < 2:
            continue
        timestamps = [n[0] for n in nodes]
        if len(set(timestamps)) < 2:
            continue
        world_points = []
        for ts, kp_idx in nodes:
            pose = infra1_poses[ts]
            world_points.append(pose[:3, :3] @ node_obs_cam[(ts, kp_idx)] + pose[:3, 3])
        if len(world_points) < 2:
            continue
        landmark_world = np.mean(np.asarray(world_points, dtype=np.float64), axis=0)
        point_3ds[point_id] = landmark_world.astype(np.float64)
        for ts, kp_idx in nodes:
            node = (ts, kp_idx)
            observations.append((int(ts), int(point_id), node_obs_cam[node], node_sqrt_info[node]))
            kpts = frame_kpts[int(ts)]
            uv = np.asarray(kpts[int(kp_idx)], dtype=np.float64)
            if point_id not in landmark_feature_uv:
                landmark_feature_uv[point_id] = {}
            landmark_feature_uv[point_id][int(ts)] = uv
        point_id += 1

    print(f"num_points={len(point_3ds)}, num_observations={len(observations)}")
    if len(point_3ds) == 0 or len(observations) == 0:
        raise RuntimeError("No valid points/observations collected")
    optimized_camera_poses, optimized_points_3d = solve_bundle_adjustment_depth_mahalanobis(
        point_3ds,
        observations,
        infra1_poses,
        infra1_intrinsics,
        constant_pose_index={int(min_timestamp): True},
    )

    delta_translation_list = []
    delta_rotation_list = []
    delta_by_timestamp = []
    for timestamp, optimized_pose in optimized_camera_poses.items():
        delta = np.linalg.inv(optimized_pose) @ infra1_poses[timestamp]
        delta_translation = delta[:3, 3]
        delta_rotation = delta[:3, :3]
        cos_theta = (np.trace(delta_rotation) - 1) / 2
        r_diff = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
        delta_translation_list.append(np.linalg.norm(delta_translation))
        delta_rotation_list.append(r_diff)
        delta_by_timestamp.append((int(timestamp), float(np.linalg.norm(delta_translation)), float(r_diff)))
    mean_delta_translation = np.mean(delta_translation_list)
    mean_delta_rotation = np.mean(delta_rotation_list)
    print(f"mean_delta_translation: {mean_delta_translation}, mean_delta_rotation: {mean_delta_rotation}")

    if save_top_rotation_match_count > 0:
        match_dir = os.path.join(output_dir, "largest_rotation_matches")
        os.makedirs(match_dir, exist_ok=True)
        edge_set = {(int(a), int(b)) for a, b in edges}
        sorted_delta = sorted(delta_by_timestamp, key=lambda x: x[2], reverse=True)
        saved = 0
        for curr_timestamp, _, rot_deg in sorted_delta:
            prev_timestamp = curr_timestamp - 1
            # pick nearest sequential edge ending at curr_timestamp
            candidates = [e for e in edge_set if e[1] == curr_timestamp]
            if not candidates:
                continue
            prev_timestamp = max(candidates, key=lambda e: e[0])[0]
            prev_depth, _, _, _, prev_loader = db.get_depth_embedding_features_images(prev_timestamp)
            curr_depth, _, _, _, curr_loader = db.get_depth_embedding_features_images(curr_timestamp)
            prev_img = prev_loader()
            curr_img = curr_loader()
            if prev_img is None or curr_img is None or prev_depth is None or curr_depth is None:
                continue
            image_shape = np.array([prev_img.shape[1], prev_img.shape[0]], dtype=np.int64)
            k0, k1, m, _, _ = match_images_with_trt(prev_img, curr_img, image_shape)
            if m.shape[0] == 0:
                continue
            vis = draw_image(prev_img, curr_img, k0, k1, m)
            save_name = (
                f"rank_{saved:03d}_curr_{curr_timestamp}_prev_{prev_timestamp}"
                f"_rot_{rot_deg:.3f}_matches_{m.shape[0]}.png"
            )
            cv2.imwrite(os.path.join(match_dir, save_name), vis)
            saved += 1
            if saved >= save_top_rotation_match_count:
                break
        print(f"saved {saved} match visualization files to {match_dir}")

    optimized_infra1_poses = {timestamp: optimized_pose for timestamp, optimized_pose in optimized_camera_poses.items()}
    np.save(os.path.join(output_dir, "poses.npy"), optimized_infra1_poses, allow_pickle=True)
    np.save(os.path.join(output_dir, "intrinsics.npy"), infra1_intrinsics, allow_pickle=True)
    print(f"save refined poses to {os.path.join(output_dir, 'poses.npy')}")
    if save_match_pair_visualization:
        print(f"saved match pair visualizations to {match_pair_dir}")

    if save_reprojection_visualization:
        reproj_dir = os.path.join(output_dir, "reprojection_compare")
        os.makedirs(reproj_dir, exist_ok=True)
        # Collect valid depth range for depth-colored dropped markers.
        valid_depth_values = []
        for dropped_items in frame_dropped_uv.values():
            for _, reason, depth_value in dropped_items:
                if reason.startswith("invalid_") and depth_value is not None and np.isfinite(depth_value) and depth_value > 0.0:
                    valid_depth_values.append(float(depth_value))
        depth_min = float(np.min(valid_depth_values)) if len(valid_depth_values) > 0 else 0.1
        depth_max = float(np.max(valid_depth_values)) if len(valid_depth_values) > 0 else 10.0
        if depth_max <= depth_min:
            depth_max = depth_min + 1.0

        def _depth_to_bgr(depth_value: float) -> tuple[int, int, int]:
            n = np.clip((depth_value - depth_min) / (depth_max - depth_min), 0.0, 1.0)
            # Blue (near) -> Red (far) in BGR.
            b = int((1.0 - n) * 255.0)
            g = int((1.0 - abs(2.0 * n - 1.0)) * 255.0)
            r = int(n * 255.0)
            return (b, g, r)
        by_frame: Dict[int, List[Tuple[np.ndarray, np.ndarray, int]]] = {}
        for landmark_id, frame_uvs in landmark_feature_uv.items():
            if landmark_id not in optimized_points_3d:
                continue
            landmark_world = np.asarray(optimized_points_3d[landmark_id], dtype=np.float64)
            for ts, feat_uv in frame_uvs.items():
                if ts not in optimized_infra1_poses:
                    continue
                pose = np.asarray(optimized_infra1_poses[ts], dtype=np.float64)
                proj_u, proj_v = project_point_to_image(landmark_world, pose, infra1_intrinsics)
                by_frame.setdefault(int(ts), []).append(
                    (np.asarray(feat_uv, dtype=np.float64), np.array([proj_u, proj_v], dtype=np.float64), int(landmark_id))
                )

        frame_timestamps = sorted(by_frame.keys())
        if save_reprojection_max_frames > 0:
            frame_timestamps = frame_timestamps[:save_reprojection_max_frames]
        saved_frames = 0
        for ts in frame_timestamps:
            _, _, _, _, img_loader = db.get_depth_embedding_features_images(ts)
            img = img_loader()
            if img is None:
                continue
            if img.ndim == 2:
                vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                vis = img.copy()
            h, w = vis.shape[:2]
            for feat_uv, proj_uv, _ in by_frame[ts]:
                fu, fv = int(round(float(feat_uv[0]))), int(round(float(feat_uv[1])))
                pu, pv = int(round(float(proj_uv[0]))), int(round(float(proj_uv[1])))
                if 0 <= fu < w and 0 <= fv < h:
                    cv2.circle(vis, (fu, fv), 2, (255, 0, 0), -1)  # original feature (blue)
                if 0 <= pu < w and 0 <= pv < h:
                    cv2.circle(vis, (pu, pv), 2, (0, 255, 0), -1)  # reprojected landmark (green)
                if 0 <= fu < w and 0 <= fv < h and 0 <= pu < w and 0 <= pv < h:
                    cv2.line(vis, (fu, fv), (pu, pv), (0, 0, 255), 1)  # residual vector (red)
            for drop_uv, drop_reason, drop_depth in frame_dropped_uv.get(int(ts), []):
                du = int(round(float(drop_uv[0])))
                dv = int(round(float(drop_uv[1])))
                if 0 <= du < w and 0 <= dv < h:
                    if drop_reason.startswith("oob_"):
                        marker_color = (255, 0, 255)  # magenta: out-of-bounds
                    elif drop_depth is None or not np.isfinite(drop_depth) or drop_depth <= 0.0:
                        marker_color = (0, 255, 255)  # yellow: invalid/zero/nan depth
                    else:
                        marker_color = _depth_to_bgr(float(drop_depth))  # depth-coded color
                    cv2.drawMarker(
                        vis,
                        (du, dv),
                        marker_color,
                        markerType=cv2.MARKER_TILTED_CROSS,
                        markerSize=8,
                        thickness=1,
                    )
            out_path = os.path.join(reproj_dir, f"{ts}.png")
            cv2.imwrite(out_path, vis)
            saved_frames += 1
        print(f"saved {saved_frames} reprojection visualization frames to {reproj_dir}")

    db.close()

if __name__ == "__main__":
    tyro.cli(main)
