import torch
import tyro
import numpy as np
import shelve
import os
import cv2
import einops
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from tqdm import tqdm
from tinynav.tinynav_cpp_bind import ba_solve
from typing import Dict, List, Tuple
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import json
import os


extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher

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


def solve_bundle_adjustment(points_3d: Dict[int, np.ndarray], 
                                observations: List[Tuple[int, int, np.ndarray]], 
                                camera_poses: Dict[int, np.ndarray], 
                                intrinsics: np.ndarray,
                                constant_pose_index: Dict[int, bool] = None,
                                relative_pose_constraints: List[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = None):
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

        optimized_camera_poses, optimized_points_3d = ba_solve(camera_poses, points_3d, observations, intrinsics, py_constant_pose_index, py_relative_pose_constraints)
        return optimized_camera_poses, optimized_points_3d

def project_point_to_image(point_3d, pose_in_world, intrinsics):
    point_in_world = np.hstack((point_3d, 1))
    point_in_camera = np.linalg.inv(pose_in_world) @ point_in_world
    x, y, z, _ = point_in_camera
    u = int(intrinsics[0, 0] * x / z + intrinsics[0, 2])
    v = int(intrinsics[1, 1] * y / z + intrinsics[1, 2])
    return u, v

def match_images(image0, image1):
    image0 = einops.rearrange(image0, "h w c -> c h w")
    image1 = einops.rearrange(image1, "h w c -> c h w")
    image0 = (torch.from_numpy(image0) / 255.0).cuda()
    image1 = (torch.from_numpy(image1) / 255.0).cuda()
    feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
    feats1 = extractor.extract(image1)
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
    matches = matches01['matches']  # indices with shape (K,2)
    points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
    points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
    descriptors0 = feats0['descriptors']
    descriptors1 = feats1['descriptors']
    return feats0['keypoints'], feats1['keypoints'], matches, descriptors0, descriptors1

def draw_image(image_left, image_right, keypoints0, keypoints1, matches):
    cv_matches = [cv2.DMatch(_queryIdx=matches[index, 0].item(), _trainIdx=matches[index, 1].item(), _imgIdx=0, _distance=0) for index in range(matches.shape[0])]
    cv_kpts_prev = [cv2.KeyPoint(x=keypoints0[index, 0].item(), y=keypoints0[index, 1].item(), size=20) for index in range(keypoints0.shape[0])]
    cv_kpts_curr = [cv2.KeyPoint(x=keypoints1[index, 0].item(), y=keypoints1[index, 1].item(), size=20) for index in range(keypoints1.shape[0])]
    output_image = cv2.drawMatches(image_left, cv_kpts_prev, image_right, cv_kpts_curr, cv_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return output_image

def main(tinynav_db_path: str):
    infra1_poses = np.load(os.path.join(tinynav_db_path, "poses.npy"), allow_pickle=True).item()
    rgb_images = shelve.open(os.path.join(tinynav_db_path, "rgb_images"), flag='r')
    T_rgb_to_infra1 = np.load(os.path.join(tinynav_db_path, "T_rgb_to_infra1.npy"), allow_pickle=True)
    rgb_intrinsics = np.load(os.path.join(tinynav_db_path, "rgb_camera_intrinsics.npy"), allow_pickle=True)
    edges = np.load(os.path.join(tinynav_db_path, "edges.npy"), allow_pickle=True)
    min_timestamp = min(infra1_poses.keys())
    rgb_poses = {}
    for timestamp in infra1_poses.keys():
        rgb_poses[timestamp] = infra1_poses[timestamp] @ T_rgb_to_infra1
    keypoints_matches = {}
    landmark_tracker = LandmarkTracker()
    for prev_timestamp, curr_timestamp in edges:
        prev_rgb_image = rgb_images[str(prev_timestamp)]
        curr_rgb_image = rgb_images[str(curr_timestamp)]
        keypoints0, keypoints1, matches, descriptors0, descriptors1 = match_images(prev_rgb_image, curr_rgb_image)
        points0 = keypoints0[matches[..., 0]]  # coordinates in image #0, shape (K,2)
        points1 = keypoints1[matches[..., 1]]  # coordinates in image #1, shape (K,2)
        keypoints_matches[(prev_timestamp, curr_timestamp)] = (points0, points1, matches)
        landmark_tracker.add_matched_frame(prev_timestamp, curr_timestamp, keypoints0.detach().cpu().numpy(), keypoints1.detach().cpu().numpy(), descriptors0.detach().cpu().numpy(), descriptors1.detach().cpu().numpy(), matches.detach().cpu().numpy())
        landmark_tracker.triangulate_landmarks(rgb_poses, rgb_intrinsics)
    print(f"landmark_tracker.observation_relations_for_ba(): {len(landmark_tracker.observation_relations_for_ba())}")
    landmark_point3d = landmark_tracker.get_landmark_point3ds()
    optimized_camera_poses, optimized_points_3d = solve_bundle_adjustment(
        landmark_point3d,
        landmark_tracker.observation_relations_for_ba(),
        rgb_poses,
        rgb_intrinsics,
        constant_pose_index={timestamp: True for timestamp in rgb_poses.keys()})

    landmark_point3d = {landmark_id: optimized_points_3d[landmark_id] for landmark_id in optimized_points_3d.keys()}

    filter_landmark_count = 0
    for landmark_id, landmark_position in landmark_point3d.items():
        error = []
        for timestamp, kp_idx in landmark_tracker.landmark_observations[landmark_id].items():
            keypoint = landmark_tracker.landmark_keypoints[landmark_id][timestamp][kp_idx]
            projected_landmark = project_point_to_image(landmark_position,optimized_camera_poses[timestamp], rgb_intrinsics)
            error.append(np.linalg.norm(keypoint - projected_landmark))
        if len(error) > 0 and np.mean(np.array(error)) > 10:
            filter_landmark_count += 1
            landmark_tracker.landmarks[landmark_id].triangulated = False
            filter_landmark_count += 1

    print(f"filter_landmark_count: {filter_landmark_count}")
    optimized_camera_poses, optimized_points_3d = solve_bundle_adjustment(landmark_point3d, landmark_tracker.observation_relations_for_ba(), optimized_camera_poses, rgb_intrinsics, constant_pose_index={min_timestamp: True})

    delta_translation_list = []
    delta_rotation_list = []
    for timestamp, optimized_pose in optimized_camera_poses.items():
        delta = np.linalg.inv(optimized_pose) @ rgb_poses[timestamp]
        delta_translation = delta[:3, 3]
        delta_rotation = delta[:3, :3]
        cos_theta = (np.trace(delta_rotation) - 1) / 2
        r_diff = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
        delta_translation_list.append(np.linalg.norm(delta_translation))
        delta_rotation_list.append(r_diff)
    mean_delta_translation = np.mean(delta_translation_list)
    mean_delta_rotation = np.mean(delta_rotation_list)
    print(f"mean_delta_translation: {mean_delta_translation}, mean_delta_rotation: {mean_delta_rotation}")

    optimized_infra1_poses = {
        timestamp: optimized_pose @ np.linalg.inv(T_rgb_to_infra1) for timestamp, optimized_pose in optimized_camera_poses.items()
    }
    np.save(os.path.join(tinynav_db_path, "poses.npy"), optimized_infra1_poses, allow_pickle=True)
    print(f"save refined poses to {os.path.join(tinynav_db_path, 'poses.npy')}")

if __name__ == "__main__":
    tyro.cli(main)
