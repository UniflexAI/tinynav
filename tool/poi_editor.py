from __future__ import annotations

import time
from pathlib import Path
from typing import TypedDict
import shelve
from nav_msgs.msg import Odometry
import nav_msgs
import numpy as np
import numpy.typing as npt
import tyro
from plyfile import PlyData
import  viser.transforms as vtf
import viser
from viser import transforms as tf
import json
from rclpy.node import Node
import rclpy
import cv2
import os

class SplatFile(TypedDict):
    centers: npt.NDArray[np.floating]
    rgbs: npt.NDArray[np.floating]
    opacities: npt.NDArray[np.floating]
    covariances: npt.NDArray[np.floating]


def load_splat_file(splat_path: Path, center: bool = False) -> SplatFile:
    start_time = time.time()
    splat_buffer = splat_path.read_bytes()
    bytes_per_gaussian = (
        # Each Gaussian is serialized as:
        # - position (vec3, float32)
        3 * 4
        # - xyz (vec3, float32)
        + 3 * 4
        # - rgba (vec4, uint8)
        + 4
        # - ijkl (vec4, uint8), where 0 => -1, 255 => 1.
        + 4
    )
    assert len(splat_buffer) % bytes_per_gaussian == 0
    num_gaussians = len(splat_buffer) // bytes_per_gaussian

    # Reinterpret cast to dtypes that we want to extract.
    splat_uint8 = np.frombuffer(splat_buffer, dtype=np.uint8).reshape(
        (num_gaussians, bytes_per_gaussian)
    )
    scales = splat_uint8[:, 12:24].copy().view(np.float32)
    wxyzs = splat_uint8[:, 28:32] / 255.0 * 2.0 - 1.0
    Rs = tf.SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    centers = splat_uint8[:, 0:12].copy().view(np.float32)
    if center:
        centers -= np.mean(centers, axis=0, keepdims=True)
    print(
        f"Splat file with {num_gaussians=} loaded in {time.time() - start_time} seconds"
    )
    return {
        "centers": centers,
        # Colors should have shape (N, 3).
        "rgbs": splat_uint8[:, 24:27] / 255.0,
        "opacities": splat_uint8[:, 27:28] / 255.0,
        # Covariances should have shape (N, 3, 3).
        "covariances": covariances,
    }


def load_ply_file(ply_file_path: Path, center: bool = False) -> SplatFile:
    start_time = time.time()

    SH_C0 = 0.28209479177387814

    plydata = PlyData.read(ply_file_path)
    v = plydata["vertex"]
    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1))
    wxyzs = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
    colors = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    # sigmoid function
    opacities = 1.0 / (1.0 + np.exp(-v["opacity"][:, None]))

    Rs = tf.SO3(wxyzs).as_matrix()
    covariances = np.einsum(
        "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
    )
    if center:
        positions -= np.mean(positions, axis=0, keepdims=True)

    num_gaussians = len(v)
    print(
        f"PLY file with {num_gaussians=} loaded in {time.time() - start_time} seconds"
    )
    return {
        "centers": positions,
        "rgbs": colors,
        "opacities": opacities,
        "covariances": covariances,
    }
    
def create_poi_ui(server,poi_list_container,poi_index:int, poi_points:dict, sphere_handle:viser.SceneHandle):
    with poi_list_container:
        with server.gui.add_folder(f"POI_{poi_index}") as poi_container:
            gui_vector3 = server.gui.add_vector3(
                "Position",
                initial_value=poi_points[poi_index]['position'],
                step=0.25,
            )
            scale = server.gui.add_slider(
                "Scale", min=0.1, max=5.0, step=0.05, initial_value=0.1
            )
            color_r_slider = server.gui.add_slider("Color R", min=0, max=255, step=1, initial_value=int(sphere_handle.color[0]))
            color_g_slider = server.gui.add_slider("Color G", min=0, max=255, step=1, initial_value=int(sphere_handle.color[1]))
            color_b_slider = server.gui.add_slider("Color B", min=0, max=255, step=1, initial_value=int(sphere_handle.color[2]))
            delete_button = server.gui.add_button("Delete POI", color=(255, 0, 0))

    def update_scale(event):
        sphere_handle.radius = scale.value
    scale.on_update(update_scale)

    def update_color(event):
        sphere_handle.color = (color_r_slider.value, color_g_slider.value, color_b_slider.value)
    color_r_slider.on_update(update_color)
    color_g_slider.on_update(update_color)
    color_b_slider.on_update(update_color)

    # Add a transform gizmo attached to the sphere
    gizmo = server.scene.add_transform_controls(f"/{poi_points[poi_index]['name']}_gizmo", position=poi_points[poi_index]['position'], wxyz=(1.0, 0.0, 0.0, 0.0))
    def on_gizmo_update(event):
        # Update sphere position when gizmo is dragged
        sphere_handle.position = event.target.position
        gui_vector3.value = event.target.position
        poi_points[poi_index]['position'] = event.target.position
        #print("Sphere moved to:", event.position)
    gizmo.on_update(on_gizmo_update)

    @delete_button.on_click
    def _(_) -> None:
        del poi_points[poi_index]
        poi_container.remove()
        sphere_handle.remove()
        gizmo.remove()

class RelocalizationPose(Node):
    def __init__(self, viser_server: viser.ViserServer):
        super().__init__('relocalization_pose')
        self.viser_server = viser_server
        self.relocalization_pose_sub = self.create_subscription(Odometry, '/map/relocalization', self.relocalization_pose_callback, 10)
        self.global_plan_sub = self.create_subscription(nav_msgs.msg.Path, '/mapping/global_plan', self.global_plan_callback, 10)

    def relocalization_pose_callback(self, msg: Odometry):
        position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        self.viser_server.scene.add_icosphere(
            "/relocalization_pose",
            color=(255, 0, 0),
            position=position,
            radius=0.1
        )

    def global_plan_callback(self, msg: Path):
        points = []
        for pose in msg.poses:
            position = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
            points.append(position)
        if len(points) < 2:
            print("Not enough points to draw line segments")
            return
        line_segments= []
        for i in range(1, len(points)):
            line_segments.append(np.array([points[i-1], points[i]]))
        line_segments = np.array(line_segments)
        N = line_segments.shape[0]
        colors = np.zeros((N, 2, 3))
        colors[:, 0, :] = (0, 255, 0)
        colors[:, 1, :] = (0, 255, 0)
        self.viser_server.scene.add_line_segments(
            "/global_plan",
            points=np.array(line_segments),
            colors=colors,
            line_width=3
        )


def main(
    tinynav_db_path: Path,
) -> None:
    server = viser.ViserServer()
    server.scene.world_axes.visible = True
    server.scene.set_up_direction("+z")
    
    # POI management
    poi_points = {}
    poi_id_counter = 0

    if os.path.exists(f"{tinynav_db_path}/pois.json"):
        with open(f"{tinynav_db_path}/pois.json", "r") as f:
            poi_points = json.load(f)
            poi_points = {int(k): v for k, v in poi_points.items()}
            for k, v in poi_points.items():
                v['position'] = np.array(v['position'])
            poi_id_counter = max(map(lambda x: int(x), poi_points.keys())) + 1
       
    
    # Add POI management UI
    with server.gui.add_folder("Points of Interest (POI)") as _:
        add_poi_button = server.gui.add_button("Add POI Point")
        add_save_poi_button = server.gui.add_button("Save POI")

        @add_save_poi_button.on_click
        def _(_) -> None:
            with open(f"{tinynav_db_path}/pois.json", "w") as f:
                json.dump(poi_points, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)


        poi_list_container = server.gui.add_folder("POI List")
        for poi_id, poi_point in poi_points.items():
            sphere_handle = server.scene.add_icosphere(
                f"/{poi_point['name']}",
                radius=0.1,
                color=(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)),
                position=poi_point['position']
            )
            create_poi_ui(server, poi_list_container,int(poi_id), poi_points, sphere_handle)

        @add_poi_button.on_click
        def _(_) -> None:
            # Get camera position as POI location
            #camera_position = server.camera.position
            nonlocal poi_id_counter
            poi_id = poi_id_counter
            poi_id_counter += 1
            poi_name = f"POI_{poi_id}"
            # Add POI to list
            poi_points[poi_id] = {
                'id': poi_id,
                'name': poi_name,
                'position': np.random.randn(3),
            }
            sphere_handle = server.scene.add_icosphere(
                f"/{poi_name}",
                radius=0.1,
                color=(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)),
                position=poi_points[poi_id]['position']
            )
            create_poi_ui(server, poi_list_container,poi_id, poi_points, sphere_handle)
    
    poses = np.load(tinynav_db_path / "poses.npy", allow_pickle=True).item()
    rgb_camera_K = np.load(tinynav_db_path / "rgb_camera_intrinsics.npy", allow_pickle=True)
    rgb_images = shelve.open(f"{tinynav_db_path}/rgb_images")
    T_rgb_to_infra1 = np.load(tinynav_db_path / "T_rgb_to_infra1.npy", allow_pickle=True)

    splat_paths = list([Path(f"{tinynav_db_path}/splat.ply")])
    fx, _, cx, cy = rgb_camera_K[0, 0], rgb_camera_K[1, 1], rgb_camera_K[0, 2], rgb_camera_K[1, 2]
    with server.gui.add_folder("cameras") as _:
        for timestamp, rgb_pose in poses.items():
            rgb_pose = rgb_pose @ T_rgb_to_infra1 
            rgb_image = rgb_images[str(timestamp)]
            _ = rgb_image.shape[:2]
            R = vtf.SO3.from_matrix(rgb_pose[:3, :3])
            t = rgb_pose[:3, 3]
            camera_frustum = server.scene.add_camera_frustum(
                name=f"/cameras/camera_{timestamp}",
                fov=float(2 * np.arctan((cx / fx))),
                scale=0.01,
                aspect=float(cx / cy),
                image=None,
                wxyz=R.wxyz,
                position=t,
                format="jpeg",
                jpeg_quality=50
            )

            def make_on_click(camera_frustum_handle,ts):
                def handler(event):
                    img = rgb_images[str(ts)]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    camera_frustum_handle.image = img
                    clients = server.get_clients()

                    for id, client in clients.items():
                        client.camera.wxyz = camera_frustum_handle.wxyz
                        client.camera.position = camera_frustum_handle.position
                        client.camera.fov = camera_frustum_handle.fov
                return handler
            camera_frustum.on_click(make_on_click(camera_frustum, timestamp))




    
    # Load splat files
    for i, splat_path in enumerate(splat_paths):
        if splat_path.suffix == ".splat":
            splat_data = load_splat_file(splat_path, center=True)
        elif splat_path.suffix == ".ply":
            splat_data = load_ply_file(splat_path, center=False)
        else:
            raise SystemExit("Please provide a filepath to a .splat or .ply file.")

        #server.scene.add_transform_controls(f"/{i}")
        gs_handle = server.scene.add_gaussian_splats(
            f"/{i}/gaussian_splats",
            centers=splat_data["centers"],
            rgbs=splat_data["rgbs"],
            opacities=splat_data["opacities"],
            covariances=splat_data["covariances"],
        )
        remove_button = server.gui.add_button(f"Remove splat object {i}")
        @remove_button.on_click
        def _(_, gs_handle=gs_handle, remove_button=remove_button) -> None:
            gs_handle.remove()
            remove_button.remove()

    rclpy.init()
    relocalization_pose_node = RelocalizationPose(server)
    try:
        rclpy.spin(relocalization_pose_node)
        relocalization_pose_node.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    tyro.cli(main)
