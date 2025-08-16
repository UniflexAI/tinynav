import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from tqdm import tqdm
import os
class POIEditorApp:
    def __init__(self, occupancy_path, meta_path):
        self.occupancy = np.load(occupancy_path)
        occupancy_meta = np.load(meta_path)
        self.occupancy_origin = occupancy_meta[:3]
        self.occupancy_resolution = occupancy_meta[3]
        self.pois = []

        # Load existing POIs if a store exists (prefer NPY, fallback to TXT)
        try:
            if os.path.exists("tinynav_db/pois.npy"):
                loaded = np.load("tinynav_db/pois.npy")
            elif os.path.exists("tinynav_db/pois.txt"):
                loaded = np.loadtxt("tinynav_db/pois.txt")
            else:
                loaded = None

            if loaded is not None:
                loaded = np.asarray(loaded, dtype=float)
                if loaded.size == 0:
                    self.pois = []
                else:
                    if loaded.ndim == 1:
                        # Handle single POI stored as a 1D array of length 3
                        loaded = loaded.reshape(1, 3)
                    # Keep as list of numpy arrays for downstream operations
                    self.pois = list(loaded)
        except Exception as e:
            print(f"Warning: Failed to load existing POIs: {e}")
        self.moving_poi_idx = None  # None if not moving, else index of POI being moved
        self.move_poi_mode = False  # Robust move POI mode
        self.dragging = False  # Track dragging state
        self.active_axis = None  # 'x', 'y', or 'z' if dragging along axis
        self.gizmo_length = 0.7
        self.gizmo_radius = 0.07
        self.gizmo_cone_height = 0.18
        self.gizmo_cone_radius = 0.13
        self.gizmo_geoms = []
        self.gizmo_colors = {'x': [1, 0, 0, 1], 'y': [0, 1, 0, 1], 'z': [0, 0, 1, 1]}
        self.gizmo_highlight_colors = {'x': [1, 0.7, 0.7, 1], 'y': [0.7, 1, 0.7, 1], 'z': [0.7, 0.7, 1, 1]}
        self.axis_drag_offset = 0.0  # in __init__

        # Prepare point cloud
        if self.occupancy.ndim == 3:
            points = []
            for x_idx in tqdm(range(self.occupancy.shape[0])):
                for y_idx in range(self.occupancy.shape[1]):
                    for z_idx in range(self.occupancy.shape[2]):
                        if self.occupancy[x_idx, y_idx, z_idx] == 2:
                            points.append([
                                x_idx * self.occupancy_resolution + self.occupancy_origin[0],
                                y_idx * self.occupancy_resolution + self.occupancy_origin[1],
                                z_idx * self.occupancy_resolution + self.occupancy_origin[2]
                            ])
            points = np.array(points)
        else:
            points = self.occupancy

        print("Number of points in point cloud:", points.shape[0])
        if points.shape[0] > 0:
            print("Point cloud min:", points.min(axis=0), "max:", points.max(axis=0))
        else:
            print("Warning: No points to display!")

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(points)
        colors = np.zeros((points.shape[0], 3))
        colors[:, 2] = points[:, 2] / np.max(points[:, 2]) * 255 if points.shape[0] > 0 else 0
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        # For POI placement
        self.center = np.mean(points, axis=0) if points.shape[0] > 0 else np.zeros(3)
        self.fixed_z = float(np.mean(points[:, 2])) if points.shape[0] > 0 else 0.0

        # Open3D GUI
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window("POI Editor", 1024, 768)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.add_geometry("PointCloud", self.pcd, rendering.MaterialRecord())
        if points.shape[0] > 0:
            self.scene.setup_camera(60, self.pcd.get_axis_aligned_bounding_box(), self.pcd.get_center())
        self.scene.set_on_mouse(self.on_mouse)
        self.window.add_child(self.scene)

        # Add POI button
        self.add_poi_button = gui.Button("Add POI")
        self.add_poi_button.set_on_clicked(self.add_poi)
        self.window.add_child(self.add_poi_button)

        # Move POI mode toggle button
        self.move_poi_button = gui.Button("Enable Move POI Mode")
        self.move_poi_button.set_on_clicked(self.toggle_move_poi_mode)
        self.window.add_child(self.move_poi_button)

        # POI List and Buttons
        self.poi_list = gui.ListView()
        self.poi_list.set_on_selection_changed(self.on_poi_selected)
        self.window.add_child(self.poi_list)

        self.delete_button = gui.Button("Delete Selected POI")
        self.delete_button.set_on_clicked(self.delete_selected_poi)
        self.window.add_child(self.delete_button)

        self.save_button = gui.Button("Save POIs")
        self.save_button.set_on_clicked(self.save_pois)
        self.window.add_child(self.save_button)

        self.selected_poi_idx = None
        self.update_layout()

        # Reflect any loaded POIs in the UI
        self.update_poi_list()
        self.update_poi_spheres()

        self.window.set_on_layout(self.update_layout)

    def update_layout(self, _=None):
        content_rect = self.window.content_rect
        self.scene.frame = gui.Rect(content_rect.x, content_rect.y, content_rect.width - 200, content_rect.height)
        self.add_poi_button.frame = gui.Rect(content_rect.get_right() - 200, content_rect.y, 200, 40)
        self.move_poi_button.frame = gui.Rect(content_rect.get_right() - 200, content_rect.y + 40, 200, 40)
        self.poi_list.frame = gui.Rect(content_rect.get_right() - 200, content_rect.y + 80, 200, content_rect.height - 200)
        self.delete_button.frame = gui.Rect(content_rect.get_right() - 200, content_rect.get_bottom() - 80, 200, 40)
        self.save_button.frame = gui.Rect(content_rect.get_right() - 200, content_rect.get_bottom() - 40, 200, 40)

    def toggle_move_poi_mode(self):
        self.move_poi_mode = not self.move_poi_mode
        if self.move_poi_mode:
            self.move_poi_button.text = "Disable Move POI Mode"
        else:
            self.move_poi_button.text = "Enable Move POI Mode"

    def project_point_to_screen(self, P_world):
        view_matrix = self.scene.scene.camera.get_view_matrix()
        proj_matrix = self.scene.scene.camera.get_projection_matrix()
        width, height = self.scene.frame.width, self.scene.frame.height
        # Convert to homogeneous coordinates
        P = np.array([*P_world, 1.0], dtype=np.float32)  # [x, y, z, 1]

        # Apply view matrix (world to camera)
        P_camera = view_matrix @ P
        # Apply projection matrix (camera to clip)
        P_clip = proj_matrix @ P_camera

        # Perspective divide (clip to NDC)
        if P_clip[3] == 0:
            return None, None, None  # Avoid divide by zero
        P_ndc = P_clip[:3] / P_clip[3]

        # NDC to screen space
        x_screen = (P_ndc[0] + 1.0) * 0.5 * width
        y_screen = (1.0 - P_ndc[1]) * 0.5 * height  # Y is flipped
        depth = P_ndc[2]

        return x_screen, y_screen, depth

    def add_poi(self):
        # Add a new POI at the center
        self.pois.append(self.center.copy())
        self.update_poi_list()
        self.update_poi_spheres()

    def update_poi_list(self):
        self.poi_list.set_items([
            f"{i}: ({poi[0]:.3f}\n{poi[1]:.3f}\n{poi[2]:.3f})" for i, poi in enumerate(self.pois)
        ])

    def update_poi_spheres(self):
        for i in range(1000):
            self.scene.scene.remove_geometry(f"POI_{i}")
        for axis in ['x', 'y', 'z']:
            self.scene.scene.remove_geometry(f"GIZMO_{axis}")
        for i, poi in enumerate(self.pois):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
            sphere.translate(poi)
            mat = rendering.MaterialRecord()
            mat.base_color = [1, 0, 0, 1]
            mat.shader = "defaultLit"
            self.scene.scene.add_geometry(f"POI_{i}", sphere, mat)
        # Draw axis arrows for selected POI
        if self.moving_poi_idx is not None and 0 <= self.moving_poi_idx < len(self.pois):
            self._draw_gizmo(self.pois[self.moving_poi_idx])

    def _draw_gizmo(self, center):
        # Remove old gizmo geoms
        for axis in ['x', 'y', 'z']:
            self.scene.scene.remove_geometry(f"GIZMO_{axis}")
        # Draw X, Y, Z arrows
        for axis, vec in zip(['x', 'y', 'z'], [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]):
            # Cylinder (shaft)
            cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=self.gizmo_radius, height=self.gizmo_length)
            cyl.compute_vertex_normals()
            cyl.rotate(self._axis_rotation(axis), center=[0,0,0])
            cyl.translate(center + vec * self.gizmo_length/2)
            # Cone (tip)
            cone = o3d.geometry.TriangleMesh.create_cone(radius=self.gizmo_cone_radius, height=self.gizmo_cone_height)
            cone.compute_vertex_normals()
            cone.rotate(self._axis_rotation(axis), center=[0,0,0])
            cone.translate(center + vec * (self.gizmo_length + self.gizmo_cone_height/2))
            # Color
            mat = rendering.MaterialRecord()
            if self.active_axis == axis:
                mat.base_color = self.gizmo_highlight_colors[axis]
            else:
                mat.base_color = self.gizmo_colors[axis]
            mat.shader = "defaultLit"
            self.scene.scene.add_geometry(f"GIZMO_{axis}", cyl + cone, mat)

    def _axis_rotation(self, axis):
        # Returns rotation matrix to align +Z to axis
        if axis == 'x':
            return o3d.geometry.get_rotation_matrix_from_xyz([0, -np.pi/2, 0])
        elif axis == 'y':
            return o3d.geometry.get_rotation_matrix_from_xyz([np.pi/2, 0, 0])
        else:
            return np.eye(3)

    def _project_point(self, point_world):
        camera = self.scene.scene.camera
        width, height = self.scene.frame.width, self.scene.frame.height
        view = np.asarray(camera.get_view_matrix())
        proj = np.asarray(camera.get_projection_matrix())
        point = np.array([*point_world, 1.0])
        point_camera = view @ point
        point_clip = proj @ point_camera
        point_ndc = point_clip[:3] / point_clip[3]
        screen_x = (point_ndc[0] * 0.5 + 0.5) * width
        screen_y = (1.0 - (point_ndc[1] * 0.5 + 0.5)) * height
        return np.array([screen_x, screen_y])

    def _mouse_to_fixed_z(self, x, y):
        # Convert screen (x, y) to a ray and intersect with z = self.fixed_z
        view = self.scene.frame
        camera = self.scene.scene.camera
        width, height = view.width, view.height
        # Get near and far points in world
        near = camera.unproject(x, y, 0.0, width, height)
        far = camera.unproject(x, y, 1.0, width, height)
        dir = far - near
        if abs(dir[2]) < 1e-6:
            return near  # Avoid division by zero, fallback
        t = (self.fixed_z - near[2]) / dir[2]
        point = near + t * dir
        return point

    def on_mouse(self, event):
        if not self.move_poi_mode:
            return gui.Widget.EventCallbackResult.IGNORED
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.buttons == 1:
            x, y = int(event.x), int(event.y)
            # If a POI is selected, check if an axis arrow is clicked
            if self.moving_poi_idx is not None and 0 <= self.moving_poi_idx < len(self.pois):
                center = self.pois[self.moving_poi_idx]
                axis_hit = None
                min_dist = float('inf')
                for axis, vec in zip(['x', 'y', 'z'], [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]):
                    arrow_tip = center + vec * (self.gizmo_length + self.gizmo_cone_height)
                    arrow_screen = self._project_point(arrow_tip)
                    dist = np.linalg.norm(np.array([x, y]) - arrow_screen)
                    if dist < 30 and dist < min_dist:  # 30 px threshold
                        min_dist = dist
                        axis_hit = axis
                if axis_hit:
                    self.active_axis = axis_hit
                    self.dragging = True
                    # Compute offset
                    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis_hit]
                    poi = self.pois[self.moving_poi_idx]
                    screen_x, screen_y, depth = self.project_point_to_screen(poi)
                    camera = self.scene.scene.camera
                    width, height = self.scene.frame.width, self.scene.frame.height
                    mouse_world = camera.unproject(x, y, depth, width, height)
                    self.axis_drag_offset = poi[axis_idx] - mouse_world[axis_idx]
                    self.update_poi_spheres()
                    return gui.Widget.EventCallbackResult.CONSUMED
            # Otherwise, select POI as before

            min_dist = float('inf')
            min_idx = None
            for i, poi in enumerate(self.pois):
                # it should project the point to the plane and then find the closest POI
                screen_x, screen_y, depth = self.project_point_to_screen(poi)
                dist = np.linalg.norm(np.array([x, y]) - np.array([screen_x, screen_y]))
                if dist < 5.0:
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = i
            if min_idx is not None:
                self.moving_poi_idx = min_idx
                self.dragging = False
                self.active_axis = None
                self.update_poi_spheres()
                return gui.Widget.EventCallbackResult.CONSUMED
            return gui.Widget.EventCallbackResult.CONSUMED
        elif event.type == gui.MouseEvent.Type.DRAG:
            if self.dragging and self.moving_poi_idx is not None and self.active_axis:
                x, y = int(event.x), int(event.y)
                poi = self.pois[self.moving_poi_idx]
                screen_x, screen_y, depth = self.project_point_to_screen(poi)
                camera = self.scene.scene.camera
                width, height = self.scene.frame.width, self.scene.frame.height
                new_world = camera.unproject(x, y, depth, width, height)
                axis_idx = {'x': 0, 'y': 1, 'z': 2}[self.active_axis]
                new_poi = poi.copy()
                # Add the offset!
                new_poi[axis_idx] = new_world[axis_idx] + self.axis_drag_offset
                self.pois[self.moving_poi_idx] = new_poi
                self.update_poi_list()
                self.update_poi_spheres()
                return gui.Widget.EventCallbackResult.CONSUMED
            return gui.Widget.EventCallbackResult.CONSUMED
        elif event.type == gui.MouseEvent.Type.BUTTON_UP:
            if self.dragging and self.moving_poi_idx is not None and self.active_axis:
                self.dragging = False
                self.active_axis = None
                self.update_poi_spheres()
                return gui.Widget.EventCallbackResult.CONSUMED
            return gui.Widget.EventCallbackResult.CONSUMED
        return gui.Widget.EventCallbackResult.CONSUMED

    def on_poi_selected(self, list_view, idx):
        self.selected_poi_idx = idx
        self.moving_poi_idx = idx

    def delete_selected_poi(self):
        if self.selected_poi_idx is not None and 0 <= self.selected_poi_idx < len(self.pois):
            del self.pois[self.selected_poi_idx]
            self.update_poi_list()
            self.update_poi_spheres()
            self.selected_poi_idx = None

    def save_pois(self):
        try:
            os.makedirs("tinynav_db", exist_ok=True)
            pois_arr = np.array(self.pois, dtype=float)
            np.save("tinynav_db/pois.npy", pois_arr)
            # Use a stable text format for readability
            np.savetxt("tinynav_db/pois.txt", pois_arr, fmt="%.6f")
            self._show_message_dialog("Save successful", "Saved POIs to tinynav_db/pois.npy and tinynav_db/pois.txt")
        except Exception as e:
            self._show_message_dialog("Save failed", f"Could not save POIs: {e}")

    def _show_message_dialog(self, title, message):
        dialog = gui.Dialog(title)
        layout = gui.Vert(0, gui.Margins(20, 20, 20, 20))
        layout.add_child(gui.Label(message))
        ok_button = gui.Button("OK")
        
        def _close_dialog():
            # Open3D's close_dialog takes no arguments
            self.window.close_dialog()

        ok_button.set_on_clicked(_close_dialog)
        layout.add_child(ok_button)
        dialog.add_child(layout)
        self.window.show_dialog(dialog)

if __name__ == "__main__":
    occupancy_path = "tinynav_db/occupancy_map.npy"
    meta_path = "tinynav_db/occupancy_map_meta.npy"
    app = POIEditorApp(occupancy_path, meta_path)
    gui.Application.instance.run() 
