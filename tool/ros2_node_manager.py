import argparse
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import os
import shutil
import threading

class Ros2NodeManager(Node):
    def __init__(self, tinynav_db_path: str = '/tinynav/tinynav_db'):
        super().__init__('ros2_node_manager')
        self.state = 'idle'
        self.processes = {}
        self.tinynav_db_path = tinynav_db_path
        self.bag_path = os.path.join(tinynav_db_path, 'bag')
        self.map_path = os.path.join(tinynav_db_path, 'map')
        self.nav_out_path = os.path.join(tinynav_db_path, 'nav_out')
        
        self.state_pub = self.create_publisher(String, '/service/state', 10)
        self.create_subscription(String, '/service/command', self._cmd_cb, 10)
        
        self.state_timer = self.create_timer(1.0, self._pub_state)
        self.process_monitor_timer = self.create_timer(2.0, self._check_processes)
        self._pub_state()
    
    def _cmd_cb(self, msg):
        cmd = msg.data.strip()
        if cmd == self.state:
            self._stop_all()
        elif cmd in ['realsense_sensor', 'realsense_bag_record', 'rosbag_build_map', 'navigation']:
            self._stop_all()
            self._start(cmd)
    
    def _start(self, mode):
        if mode == 'realsense_sensor':
            self._start_realsense_sensor()
        elif mode == 'realsense_bag_record':
            self._start_realsense_bag_record()
        elif mode == 'rosbag_build_map':
            self._start_rosbag_build_map()
        elif mode == 'navigation':
            self._start_navigation()
        self.state = mode
        self._pub_state()
    
    def _get_realsense_cmd(self):
        return [
            'ros2', 'launch', 'realsense2_camera', 'rs_launch.py',
            'initial_reset:=true',
            'depth_module.auto_exposure_limit:=1000',
            'tf_publish_rate:=1.0',
            'publish_tf:=true',
            'rgb_camera.color_profile:=640x360x30',
            'unite_imu_method:=2'
        ]
    
    def _start_realsense_sensor(self):
        self.processes['realsense'] = self._spawn(self._get_realsense_cmd())
    
    def _start_realsense_bag_record(self):
        if os.path.exists(self.bag_path):
            shutil.rmtree(self.bag_path)
        
        self.processes['realsense'] = self._spawn(self._get_realsense_cmd())
        
        topics = [
            '/camera/camera/infra1/camera_info',
            '/camera/camera/infra1/image_rect_raw',
            '/camera/camera/infra1/metadata',
            '/camera/camera/infra2/camera_info',
            '/camera/camera/infra2/image_rect_raw',
            '/camera/camera/infra2/metadata',
            '/camera/camera/extrinsics/depth_to_infra1',
            '/camera/camera/extrinsics/depth_to_infra2',
            '/camera/camera/accel/sample',
            '/camera/camera/gyro/sample',
            '/camera/camera/color/image_raw',
            '/camera/camera/color/camera_info',
            '/camera/camera/imu',
            '/tf',
            '/cmd_vel',
            '/mapping/global_plan',
            '/mapping/poi',
            '/mapping/poi_change',
            '/planning/trajectory_path',
            '/planning/occupied_voxels'
        ]
        cmd_bag = ['ros2', 'bag', 'record', '--max-cache-size', '2147483648', '-o', self.bag_path] + topics
        self.processes['bag_record'] = self._spawn(cmd_bag)
    
    def _start_rosbag_build_map(self):
        bag_file = os.path.join(self.bag_path, 'bag_0.db3')
        if not os.path.exists(bag_file):
            self.get_logger().warn(f'Bag file not found: {bag_file}')
            return
        
        cmd_perception = ['uv', 'run', 'python', '/tinynav/tinynav/core/perception_node.py']
        self.processes['perception'] = self._spawn(cmd_perception)
        
        cmd_build = [
            'uv', 'run', 'python', '/tinynav/tinynav/core/build_map_node.py',
            '--map_save_path', self.map_path,
            '--bag_file', bag_file
        ]
        self.processes['build_map'] = self._spawn(cmd_build)
        
        def wait_and_convert():
            proc_build = self.processes.get('build_map')
            if proc_build:
                proc_build.wait()
            cmd_convert = [
                'uv', 'run', 'python', '/tinynav/tool/convert_to_colmap_format.py',
                '--input_dir', self.map_path,
                '--output_dir', self.map_path
            ]
            subprocess.run(cmd_convert)
            self._stop_all()
            self.state = 'idle'
            self._pub_state()
        
        threading.Thread(target=wait_and_convert, daemon=True).start()
    
    def _start_navigation(self):
        if os.path.exists('nav_bag'):
            shutil.rmtree('nav_bag')
        
        cmd_perception = ['uv', 'run', 'python', '/tinynav/tinynav/core/perception_node.py']
        self.processes['perception'] = self._spawn(cmd_perception)
        
        cmd_planning = ['uv', 'run', 'python', '/tinynav/tinynav/core/planning_node.py']
        self.processes['planning'] = self._spawn(cmd_planning)
        
        cmd_control = ['uv', 'run', 'python', '/tinynav/tinynav/platforms/cmd_vel_control.py']
        self.processes['control'] = self._spawn(cmd_control)
        
        cmd_map = [
            'uv', 'run', 'python', '/tinynav/tinynav/core/map_node.py',
            '--tinynav_db_path', self.nav_out_path,
            '--tinynav_map_path', self.map_path
        ]
        self.processes['map'] = self._spawn(cmd_map)
        
        self.processes['realsense'] = self._spawn(self._get_realsense_cmd())
        
        topics = [
            '/tf', '/cmd_vel', '/mapping/global_plan', '/mapping/poi',
            '/mapping/poi_change', '/planning/trajectory_path',
            '/planning/occupied_voxels',
            '/slam/odometry',
            '/mapping/pointcloud_markers'
        ]
        cmd_bag = ['ros2', 'bag', 'record', '--max-cache-size', '2147483648', '-o', 'nav_bag'] + topics
        self.processes['bag_record'] = self._spawn(cmd_bag)
    
    def _spawn(self, cmd):
        env = os.environ.copy()
        return subprocess.Popen(cmd, env=env, preexec_fn=os.setsid)
    
    def _stop_all(self):
        for name, proc in list(self.processes.items()):
            if proc and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), 15)
                    proc.wait(timeout=2)
                except:
                    try:
                        proc.kill()
                    except:
                        pass
        self.processes.clear()
        if self.state != 'idle':
            self.state = 'idle'
            self._pub_state()
    
    def _check_processes(self):
        if self.state == 'idle' or not self.processes:
            return
        
        allow_exit = {'build_map'} if self.state == 'rosbag_build_map' else set()
        failed = []
        
        for name, proc in list(self.processes.items()):
            if not proc:
                continue
            retcode = proc.poll()
            if retcode is not None:
                if retcode == 0 and name in allow_exit:
                    self.processes.pop(name, None)
                else:
                    failed.append((name, retcode))
        
        if failed:
            for name, retcode in failed:
                self.get_logger().error(f'Process {name} exited with code {retcode}')
            self._stop_all()
            failed_names = ','.join(name for name, _ in failed)
            self.state = f'error:{failed_names}'
            self._pub_state()
    
    def _pub_state(self):
        self.state_pub.publish(String(data=self.state))
    
    def destroy_node(self):
        self._stop_all()
        super().destroy_node()

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--tinynav_db_path', type=str, default='/tinynav/tinynav_db')
    parsed_args, unknown_args = parser.parse_known_args(args)
    rclpy.init(args=unknown_args)
    node = Ros2NodeManager(tinynav_db_path=parsed_args.tinynav_db_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

