import argparse
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import os
import shutil
import threading
from datetime import datetime

class Ros2NodeManager(Node):
    def __init__(self, tinynav_db_path: str = '/tinynav/tinynav_db'):
        super().__init__('ros2_node_manager')
        self.state = 'idle'
        self.processes = {}
        self.tinynav_db_path = tinynav_db_path
        self.bag_path = os.path.join(tinynav_db_path, 'bag')
        self.map_path = os.path.join(tinynav_db_path, 'map')
        self.nav_out_path = os.path.join(tinynav_db_path, 'nav_out')
        self.ros_domain_id = os.environ.get('ROS_DOMAIN_ID', '0')
        self.get_logger().info(f'Using ROS_DOMAIN_ID={self.ros_domain_id}')
        
        self.state_pub = self.create_publisher(String, '/service/state', 10)
        self.create_subscription(String, '/service/command', self._cmd_cb, 10)
        
        self.state_timer = self.create_timer(1.0, self._pub_state)
        self.process_monitor_timer = self.create_timer(2.0, self._check_processes)
        self._child_log_files = []
        self._pub_state()
    
    def _cmd_cb(self, msg):
        cmd = msg.data.strip()
        if cmd == 'stop_navigation':
            self._stop_all()
        elif cmd == self.state:
            self._stop_all()
        elif cmd in ['realsense_bag_record', 'rosbag_build_map', 'navigation']:
            self._stop_all()
            self._start(cmd)
    
    def _start(self, mode):
        if mode == 'realsense_bag_record':
            self._start_realsense_bag_record()
        elif mode == 'rosbag_build_map':
            self._start_rosbag_build_map()
        elif mode == 'navigation':
            self._start_navigation()
        self.state = mode
        self._pub_state()
    
    def _start_realsense_bag_record(self):
        if os.path.exists(self.bag_path):
            shutil.rmtree(self.bag_path)
        
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
            '/camera/camera/color/image_rect_raw/compressed',
            '/camera/camera/imu',
            '/tf',
            '/tf_static',
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
        domain_env = {'ROS_DOMAIN_ID': self.ros_domain_id} if self.ros_domain_id is not None else {}
        
        cmd_perception = ['uv', 'run', 'python', '/tinynav/tinynav/core/perception_node.py']
        self.processes['perception'] = self._spawn(cmd_perception, extra_env=domain_env)
        
        cmd_build = [
            'uv', 'run', 'python', '/tinynav/tinynav/core/build_map_node.py',
            '--map_save_path', self.map_path,
            '--bag_file', bag_file
        ]
        self.processes['build_map'] = self._spawn(cmd_build, extra_env=domain_env)
        
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
        
        log_dir = os.path.join(self.tinynav_db_path, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.get_logger().info(f'Navigation child logs under {log_dir} (session {log_ts})')
        
        cmd_perception = ['uv', 'run', 'python', '/tinynav/tool/looper_bridge_node.py']
        self.processes['perception'] = self._spawn(
            cmd_perception, log_path=os.path.join(log_dir, f'{log_ts}_uv_run_looper_bridge_node.log'))
        
        cmd_planning = ['uv', 'run', 'python', '/tinynav/tinynav/core/planning_node.py']
        self.processes['planning'] = self._spawn(
            cmd_planning, log_path=os.path.join(log_dir, f'{log_ts}_uv_run_planning_node.log'))
        
        cmd_control = ['uv', 'run', 'python', '/tinynav/tinynav/platforms/cmd_vel_control.py']
        self.processes['control'] = self._spawn(
            cmd_control, log_path=os.path.join(log_dir, f'{log_ts}_uv_run_cmd_vel_control.log'))
        
        cmd_map = [
            'uv', 'run', 'python', '/tinynav/tinynav/core/map_node.py',
            '--tinynav_db_path', self.nav_out_path,
            '--tinynav_map_path', self.map_path,
            '--freeze_relocalization_after_first_success'
        ]
        self.processes['map'] = self._spawn(
            cmd_map, log_path=os.path.join(log_dir, f'{log_ts}_uv_run_map_node.log'))
        
        disable_manager_rosbridge = os.environ.get('DISABLE_MANAGER_ROSBRIDGE', '0') == '1'
        if disable_manager_rosbridge:
            self.get_logger().info('Skip rosbridge launch in manager (DISABLE_MANAGER_ROSBRIDGE=1)')
        else:
            rosbridge_port = os.environ.get('ROSBRIDGE_PORT', '9090')
            self.processes['rosbridge'] = self._spawn([
                'ros2', 'launch', 'rosbridge_server', 'rosbridge_websocket_launch.xml',
                f'port:={rosbridge_port}',
            ], log_path=os.path.join(log_dir, f'{log_ts}_ros2_launch_rosbridge_websocket.log'))

        topics = [
            '/tf', '/cmd_vel', '/mapping/global_plan', '/mapping/poi',
            '/mapping/poi_change', '/planning/trajectory_path',
            '/planning/occupied_voxels',
            '/slam/odometry',
            '/control/target_pose',
        ]
        cmd_bag = ['ros2', 'bag', 'record', '--max-cache-size', '2147483648', '-o', 'nav_bag'] + topics
        #self.processes['bag_record'] = self._spawn(cmd_bag)
    
    def _spawn(self, cmd, extra_env=None, log_path=None):
        env = os.environ.copy()
        env['ROS_DOMAIN_ID'] = self.ros_domain_id
        if extra_env:
            env.update(extra_env)
        popen_kw = {'env': env, 'preexec_fn': os.setsid}
        log_fp = None
        if log_path:
            log_fp = open(log_path, 'w', encoding='utf-8', errors='replace', buffering=1)
            self._child_log_files.append(log_fp)
            popen_kw['stdout'] = log_fp
            popen_kw['stderr'] = subprocess.STDOUT
        try:
            return subprocess.Popen(cmd, **popen_kw)
        except Exception:
            if log_fp:
                self._child_log_files.remove(log_fp)
                log_fp.close()
            raise
    
    def _stop_all(self):
        for fp in self._child_log_files:
            try:
                fp.close()
            except Exception:
                pass
        self._child_log_files.clear()
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
