"""Attach synchronized observed recovery to the existing velocity controller."""
import math
import json
import time
import numpy as np
import message_filters
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from scipy.spatial.transform import Rotation
from tinynav.core.recovery.runtime import RecoveryRuntime


class RecoveryAdapter:
    def __init__(self, node, robot):
        self.node = node
        self.runtime = RecoveryRuntime(robot)
        self.robot = robot
        self.camera = None
        self.pose_at = None
        self.pose_xy = None
        self.pose_yaw = None
        self.pose_stamp = None
        self.bridge = CvBridge()
        self.status_pub = node.create_publisher(String,'/navigation/recovery/status',10)
        node.create_subscription(CameraInfo,'/camera/camera/infra2/camera_info',self.info,10)
        node.create_subscription(Odometry,'/control/target_pose',self.target,10)
        self.depth = message_filters.Subscriber(node,Image,'/slam/depth')
        self.odom = message_filters.Subscriber(node,Odometry,'/slam/odometry_visual')
        self.sync = message_filters.TimeSynchronizer([self.depth,self.odom],10)
        self.sync.registerCallback(self.observe)
        self.last_status = None
        self.last_active = False
        self.target_anchor = None

    def info(self, msg):
        if msg.k[0]>0 and msg.k[4]>0:
            self.camera = {'fx':msg.k[0],'fy':msg.k[4],'cx':msg.k[2],'cy':msg.k[5],'ground_z':0}

    def target(self, msg):
        p=msg.pose.pose.position
        if msg.header.frame_id not in ('world','odom',''):
            self.runtime.reset();self.pose_stamp=None;return
        target=[p.x,p.y,p.z]
        if not all(math.isfinite(v) for v in target):
            self.runtime.reset();return
        anchor=getattr(self,'target_anchor',None)
        if self.runtime.target is None or anchor is None or math.dist(anchor,target)>.25:
            self.runtime.reset(target);self.pose_stamp=None;self.target_anchor=list(target)
        else:
            self.runtime.target=list(target)

    def observe(self, depth_msg, odom_msg):
        if self.camera is None:return
        stamp=odom_msg.header.stamp.sec+odom_msg.header.stamp.nanosec*1e-9
        ros_now=self.node.get_clock().now().nanoseconds*1e-9
        if not -.1 <= ros_now-stamp <= .5:return
        if self.pose_stamp is not None and stamp<=self.pose_stamp:return
        self.pose_stamp=stamp
        p=odom_msg.pose.pose.position;q=odom_msg.pose.pose.orientation
        quat=np.array([q.x,q.y,q.z,q.w],dtype=float)
        if not np.isfinite(quat).all() or np.linalg.norm(quat)<.5:return
        T=np.eye(4);T[:3,:3]=Rotation.from_quat(quat).as_matrix();T[:3,3]=[p.x,p.y,p.z]
        if depth_msg.encoding not in ('32FC1','16UC1','mono16'):return
        depth=self.bridge.imgmsg_to_cv2(depth_msg,desired_encoding='passthrough').astype(np.float32)
        if depth_msg.encoding in ('16UC1','mono16'):depth*=.001
        depth[(~np.isfinite(depth)) | (depth<=0) | (depth>8)]=0
        center=T[:3,3]-T[:3,:3]@self.robot.cam_offset_3d
        forward=T[:3,2]
        if not np.isfinite(center).all():return
        stride=max(1,int(np.ceil(max(depth.shape[1]/160,depth.shape[0]/100))))
        camera={**self.camera,'fx':self.camera['fx']/stride,'fy':self.camera['fy']/stride,
                'cx':self.camera['cx']/stride,'cy':self.camera['cy']/stride,
                'ground_z':T[2,3]+self.robot.obstacle.robot_z_bottom-.05}
        now=time.monotonic()
        self.runtime.observe(depth[::stride,::stride],T,camera,now)
        self.pose_at=now;self.pose_xy=center[:2].tolist()
        self.pose_yaw=float(np.degrees(np.arctan2(forward[1],forward[0])))

    def command(self, active, paused):
        if active and not self.last_active:
            self.runtime.reset(self.runtime.target);self.pose_stamp=None
        self.last_active=active
        now=time.monotonic()
        age=now-self.pose_at if self.pose_at is not None else float('inf')
        value=self.runtime.tick(self.pose_xy or [0,0],self.pose_yaw or 0,now,active,paused,age)
        status=self.runtime.status()
        encoded=json.dumps(status,allow_nan=False)
        if encoded!=self.last_status:
            self.status_pub.publish(String(data=encoded));self.last_status=encoded
        if value is None:return None
        cmd=Twist();cmd.linear.x,cmd.angular.z=map(float,value)
        return cmd
