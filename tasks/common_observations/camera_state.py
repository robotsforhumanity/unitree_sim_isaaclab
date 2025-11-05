# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
camera state
"""     

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import numpy as np
import sys
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# add the project root directory to the path, so that the shared memory tool can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from image_server.shared_memory_utils import MultiImageWriter, SharedMemoryWriter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# create the global multi-image shared memory writer

multi_image_writer = MultiImageWriter()
image_head_depth = SharedMemoryWriter(shm_name="image_head_depth",shm_size=640 * 480 * 1 * 4 + 1024)
image_head_segmentation = SharedMemoryWriter(shm_name="image_head_segmentation",shm_size=640 * 480 * 4 * 1 + 1024)

class CameraRos2Publisher(Node):
    def __init__(self):
        super().__init__('camera_publisher_node')
        self.bridge = CvBridge()
        self.pub_head = self.create_publisher(Image, '/camera/head/image_raw', 10)
        self.pub_left = self.create_publisher(Image, '/camera/left/image_raw', 10)
        self.pub_right = self.create_publisher(Image, '/camera/right/image_raw', 10)
        self.pub_camera = self.create_publisher(Image, '/camera/perspective/image_raw', 10)
        self.get_logger().info("Camera ROS2 publishers ready.")

    def publish_images(self, images_rgb: dict):
        for name, img_np in images_rgb.items():
            msg = self.bridge.cv2_to_imgmsg(np.uint8(img_np), encoding='rgb8')
            if name == "head":
                self.pub_head.publish(msg)
            elif name == "left":
                self.pub_left.publish(msg)
            elif name == "right":
                self.pub_right.publish(msg)
            elif name == "perspective":
                self.pub_camera.publish(msg)
            else:
                # Publicar en un tópico genérico
                self.create_publisher(Image, f'/camera/{name}/image_raw', 10).publish(msg)

# Creamos un nodo ROS2 global (inicializa una sola vez)
rclpy.init(args=None)
ros2_publisher = CameraRos2Publisher()
def get_camera_image(
    env: ManagerBasedRLEnv,
) -> dict:
    # pass
    """get multiple camera images and write them to shared memory
    
    Args:
        env: ManagerBasedRLEnv - reinforcement learning environment instance
    
    Returns:
        dict: dictionary containing multiple camera images
    """
    # get the camera images
    images_rgb = {}
    #images_depth = {}
    #images_segmentation = {}
    
    # env.sim.render()
    
    # Head camera (front camera)
    if "front_camera" in env.scene.keys():
        head_image = env.scene["front_camera"].data.output["rgb"][0]  # [batch, height, width, 3]
        #head_depth = env.scene["front_camera"].data.output["depth"][0]
        #head_segmentation = env.scene["front_camera"].data.output["semantic_segmentation"][0]
        images_rgb["head"] = head_image.cpu().numpy()
        #images_depth["depth"] = head_depth.cpu().numpy()
        #images_segmentation["segmentation"] = head_segmentation.cpu().numpy()
        #print(head_depth.dtype, head_depth.min(), head_depth.max())
    
    # Left camera (left wrist camera)
    if "left_wrist_camera" in env.scene.keys():
        left_image = env.scene["left_wrist_camera"].data.output["rgb"][0]
        images_rgb["left"] = left_image.cpu().numpy()
    
    # Right camera (right wrist camera)  
    if "right_wrist_camera" in env.scene.keys():
        right_image = env.scene["right_wrist_camera"].data.output["rgb"][0]
        images_rgb["right"] = right_image.cpu().numpy()

    if "perspective_camera" in env.scene.keys():
        perspective_image = env.scene["perspective_camera"].data.output["rgb"][0]
        images_rgb["perspective"] = perspective_image.cpu().numpy()
    
    # if no camera with the specified name is found, try other common camera names
    if not images_rgb:
        # try to find other possible camera names
        available_cameras = [name for name in env.scene.keys() if "camera" in name.lower()]
        print(f"[camera_state] No standard cameras found. Available cameras: {available_cameras}")
        
        # if there are available cameras, use the first three as head, left, right
        for i, camera_name in enumerate(available_cameras[:3]):
            camera_image = env.scene[camera_name].data.output["rgb"][0]
            
            if i == 0:
                images_rgb["head"] = camera_image.cpu().numpy()
            elif i == 1:
                images_rgb["left"] = camera_image.cpu().numpy()
            elif i == 2:
                images_rgb["right"] = camera_image.cpu().numpy()
    
    # write the multi-image data to shared memory
    if images_rgb:
        success = multi_image_writer.write_images(images_rgb)
  
        ros2_publisher.publish_images(images_rgb)

        rclpy.spin_once(ros2_publisher, timeout_sec=0.0)

    #if images_depth:
    #    success_depth = image_head_depth.write_image(images_depth["depth"])

    #if images_segmentation:
    #    success_segmentation = image_head_segmentation.write_image(images_segmentation["segmentation"])
    else:
        print("[camera_state] No camera images found in the environment")
    
    return torch.zeros((1, 480, 640, 3))

