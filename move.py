#!/usr/bin/env python

import rospy
import cv2
import os
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

folder_path = "./captured_images"  
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

class TurtlebotImageCapture:

    def __init__(self):
        self.bridge = CvBridge()
        self.image_received = False
        self.cv_image = None
        
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.image_subscriber = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)

    def image_callback(self, img_msg):
        self.cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        self.image_received = True

    def rotate_bot(self, angle_degrees):
        vel_msg = Twist()

        time_for_45_degrees = 1.0  

        vel_msg.angular.z = angle_degrees * (3.14159 / 180.0) / time_for_45_degrees
        end_time = rospy.Time.now() + rospy.Duration(time_for_45_degrees)
        
        while rospy.Time.now() < end_time:
            self.velocity_publisher.publish(vel_msg)

        vel_msg.angular.z = 0
        self.velocity_publisher.publish(vel_msg)

        # Pause after rotation
        rospy.sleep(2.0)  

    def save_image(self, filename):
        while not self.image_received:
            rospy.sleep(0.1)
        file_path = os.path.join(folder_path, filename)
        cv2.imwrite(file_path, self.cv_image)
        self.image_received = False

    def execute(self):
        for i in range(8):
            self.save_image(f'image_{i}.png')
            self.rotate_bot(45)


if __name__ == "__main__":
    rospy.init_node('capture_images', anonymous=True)
    image_capturer = TurtlebotImageCapture()
    image_capturer.execute()
    rospy.spin()
