#!/usr/bin/env python

import rospy
import cv2
import os
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, Quaternion
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import tf.transformations

folder_path = "./captured_images"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

class TurtlebotImageCapture:

    def __init__(self):
        self.bridge = CvBridge()
        self.image_received = False
        self.cv_image = None
        self.orientation = None
        
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.image_subscriber = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        self.odom_subscriber = rospy.Subscriber('/odom', Odometry, self.odom_callback)

    def image_callback(self, img_msg):
        self.cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        self.image_received = True

    def odom_callback(self, data):
        self.orientation = data.pose.pose.orientation  

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
        
        # Save orientation with the image
        orientation_filename = filename.replace(".png", ".txt")
        orientation_path = os.path.join(folder_path, orientation_filename)
        with open(orientation_path, 'w') as f:
            f.write(str(self.orientation.x) + "," +
                    str(self.orientation.y) + "," +
                    str(self.orientation.z) + "," +
                    str(self.orientation.w))
        
        self.image_received = False

    def execute(self):
        for i in range(8):
            self.save_image(f'image_{i}.png')
            self.rotate_bot(45)

    def move_to_orientation(self, target_orientation):
        # Convert the target and current orientations to Euler angles
        _, _, target_yaw = tf.transformations.euler_from_quaternion([target_orientation.x, target_orientation.y, target_orientation.z, target_orientation.w])
        _, _, current_yaw = tf.transformations.euler_from_quaternion([self.orientation.x, self.orientation.y, self.orientation.z, self.orientation.w])

        # Calculate the error (angular difference)
        error = target_yaw - current_yaw

        # Control parameters
        Kp = 0.5
        error_threshold = 0.1  
        max_angular_speed = 1.0  

        vel_msg = Twist()

        while abs(error) > error_threshold:
            angular_speed = Kp * error

            # Clip the angular speed to the max value
            if angular_speed > max_angular_speed:
                angular_speed = max_angular_speed
            elif angular_speed < -max_angular_speed:
                angular_speed = -max_angular_speed

            vel_msg.angular.z = angular_speed
            self.velocity_publisher.publish(vel_msg)

            # Give some time for the robot to rotate and get a new orientation value from /odom
            rospy.sleep(0.1)

            # Update the current orientation and error
            _, _, current_yaw = tf.transformations.euler_from_quaternion([self.orientation.x, self.orientation.y, self.orientation.z, self.orientation.w])
            error = target_yaw - current_yaw

       
        vel_msg.angular.z = 0
        self.velocity_publisher.publish(vel_msg)
    def adjust_to_saved_orientation(self, filename):
        # Construct the full path to the file
        filepath = os.path.join(folder_path, filename)
        
       
        with open(filepath, 'r') as f:
            x, y, z, w = map(float, f.readline().split(','))
    
        target_orientation = Quaternion(x=x, y=y, z=z, w=w)
        self.move_to_orientation(target_orientation)
if __name__ == "__main__":
    rospy.init_node('capture_images', anonymous=True)
    image_capturer = TurtlebotImageCapture()
    image_capturer.execute()
    #Here I will have the Clip Model stuff
    
    
    image_capturer.adjust_to_saved_orientation("image_4.txt")

    rospy.spin()