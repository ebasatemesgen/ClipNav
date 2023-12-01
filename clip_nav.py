#!/usr/bin/env python

import rospy
import cv2
import os
import torch
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, Quaternion
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import tf.transformations
from PIL import Image as PILImage
from transformers import CLIPProcessor, CLIPModel



folder_path = "./captured_images"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Initialize CLIP model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

class TurtlebotImageCapture:




    def __init__(self):
        self.bridge = CvBridge()
        self.image_received = False
        self.cv_image = None
        self.orientation = None
        
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.image_subscriber = rospy.Subscriber('/camera/image', Image, self.image_callback)
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
        
    def execute_and_compare(self, command):
        max_similarity = -1
        best_orientation = None

        for i in range(8):
            image_name = f'image_{i}.png'
            self.save_image(image_name)
            self.rotate_bot(45)

            similarity = self.compare_image_with_text(image_name, command)
            if similarity > max_similarity:
                max_similarity = similarity
                best_orientation = self.read_orientation_from_file(image_name)

        if best_orientation:
            self.move_to_orientation(best_orientation)

    def compare_image_with_text(self, image_name, text):
        image_path = os.path.join(folder_path, image_name)
        image = PILImage.open(image_path)

        # Process image and text with CLIP
        image_input = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        text_input = processor(text=text, return_tensors="pt")["input_ids"].to(device)

        # Get features
        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=image_input)
            text_features = model.get_text_features(input_ids=text_input)

        # Calculate similarity
        similarity = torch.cosine_similarity(text_features, image_features).cpu().numpy()[0]
        print(f"Similarity between '{text}' and '{image_name}': {similarity}")
        return similarity

    def read_orientation_from_file(self, image_name):
        orientation_file = image_name.replace(".png", ".txt")
        orientation_path = os.path.join(folder_path, orientation_file)
        with open(orientation_path, 'r') as file:
            x, y, z, w = map(float, file.readline().split(','))
            return Quaternion(x=x, y=y, z=z, w=w)

    def move_forward_for_duration(self, speed, duration):
        """
        Move the robot forward at a specified speed for a specified duration.
        """
        vel_msg = Twist()
        vel_msg.linear.x = speed
        end_time = rospy.Time.now() + rospy.Duration(duration)
        
        while rospy.Time.now() < end_time:
            self.velocity_publisher.publish(vel_msg)
        
        # Stop the robot after moving
        self.velocity_publisher.publish(Twist())

if __name__ == "__main__":
    rospy.init_node('clip_navigation', anonymous=True)
    image_capturer = TurtlebotImageCapture()

    # Input command
    command = input("Enter the command (e.g., 'Move to the brick'): ")

    image_capturer.execute_and_compare(command)

    # Define speed and duration
    speed = 0.2  # Adjust this speed as needed
    duration = 5  # Duration in seconds, adjust as needed based on your distance requirement
    image_capturer.move_forward_for_duration(speed, duration)

    rospy.spin()
