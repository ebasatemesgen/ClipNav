    #!/usr/bin/env python

    import rospy
    import cv2
    import os
    import torch
    import numpy as np
    import math
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    import tf
    from PIL import Image as PILImage
    from transformers import CLIPProcessor, CLIPModel

    # Folder for captured images
    folder_path = "./captured_images"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Initialize CLIP model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    class PID:
        def __init__(self, P, I, D):
            self.Kp = P
            self.Ki = I
            self.Kd = D
            self.integral = 0
            self.previous_error = 0

        def update(self, error, delta_time):
            self.integral += error * delta_time
            derivative = (error - self.previous_error) / delta_time
            self.previous_error = error
            return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

    class TurtlebotImageCapture:
        def __init__(self):
            self.bridge = CvBridge()
            self.image_received = False
            self.cv_image = None
            self.current_odom = None

            rospy.init_node('clip_navigation', anonymous=True)
            self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
            self.odom_subscriber = rospy.Subscriber('/odom', Odometry, self.odom_callback)
            self.image_subscriber = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)

        def image_callback(self, img_msg):
            self.cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            self.image_received = True

        def odom_callback(self, data):
            self.current_odom = data

        def get_yaw_from_odom(self):
            orientation_q = self.current_odom.pose.pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            (_, _, yaw) = tf.transformations.euler_from_quaternion(orientation_list)
            return yaw

        def rotate_robot(self, target_angle_deg):
            pid = PID(P=0.6, I=0.01, D=0.1)  # Tune these values
            rate = rospy.Rate(10)  # 10 Hz
            tolerance = 0.01  # Radians

            target_angle_rad = math.radians(target_angle_deg)
            current_yaw = self.get_yaw_from_odom()
            target_yaw = current_yaw + target_angle_rad

            # Normalize the target angle
            target_yaw = math.atan2(math.sin(target_yaw), math.cos(target_yaw))

            vel_msg = Twist()

            while not rospy.is_shutdown():
                current_yaw = self.get_yaw_from_odom()
                error = target_yaw - current_yaw
                error = math.atan2(math.sin(error), math.cos(error))  # Normalize error

                if abs(error) < tolerance:
                    break

                control = pid.update(error, 1.0 / 10.0)  # Assuming rate is 10 Hz
                vel_msg.angular.z = control
                self.velocity_publisher.publish(vel_msg)
                rate.sleep()

            # Stop the robot
            vel_msg.angular.z = 0
            self.velocity_publisher.publish(vel_msg)
            
        def control_robot(self, linear_velocity, distance):
            initial_position = self.current_odom.pose.pose.position
            vel_msg = Twist()
            vel_msg.linear.x = linear_velocity

            while not rospy.is_shutdown():
                current_position = self.current_odom.pose.pose.position
                distance_moved = math.sqrt(
                    (current_position.x - initial_position.x) ** 2 +
                    (current_position.y - initial_position.y) ** 2
                )

                if distance_moved >= distance:
                    break
                self.velocity_publisher.publish(vel_msg)

            # Stop the robot
            vel_msg.linear.x = 0
            self.velocity_publisher.publish(vel_msg)
            
        def save_image(self, filename):
            while not self.image_received:
                rospy.sleep(0.1)
            file_path = os.path.join(folder_path, filename)
            cv2.imwrite(file_path, self.cv_image)
            self.image_received = False

        def execute_and_compare(self, command):
            max_similarity = -1
            best_angle = None
            num_steps = 8
            angle_increment = 360 / num_steps
            captured_angles = []
            current_angle = 0
            for i in range(num_steps):
                # Rotate robot to the next increment and stop

                # Capture and save image
                image_name = f'image_{i}.png'
                self.save_image(image_name)
                rospy.sleep(1)  # Pause to stabilize before capturing image
                # Compare image with text
                similarity = self.compare_image_with_text(image_name, command)
                rospy.loginfo(f"Captured at step {i} - Similarity: {similarity}")
                
                
                
                

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_angle = current_angle
                
                rospy.loginfo(f"Captured at {current_angle}")
                current_angle = (i+1) * angle_increment
                
                captured_angles.append(current_angle)


                self.rotate_robot(angle_increment)
            

            
            # Log captured angles
            rospy.loginfo(f"This is the all the angle Captured angles: {captured_angles}")

            rospy.loginfo(f"this is my current angle: {current_angle}")
            # Rotate to best angle
            if best_angle is not None:
                rospy.loginfo(f"Rotating to best angle: {best_angle} degrees, this is the direction the robot is rotating at {best_angle - current_angle}")
                self.rotate_robot(best_angle - current_angle)  # Adjust to rotate from the current angle

            # Move forward towards the target
            move_distance = 0.5  # Set the distance to move towards the target
            self.control_robot(0.2, move_distance)  # Adjust linear speed as needed


        def compare_image_with_text(self, image_name, text):
            image_path = os.path.join(folder_path, image_name)
            image = PILImage.open(image_path)
            image_input = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
            text_input = processor(text=text, return_tensors="pt")["input_ids"].to(device)

            with torch.no_grad():
                image_features = model.get_image_features(pixel_values=image_input)
                text_features = model.get_text_features(input_ids=text_input)

            similarity = torch.cosine_similarity(text_features, image_features).cpu().numpy()[0]
            rospy.loginfo(f"Similarity between '{text}' and '{image_name}': {similarity}")
            return similarity

    if __name__ == "__main__":
        try:
            image_capturer = TurtlebotImageCapture()
            while True:
                command = input("Enter the command (e.g., 'Move to the brick'), or type 'exit' to quit: ")
                if command.lower() == 'exit':
                    break
                image_capturer.execute_and_compare(command)
        except rospy.ROSInterruptException:
            rospy.loginfo("Navigation node terminated.")
        except KeyboardInterrupt:
            rospy.loginfo("Program interrupted by user, shutting down.")
        finally:
            rospy.loginfo("Shutting down the navigation node.")
